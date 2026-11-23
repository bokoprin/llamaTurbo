#!/usr/bin/env python3
"""Benchmark runner for Gemma 4 26B A4B QAT optimization on RTX 4080."""
import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time

VERIF_DIR = pathlib.Path('/home/shinsuke/llm-benchmark-setup/gemma31-rtx4080-20261123/verification')
ROOT_DIR = pathlib.Path('/home/shinsuke/llm-benchmark-setup/gemma26b-rtx4080')
OPT_DIR = ROOT_DIR / 'opt'
sys.path.insert(0, str(VERIF_DIR))

import benchmark as bm  # noqa: E402

MODEL_PATH = '/home/shinsuke/llm-benchmark-setup/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
ORIG_BIN = '/home/shinsuke/llm-benchmark-setup/src/llama.cpp/build/bin/llama-server'
BASE_ARGS = [
    '-m', MODEL_PATH,
    '--host', '127.0.0.1',
    '--port', '18192',
    '--ctx-size', '32768',
    '--cache-type-k', 'q4_0',
    '--cache-type-v', 'q4_0',
    '--flash-attn', 'on',
    '--parallel', '1',
    '-lv', '4',
]


def stop_port_18191():
    """Stop any llama-server running on port 18191."""
    subprocess.run(['pkill', '-x', 'llama-server'], check=False)
    time.sleep(2)


def restore_port_18191():
    """Restore original 31B llama-server on port 18191 if not already running."""
    try:
        if bm.api(18191, '/health', timeout=2).get('status') == 'ok':
            return
    except Exception:
        pass
    argv = [
        ORIG_BIN,
        '-m', '/home/shinsuke/llm-benchmark-setup/models/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf',
        '--host', '127.0.0.1',
        '--port', '18191',
        '--ctx-size', '32768',
        '--cache-type-k', 'q4_0',
        '--cache-type-v', 'q4_0',
        '--flash-attn', 'on',
        '--parallel', '1',
        '--fit', 'on',
    ]
    env = os.environ.copy()
    cwd = pathlib.Path('/home/shinsuke/llm-benchmark-setup')
    proc = bm.start(argv, env, cwd, OPT_DIR / 'original-restored.log')
    bm.ready(18191, proc)


class VramMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.min_free_mib = 999999.0
        self.max_used_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.free,memory.used', '--format=csv,noheader,nounits'],
                    text=True,
                    timeout=3,
                ).strip().splitlines()[0]
                free_s, used_s = [float(x.strip()) for x in out.split(',')]
                if free_s < self.min_free_mib:
                    self.min_free_mib = free_s
                if used_s > self.max_used_mib:
                    self.max_used_mib = used_s
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        return {
            'min_free_mib': round(self.min_free_mib, 1),
            'max_used_mib': round(self.max_used_mib, 1),
        }


def parse_log_placement(log_path: pathlib.Path):
    txt = log_path.read_text(errors='replace')
    offload = re.findall(r'offloaded (\d+/\d+) layers to GPU', txt)
    fit_free = re.findall(r'fit_params:.*?free:\s*([\d.]+)\s*MiB', txt)
    return {
        'offloaded': offload,
        'fit_free_mib': fit_free[-1] if fit_free else None,
    }


def run_variant(name, server_bin, extra_flags, env_overrides=None, repeats=1, do_confirm=False, no_fit=False):
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    fit_args = [] if no_fit else ['--fit', 'on']
    argv = [server_bin] + BASE_ARGS + fit_args + extra_flags
    log_path = OPT_DIR / f'server-{name}.log'
    if log_path.exists():
        log_path.unlink()

    mon = VramMonitor(interval=0.5)
    mon.start()
    child = bm.start(argv, env, ROOT_DIR, log_path)
    try:
        bm.ready(18192, child)
        rows = []
        for trial in range(1, repeats + 1):
            for case, prompt in zip(('short', '4k', '12k'), bm.workloads()):
                resp, wall = bm.request(18192, prompt, 64)
                rows.append({
                    'trial': trial,
                    'case': case,
                    'wall_s': round(wall, 3),
                    'timings': resp['timings'],
                })
        summary = bm.summarize(rows)
        confirm_info = {}
        if do_confirm:
            q_ok = bm.quality(18192, name)
            long_p = (
                'Read the notes and summarize.\n'
                + ''.join(
                    f'Line {i}: Heavy workloads need observability and realistic performance measurements.\n'
                    for i in range(1650)
                )
                + 'Answer briefly: '
            )
            c_resp, c_wall = bm.request(18192, long_p, 8)
            confirm_info = {
                'quality_ok': q_ok,
                'long_prompt_n': c_resp['timings']['prompt_n'],
                'long_wall_s': round(c_wall, 3),
                'long_prefill': round(c_resp['timings']['prompt_per_second'], 2),
            }
        vram = mon.stop()
        placement = parse_log_placement(log_path)
        return {
            'name': name,
            'server_bin': server_bin,
            'extra_flags': extra_flags,
            'env': env_overrides or {},
            'metrics': summary,
            'vram': vram,
            'placement': placement,
            'confirm': confirm_info,
        }
    finally:
        mon.stop()
        bm.stop_child(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('spec_json', help='JSON file with list of variants')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--confirm', action='store_true')
    ap.add_argument('--out', default=str(OPT_DIR / 'results-26b.json'))
    ap.add_argument('--keep-18191-stopped', action='store_true')
    args = ap.parse_args()

    specs = json.loads(pathlib.Path(args.spec_json).read_text())
    stop_port_18191()
    results = []
    try:
        for spec in specs:
            name = spec['name']
            print(f'=== Running {name} (repeats={args.repeats}) ===', flush=True)
            t0 = time.monotonic()
            try:
                res = run_variant(
                    name=name,
                    server_bin=spec.get('server_bin', ORIG_BIN),
                    extra_flags=spec['extra_flags'],
                    env_overrides=spec.get('env', {}),
                    repeats=args.repeats,
                    do_confirm=args.confirm,
                    no_fit=spec.get('no_fit', False),
                )
                res['elapsed_s'] = round(time.monotonic() - t0, 1)
                results.append(res)
                m12 = res['metrics']['12k']
                m4 = res['metrics']['4k']
                ms = res['metrics']['short']
                print(
                    f"  OK {name}: 12k prefill={m12['prefill']} decode={m12['decode']} wall={m12['wall_s']}s | "
                    f"4k={m4['wall_s']}s | short={ms['wall_s']}s | min_free={res['vram']['min_free_mib']} MiB | "
                    f"offload={res['placement']['offloaded']}",
                    flush=True,
                )
            except Exception as exc:
                print(f'  FAIL {name}: {exc}', flush=True)
                results.append({'name': name, 'error': str(exc)})
            pathlib.Path(args.out).write_text(json.dumps(results, indent=2) + '\n')
    finally:
        if not args.keep_18191_stopped:
            restore_port_18191()


if __name__ == '__main__':
    main()
