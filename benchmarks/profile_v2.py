#!/usr/bin/env python3
"""Capture 12K prefill nsys profile for v2 winner and restore port 18191."""
import pathlib
import subprocess
import sys
import time

VERIF_DIR = pathlib.Path('/home/shinsuke/llm-benchmark-setup/gemma31-rtx4080-20261123/verification')
ROOT_DIR = pathlib.Path('/home/shinsuke/llm-benchmark-setup/gemma31-rtx4080-20261123')
OPT_DIR = ROOT_DIR / 'opt-v2'
sys.path.insert(0, str(VERIF_DIR))
sys.path.insert(0, str(OPT_DIR))

import benchmark as bm  # noqa: E402
import bench_v2 as b2   # noqa: E402


def main():
    saved = b2.stop_port_18191()
    try:
        env = dict(**b2.os.environ)
        env['LD_LIBRARY_PATH'] = str(ROOT_DIR / 'cuda-sm89')
        env['GGML_CUDA_REGISTER_HOST'] = '1'
        extra = [
            '--fit', 'on',
            '--fit-target', '1024',
            '--ubatch-size', '1024',
            '--load-mode', 'none',
            '--spec-type', 'draft-mtp',
            '--spec-draft-model', str(OPT_DIR / 'mtp-gemma-4-31B-it-Q4_0.gguf'),
            '--spec-draft-n-max', '5',
            '--spec-draft-n-ubatch', '64',
        ]
        argv = [b2.ORIG_BIN] + b2.BASE_ARGS + extra
        out_base = OPT_DIR / 'nsys-prefill-v2'
        cmd = [
            'nsys', 'profile',
            '--trace=cuda', '--sample=none', '--cpuctxsw=none',
            '--delay=9', '--duration=26', '--kill=sigterm', '--force-overwrite=true',
            '--output=' + str(out_base),
            *argv,
        ]
        t0 = time.monotonic()
        child = bm.start(cmd, env, ROOT_DIR, out_base.with_suffix('.log'))
        try:
            bm.ready(18192, child)
            time.sleep(max(0.0, 9.8 - (time.monotonic() - t0)))
            resp, wall = bm.request(18192, bm.workloads()[-1], 1)
            print('profile_probe:', round(wall, 3), resp['timings'], flush=True)
            child.wait(timeout=48)
            rep = out_base.with_suffix('.nsys-rep')
            reports = 'cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum'
            with out_base.with_suffix('.stats.txt').open('w') as f:
                subprocess.run(
                    ['nsys', 'stats', '--report', reports, str(rep)],
                    stdout=f, stderr=subprocess.STDOUT, timeout=45, check=True,
                )
            print('nsys stats written:', out_base.with_suffix('.stats.txt'), flush=True)
        finally:
            bm.stop_child(child)
    finally:
        b2.restore_port_18191(saved)
        r = bm.api(18191, '/completion', {'prompt': 'Hello', 'n_predict': 4, 'temperature': 0}, timeout=30)
        print('port 18191 restored OK, content:', repr(r.get('content', '')), flush=True)


if __name__ == '__main__':
    main()
