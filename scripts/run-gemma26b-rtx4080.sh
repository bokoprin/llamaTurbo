#!/usr/bin/env bash
set -euo pipefail

# Optimized Gemma 4 26B A4B QAT (MoE) launcher for RTX 4080 (16 GB VRAM, sm_89)
LLAMA_BIN="${LLAMA_BIN:-/home/shinsuke/llm-benchmark-setup/src/llama.cpp/build/bin/llama-server}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-/home/shinsuke/llm-benchmark-setup/gemma31-rtx4080-20261123/cuda-sm89}"
MODEL_PATH="${MODEL_PATH:-/home/shinsuke/llm-benchmark-setup/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
DRAFT_PATH="${DRAFT_PATH:-/home/shinsuke/llm-benchmark-setup/gemma26b-rtx4080/opt/mtp-gemma-4-26B-A4B-it-Q4_0.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18191}"
CTX_SIZE="${CTX_SIZE:-32768}"

export LD_LIBRARY_PATH="$CUDA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$LLAMA_BIN" \
  -m "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size "$CTX_SIZE" \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --parallel 1 \
  --fit off \
  -ngl 99 \
  --ubatch-size 1024 \
  --spec-type draft-mtp \
  --spec-draft-model "$DRAFT_PATH" \
  --spec-draft-n-max 7 \
  --spec-draft-n-ubatch 64 \
  "$@"
