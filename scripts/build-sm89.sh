#!/usr/bin/env bash
set -euo pipefail

# Reproducible RTX 4080 (sm_89) build script for llama.cpp (ggml-cuda + llama-server)
SRC_DIR="${SRC_DIR:-$HOME/llm-benchmark-setup/src/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$PWD/build-sm89}"
NVCC_BIN="${NVCC_BIN:-/usr/local/cuda/bin/nvcc}"
JOBS="${JOBS:-$(nproc)}"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_FORCE_MMQ=OFF \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16" \
  -DGGML_BUILD_TESTS=OFF \
  -DGGML_BUILD_EXAMPLES=OFF

cmake --build "$BUILD_DIR" --target llama-server ggml-cuda -j"$JOBS"
