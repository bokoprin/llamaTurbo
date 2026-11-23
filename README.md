# llamaTurbo

RTX 4080 (`sm_89`, 16 GB VRAM) 上で **Gemma 4 31B QAT**（`gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`、約17 GB）を部分オフロードで高速推論するための `llama.cpp` 最適化設定・ベンチマーク・Nsightプロファイル記録をまとめたリポジトリです。

## ベンチマーク結果（2回計測中央値・`n_predict=64`・`cache_prompt=False`）

- 検証環境: NVIDIA GeForce RTX 4080 (16 GB VRAM) + Intel Core i9-14900KF (24C/32T, 62 GB RAM)
- コンテキスト設定: `32,768` (`--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on`)

| 入力長 | 指標 | 元構成 (`base86`) | v1 (Codex最終候補) | **v2 (llamaTurbo)** | 元構成比 | v1比 |
|---|---|---:|---:|---:|---:|---|
| **Short** (115 tok) | Prefill (tok/s) | 192.6 | 184.2 | **228.7** | +18.7% | +24.2% |
| | Decode (tok/s) | 9.68 | 16.05 | **17.52** | +81.0% | +9.2% |
| | 全体時間 | 7.22秒 | 4.66秒 | **4.21秒** | **-41.7%** | -9.7% |
| **4K** (4,004 tok) | Prefill (tok/s) | 793.6 | 1,008.9 | **1,134.7** | +43.0% | +12.5% |
| | Decode (tok/s) | 8.83 | 15.12 | **15.54** | +76.0% | +2.8% |
| | 全体時間 | 12.26秒 | 8.21秒 | **7.66秒** | **-37.5%** | -6.8% |
| **12K** (11,924 tok) | Prefill (tok/s) | 806.9 | 1,033.9 | **1,189.7** | +47.4% | +15.1% |
| | Decode (tok/s) | 8.26 | 14.25 | **16.67** | +101.8% | +17.0% |
| | 全体時間 | 22.63秒 | 16.18秒 | **14.02秒** | **-38.0%** | **-13.4%** |

## 高速化のポイント

1. **MTP 補助モデルの `Q4_0` 化 + GPU層数拡大 (`47/61 → 48/61`)**
   - `mtp-gemma-4-31B-it-Q8_0.gguf`（約476 MiB）から `mtp-gemma-4-31B-it-Q4_0.gguf`（267 MiB）へ変更し、`--fit-target 1024` と組み合わせることで、本体の GPU オフロードを **48/61層**（+ MTP `5/5` 層）に増やしつつ、最低空き VRAM **427.0 MiB** を確保。
2. **ホストメモリピン留め (`--load-mode none` + `GGML_CUDA_REGISTER_HOST=1`)**
   - CPU側に残る13層の重み・バッファを `cudaHostRegister` でページロックして PCIe DMA 転送を高速化。
   - Nsight（12K Prefill）計測において、CPU→GPU 転送時間が **3,785.3 ms → 2,437.4 ms**（-35.6%）、転送量が **47.4 GB → 43.88 GB**、`cudaMemcpyAsync` の CUDA API 時間比率が **36.3% → 1.7%** へ減少。
3. **MTP 投機デコードの最適化 (`--spec-draft-n-max 5 --spec-draft-n-ubatch 64`)**
   - `Q4_0` MTP でも高い受理率（12K で `98/138` = 71.0%）を維持し、Decode 速度を向上。
4. **安全性・品質検証**
   - 32K コンテキストで **26,952 トークン**の長文プロンプト完走（26.043秒、空き VRAM 427.0 MiB）。
   - JSON出力・計算スモークチェック合格（`quality_ok: true`）。

## クイックスタート

1. `llama.cpp` を `sm_89` 向けにビルド（任意・cuBLAS版推奨）:
   ```bash
   ./scripts/build-sm89.sh
   ```
2. MTP 補助モデル（`mtp-gemma-4-31B-it-Q4_0.gguf`）を `unsloth/gemma-4-31B-it-qat-GGUF` から取得:
   ```bash
   curl -fL -o mtp-gemma-4-31B-it-Q4_0.gguf \
     https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/resolve/main/MTP/mtp-gemma-4-31B-it-Q4_0.gguf
   ```
3. サーバー起動:
   ```bash
   MODEL_PATH=/path/to/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf \
   DRAFT_PATH=/path/to/mtp-gemma-4-31B-it-Q4_0.gguf \
   ./scripts/run-gemma31-rtx4080-v2.sh
   ```

## ディレクトリ構成

```text
llamaTurbo/
├── README.md
├── LICENSE
├── .gitignore
├── scripts/
│   ├── build-sm89.sh                 # RTX 4080 (sm_89) 向けビルドスクリプト
│   └── run-gemma31-rtx4080-v2.sh     # 最適化済み起動スクリプト
├── benchmarks/
│   ├── bench_v2.py                   # ベンチマーク・VRAM監視ハーネス
│   ├── profile_v2.py                 # Nsight 12K プロファイル取得スクリプト
│   └── specs/                        # 各フェーズの探索条件 JSON
└── results/
    ├── results-v2.json               # 最終2回中央値 + 26K耐性 + 品質結果
    ├── phase_a_results.json          # Phase A (MTP Q4_0, fit-target, threads, n-max)
    ├── phase_a2_results.json         # Phase A2 (ubatch, threads-batch, -ot 検証)
    ├── phase_a3_results.json         # Phase A3 (GGML_CUDA_REGISTER_HOST=1 検証)
    ├── phase_b_results.json          # Phase B (cuBLAS vs FORCE_MMQ A/B 比較)
    └── nsys-prefill-v2.stats.txt     # Nsight 12K Prefill 統計レポート
```
