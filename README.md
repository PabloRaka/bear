# 🐻 Mesosfer Bear AI

**Mesosfer Bear AI** is a state-of-the-art, high-efficiency autoregressive language model built on a Llama-style decoder-only transformer architecture. It features a custom **Kimi-K3 XTML BPE Tokenizer** (60K vocabulary) with a high-performance **Rust backend (`tiktoken`)**, a multi-domain dataset streaming pipeline, a phased **4-Stage training & evaluation curriculum**, a **Hugging Face benchmark evaluation suite**, and an **interactive real-time streaming Chat CLI**.

---

## 🌟 Key Architecture & Highlights

- **Model Architecture**: Llama-style decoder-only Transformer (~240M parameters, 16 layers, `d_model=1024`, `ffn_hidden=2816`).
- **Attention & Scaling**: Grouped Query Attention (GQA, 4:1 query/KV ratio), Rotary Position Embeddings (RoPE, $\theta=10000$), SwiGLU non-linear activations, RMSNorm pre-normalization.
- **Hardware Acceleration**: Automatic FlashAttention-2 / PyTorch SDPA kernel selection, mixed-precision AMP (BF16 / FP16), optimized for NVIDIA CUDA & AMD Instinct MI300X (ROCm 6.0+).
- **Custom Tokenizer**: 60,000 vocabulary size based on Kimi-K3 BPE with Rust `tiktoken` acceleration (**4.1M+ tokens/sec**, 29× faster than pure-Python), competitive **3.43 bytes/token** compression ratio, and native XTML chat template markup support (`<|open|>message...<|close|>`).
- **4-Stage Training Pipeline**: Pre-training $\rightarrow$ Continual Pre-Training (CPT) $\rightarrow$ Supervised Fine-Tuning (SFT) $\rightarrow$ Safety Alignment, with mandatory stage-by-stage benchmark gates.
- **Evaluation & Benchmarks**: Real-time evaluation during training (nanoGPT-style Cross-Entropy Loss, Perplexity, Bits-Per-Byte / BPB) plus Hugging Face standardized stage benchmarks.

---

## 📦 Installation & Setup (via `uv`)

We use [`uv`](https://github.com/astral-sh/uv) for fast, deterministic, and reproducible environment synchronization.

```bash
# 1. Clone the repository
git clone https://github.com/mesosfer/bear.git
cd bear

# 2. Synchronize base environment & dependencies
uv sync

# 3a. For NVIDIA CUDA GPUs:
uv sync --extra gpu

# 3b. For AMD Instinct MI300X (ROCm 6.0+ CDNA 3):
GPU_ARCHS="gfx942" uv pip install flash-attn --no-build-isolation
```

---

## 🚀 End-to-End Workflow: Step-by-Step Guide

```
+---------------------------------------------------------------------------------------------------+
|                                     BEAR AI 4-STAGE PIPELINE                                      |
|                                                                                                   |
|  [Step 1: Datasets] -> [Step 2: Train Tokenizer] -> [Step 3: Stage-by-Stage Train & Eval]          |
|  (22 HF Sources)       (60K Vocab / BPE)            |                                             |
|                                                     +--> Stage 1: Pretrain  --> Eval Pretrain     |
|                                                     +--> Stage 2: CPT       --> Eval CPT          |
|  [Step 4: Interactive Real-Time Chat CLI]           +--> Stage 3: SFT       --> Eval SFT          |
|  (Streaming REPL with XTML & Thinking)              +--> Stage 4: Safety    --> Eval Safety       |
+---------------------------------------------------------------------------------------------------+
```

---

### Step 1: Download & Prepare Multi-Stage Datasets

You can either pull the pre-processed, deterministic-shuffled, and partitioned dataset directly from the official Bear AI Hugging Face repository, or download and process the raw shards from the 22 upstream repositories:

#### Option A: Pull Pre-processed Dataset from Hugging Face (Recommended & Fastest)
```bash
# Download all 4 stages (~20 GB) into storage/dataset/
uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage all

# Or download a specific stage
uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage pretrain
uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage cpt
uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage sft
uv run python -m data.hf_download --repo Dummy9898/bear-dataset --stage safety

# Dry-run inspection
uv run python -m data.hf_download --dry-run
```

#### Option B: Download & Process Raw Shards from 22 Upstream Repositories
```bash
# Download and prepare all 4 stages
uv run python -m data.dataset --stage all

# Quick dry-run inspection (inspect shards without downloading)
uv run python -m data.dataset --dry-run
```

*(Optional) Export and upload your own dataset corpus to Hugging Face Hub:*
```bash
uv run python -m data.hf_upload --repo <your-username>/bear-dataset --stage all
```

---

### Step 2: Train & Benchmark the Custom Tokenizer

Train the 60,000 vocabulary BearTokenizer using BPE with inverted-index acceleration across all downloaded multi-domain shards:

```bash
# 1. Train Tokenizer on the local dataset corpus
uv run python -m train.tok_train --vocab-size 60000 --min-frequency 2

# 2. Run 3-Way Tokenizer Benchmark (Bear AI vs Kimi-K3 vs Qwen-3.8)
uv run python -m benchmark.compare_tokenizers
```

The trained artifact is saved directly to `storage/tokenizer/bear_tokenizer.json`.

---

### Step 3: Phased 4-Stage Training & Mandatory Stage Evaluation

Train the model sequentially through the 4-stage curriculum. **Always run the stage evaluation benchmark immediately after each training stage completes** before advancing to the next stage.

```
+-------------------------------------------------------------------------------+
|  Stage 1: Pretrain   --->  Evaluate Pretrain  (PPL, Loss, BPB)                |
|         │                                                                     |
|         ▼                                                                     |
|  Stage 2: CPT        --->  Evaluate CPT       (HumanEval Code, Math, CLI)     |
|         │                                                                     |
|         ▼                                                                     |
|  Stage 3: SFT        --->  Evaluate SFT       (XTML Chat, Multi-turn CoT)     |
|         │                                                                     |
|         ▼                                                                     |
|  Stage 4: Safety     --->  Evaluate Safety    (Harm Refusal & Guardrails)     |
+-------------------------------------------------------------------------------+
```

#### 🔹 Stage 1: Foundational Pre-training
*General knowledge, large-scale multi-domain language modeling (Chinchilla Optimal: ~6.29B tokens, LR: 3e-4, Warmup: 500 steps).*

```bash
# 1. Train Pre-training Stage (Saves to storage/models/pretrain/)
uv run python -m train.pretrain --batch-size 16 --grad-accum 8 --max-steps 12000

# 2. Evaluate Pre-training Checkpoint (PPL, Loss, Bits-Per-Byte on Wikipedia/MMLU/ARC)
uv run python -m eval.evaluator --stage pretrain
```

---

#### 🔹 Stage 2: Continual Pre-Training (CPT)
*Domain specialization in Python/TypeScript source code, OpenWebMath, LaTeX proofs, and PowerShell/Bash CLI scripting (LR: 1e-4).*

```bash
# 1. Train CPT Stage (Auto-resumes from storage/models/pretrain/bear_final.pt, saves to storage/models/cpt/)
uv run python -m train.cpt --max-steps 5000

# 2. Evaluate CPT Checkpoint (HumanEval coding, GSM8K arithmetic, LaTeX math, CLI)
uv run python -m eval.evaluator --stage cpt
```

---

#### 🔹 Stage 3: Supervised Fine-Tuning (SFT)
*Instruction following, multi-turn dialogues, and Chain-of-Thought reasoning using Kimi-K3 XTML markup (LR: 2e-5, Weight Decay: 0.01).*

```bash
# 1. Train SFT Stage (Auto-resumes from storage/models/cpt/bear_final.pt, saves to storage/models/sft/)
uv run python -m train.sft --max-steps 3000

# 2. Evaluate SFT Checkpoint (XTML format compliance, keyword recall, reasoning CoT)
uv run python -m eval.evaluator --stage sft
```

---

#### 🔹 Stage 4: Safety & Guardrail Alignment
*Refusal of harmful, unauthorized, or exploitative requests while preserving helpfulness on benign security education queries (LR: 5e-6).*

```bash
# 1. Train Safety Stage (Auto-resumes from storage/models/sft/bear_final.pt, saves to storage/models/safety/)
uv run python -m train.safety --max-steps 1500

# 2. Evaluate Safety Checkpoint (Refusal Accuracy & Benign Pass Rate)
uv run python -m eval.evaluator --stage safety
```

---

#### 🏆 Run Full End-to-End Evaluation (All Stages)
Run all 4 stage benchmark suites simultaneously to produce a comprehensive model scorecard:

```bash
uv run python -m eval.evaluator --stage all
```

All evaluation results and metrics are automatically saved as JSON reports under `storage/eval/eval_report_<stage>_<timestamp>.json`.

---

### Step 4: Interactive Real-Time Chat CLI

Test and interact with your trained model directly via terminal with real-time streaming output:

```bash
# Start chat session with a trained checkpoint
uv run python -m chat.cli --checkpoint storage/models/bear_final.pt

# Custom sampling parameters
uv run python -m chat.cli --checkpoint storage/models/bear_final.pt --temperature 0.7 --top-p 0.9 --top-k 40

# Instant dry-run / mock mode
uv run python -m chat.cli --dry-run
```

#### In-Chat Interactive Commands:
- `/clear` — Reset conversation history
- `/system <prompt>` — Update system instruction prompt dynamically
- `/thinking <on|off>` — Toggle XTML `<|open|>thought...<|close|>` reasoning format
- `/temp <value>` — Adjust sampling temperature (e.g., `/temp 0.7`)
- `/history` — Display active conversation dialogue tree
- `/exit` or `/quit` — Exit chat session

---

### Step 5: Interactive Model Packaging & Hugging Face Hub Release

Package and commit trained checkpoints per stage (complete with 60k Kimi-K3 tokenizer, architecture config, generation config, standalone `inference.py`, `chat/cli.py`, self-contained engine, and benchmark scorecards) directly to Hugging Face Model Hub:

```bash
# 1. Interactive terminal wizard (Select stage, checkpoint, and repo interactively)
uv run python -m scripts.commit_model

# 2. Or direct CLI commit
uv run python -m scripts.commit_model --stage pretrain --checkpoint storage/models/bear_final.pt --repo Dummy9898/bear-240m-pretrain

# 3. Export standalone bundle locally to storage/export/ without uploading
uv run python -m scripts.commit_model --stage pretrain --export-only

# 4. List all available local checkpoints
uv run python -m scripts.commit_model --list
```

---

## 🧪 Running Unit Tests

Execute the complete test suite (covering dataloaders, engine, tokenizers, transformers, evaluator, and chat CLI):

```bash
uv run pytest tests/ -q
```

---

## 📂 Project Directory Structure

```
bear/
├── benchmark/
│   └── compare_tokenizers.py       # 3-Way Tokenizer Benchmark (Bear vs Kimi vs Qwen)
├── chat/
│   ├── __init__.py
│   └── cli.py                      # Interactive Streaming Chat CLI REPL
├── data/
│   ├── dataloader.py               # Sequence Packing Streaming DataLoader (4096 tokens)
│   ├── dataset.py                  # Multi-domain raw dataset downloader & preprocessor
│   ├── hf_download.py              # Automated Hugging Face Dataset Downloader & Restorer
│   └── hf_upload.py                # Automated Hugging Face Hub Dataset Exporter
├── engine/
│   ├── engine.py                   # Training loop, optimizer, cosine LR, validation BPB
│   ├── flashattion.py              # FlashAttention-2 / SDPA kernel backend selection
│   ├── tokenizer.py                # Kimi-K3 XTML BearTokenizer (Rust tiktoken backend)
│   └── transformer.py              # Llama-style decoder-only BearTransformer
├── eval/
│   └── evaluator.py                # Stage-specific benchmark runner & reporter
├── scripts/
│   ├── __init__.py
│   └── commit_model.py             # Interactive Model Release, Bundling & Hub Committer
├── task/
│   ├── pretrain_tasks.json         # HF Pretrain Benchmark Tasks (Wikipedia, MMLU, ARC)
│   ├── cpt_tasks.json              # HF CPT Technical Tasks (HumanEval, GSM8K, PowerShell)
│   ├── sft_tasks.json              # HF SFT Instruction Tasks (Alpaca-ID, UltraChat, Platypus)
│   └── safety_tasks.json           # HF Safety Tasks (PKU-SafeRLHF, HH-RLHF, JailbreakHub)
├── tests/
│   ├── test_chat.py
│   ├── test_commit_model.py
│   ├── test_dataloader.py
│   ├── test_dataset.py
│   ├── test_engine.py
│   ├── test_eval.py
│   ├── test_hf_download.py
│   ├── test_hf_upload.py
│   ├── test_tokenizer.py
│   └── test_transformer.py
├── train/
│   ├── pretrain.py                 # Stage 1 Pre-training CLI
│   ├── cpt.py                      # Stage 2 Continual Pre-Training CLI
│   ├── sft.py                      # Stage 3 Supervised Fine-Tuning CLI
│   ├── safety.py                   # Stage 4 Safety Alignment CLI
│   └── tok_train.py                # Tokenizer BPE Training CLI
├── pyproject.toml                  # PEP 621 package & dependency definitions
├── uv.lock                         # Exact pinned dependency lockfile
└── README.md                       # Documentation & Quickstart Guide
```

---

## 📜 License

Distributed under the **Apache-2.0 License**. See `LICENSE` for details.
