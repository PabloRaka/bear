"""
Mesosfer Bear AI: Interactive Model Packaging, Hub Exporter & Release Committer

Bundles everything required to run and deploy the Bear AI model at each stage:
  1. Model Checkpoint & PyTorch Weights (`bear_model.pt`, `model_card.json`)
  2. Custom Kimi-K3 XTML 60K BPE Tokenizer (`bear_tokenizer.json`)
  3. Model Architecture Config (`config.json`) & Generation Config (`generation_config.json`)
  4. Complete Self-Contained Engine (`transformer.py`, `tokenizer.py`, `flashattion.py`)
  5. Standalone Inference Script (`inference.py`) & Interactive Chat REPL (`cli.py`)
  6. Stage Benchmark Evaluation Scorecard (`eval_report.json`)
  7. Rich Hugging Face Model Card (`README.md`)

Usage:
  Interactive Mode:
    uv run python -m scripts.commit_model

  CLI Flags Mode:
    uv run python -m scripts.commit_model --stage pretrain --checkpoint storage/models/bear_final.pt --repo Dummy9898/bear-240m-pretrain
    uv run python -m scripts.commit_model --stage cpt --export-only
    uv run python -m scripts.commit_model --list
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Dict, Tuple

import torch

from data.hf_upload import load_hf_token
from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer
from engine.flashattion import get_attn_backend


# Stage Definitions
STAGES = {
    "1": ("pretrain", "Stage 1: Foundational Pre-training", "Pre-trained on 6.3B tokens of general multilingual corpus"),
    "2": ("cpt", "Stage 2: Continual Pre-Training (CPT)", "Domain-specialized on Python, TypeScript, Math, LaTeX, and CLI scripts"),
    "3": ("sft", "Stage 3: Supervised Fine-Tuning (SFT)", "Dialogue tuned with Kimi-K3 XTML markup & CoT reasoning"),
    "4": ("safety", "Stage 4: Safety & Guardrail Alignment", "Aligned with harm refusal and security guardrails"),
}


def find_available_checkpoints(model_dir: Path = Path("storage/models")) -> List[Tuple[Path, int, float, str]]:
    """Scan and return list of (path, step, loss, size_str) sorted by step."""
    if not model_dir.exists():
        return []
    
    ckpts = []
    for f in model_dir.glob("*.pt"):
        size_mb = f.stat().st_size / (1024 * 1024)
        step = 0
        loss = float("nan")
        try:
            ckpt_data = torch.load(f, map_location="cpu", weights_only=False)
            step = ckpt_data.get("step", 0)
            loss = ckpt_data.get("loss", float("nan"))
        except Exception:
            pass
        ckpts.append((f, step, loss, f"{size_mb:.1f} MB"))
    
    # Sort by step descending, or name
    ckpts.sort(key=lambda x: (x[1], x[0].name), reverse=True)
    return ckpts


def find_latest_eval_report(stage: str, eval_dir: Path = Path("storage/eval")) -> Optional[dict]:
    """Find and load the latest benchmark JSON evaluation report for the stage."""
    if not eval_dir.exists():
        return None
    
    matching = list(eval_dir.glob(f"eval_report_{stage}_*.json"))
    if not matching:
        matching = list(eval_dir.glob("eval_report_all_*.json"))
    if not matching:
        return None
    
    # Sort by modification time descending
    matching.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    try:
        with open(matching[0], "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def generate_hf_model_card(
    stage: str,
    stage_title: str,
    stage_desc: str,
    model_config: BearConfig,
    param_count: int,
    step: int,
    loss: float,
    eval_report: Optional[dict] = None,
) -> str:
    """Generate professional, markdown model card for Hugging Face Hub."""
    
    eval_table = ""
    if eval_report and "benchmarks" in eval_report:
        eval_table = "\n### 📈 Benchmark Evaluation Scorecard\n\n"
        eval_table += "| Benchmark Stage | Loss | Perplexity (PPL) | Bits-Per-Byte (BPB) | Total Tasks |\n"
        eval_table += "|---|---|---|---|---|\n"
        for s_key, b in eval_report["benchmarks"].items():
            eval_table += f"| **{b.get('stage', s_key).upper()}** | `{b.get('avg_loss', 0.0):.4f}` | `{b.get('perplexity', 0.0):.2f}` | `{b.get('bits_per_byte', 0.0):.3f}` | {len(b.get('tasks', []))} tasks |\n"

    return f"""---
language:
- id
- en
- code
license: apache-2.0
tags:
- mesosfer
- bear-ai
- llama-architecture
- text-generation
- pytorch
- transformers
- causal-lm
pipeline_tag: text-generation
---

# 🐻 Mesosfer Bear AI - {stage_title}

**Mesosfer Bear AI ({param_count/1e6:.1f}M)** is a high-efficiency autoregressive decoder-only language model built on a Llama-style architecture. This repository contains the official model weights and runtime engine for **{stage_title}** ({stage_desc}).

---

## 🌟 Model Architecture Highlights

- **Parameters**: **{param_count:,} ({param_count/1e6:.1f}M)**
- **Layers / Depth**: **{model_config.n_layers}** transformer blocks
- **Hidden Dimension (`d_model`)**: **{model_config.d_model}**
- **FFN Hidden Dimension**: **{model_config.ffn_hidden}** (SwiGLU activation)
- **Attention Heads**: **{model_config.n_heads}** Query heads / **{model_config.n_kv_heads}** KV heads (**Grouped Query Attention 4:1**)
- **Context Length**: **{getattr(model_config, 'max_seq_len', getattr(model_config, 'context_length', 4096))}** tokens
- **Positional Encoding**: Rotary Position Embeddings (RoPE, $\\theta=10000$)
- **Tokenizer**: 60,000 vocabulary based on Kimi-K3 BPE with native XTML markup (`<|open|>...<|close|>`) and Rust `tiktoken` acceleration.
- **Training Step**: Step **{step:,}** (Loss: `{loss:.4f}`)
{eval_table}
---

## 🚀 Quickstart: Running Inference

You can run text generation and chat streaming immediately with zero external frameworks:

### 1. Installation
```bash
git clone https://huggingface.co/{{REPO_ID}}
cd {{REPO_NAME}}
pip install torch tiktoken
```

### 2. Standalone Inference Script
```bash
python inference.py --prompt "Jelaskan konsep machine learning secara singkat:"
```

### 3. Interactive Streaming Chat CLI
```bash
python cli.py --temperature 0.7 --top-p 0.9
```

### 4. Python API Usage
```python
from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer
import torch

# Load Tokenizer & Model
tokenizer = BearTokenizer.load("bear_tokenizer.json")
config = BearConfig.from_dict(torch.load("config.json"))
model = BearTransformer(config)

checkpoint = torch.load("bear_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
model.eval()

# Chat format
conversation = [
    {{"role": "system", "content": "Anda adalah asisten AI Bear yang cerdas dan ramah."}},
    {{"role": "user", "content": "Halo! Siapa kamu?"}}
]
prompt = tokenizer.apply_chat_template(conversation, thinking=True)
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

output_ids = model.generate(input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9)
response = tokenizer.decode(output_ids[0].tolist())
print(response)
```

---

## 📜 License
Distributed under the **Apache-2.0 License**. Developed by **Mesosfer Team**.
"""


def generate_standalone_inference_py() -> str:
    """Generate a clean standalone inference.py script."""
    return """\"\"\"
Standalone Inference Runner for Mesosfer Bear AI
\"\"\"
import os
import sys
import json
import argparse
import torch

from engine.transformer import BearTransformer, BearConfig
from engine.tokenizer import BearTokenizer


def main():
    parser = argparse.ArgumentParser(description="Mesosfer Bear AI Standalone Inference")
    parser.add_argument("--prompt", type=str, default="Halo, jelaskan apa itu kecerdasan buatan dalam 2 kalimat.")
    parser.add_argument("--checkpoint", type=str, default="bear_model.pt")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--tokenizer", type=str, default="bear_tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--thinking", action="store_true", help="Enable XTML thinking mode")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Loading Bear AI model on {device}...")

    # 1. Load Tokenizer
    tokenizer = BearTokenizer.load(args.tokenizer) if os.path.exists(args.tokenizer) else BearTokenizer()

    # 2. Load Config & Model
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = BearConfig.from_dict(cfg_dict)
    model = BearTransformer(config)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Format Prompt
    conv = [
        {"role": "system", "content": "Anda adalah asisten AI Bear yang cerdas, ringkas, dan ramah."},
        {"role": "user", "content": args.prompt}
    ]
    formatted = tokenizer.apply_chat_template(conv, thinking=args.thinking)
    input_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device=device)

    print(f"\\nPrompt: {args.prompt}\\n" + "=" * 60)
    print("Generating response (streaming):\\n")

    # 4. Generate
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    
    generated_text = tokenizer.decode(out[0].tolist())
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
"""


def package_and_export_bundle(
    checkpoint_path: Path,
    stage: str,
    output_dir: Path,
    repo_id: str,
) -> Tuple[bool, int]:
    """Package model checkpoint, tokenizer, engine, config, and standalone runners into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n-> Packaging model bundle into '{output_dir.resolve()}'...")

    # 1. Load checkpoint to extract info & verify
    ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state", ckpt_data)
    step = ckpt_data.get("step", 0)
    loss = ckpt_data.get("loss", float("nan"))
    
    # Extract config
    if "model_config" in ckpt_data:
        m_cfg = ckpt_data["model_config"]
        if isinstance(m_cfg, dict):
            config = BearConfig.from_dict(m_cfg) if hasattr(BearConfig, "from_dict") else BearConfig(**{k: v for k, v in m_cfg.items() if hasattr(BearConfig, k)})
        elif isinstance(m_cfg, BearConfig):
            config = m_cfg
        else:
            config = BearConfig()
    else:
        config = BearConfig()
    
    model = BearTransformer(config)
    param_count = model.param_count()

    # 2. Save Clean Model Checkpoint (strip heavy optimizer states if standalone weights)
    clean_ckpt = {
        "model_state": state_dict,
        "step": step,
        "loss": loss,
        "stage": stage,
        "model_config": asdict(config),
    }
    torch.save(clean_ckpt, output_dir / "bear_model.pt")
    print("   [OK] Model weights saved: bear_model.pt")

    # 3. Save Architecture config.json & generation_config.json
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    print("   [OK] Architecture config saved: config.json")

    generation_config = {
        "model_type": "bear_transformer",
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, "max_seq_len", getattr(config, "context_length", 4096)),
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 12,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(generation_config, f, indent=2)
    print("   [OK] Generation config saved: generation_config.json")

    # 4. Copy Tokenizer
    tok_src = Path("storage/tokenizer/bear_tokenizer.json")
    if tok_src.exists():
        shutil.copy2(tok_src, output_dir / "bear_tokenizer.json")
        print("   [OK] Tokenizer copied: bear_tokenizer.json")

    # 5. Bundle Runtime Engine Files
    engine_dest = output_dir / "engine"
    engine_dest.mkdir(exist_ok=True)
    for eng_file in ["transformer.py", "tokenizer.py", "flashattion.py", "__init__.py"]:
        src_f = Path("engine") / eng_file
        if src_f.exists():
            shutil.copy2(src_f, engine_dest / eng_file)
    print("   [OK] Self-contained engine bundled: engine/ (transformer, tokenizer, flashattention)")

    # 6. Bundle Standalone Inference & Chat CLI
    with open(output_dir / "inference.py", "w", encoding="utf-8") as f:
        f.write(generate_standalone_inference_py())
    
    chat_src = Path("chat/cli.py")
    if chat_src.exists():
        shutil.copy2(chat_src, output_dir / "cli.py")
    print("   [OK] Standalone runners created: inference.py & cli.py")

    # 7. Attach Benchmark Scorecard & Model Card README.md
    eval_report = find_latest_eval_report(stage)
    if eval_report:
        with open(output_dir / "eval_report.json", "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)
        print("   [OK] Benchmark scorecard attached: eval_report.json")

    stage_title = STAGES.get(stage, (stage, stage.upper(), ""))[1]
    stage_desc = STAGES.get(stage, (stage, stage.upper(), ""))[2]
    
    readme_content = generate_hf_model_card(
        stage=stage,
        stage_title=stage_title,
        stage_desc=stage_desc,
        model_config=config,
        param_count=param_count,
        step=step,
        loss=loss,
        eval_report=eval_report,
    )
    readme_content = readme_content.replace("{{REPO_ID}}", repo_id).replace("{{REPO_NAME}}", repo_id.split("/")[-1])
    
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("   [OK] Hugging Face Model Card generated: README.md")

    # Calculate total bundle size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    file_count = len(list(output_dir.rglob("*")))
    print(f"\n[SUCCESS] Bundle ready! Total: {file_count} files ({total_size / (1024 * 1024):.1f} MB)")
    
    return True, param_count


def upload_model_bundle_to_hf(
    bundle_dir: Path,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
) -> bool:
    """Upload exported model bundle to Hugging Face Model Hub in a single atomic commit."""
    from huggingface_hub import HfApi

    auth_token = load_hf_token(token)
    api = HfApi(token=auth_token)

    print(f"\n-> Connecting to Hugging Face Hub for repo '{repo_id}'...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"   [OK] Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"   [ERROR] Failed to access/create repository '{repo_id}': {e}")
        return False

    print(f"-> Uploading model bundle in ONE SINGLE ATOMIC COMMIT...")
    try:
        api.upload_folder(
            folder_path=str(bundle_dir.resolve()),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Release Mesosfer Bear AI Model checkpoint ({bundle_dir.name})",
        )
        print(f"\n🎉 [SUCCESS] Model successfully published & committed!")
        print(f"Model Hub URL: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Upload failed: {e}")
        return False


def interactive_menu():
    """Interactive guided terminal wizard."""
    print("=" * 75)
    print(" 🐻 MESOSFER BEAR AI — INTERACTIVE MODEL RELEASE & HUB COMMITTER")
    print("=" * 75)

    # 1. Select Stage
    print("\n[Step 1] Select Training Stage to Commit:")
    for k, (s_name, s_title, s_desc) in STAGES.items():
        print(f"  [{k}] {s_title}")
        print(f"      Desc: {s_desc}")
    print("  [5] Custom Checkpoint Path")
    print("  [0] Exit")

    choice = input("\nEnter choice [1-5, default 1]: ").strip() or "1"
    if choice == "0":
        print("Aborted.")
        return

    custom_ckpt = None
    if choice in STAGES:
        selected_stage = STAGES[choice][0]
    elif choice == "5":
        selected_stage = "custom"
        custom_ckpt = input("Enter path to custom checkpoint (.pt): ").strip()
    else:
        print("Invalid choice.")
        return

    # 2. Select Checkpoint
    ckpts = find_available_checkpoints()
    if not ckpts and not custom_ckpt:
        print(f"\n[ERROR] No checkpoints found in storage/models/! Make sure training has completed.")
        return

    target_ckpt = None
    if custom_ckpt:
        target_ckpt = Path(custom_ckpt)
    else:
        print(f"\n[Step 2] Available Checkpoints in storage/models/:")
        for idx, (c_path, step, loss, size_str) in enumerate(ckpts, 1):
            loss_str = f"{loss:.4f}" if not torch.isnan(torch.tensor(loss)) else "N/A"
            print(f"  [{idx}] {c_path.name:<28} | Step: {step:>6d} | Loss: {loss_str} | Size: {size_str}")
        
        c_choice = input(f"\nSelect checkpoint [1-{len(ckpts)}, default 1 ({ckpts[0][0].name})]: ").strip() or "1"
        try:
            target_ckpt = ckpts[int(c_choice) - 1][0]
        except Exception:
            target_ckpt = ckpts[0][0]

    print(f"\nSelected Checkpoint: {target_ckpt}")

    # 3. Target Hugging Face Repository
    default_repo = f"Dummy9898/bear-240m-{selected_stage}"
    repo_input = input(f"\n[Step 3] Hugging Face Model Repository ID [default: {default_repo}]: ").strip()
    target_repo = repo_input or default_repo

    # 4. Visibility
    vis_input = input("[Step 4] Repository Visibility (1=Private, 2=Public) [default: 1]: ").strip() or "1"
    is_private = (vis_input == "1")

    # 5. Export Directory
    bundle_dir = Path("storage/export") / f"bear_{selected_stage}"
    
    # 6. Action Confirmation
    print("\n" + "-" * 75)
    print("SUMMARY OF ACTIONS:")
    print(f"  • Stage           : {selected_stage.upper()}")
    print(f"  • Source Checkpoint: {target_ckpt}")
    print(f"  • Target Repo     : {target_repo}")
    print(f"  • Visibility      : {'Private' if is_private else 'Public'}")
    print(f"  • Export Directory: {bundle_dir.resolve()}")
    print("-" * 75)

    confirm = input("\nProceed with packaging and commit? (y/n) [default: y]: ").strip().lower() or "y"
    if confirm != "y":
        print("Cancelled.")
        return

    # Package Bundle
    success, params = package_and_export_bundle(
        checkpoint_path=target_ckpt,
        stage=selected_stage,
        output_dir=bundle_dir,
        repo_id=target_repo,
    )

    if not success:
        return

    # Upload to HF
    upload_now = input("\nUpload model bundle to Hugging Face Hub now? (y/n) [default: y]: ").strip().lower() or "y"
    if upload_now == "y":
        upload_model_bundle_to_hf(
            bundle_dir=bundle_dir,
            repo_id=target_repo,
            private=is_private,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Mesosfer Bear AI: Interactive Model Release & Hub Committer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", type=str, choices=["pretrain", "cpt", "sft", "safety", "custom"], help="Training stage")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--repo", type=str, help="Target Hugging Face model repository ID")
    parser.add_argument("--public", action="store_true", help="Set repository to public (default: private)")
    parser.add_argument("--export-only", action="store_true", help="Only create local bundle in storage/export without uploading")
    parser.add_argument("--list", action="store_true", help="List available local checkpoints and exit")

    args = parser.parse_args()

    if args.list:
        ckpts = find_available_checkpoints()
        print("\nAvailable checkpoints in storage/models/:")
        for c_path, step, loss, size_str in ckpts:
            loss_str = f"{loss:.4f}" if not torch.isnan(torch.tensor(loss)) else "N/A"
            print(f"  - {c_path.name:<28} | Step: {step:>6d} | Loss: {loss_str} | Size: {size_str}")
        return

    # If no flags passed, run interactive wizard
    if not args.stage and not args.checkpoint:
        interactive_menu()
        return

    # CLI Flags execution
    stage = args.stage or "pretrain"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path("storage/models/bear_final.pt")
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    repo_id = args.repo or f"Dummy9898/bear-240m-{stage}"
    bundle_dir = Path("storage/export") / f"bear_{stage}"

    success, _ = package_and_export_bundle(
        checkpoint_path=ckpt_path,
        stage=stage,
        output_dir=bundle_dir,
        repo_id=repo_id,
    )

    if success and not args.export_only:
        upload_model_bundle_to_hf(
            bundle_dir=bundle_dir,
            repo_id=repo_id,
            private=not args.public,
        )


if __name__ == "__main__":
    main()
