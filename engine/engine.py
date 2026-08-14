"""
Mesosfer Bear AI - Training Engine

Full training loop for pre-training and continued pre-training:
  - AdamW with decoupled weight decay
  - Cosine LR schedule with linear warmup
  - Mixed precision (AMP) with bf16/fp16
  - Gradient clipping & accumulation
  - Checkpoint save/resume
  - Throughput logging (tokens/sec)
"""

import os
import time
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from engine.transformer import BearTransformer, BearConfig
from engine.flashattion import get_attn_backend


# -- Model Identity ---------------------------------------------------------
# ponytail: single source of truth for model naming and tags

BEAR_MODEL_IDENTITY = {
    "model_name": "Bear",
    "model_full_name": "Mesosfer Bear AI",
    "model_version": "0.1.0",
    "author": "Mesosfer Team",
    "architecture": "BearTransformer",
    "architecture_tags": [
        "decoder-only",
        "llama-style",
        "rope",
        "gqa",
        "swiglu",
        "rmsnorm",
        "flash-attention",
    ],
    "license": "Apache-2.0",
    "base_model": None,  # set to parent checkpoint path for CPT/SFT
    "language": ["id", "en"],
    "pipeline_tag": "text-generation",
    "library_name": "mesosfer-bear",
    "tags": [
        "bear",
        "mesosfer",
        "causal-lm",
        "indonesian",
        "multilingual",
        "code",
        "math",
    ],
}


# -- Training Config --------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Optimization
    lr: float = 3e-4
    min_lr: float = 3e-5          # cosine schedule floor (10% of peak)
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 50000

    # Batching
    batch_size: int = 4
    grad_accum_steps: int = 8     # effective batch = batch_size * grad_accum_steps

    # Precision
    dtype: str = "bfloat16"       # "bfloat16", "float16", or "float32"

    # Checkpointing & Evaluation
    checkpoint_dir: str = "storage/models"
    save_every: int = 1000        # save checkpoint every N steps
    eval_every: int = 1000        # evaluate on validation set every N steps
    eval_batches: int = 50        # number of batches to evaluate
    log_every: int = 10           # print loss every N steps

    # Data & Metrics
    context_length: int = 4096
    stage: str = "pretrain"
    dataset_dir: str = "storage/dataset"
    seed: int = 42
    bytes_per_token: float = 3.43 # average compression ratio for BPB calculation


# -- LR Schedule -----------------------------------------------------------

def cosine_lr(step: int, config: TrainConfig) -> float:
    """Cosine annealing with linear warmup. Returns learning rate for given step."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps

    if step >= config.max_steps:
        return config.min_lr

    # Cosine decay from lr to min_lr
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.lr - config.min_lr)


# -- Optimizer factory ------------------------------------------------------

def create_optimizer(model: BearTransformer, config: TrainConfig) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with decoupled weight decay.
    No weight decay on: biases, norms, embeddings.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # ponytail: simple rule — 1D params (bias, norm weight) get no decay
        if param.dim() <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        fused=torch.cuda.is_available(),  # faster fused kernel on CUDA
    )


# -- Checkpoint I/O ---------------------------------------------------------

def save_checkpoint(
    model: BearTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    model_config: BearConfig,
    train_config: TrainConfig,
    path: Path,
):
    """Save full training state for resumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "bear_identity": BEAR_MODEL_IDENTITY,
        "step": step,
        "loss": loss,
        "param_count": model.param_count(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    # Atomic save: write to .tmp then rename
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    if path.exists():
        path.unlink()
    tmp_path.rename(path)
    print(f"  [CHECKPOINT] Saved at step {step} -> {path}")


def load_checkpoint(
    path: Path,
    model: BearTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> int:
    """Load checkpoint and return the step number to resume from."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("nan"))
    print(f"  [CHECKPOINT] Resumed from step {step} (loss={loss:.4f}) <- {path}")
    return step


# -- Training Loop ----------------------------------------------------------

def train(
    model: BearTransformer,
    train_loader,
    val_loader=None,
    config: Optional[TrainConfig] = None,
    device: str = "auto",
    resume_from: Optional[str] = None,
):
    """
    Main training loop.

    Args:
        model: BearTransformer instance.
        train_loader: DataLoader yielding (input_ids, labels) batches.
        val_loader: Optional validation DataLoader.
        config: TrainConfig instance.
        device: "auto", "cuda", "cpu", or "mps".
        resume_from: Path to checkpoint file to resume from.
    """
    config = config or TrainConfig()

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = model.to(device)

    # AMP setup
    use_amp = (config.dtype != "float32") and (device == "cuda")
    amp_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    scaler = GradScaler(enabled=(use_amp and config.dtype == "float16"))
    # ponytail: bf16 doesn't need GradScaler (no inf/nan scaling issues)

    optimizer = create_optimizer(model, config)
    start_step = 0

    if resume_from and Path(resume_from).exists():
        start_step = load_checkpoint(Path(resume_from), model, optimizer, device)

    # Print training info
    tokens_per_step = config.batch_size * config.grad_accum_steps * config.context_length
    print(f"\n{'='*60}")
    print(f"  Mesosfer Bear AI - Training Engine")
    print(f"{'='*60}")
    print(f"  Model params : {model.param_count():,} ({model.param_count()/1e6:.1f}M)")
    print(f"  Attention     : {get_attn_backend()}")
    print(f"  Device        : {device}")
    print(f"  Precision     : {config.dtype}")
    print(f"  LR            : {config.lr} -> {config.min_lr} (cosine, {config.warmup_steps} warmup)")
    print(f"  Batch size    : {config.batch_size} x {config.grad_accum_steps} accum = {config.batch_size * config.grad_accum_steps} effective")
    print(f"  Tokens/step   : {tokens_per_step:,}")
    print(f"  Context length: {config.context_length}")
    print(f"  Max steps     : {config.max_steps}")
    print(f"  Resume step   : {start_step}")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()

    running_loss = 0.0
    tokens_seen = 0
    t0 = time.time()
    data_iter = iter(train_loader)
    step = start_step

    while step < config.max_steps:
        # Update learning rate
        lr = cosine_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        micro_loss_sum = 0.0
        for micro_step in range(config.grad_accum_steps):
            try:
                input_ids, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, labels = next(data_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, labels)
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            micro_loss_sum += loss.item()
            tokens_seen += input_ids.numel()

        # Gradient clipping
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += micro_loss_sum
        step += 1

        # Logging
        if step % config.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / config.log_every
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            print(
                f"  step {step:>6d}/{config.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"elapsed {elapsed:.1f}s"
            )
            running_loss = 0.0
            tokens_seen = 0
            t0 = time.time()

        # Checkpoint
        if step % config.save_every == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"bear_step_{step}.pt"
            save_checkpoint(model, optimizer, step, micro_loss_sum, model.config, config, ckpt_path)

        # Validation Evaluation (nanoGPT style: loss, ppl, bpb, bpt)
        if val_loader is not None and (step % config.eval_every == 0 or step == config.max_steps):
            val_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
                max_batches=config.eval_batches,
                bytes_per_token=config.bytes_per_token,
            )
            print(
                f"  [VAL] step {step:>6d} | "
                f"loss {val_metrics['loss']:.4f} | "
                f"ppl {val_metrics['ppl']:.2f} | "
                f"bpb {val_metrics['bpb']:.3f} | "
                f"bpt {val_metrics['bpt']:.3f}"
            )
            model.train()

    # Final save
    final_path = Path(config.checkpoint_dir) / "bear_final.pt"
    save_checkpoint(model, optimizer, step, micro_loss_sum, model.config, config, final_path)

    # Save human-readable model card alongside checkpoint
    model_card = {
        **BEAR_MODEL_IDENTITY,
        "param_count": model.param_count(),
        "param_count_human": f"{model.param_count()/1e6:.1f}M",
        "training_stage": config.stage,
        "training_steps": step,
        "final_loss": micro_loss_sum,
        "context_length": config.context_length,
        "model_config": asdict(model.config),
        "attention_backend": get_attn_backend(),
    }
    card_path = Path(config.checkpoint_dir) / "bear_model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    print(f"\n  Training complete.")
    print(f"  Final checkpoint : {final_path}")
    print(f"  Model card       : {card_path}")


# -- Evaluation -------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: BearTransformer,
    val_loader,
    device: str = "cpu",
    amp_dtype=torch.bfloat16,
    use_amp: bool = False,
    max_batches: int = 50,
    bytes_per_token: float = 3.43,
) -> dict:
    """
    Compute validation metrics over max_batches:
      - loss: Cross-entropy loss in nats
      - ppl: Perplexity (exp(loss))
      - bpb: Bits Per Byte (loss / (ln(2) * bytes_per_token))
      - bpt: Bits Per Token (loss / ln(2))
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for i, (input_ids, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(input_ids, labels)

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    bpt = avg_loss / math.log(2)           # bits per token
    bpb = bpt / max(bytes_per_token, 1e-5) # bits per byte
    ppl = math.exp(min(avg_loss, 20.0))    # clamp to avoid math range overflow

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bpb": bpb,
        "bpt": bpt,
    }
