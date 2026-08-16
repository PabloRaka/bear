"""
Mesosfer Bear AI - Stage 1: Pre-training CLI

Foundational language modeling on large-scale multi-domain corpus.
High LR, long warmup, many steps.

Usage:
  uv run python -m train.pretrain
  uv run python -m train.pretrain --batch-size 16 --grad-accum 8 --max-steps 12000
  uv run python -m train.pretrain --resume storage/models/bear_step_1000.pt
  uv run python -m train.pretrain --dry-run
"""

import argparse
import sys

import torch

from engine.transformer import BearTransformer, BearConfig
from engine.engine import TrainConfig, train
from engine.flashattion import get_attn_backend
from data.dataloader import create_dataloader


STAGE = "pretrain"


def main():
    parser = argparse.ArgumentParser(
        description="Mesosfer Bear AI - Stage 1: Pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--vocab-size", type=int, default=60000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=2816)
    parser.add_argument("--context-length", type=int, default=4096)

    # Training — pretrain defaults: high LR, long warmup
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=12000, help="Max steps (12k steps ~6.29B tokens, Chinchilla optimal for 240M params)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Data
    parser.add_argument("--dataset-dir", type=str, default="storage/dataset")

    # Checkpointing & Evaluation
    parser.add_argument("--checkpoint-dir", type=str, default="storage/models/pretrain")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate validation loss/BPB every N steps")
    parser.add_argument("--eval-batches", type=int, default=50, help="Number of batches to evaluate")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    parser.add_argument("--seed", type=int, default=42)

    # Quick test
    parser.add_argument("--dry-run", action="store_true", help="Run 5 steps with tiny config for sanity check")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dry-run: override with tiny config
    if args.dry_run:
        args.vocab_size = 512
        args.d_model = 64
        args.n_layers = 2
        args.n_heads = 4
        args.n_kv_heads = 2
        args.ffn_hidden = 128
        args.context_length = 128
        args.max_steps = 5
        args.batch_size = 2
        args.grad_accum = 1
        args.warmup_steps = 2
        args.save_every = 5
        args.eval_every = 5
        args.eval_batches = 2
        args.log_every = 1
        args.dtype = "float32"
        print("[DRY-RUN] Using tiny model config for sanity check\n")

    # Build model config
    model_config = BearConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden=args.ffn_hidden,
        max_seq_len=args.context_length,
    )

    # Build training config
    train_config = TrainConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        context_length=args.context_length,
        stage=STAGE,
        dataset_dir=args.dataset_dir,
        seed=args.seed,
    )

    # Load tokenizer
    from engine.tokenizer import BearTokenizer
    from pathlib import Path

    tok_path = Path("storage/tokenizer/bear_tokenizer.json")
    if tok_path.exists():
        tokenizer = BearTokenizer.load(str(tok_path))
        print(f"  Loaded trained tokenizer: vocab_size={tokenizer.vocab_size}")
        model_config.vocab_size = tokenizer.vocab_size
    else:
        tokenizer = BearTokenizer()
        print(f"  Using base tokenizer: vocab_size={tokenizer.vocab_size}")
        if not args.dry_run:
            model_config.vocab_size = tokenizer.vocab_size

    # Create model
    model = BearTransformer(model_config)

    # Create dataloaders
    print(f"  Loading dataset: stage={STAGE}, split=train...")
    train_loader = create_dataloader(
        tokenizer=tokenizer,
        stage=STAGE,
        split="train",
        context_length=args.context_length,
        batch_size=args.batch_size,
        base_dir=args.dataset_dir,
        seed=args.seed,
    )

    val_loader = None
    try:
        val_loader = create_dataloader(
            tokenizer=tokenizer,
            stage=STAGE,
            split="val",
            context_length=args.context_length,
            batch_size=args.batch_size,
            base_dir=args.dataset_dir,
            seed=args.seed,
        )
        print(f"  Loaded validation set")
    except FileNotFoundError:
        print(f"  No validation set found, skipping eval")

    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
