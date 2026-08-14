"""Tests for engine.engine — training loop, LR schedule, optimizer, checkpointing."""

import torch
import pytest
from pathlib import Path

from engine.transformer import BearTransformer, BearConfig
from engine.engine import (
    TrainConfig,
    cosine_lr,
    create_optimizer,
    save_checkpoint,
    load_checkpoint,
    evaluate,
)


@pytest.fixture
def small_config():
    return BearConfig(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4,
        n_kv_heads=2, ffn_hidden=128, max_seq_len=64,
    )


@pytest.fixture
def model(small_config):
    return BearTransformer(small_config)


@pytest.fixture
def train_config():
    return TrainConfig(
        lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100,
        batch_size=2, grad_accum_steps=1, dtype="float32",
        log_every=5, save_every=50, context_length=32,
    )


def _make_fake_loader(vocab_size, batch_size, seq_len, n_batches=20):
    """Generate a simple list of (input_ids, labels) batches."""
    batches = []
    for _ in range(n_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        batches.append((x, y))
    return batches


# -- LR Schedule --

def test_cosine_lr_warmup():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    # Step 0: should be ~lr/warmup_steps
    lr0 = cosine_lr(0, cfg)
    assert lr0 < cfg.lr
    assert lr0 > 0

    # Step 9 (end of warmup): should be close to peak
    lr9 = cosine_lr(9, cfg)
    assert lr9 > lr0
    assert lr9 <= cfg.lr


def test_cosine_lr_decay():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    lr_mid = cosine_lr(55, cfg)
    lr_end = cosine_lr(99, cfg)
    assert lr_mid > lr_end
    assert lr_end >= cfg.min_lr


def test_cosine_lr_at_max():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    lr = cosine_lr(100, cfg)
    assert lr == cfg.min_lr


# -- Optimizer --

def test_create_optimizer_param_groups(model, train_config):
    opt = create_optimizer(model, train_config)
    assert len(opt.param_groups) == 2
    # Group 0: weight decay params (2D+ tensors)
    assert opt.param_groups[0]["weight_decay"] == train_config.weight_decay
    # Group 1: no decay params (1D tensors like norms)
    assert opt.param_groups[1]["weight_decay"] == 0.0


# -- Checkpoint --

def test_save_and_load_checkpoint(model, small_config, train_config, tmp_path):
    opt = create_optimizer(model, train_config)

    # Run one forward+backward to populate optimizer state
    x = torch.randint(0, small_config.vocab_size, (2, 32))
    y = torch.randint(0, small_config.vocab_size, (2, 32))
    _, loss = model(x, y)
    loss.backward()
    opt.step()

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(model, opt, step=42, loss=loss.item(),
                    model_config=small_config, train_config=train_config, path=ckpt_path)

    assert ckpt_path.exists()

    # Load into a fresh model
    model2 = BearTransformer(small_config)
    opt2 = create_optimizer(model2, train_config)
    step = load_checkpoint(ckpt_path, model2, opt2)
    assert step == 42

    # Weights should match
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_atomic(model, small_config, train_config, tmp_path):
    """Verify no .tmp file remains after save."""
    ckpt_path = tmp_path / "atomic_test.pt"
    opt = create_optimizer(model, train_config)
    save_checkpoint(model, opt, step=1, loss=0.5,
                    model_config=small_config, train_config=train_config, path=ckpt_path)
    assert ckpt_path.exists()
    assert not ckpt_path.with_suffix(".tmp").exists()


# -- Evaluate --

def test_evaluate(model, small_config):
    loader = _make_fake_loader(small_config.vocab_size, 2, 32, n_batches=5)
    metrics = evaluate(model, loader, device="cpu", max_batches=3, bytes_per_token=3.43)
    assert isinstance(metrics, dict)
    assert "loss" in metrics and metrics["loss"] > 0
    assert "ppl" in metrics and metrics["ppl"] > 1.0
    assert "bpb" in metrics and metrics["bpb"] > 0
    assert "bpt" in metrics and metrics["bpt"] > 0
