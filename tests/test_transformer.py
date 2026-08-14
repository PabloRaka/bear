"""Tests for engine.transformer — model architecture verification."""

import torch
import pytest

from engine.transformer import (
    BearTransformer,
    BearConfig,
    RMSNorm,
    precompute_rope_freqs,
    apply_rope,
)


@pytest.fixture
def small_config():
    """Tiny config for fast tests."""
    return BearConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden=128,
        max_seq_len=128,
    )


@pytest.fixture
def model(small_config):
    return BearTransformer(small_config)


# -- RMSNorm --

def test_rmsnorm_output_shape():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == (2, 8, 64)


def test_rmsnorm_unit_norm():
    """After RMSNorm, the RMS of each vector should be ~1 (before weight scaling)."""
    norm = RMSNorm(64)
    norm.weight.data.fill_(1.0)
    x = torch.randn(1, 1, 64) * 10  # large values
    out = norm(x)
    rms = out.float().pow(2).mean().sqrt().item()
    assert 0.8 < rms < 1.2


# -- RoPE --

def test_rope_freqs_shape():
    freqs = precompute_rope_freqs(head_dim=64, max_seq_len=128)
    assert freqs.shape == (128, 32)  # head_dim//2
    assert freqs.is_complex()


def test_apply_rope_preserves_shape():
    freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
    x = torch.randn(2, 4, 32, 16)  # (batch, heads, seq, head_dim)
    out = apply_rope(x, freqs)
    assert out.shape == x.shape


def test_rope_different_positions_give_different_values():
    """Same vector at different positions should produce different outputs."""
    freqs = precompute_rope_freqs(head_dim=16, max_seq_len=32)
    x = torch.ones(1, 1, 32, 16)
    out = apply_rope(x, freqs)
    # Position 0 and position 1 should differ
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])


# -- Model architecture --

def test_param_count(model, small_config):
    count = model.param_count()
    assert count > 0
    # Sanity: with tie_weights, embedding params counted once
    embed_params = small_config.vocab_size * small_config.d_model
    assert count >= embed_params


def test_forward_shape(model, small_config):
    x = torch.randint(0, small_config.vocab_size, (2, 32))
    logits, loss = model(x)
    assert logits.shape == (2, 32, small_config.vocab_size)
    assert loss is None


def test_forward_with_labels(model, small_config):
    x = torch.randint(0, small_config.vocab_size, (2, 32))
    labels = torch.randint(0, small_config.vocab_size, (2, 32))
    logits, loss = model(x, labels)
    assert logits.shape == (2, 32, small_config.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_loss_decreases_with_gradient_step(small_config):
    """One gradient step should reduce loss — confirms gradients flow correctly."""
    model = BearTransformer(small_config)
    x = torch.randint(0, small_config.vocab_size, (4, 32))
    labels = torch.randint(0, small_config.vocab_size, (4, 32))

    _, loss_before = model(x, labels)
    loss_before.backward()

    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad

    _, loss_after = model(x, labels)
    assert loss_after.item() < loss_before.item()


def test_generate(model, small_config):
    prompt = torch.randint(0, small_config.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=4)
    assert out.shape == (1, 12)  # 8 prompt + 4 generated


def test_generate_greedy(model, small_config):
    prompt = torch.randint(0, small_config.vocab_size, (1, 4))
    out = model.generate(prompt, max_new_tokens=2, temperature=0.0)
    assert out.shape == (1, 6)


def test_seq_len_assertion(model, small_config):
    """Should raise on sequences exceeding max_seq_len."""
    x = torch.randint(0, small_config.vocab_size, (1, small_config.max_seq_len + 1))
    with pytest.raises(AssertionError):
        model(x)


def test_weight_tying(model):
    """Embedding and LM head should share the same weight tensor."""
    assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_no_weight_tying():
    config = BearConfig(vocab_size=256, d_model=64, n_layers=1, n_heads=2, n_kv_heads=1, ffn_hidden=64, max_seq_len=32, tie_weights=False)
    model = BearTransformer(config)
    assert model.tok_emb.weight.data_ptr() != model.lm_head.weight.data_ptr()


def test_default_config_param_count():
    """Default config should be ~240M params (250M target)."""
    config = BearConfig()
    model = BearTransformer(config)
    count = model.param_count()
    assert 200_000_000 < count < 300_000_000, f"Expected ~250M params, got {count:,}"
