"""
Mesosfer Bear AI - Flash Attention Backend

Provides a unified attention function that auto-selects the best backend:
  1. flash-attn package (Dao-AILab) — fastest, native GQA, no KV repeat needed
  2. PyTorch SDPA — auto-dispatches to Flash/Memory-efficient/Math kernels

Usage:
    from engine.flashattion import bear_attention, get_attn_backend
"""

from typing import Optional

import torch
import torch.nn.functional as F

# -- Backend detection ------------------------------------------------------

try:
    from flash_attn import flash_attn_func  # type: ignore
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


def get_attn_backend() -> str:
    """Return which attention backend will be used."""
    if FLASH_ATTN_AVAILABLE:
        return "flash-attn (Dao-AILab)"
    return "PyTorch SDPA (auto Flash/Memory-efficient/Math)"


# -- Unified attention function ---------------------------------------------

def bear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_rep: int = 1,
    dropout_p: float = 0.0,
    causal: bool = True,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with automatic backend selection.

    Args:
        q: (B, T, n_heads, head_dim)     — query, already RoPE'd
        k: (B, T, n_kv_heads, head_dim)  — key, already RoPE'd
        v: (B, T, n_kv_heads, head_dim)  — value
        n_rep: repeat factor for GQA (n_heads // n_kv_heads)
        dropout_p: attention dropout probability
        causal: use causal (autoregressive) masking
        mask: explicit attention mask (only used in SDPA path, ignored by flash-attn)

    Returns:
        (B, T, n_heads, head_dim) attention output
    """
    if FLASH_ATTN_AVAILABLE and q.is_cuda:
        # flash_attn_func takes (B, T, H, D) and handles GQA natively
        # — no repeat_interleave needed, saves VRAM
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    # SDPA path: needs (B, H, T, D) layout and expanded KV heads
    q = q.transpose(1, 2)  # (B, H, T, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        is_causal=(causal and mask is None),
        dropout_p=dropout_p,
    )

    return out.transpose(1, 2)  # back to (B, T, H, D)
