"""
Mesosfer Bear AI - Flash Attention Backend

Provides a unified attention function that auto-selects the best backend:
  1. flash-attn package (Dao-AILab / ROCm Triton) — fastest, native GQA
  2. PyTorch SDPA — auto-dispatches to native FlashAttention / AOTriton kernels

Usage:
    from engine.flashattion import bear_attention, get_attn_backend
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F


def _is_rocm() -> bool:
    """Return True if running on AMD ROCm (HIP)."""
    return bool(getattr(torch.version, "hip", None))


# -- Backend detection ------------------------------------------------------

if _is_rocm():
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

try:
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except ImportError:
        from flash_attn.flash_attn_interface import flash_attn_func  # type: ignore
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


def get_attn_backend() -> str:
    """Return which attention backend will be used."""
    if FLASH_ATTN_AVAILABLE:
        return "flash-attn v2 (Dao-AILab / ROCm Triton)"
    if torch.cuda.is_available():
        device_label = "ROCm AOTriton" if _is_rocm() else "CUDA FlashAttention"
        return f"PyTorch SDPA (Native {device_label} Kernel)"
    return "PyTorch SDPA (CPU Kernel)"


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
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)

    # SDPA path: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)  # (B, H, T, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # PyTorch 2.5+ SDPA supports enable_gqa parameter directly
    enable_gqa = (n_rep > 1)

    try:
        # Try native GQA parameter in SDPA (PyTorch 2.5+)
        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask,
            is_causal=(causal and mask is None),
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
        )
    except TypeError:
        # Fallback for PyTorch versions where enable_gqa isn't available
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
