"""
Mesosfer Bear AI - Transformer Language Model

Llama-style decoder-only transformer with:
  - RMSNorm (pre-norm)
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA) with Flash Attention
  - SwiGLU FFN
  - Weight-tied embedding ↔ LM head

Default config targets ~250M params at depth 16.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.flashattion import bear_attention


# -- Model Config -----------------------------------------------------------

@dataclass
class BearConfig:
    """
    ~250M param config (weight-tied):
      embed 60K*1024 = 61M
      16 layers * ~11M = 179M
      total ≈ 240M
    """
    vocab_size: int = 60000
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16        # query heads
    n_kv_heads: int = 4      # key/value heads (GQA ratio 4:1)
    ffn_hidden: int = 2816   # SwiGLU hidden dim (≈ 4*d_model*2/3 rounded to 256)
    max_seq_len: int = 4096
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0     # ponytail: 0 for pre-training, set >0 for fine-tuning
    tie_weights: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "BearConfig":
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# -- RMSNorm ----------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# -- Rotary Position Embedding (RoPE) --------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies: shape (max_seq_len, head_dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to x.
    x: (batch, n_heads, seq_len, head_dim)
    freqs: (seq_len, head_dim//2) complex
    """
    # Reshape to pairs of floats -> complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


# -- Grouped Query Attention ------------------------------------------------

class GQAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).
    Delegates to bear_attention() from engine.flashattion for backend selection.
    """

    def __init__(self, config: BearConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout_p = config.dropout

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE (needs B, H, T, D layout)
        q = apply_rope(q.transpose(1, 2), rope_freqs).transpose(1, 2)  # -> (B, T, H, D)
        k = apply_rope(k.transpose(1, 2), rope_freqs).transpose(1, 2)

        drop_p = self.dropout_p if self.training else 0.0

        attn_out = bear_attention(q, k, v, n_rep=self.n_rep, dropout_p=drop_p, causal=True, mask=mask)

        attn_out = attn_out.contiguous().view(B, T, -1)
        return self.wo(attn_out)


# -- SwiGLU Feed-Forward ---------------------------------------------------

class SwiGLU(nn.Module):
    """
    SwiGLU FFN (Shazeer 2020): gate(x) * up(x), then down projection.
    More parameter-efficient than standard FFN at same quality.
    """

    def __init__(self, config: BearConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, config.ffn_hidden, bias=False)
        self.w_down = nn.Linear(config.ffn_hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# -- Transformer Block -----------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn → residual → RMSNorm → FFN → residual."""

    def __init__(self, config: BearConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GQAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# -- Full Model -------------------------------------------------------------

class BearTransformer(nn.Module):
    """
    Mesosfer Bear AI decoder-only transformer.

    Architecture: Llama-style with RoPE + GQA + SwiGLU + RMSNorm.
    Outputs raw logits (no softmax) for use with CrossEntropyLoss.
    """

    def __init__(self, config: BearConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (registered as buffer, moves with .to(device))
        head_dim = config.d_model // config.n_heads
        rope_freqs = precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        """Small normal init + scaled residual init for deep networks."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale down residual projections by 1/sqrt(2*n_layers) for training stability
        # ponytail: only apply to output projections (wo, w_down)
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.wo.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.w_down.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) target IDs for loss computation (optional)

        Returns:
            (logits, loss) — loss is None if labels not provided.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        x = self.drop(self.tok_emb(input_ids))
        rope_freqs = self.rope_freqs[:T]

        for layer in self.layers:
            x = layer(x, rope_freqs)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    def param_count(self) -> int:
        """Total trainable parameters (excluding tied duplicates)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr())
                total += p.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation (no KV cache — good enough for <4K tokens).
        ponytail: KV cache adds ~150 lines of complexity. Add when inference speed matters.
        """
        for _ in range(max_new_tokens):
            # Crop to max context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # last position

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids
