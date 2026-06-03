from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Expected shapes:
    - q: (batch, heads, target_len, head_dim)
    - k: (batch, heads, source_len, head_dim)
    - v: (batch, heads, source_len, head_dim)
    - mask (optional): broadcastable to (batch, heads, target_len, source_len)
      - bool mask: True means token is allowed to attend
      - float mask: additive mask values (e.g. 0 or -inf)
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))
            else:
                scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attended = torch.matmul(attn_weights, v)
        return attended, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention built from linear projections + scaled attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.last_shapes: Dict[str, Tuple[int, ...]] = {}

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def _expand_mask(self, mask: Tensor, batch_size: int, seq_len: int) -> Tensor:
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() == 4:
            return mask
        raise ValueError("mask must have shape (L,S), (B,L,S), or (B,H,L,S)")

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        self.last_shapes = {
            "input": tuple(x.shape),
            "q": tuple(q.shape),
            "k": tuple(k.shape),
            "v": tuple(v.shape),
        }

        attn_mask = None
        if mask is not None:
            attn_mask = self._expand_mask(mask, batch_size, seq_len)

        attended, attn_weights = self.attention(q=q, k=k, v=v, mask=attn_mask)
        out = self._merge_heads(attended)
        out = self.out_proj(out)
        return out, attn_weights
