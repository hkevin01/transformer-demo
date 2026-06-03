from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from src.model.attention import MultiHeadSelfAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """Encoder block: self-attention + feed-forward with residual and layer norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = PositionwiseFeedForward(
            embed_dim=embed_dim,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        attn_out, attn_weights = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights
