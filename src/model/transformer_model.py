from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from src.model.transformer_block import TransformerEncoderBlock


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        return self.token_embedding(tokens) + self.pos_embedding(positions)


class TransformerSequenceClassifier(nn.Module):
    """Sequence classifier based on Transformer encoder blocks."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionalEmbedding(vocab_size, embed_dim, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.attention_maps: List[Tensor] = []

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        x = self.embedding(tokens)
        maps: List[Tensor] = []
        for block in self.blocks:
            x, attn_weights = block(x, mask=mask)
            maps.append(attn_weights)

        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        self.attention_maps = maps
        return logits, maps


class BaselineSequenceClassifier(nn.Module):
    """Baseline model without self-attention for comparison."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = TokenPositionalEmbedding(vocab_size, embed_dim, max_seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens)
        pooled = x.mean(dim=1)
        return self.mlp(pooled)
