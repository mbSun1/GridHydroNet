"""Transformer blocks shared by the sequence models."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.attention import ProbSparseSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, feedforward_size, dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(values)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = values + attended
        return values + self.feedforward(self.norm_feedforward(values))


class SparseTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        feedforward_size: int,
        dropout: float,
        factor: int = 5,
    ):
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = ProbSparseSelfAttention(d_model, heads, factor, dropout)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, feedforward_size, dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.feedforward(self.norm_feedforward(values))
