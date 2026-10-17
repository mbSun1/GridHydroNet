"""Segment embedding and two-stage temporal/cross-variable attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentEmbedding(nn.Module):
    def __init__(self, patch_length: int, d_model: int):
        super().__init__()
        self.patch_length = int(patch_length)
        self.projection = nn.Linear(patch_length, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, variables = values.shape
        padding = (-length) % self.patch_length
        channels_first = values.transpose(1, 2)
        if padding:
            channels_first = F.pad(channels_first, (padding, 0), mode="replicate")
        patches = channels_first.unfold(-1, self.patch_length, self.patch_length)
        return self.projection(patches)


class TwoStageAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.variable_attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.variable_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, variables, segments, channels = values.shape
        temporal = values.reshape(batch * variables, segments, channels)
        normalized = self.temporal_norm(temporal)
        temporal = temporal + self.temporal_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        temporal = temporal.reshape(batch, variables, segments, channels)
        crossed = temporal.permute(0, 2, 1, 3).reshape(
            batch * segments, variables, channels
        )
        normalized = self.variable_norm(crossed)
        crossed = crossed + self.variable_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        crossed = crossed.reshape(batch, segments, variables, channels).permute(0, 2, 1, 3)
        return crossed + self.feedforward(self.output_norm(crossed))
