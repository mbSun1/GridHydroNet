"""Value, calendar, and sinusoidal position embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, maximum_length: int):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        values = torch.zeros(maximum_length, d_model, dtype=torch.float32)
        values[:, 0::2] = torch.sin(position * divisor)
        values[:, 1::2] = torch.cos(position * divisor[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values.unsqueeze(0), persistent=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[1] > self.values.shape[1]:
            raise ValueError("Sequence exceeds configured positional embedding length")
        return self.values[:, : sequence.shape[1]].to(dtype=sequence.dtype)


class SequenceEmbedding(nn.Module):
    def __init__(
        self,
        input_size: int,
        time_feature_size: int,
        d_model: int,
        dropout: float,
        maximum_length: int,
    ):
        super().__init__()
        self.value_projection = nn.Linear(input_size, d_model)
        self.time_projection = nn.Linear(time_feature_size, d_model, bias=False)
        self.position = SinusoidalPositionEmbedding(d_model, maximum_length)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.value_projection(forcing_window)
        embedded = embedded + self.position(embedded)
        if time_features is not None:
            embedded = embedded + self.time_projection(time_features)
        return self.dropout(embedded)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence), dim=1)
        return torch.sum(sequence * weights, dim=1)
