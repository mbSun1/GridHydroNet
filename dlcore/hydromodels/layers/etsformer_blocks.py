"""Growth and frequency-domain seasonal extraction for ETSformer."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.decomposition import ExponentialSmoothing


class ETSBlock(nn.Module):
    def __init__(self, d_model: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.smoothing = ExponentialSmoothing(d_model)
        self.growth_projection = nn.Linear(d_model, d_model)
        self.season_projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        level = self.smoothing(values)
        growth = torch.diff(level, dim=1, prepend=level[:, :1])
        residual = values - level
        spectrum = torch.fft.rfft(residual, dim=1)
        seasonal = torch.fft.irfft(spectrum, n=values.shape[1], dim=1)
        combined = level + self.growth_projection(growth) + self.season_projection(seasonal)
        return combined + self.feedforward(self.norm(combined))
