"""DLinear trend-seasonal decomposition adapted to outlet regression."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.decomposition import SeriesDecomposition
from dlcore.hydromodels.model_config import ModelConfig


class DLinearRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        feedforward_size: int = 256,
        moving_average: int = 25,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "feedforward_size": feedforward_size,
            "moving_average": moving_average,
        }
        self.decomposition = SeriesDecomposition(moving_average)
        self.seasonal_projection = nn.Linear(config.lookback, d_model)
        self.trend_projection = nn.Linear(config.lookback, d_model)
        self.head = nn.Sequential(
            nn.Linear(config.input_size * d_model * 2, feedforward_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(feedforward_size, config.output_size),
        )

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forcing_window.shape[1] != self.seasonal_projection.in_features:
            raise ValueError("DLinear input length differs from configured lookback")
        seasonal, trend = self.decomposition(forcing_window)
        seasonal = self.seasonal_projection(seasonal.transpose(1, 2))
        trend = self.trend_projection(trend.transpose(1, 2))
        features = torch.cat([seasonal, trend], dim=-1).flatten(1)
        return self.head(features)
