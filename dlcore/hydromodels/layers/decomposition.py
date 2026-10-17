"""Trend, seasonal, and growth decomposition primitives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        channels_first = values.transpose(1, 2)
        padded = F.pad(channels_first, (left, right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(values)
        return values - trend, trend


class ExponentialSmoothing(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.logit_alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.logit_alpha).view(1, 1, -1)
        states = [values[:, 0:1]]
        for index in range(1, values.shape[1]):
            states.append(alpha * values[:, index : index + 1] + (1.0 - alpha) * states[-1])
        return torch.cat(states, dim=1)
