"""Sequence-consistent dropout used by recurrent models."""

from __future__ import annotations

import torch
from torch import nn


class LockedDropout(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return values
        keep = 1.0 - self.probability
        mask = values.new_empty(values.shape[0], 1, values.shape[2]).bernoulli_(keep)
        return values * mask.div(keep)
