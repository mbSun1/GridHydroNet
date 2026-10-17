"""Losses used by all deep-learning rainfall-runoff models."""

from __future__ import annotations

import torch


def finite_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        raise ValueError("Batch contains no finite targets")
    return torch.mean((prediction[valid] - target[valid]) ** 2)
