"""Shared HBV92 parameter head and grid-cell tensor utilities."""

from __future__ import annotations

import torch
from torch import nn


class HBV92ParameterHead(nn.Module):
    """Map one cell embedding to 12 raw HBV92 values with the original MLP."""

    def __init__(self, feature_size: int, width: int = 16):
        super().__init__()
        if min(feature_size, width) < 1:
            raise ValueError("Parameter-head dimensions must be positive")
        self.feature_size = int(feature_size)
        self.width = int(width)
        self.parameter_bounds = (
            ("beta", 1.0, 6.0),
            ("fc", 50.0, 1000.0),
            ("k0", 0.05, 0.9),
            ("k1", 0.01, 0.5),
            ("k2", 0.001, 0.2),
            ("lp", 0.2, 1.0),
            ("perc", 0.0, 10.0),
            ("uzl", 0.0, 100.0),
            ("tt", -2.5, 2.5),
            ("cfmax", 0.5, 10.0),
            ("cfr", 0.0, 0.1),
            ("cwh", 0.0, 0.2),
        )
        self.network = nn.Sequential(
            nn.LayerNorm(self.feature_size),
            nn.Linear(self.feature_size, self.width),
            nn.SiLU(),
            nn.Linear(self.width, len(self.parameter_bounds)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def map_parameters(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply differentiable physical bounds to the 12 raw fields."""
        if raw.shape[1] != len(self.parameter_bounds):
            raise ValueError("raw parameter tensor must have 12 channels")
        bounded = {}
        for index, (name, lower, upper) in enumerate(self.parameter_bounds):
            fraction = torch.sigmoid(raw[:, index:index + 1])
            bounded[name] = lower + (upper - lower) * fraction
        return bounded

    def checkpoint_configuration(self) -> dict:
        return {
            "type": "shared_two_layer_mlp_joint_output",
            "width": self.width,
            "activation": "silu",
            "output_parameters": len(self.parameter_bounds),
        }


def extract_active_cells(
    terrain: torch.Tensor,
    climate_sequence: torch.Tensor,
    basin_mask: torch.Tensor,
    *,
    terrain_channels: int,
    climate_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    """Validate grid tensors and return active-cell climate and terrain arrays."""
    if terrain.ndim != 4:
        raise ValueError("terrain must have shape [B,C,H,W]")
    if climate_sequence.ndim != 5:
        raise ValueError("climate_sequence must have shape [B,L,C,H,W]")
    batch, channels, height, width = terrain.shape
    if channels != terrain_channels:
        raise ValueError(
            f"terrain channels={channels}; expected {terrain_channels}"
        )
    if (
        climate_sequence.shape[0] != batch
        or climate_sequence.shape[2] != climate_channels
        or climate_sequence.shape[-2:] != (height, width)
    ):
        raise ValueError(
            "climate sequence does not match terrain: "
            f"{tuple(climate_sequence.shape)} vs {tuple(terrain.shape)}"
        )
    if climate_sequence.shape[1] < 1:
        raise ValueError("climate_sequence must contain at least one step")
    if basin_mask.ndim == 4 and basin_mask.shape[1] == 1:
        basin_mask = basin_mask[:, 0]
    if basin_mask.shape != (batch, height, width):
        raise ValueError("basin_mask must have shape [B,H,W]")
    valid = basin_mask.to(torch.bool).reshape(-1)
    if not valid.any():
        raise ValueError("basin_mask contains no active grid cells")

    length = climate_sequence.shape[1]
    climate_by_cell = climate_sequence.permute(0, 3, 4, 1, 2).reshape(
        batch * height * width, length, climate_channels
    )[valid]
    terrain_by_cell = terrain.permute(0, 2, 3, 1).reshape(
        batch * height * width, channels
    )[valid]
    if not torch.isfinite(climate_by_cell).all() or not torch.isfinite(terrain_by_cell).all():
        raise ValueError("Parameter-generator inputs must be finite within the basin")
    return climate_by_cell, terrain_by_cell, valid, (batch, height, width)


def restore_parameter_grid(
    raw_valid: torch.Tensor,
    valid: torch.Tensor,
    grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Restore active-cell raw parameters to a masked ``[B,12,H,W]`` grid."""
    batch, height, width = grid_shape
    raw_flat = raw_valid.new_zeros(batch * height * width, raw_valid.shape[-1])
    raw_flat[valid] = raw_valid
    return raw_flat.view(batch, height, width, -1).permute(0, 3, 1, 2)
