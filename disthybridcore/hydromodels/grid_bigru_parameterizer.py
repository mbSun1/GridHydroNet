"""Shared per-cell BiGRU encoder for static HBV92 parameter generation."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .hbv92_parameter_head import (
    HBV92ParameterHead,
    extract_active_cells,
    restore_parameter_grid,
)


class GridBiGRUParameterizer(nn.Module):
    """Generate one time-invariant HBV92 parameter set per active grid cell."""

    def __init__(
        self,
        terrain_channels: int,
        hidden: int = 16,
        layers: int = 1,
        climate_channels: int = 3,
        cell_batch_size: int = 256,
    ):
        super().__init__()
        if terrain_channels <= 0:
            raise ValueError("Grid HBV parameterization requires static attributes")
        if hidden <= 0 or layers <= 0:
            raise ValueError("BiGRU hidden size and layer count must be positive")
        if climate_channels != 3:
            raise ValueError("HBV climate sequence must contain P, T, and PET")
        if cell_batch_size < 1:
            raise ValueError("BiGRU cell batch size must be positive")

        self.terrain_channels = int(terrain_channels)
        self.climate_channels = int(climate_channels)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.cell_batch_size = int(cell_batch_size)
        self.input_projection = nn.Linear(
            self.climate_channels + self.terrain_channels,
            self.hidden,
        )
        self.climate_encoder = nn.GRU(
            input_size=self.hidden,
            hidden_size=self.hidden,
            num_layers=self.layers,
            batch_first=True,
            bidirectional=True,
        )
        feature_size = 4 * self.hidden + self.terrain_channels
        self.parameter_head = HBV92ParameterHead(
            feature_size=feature_size,
            width=self.hidden,
        )
        self.parameter_bounds = self.parameter_head.parameter_bounds

    def forward(
        self,
        terrain: torch.Tensor,
        climate_sequence: torch.Tensor,
        basin_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        climate, static, valid, grid_shape = extract_active_cells(
            terrain,
            climate_sequence,
            basin_mask,
            terrain_channels=self.terrain_channels,
            climate_channels=self.climate_channels,
        )
        chunks = []
        for start in range(0, static.shape[0], self.cell_batch_size):
            climate_chunk = climate[start:start + self.cell_batch_size]
            static_chunk = static[start:start + self.cell_batch_size]
            if self.training and torch.is_grad_enabled():
                encoded = checkpoint(
                    self.encode_cells,
                    climate_chunk,
                    static_chunk,
                    use_reentrant=False,
                )
            else:
                encoded = self.encode_cells(climate_chunk, static_chunk)
            chunks.append(encoded)
        raw = restore_parameter_grid(torch.cat(chunks, dim=0), valid, grid_shape)
        return self.parameter_head.map_parameters(raw)

    def encode_cells(
        self,
        climate_by_cell: torch.Tensor,
        terrain_by_cell: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one cell batch and return 12 raw parameter values per cell."""
        static_sequence = terrain_by_cell[:, None].expand(
            -1, climate_by_cell.shape[1], -1
        )
        encoder_input = torch.cat((climate_by_cell, static_sequence), dim=-1)
        projected_input = torch.relu(self.input_projection(encoder_input))
        sequence_output, _ = self.climate_encoder(projected_input)
        temporal_mean = sequence_output.mean(dim=1)
        temporal_std = sequence_output.std(dim=1, unbiased=False).clamp_min(1.0e-6)
        features = torch.cat((temporal_mean, temporal_std, terrain_by_cell), dim=-1)
        return self.parameter_head(features)

    def checkpoint_configuration(self) -> dict:
        return {
            "type": "bigru",
            "hidden": self.hidden,
            "layers": self.layers,
            "climate_channels": self.climate_channels,
            "temporal_pooling": "mean_std",
            "parameter_head": self.parameter_head.checkpoint_configuration(),
        }
