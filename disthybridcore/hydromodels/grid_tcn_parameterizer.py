"""Causal temporal-convolution encoder for grid-cell HBV92 parameters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hbv92_parameter_head import (
    HBV92ParameterHead,
    extract_active_cells,
    restore_parameter_grid,
)


class CausalTCNBlock(nn.Module):
    """One residual dilated convolution with left-only temporal padding."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(channels, dilation, kernel_size) < 1:
            raise ValueError("TCN block dimensions and dilation must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("TCN dropout must be in [0,1)")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.convolution = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != self.channels:
            raise ValueError("TCN block input must have shape [N,channels,L]")
        update = self.convolution(F.pad(sequence, (self.left_padding, 0)))
        return sequence + self.dropout(self.activation(update))


class GridTCNParameterizer(nn.Module):
    """Generate static HBV92 fields from causal daily climate features."""

    def __init__(
        self,
        terrain_channels: int,
        climate_channels: int = 3,
        cell_batch_size: int = 256,
        temporal_channels: int = 32,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        ),
        dropout: float = 0.1,
        parameter_head_width: int = 16,
    ):
        super().__init__()
        if terrain_channels <= 0:
            raise ValueError("Grid HBV parameterization requires static attributes")
        if climate_channels != 3:
            raise ValueError("HBV climate sequence must contain P, T, and PET")
        if min(
            cell_batch_size, temporal_channels, kernel_size, parameter_head_width,
        ) < 1:
            raise ValueError("TCN dimensions must be positive")
        if not dilations or any(value < 1 for value in dilations):
            raise ValueError("TCN dilations must be a nonempty positive sequence")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("TCN dropout must be in [0,1)")

        self.terrain_channels = int(terrain_channels)
        self.climate_channels = int(climate_channels)
        self.cell_batch_size = int(cell_batch_size)
        self.temporal_channels = int(temporal_channels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout_probability = float(dropout)
        self.parameter_head_width = int(parameter_head_width)
        self.receptive_field_days = 1 + (self.kernel_size - 1) * sum(
            self.dilations
        )

        self.input_projection = nn.Conv1d(
            self.climate_channels + self.terrain_channels,
            self.temporal_channels,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList([
            CausalTCNBlock(
                self.temporal_channels,
                dilation,
                kernel_size=self.kernel_size,
                dropout=self.dropout_probability,
            )
            for dilation in self.dilations
        ])
        self.parameter_head = HBV92ParameterHead(
            feature_size=2 * self.temporal_channels + self.terrain_channels,
            width=self.parameter_head_width,
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
        """Encode one active-cell batch into 12 raw parameter values."""
        static_sequence = terrain_by_cell[:, None].expand(
            -1, climate_by_cell.shape[1], -1
        )
        sequence = torch.cat((climate_by_cell, static_sequence), dim=-1)
        sequence = self.input_projection(sequence.transpose(1, 2))
        for block in self.temporal_blocks:
            sequence = block(sequence)
        temporal_mean = sequence.mean(dim=2)
        temporal_std = sequence.std(dim=2, unbiased=False).clamp_min(1.0e-6)
        features = torch.cat(
            (temporal_mean, temporal_std, terrain_by_cell), dim=1
        )
        return self.parameter_head(features)

    def checkpoint_configuration(self) -> dict:
        return {
            "type": "tcn",
            "temporal_channels": self.temporal_channels,
            "kernel_size": self.kernel_size,
            "dilations": list(self.dilations),
            "residual_blocks": len(self.temporal_blocks),
            "receptive_field_days": self.receptive_field_days,
            "dropout": self.dropout_probability,
            "climate_channels": self.climate_channels,
            "terrain_conditioning": "daily_input_concat_and_head_concat",
            "causal_padding": "left_only",
            "temporal_pooling": "mean_std",
            "parameter_head": self.parameter_head.checkpoint_configuration(),
        }
