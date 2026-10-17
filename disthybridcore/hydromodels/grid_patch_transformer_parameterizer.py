"""Patch Transformer encoder for static grid-cell HBV92 parameter generation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hbv92_parameter_head import (
    HBV92ParameterHead,
    extract_active_cells,
    restore_parameter_grid,
)


class GridPatchTransformerParameterizer(nn.Module):
    """Encode learned 30-day climate patches independently at each active cell."""

    def __init__(
        self,
        terrain_channels: int,
        climate_channels: int = 3,
        cell_batch_size: int = 256,
        patch_days: int = 30,
        model_width: int = 64,
        attention_heads: int = 4,
        layers: int = 2,
        feedforward_width: int = 128,
        dropout: float = 0.1,
        parameter_head_width: int = 16,
    ):
        super().__init__()
        if terrain_channels <= 0:
            raise ValueError("Grid HBV parameterization requires static attributes")
        if climate_channels != 3:
            raise ValueError("HBV climate sequence must contain P, T, and PET")
        if min(
            cell_batch_size, patch_days, model_width, attention_heads,
            layers, parameter_head_width,
        ) < 1:
            raise ValueError("Patch Transformer dimensions must be positive")
        if model_width % attention_heads:
            raise ValueError("Transformer model width must be divisible by attention heads")
        if model_width % 2:
            raise ValueError("Sinusoidal position encoding requires an even model width")
        if feedforward_width < model_width:
            raise ValueError("Transformer feedforward width must be at least model width")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Transformer dropout must be in [0,1)")

        self.terrain_channels = int(terrain_channels)
        self.climate_channels = int(climate_channels)
        self.cell_batch_size = int(cell_batch_size)
        self.patch_days = int(patch_days)
        self.model_width = int(model_width)
        self.attention_heads = int(attention_heads)
        self.layers = int(layers)
        self.feedforward_width = int(feedforward_width)
        self.dropout = float(dropout)
        self.parameter_head_width = int(parameter_head_width)
        self.patch_embedding = nn.Conv1d(
            self.climate_channels,
            self.model_width,
            kernel_size=self.patch_days,
            stride=self.patch_days,
        )
        self.terrain_embedding = nn.Linear(self.terrain_channels, self.model_width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=self.attention_heads,
            dim_feedforward=self.feedforward_width,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.climate_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.layers,
            norm=nn.LayerNorm(self.model_width),
            enable_nested_tensor=False,
        )
        feature_size = 2 * self.model_width + self.terrain_channels
        self.parameter_head = HBV92ParameterHead(
            feature_size=feature_size,
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
        """Return 12 raw values from causal-order-preserving climate patches."""
        sequence = climate_by_cell.transpose(1, 2)
        padding = (-sequence.shape[-1]) % self.patch_days
        if padding:
            sequence = F.pad(sequence, (0, padding))
        tokens = self.patch_embedding(sequence).transpose(1, 2)
        positions = torch.arange(
            tokens.shape[1], dtype=tokens.dtype, device=tokens.device
        )[:, None]
        frequencies = torch.exp(
            torch.arange(
                0, tokens.shape[2], 2,
                dtype=tokens.dtype,
                device=tokens.device,
            ) * (-math.log(10000.0) / tokens.shape[2])
        )
        angles = positions * frequencies[None]
        position_encoding = torch.empty_like(tokens[0])
        position_encoding[:, 0::2] = torch.sin(angles)
        position_encoding[:, 1::2] = torch.cos(angles)
        tokens = (
            tokens
            + position_encoding[None]
            + self.terrain_embedding(terrain_by_cell)[:, None]
        )
        encoded = self.climate_encoder(tokens)
        temporal_mean = encoded.mean(dim=1)
        temporal_std = encoded.std(dim=1, unbiased=False).clamp_min(1.0e-6)
        features = torch.cat((temporal_mean, temporal_std, terrain_by_cell), dim=-1)
        return self.parameter_head(features)

    def checkpoint_configuration(self) -> dict:
        return {
            "type": "patch_transformer",
            "patch_days": self.patch_days,
            "model_width": self.model_width,
            "attention_heads": self.attention_heads,
            "layers": self.layers,
            "feedforward_width": self.feedforward_width,
            "dropout": self.dropout,
            "climate_channels": self.climate_channels,
            "position_encoding": "sinusoidal",
            "terrain_conditioning": "additive_token_bias_and_head_concat",
            "temporal_pooling": "mean_std",
            "parameter_head": self.parameter_head.checkpoint_configuration(),
        }
