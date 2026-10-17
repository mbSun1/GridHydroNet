"""Residual causal temporal convolution model for outlet runoff regression."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dlcore.hydromodels.model_config import ModelConfig


class CausalTemporalBlock(nn.Module):
    """Two residual causal convolutions at one dilation scale."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if min(channels, dilation, kernel_size) < 1:
            raise ValueError("TCN channels, dilation, and kernel size must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("TCN dropout must be in the interval [0, 1)")
        self.channels = int(channels)
        self.dilation = int(dilation)
        self.kernel_size = int(kernel_size)
        self.left_padding = (self.kernel_size - 1) * self.dilation
        self.first_convolution = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.second_convolution = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def causal_convolution(
        self,
        sequence: torch.Tensor,
        convolution: nn.Conv1d,
    ) -> torch.Tensor:
        padded = F.pad(sequence, (self.left_padding, 0))
        return convolution(padded)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != self.channels:
            raise ValueError("TCN block input must have shape [batch, channels, time]")
        update = self.causal_convolution(sequence, self.first_convolution)
        update = self.dropout(self.activation(update))
        update = self.causal_convolution(update, self.second_convolution)
        update = self.dropout(self.activation(update))
        return sequence + update


class TCNRunoffModel(nn.Module):
    """Predict target-day outlet runoff from a causal forcing history."""

    def __init__(
        self,
        config: ModelConfig,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        head_hidden_size: int = 128,
    ):
        super().__init__()
        config.validate()
        if min(hidden_channels, kernel_size, head_hidden_size) < 1:
            raise ValueError("TCN dimensions must be positive")
        if not dilations or any(dilation < 1 for dilation in dilations):
            raise ValueError("TCN dilations must be a nonempty positive sequence")
        receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        self.architecture_parameters = {
            "hidden_channels": hidden_channels,
            "kernel_size": kernel_size,
            "dilations": list(dilations),
            "residual_blocks": len(dilations),
            "convolutions_per_block": 2,
            "receptive_field_days": receptive_field,
            "head_hidden_size": head_hidden_size,
            "causal_padding": "left_only",
        }
        self.lookback = int(config.lookback)
        self.input_projection = nn.Conv1d(
            config.input_size,
            hidden_channels,
            kernel_size=1,
        )
        self.temporal_blocks = nn.ModuleList([
            CausalTemporalBlock(
                hidden_channels,
                dilation,
                kernel_size=kernel_size,
                dropout=config.dropout,
            )
            for dilation in dilations
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, head_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden_size, config.output_size),
        )

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if forcing_window.ndim != 3:
            raise ValueError("TCN forcing input must have shape [batch, time, features]")
        if forcing_window.shape[1] != self.lookback:
            raise ValueError("TCN input length differs from configured lookback")
        if forcing_window.shape[2] != self.input_projection.in_channels:
            raise ValueError("TCN forcing feature count differs from model input size")
        sequence = self.input_projection(forcing_window.transpose(1, 2))
        for block in self.temporal_blocks:
            sequence = block(sequence)
        target_day_features = sequence[:, :, -1]
        return self.head(target_day_features)
