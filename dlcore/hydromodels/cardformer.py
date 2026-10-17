"""CARDformer channel-aligned patch attention for hydrological regression."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dlcore.hydromodels.layers.crossformer_blocks import TwoStageAttentionBlock
from dlcore.hydromodels.layers.decomposition import SeriesDecomposition
from dlcore.hydromodels.model_config import ModelConfig


class CARDformerRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        encoder_layers: int = 2,
        attention_heads: int = 2,
        feedforward_size: int = 256,
        patch_length: int = 16,
        patch_stride: int = 8,
        moving_average: int = 25,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "moving_average": moving_average,
        }
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.decomposition = SeriesDecomposition(moving_average)
        self.patch_projection = nn.Linear(patch_length * 2, d_model)
        patch_count = max(
            1,
            math.ceil(max(0, config.lookback - patch_length) / patch_stride) + 1,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.input_size, patch_count, d_model)
        )
        self.channel_embedding = nn.Embedding(config.input_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TwoStageAttentionBlock(
                    d_model,
                    attention_heads,
                    feedforward_size,
                    config.dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.input_size * d_model),
            nn.Linear(config.input_size * d_model, feedforward_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(feedforward_size, config.output_size),
        )

    def make_patches(self, values: torch.Tensor) -> torch.Tensor:
        channels_first = values.transpose(1, 2)
        length = channels_first.shape[-1]
        if length < self.patch_length:
            channels_first = F.pad(
                channels_first, (self.patch_length - length, 0), mode="replicate"
            )
        remainder = (channels_first.shape[-1] - self.patch_length) % self.patch_stride
        if remainder:
            channels_first = F.pad(
                channels_first, (self.patch_stride - remainder, 0), mode="replicate"
            )
        return channels_first.unfold(-1, self.patch_length, self.patch_stride)

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seasonal, trend = self.decomposition(forcing_window)
        patches = torch.cat([self.make_patches(seasonal), self.make_patches(trend)], dim=-1)
        values = self.patch_projection(patches)
        if values.shape[2] != self.position_embedding.shape[2]:
            raise ValueError("CARDformer patch count differs from configured lookback")
        values = values + self.position_embedding
        values = values + self.channel_embedding.weight[None, :, None, :]
        for block in self.blocks:
            values = block(values)
        return self.head(values.mean(dim=2).flatten(1))
