"""Crossformer segment-wise temporal and cross-variable attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from dlcore.hydromodels.layers.crossformer_blocks import (
    SegmentEmbedding,
    TwoStageAttentionBlock,
)
from dlcore.hydromodels.model_config import ModelConfig


class CrossformerRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        encoder_layers: int = 2,
        attention_heads: int = 2,
        feedforward_size: int = 256,
        patch_length: int = 16,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
            "patch_length": patch_length,
        }
        self.segment_embedding = SegmentEmbedding(patch_length, d_model)
        segment_count = math.ceil(config.lookback / patch_length)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.input_size, segment_count, d_model)
        )
        self.variable_embedding = nn.Embedding(config.input_size, d_model)
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

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.segment_embedding(forcing_window)
        if values.shape[2] != self.position_embedding.shape[2]:
            raise ValueError("Crossformer segment count differs from configured lookback")
        values = values + self.position_embedding
        values = values + self.variable_embedding.weight[None, :, None, :]
        for block in self.blocks:
            values = block(values)
        pooled = values.mean(dim=2).flatten(1)
        return self.head(pooled)
