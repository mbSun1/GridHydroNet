"""Informer with deterministic ProbSparse attention and sequence distillation."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.embedding import AttentionPooling, SequenceEmbedding
from dlcore.hydromodels.layers.transformer_blocks import SparseTransformerBlock
from dlcore.hydromodels.model_config import ModelConfig


class InformerRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        encoder_layers: int = 2,
        attention_heads: int = 2,
        feedforward_size: int = 256,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
        }
        self.embedding = SequenceEmbedding(
            config.input_size,
            config.time_feature_size,
            d_model,
            config.dropout,
            config.lookback,
        )
        self.blocks = nn.ModuleList(
            [
                SparseTransformerBlock(
                    d_model,
                    attention_heads,
                    feedforward_size,
                    config.dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.distillers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                )
                for _ in range(max(0, encoder_layers - 1))
            ]
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Linear(d_model, config.output_size)

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.embedding(forcing_window, time_features)
        for index, block in enumerate(self.blocks):
            values = block(values)
            if index < len(self.distillers):
                values = self.distillers[index](values.transpose(1, 2)).transpose(1, 2)
        return self.head(self.pool(values))
