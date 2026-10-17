"""Pyraformer using a learned multi-resolution temporal pyramid."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.embedding import AttentionPooling, SequenceEmbedding
from dlcore.hydromodels.layers.pyraformer_blocks import PyramidEncoder
from dlcore.hydromodels.model_config import ModelConfig


class PyraformerRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        encoder_layers: int = 2,
        attention_heads: int = 2,
        feedforward_size: int = 256,
        pyramid_levels: int = 2,
        pyramid_window: int = 4,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
            "pyramid_levels": pyramid_levels,
            "pyramid_window": pyramid_window,
        }
        self.embedding = SequenceEmbedding(
            config.input_size,
            config.time_feature_size,
            d_model,
            config.dropout,
            config.lookback,
        )
        self.pyramid = PyramidEncoder(
            d_model, levels=pyramid_levels, window_size=pyramid_window
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attention_heads,
            dim_feedforward=feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=encoder_layers, enable_nested_tensor=False
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Linear(d_model, config.output_size)

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.embedding(forcing_window, time_features)
        values = self.encoder(self.pyramid(values))
        return self.head(self.pool(values))
