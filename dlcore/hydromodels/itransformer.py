"""iTransformer with meteorological variables represented as tokens."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.model_config import ModelConfig


class ITransformerRunoffModel(nn.Module):
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
        if d_model % attention_heads != 0:
            raise ValueError("iTransformer d_model must be divisible by attention_heads")
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
        }
        self.lookback = config.lookback
        self.input_size = config.input_size
        self.history_projection = nn.Linear(config.lookback, d_model)
        self.variable_embedding = nn.Embedding(config.input_size, d_model)
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
        if forcing_window.shape[1] != self.lookback:
            raise ValueError("iTransformer input length differs from configured lookback")
        tokens = self.history_projection(forcing_window.transpose(1, 2))
        tokens = tokens + self.variable_embedding.weight.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.head(encoded.flatten(1))
