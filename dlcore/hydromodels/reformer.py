"""Reformer LSH attention with automatic padding for a 365-day window."""

from __future__ import annotations

import torch
from reformer_pytorch import LSHSelfAttention
from torch import nn
from torch.nn import functional as F

from dlcore.hydromodels.layers.embedding import AttentionPooling, SequenceEmbedding
from dlcore.hydromodels.layers.transformer_blocks import FeedForward
from dlcore.hydromodels.model_config import ModelConfig


class ReformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        attention_heads: int,
        feedforward_size: int,
        bucket_size: int,
        hashes: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = LSHSelfAttention(
            dim=d_model,
            heads=attention_heads,
            bucket_size=bucket_size,
            n_hashes=hashes,
            causal=False,
            dropout=dropout,
            post_attn_dropout=dropout,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, feedforward_size, dropout)

    def forward(self, values: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(
            self.attention_norm(values), input_mask=input_mask
        )
        return values + self.feedforward(self.feedforward_norm(values))


class ReformerRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        d_model: int = 256,
        encoder_layers: int = 2,
        attention_heads: int = 2,
        feedforward_size: int = 256,
        bucket_size: int = 4,
        hashes: int = 4,
    ):
        super().__init__()
        config.validate()
        self.architecture_parameters = {
            "d_model": d_model,
            "encoder_layers": encoder_layers,
            "attention_heads": attention_heads,
            "feedforward_size": feedforward_size,
            "bucket_size": bucket_size,
            "hashes": hashes,
        }
        divisor = 2 * bucket_size
        self.padded_length = ((config.lookback + divisor - 1) // divisor) * divisor
        self.embedding = SequenceEmbedding(
            config.input_size,
            config.time_feature_size,
            d_model,
            config.dropout,
            self.padded_length,
        )
        self.blocks = nn.ModuleList(
            [
                ReformerBlock(
                    d_model,
                    attention_heads,
                    feedforward_size,
                    bucket_size,
                    hashes,
                    config.dropout,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.pool = AttentionPooling(d_model)
        self.head = nn.Linear(d_model, config.output_size)

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actual_length = forcing_window.shape[1]
        padding = self.padded_length - actual_length
        if padding < 0:
            raise ValueError("Reformer input exceeds configured lookback")
        if padding:
            forcing_window = F.pad(forcing_window, (0, 0, 0, padding))
            if time_features is not None:
                time_features = F.pad(time_features, (0, 0, 0, padding))
        values = self.embedding(forcing_window, time_features)
        input_mask = torch.arange(self.padded_length, device=values.device)[None, :]
        input_mask = input_mask < actual_length
        input_mask = input_mask.expand(values.shape[0], -1)
        for block in self.blocks:
            values = block(values, input_mask)
        return self.head(self.pool(values[:, :actual_length]))
