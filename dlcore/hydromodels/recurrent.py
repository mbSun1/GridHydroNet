"""LSTM-mask adaptation plus BiLSTM, GRU, BiGRU, and vanilla RNN."""

from __future__ import annotations

import torch
from torch import nn

from dlcore.hydromodels.layers.dropout import LockedDropout
from dlcore.hydromodels.model_config import ModelConfig


class RecurrentRunoffModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        cell_type: str,
        bidirectional: bool = False,
        hidden_size: int = 256,
        encoder_layers: int = 2,
    ):
        super().__init__()
        config.validate()
        cells = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        if cell_type not in cells:
            raise ValueError(f"Unsupported recurrent cell: {cell_type}")
        self.architecture_parameters = {
            "cell_type": cell_type,
            "bidirectional": bidirectional,
            "hidden_size": hidden_size,
            "encoder_layers": encoder_layers,
        }
        recurrent_dropout = config.dropout if encoder_layers > 1 else 0.0
        arguments = {
            "input_size": config.input_size,
            "hidden_size": hidden_size,
            "num_layers": encoder_layers,
            "batch_first": True,
            "dropout": recurrent_dropout,
            "bidirectional": bidirectional,
        }
        if cell_type == "rnn":
            arguments["nonlinearity"] = "tanh"
        self.recurrent = cells[cell_type](**arguments)
        self.locked_dropout = LockedDropout(config.dropout)
        directions = 2 if bidirectional else 1
        self.layers = encoder_layers
        self.directions = directions
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * directions),
            nn.Linear(hidden_size * directions, hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.output_size),
        )

    def forward(
        self,
        forcing_window: torch.Tensor,
        time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.locked_dropout(forcing_window)
        result = self.recurrent(values)
        state = result[1][0] if isinstance(result[1], tuple) else result[1]
        state = state.reshape(self.layers, self.directions, values.shape[0], -1)[-1]
        state = state.transpose(0, 1).reshape(values.shape[0], -1)
        return self.head(state)
