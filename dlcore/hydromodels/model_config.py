"""Immutable public configuration shared by all model families."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    lookback: int = 365
    input_size: int = 3
    time_feature_size: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    @property
    def output_size(self) -> int:
        return 1

    def validate(self) -> None:
        if self.lookback < 2 or self.input_size != 3:
            raise ValueError("Model requires lookback >= 2 and three forcing inputs")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        values = dict(payload)
        values.pop("output_size", None)
        values.pop("pred_len", None)
        for name in (
            "d_model",
            "hidden_size",
            "encoder_layers",
            "decoder_layers",
            "attention_heads",
            "feedforward_size",
            "moving_average",
            "patch_length",
            "patch_stride",
            "reformer_bucket_size",
            "reformer_hashes",
        ):
            values.pop(name, None)
        config = cls(**values)
        config.validate()
        return config
