"""Default structural and optimization settings for DistHydroNet."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    parameter_encoder: str = "bigru"
    hbv_parameter_hidden: int = 16
    hbv_parameter_layers: int = 1
    hbv_parameter_context_block_days: int = 1
    hbv_parameter_cell_batch_size: int = 256

    weight_decay: float = 1.0e-5
    gradient_clip_val: float = 1.0
