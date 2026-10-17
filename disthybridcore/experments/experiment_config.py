"""Resolve internal defaults and validate experiment settings."""

from argparse import Namespace
import math
from pathlib import Path
import re
import unicodedata

import pandas as pd

from disthybridcore.hydromodels.model_config import ModelConfig


def normalize_basin_name(name: str) -> str:
    display_name = str(name).strip()
    ascii_name = unicodedata.normalize("NFKD", display_name).encode(
        "ascii", "ignore"
    ).decode("ascii")
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    if not identifier:
        raise ValueError("basin_name must contain a letter or number usable in a directory name")
    return identifier


def internal_defaults(project_root: Path) -> dict:
    model = ModelConfig()
    return {
        "results_dir": str((project_root / "outputs" / "hybrid").resolve()),
        "precision": "32-true", "num_workers": 0, "pin_memory": True,
        "deterministic": True, "generate_figures": True,
        "batch_size": 1,
        "parameter_encoder": model.parameter_encoder,
        "hbv_parameter_hidden": model.hbv_parameter_hidden,
        "hbv_parameter_layers": model.hbv_parameter_layers,
        "hbv_parameter_cell_batch_size": model.hbv_parameter_cell_batch_size,
        "hbv_parameter_context_block_days": model.hbv_parameter_context_block_days,
        "weight_decay": model.weight_decay,
        "gradient_clip_val": model.gradient_clip_val,
    }


def validate_experiment_parameters(params: Namespace) -> None:
    if params.parameter_encoder not in {"bigru", "tcn", "patch_transformer"}:
        raise ValueError(
            "parameter_encoder must be bigru, tcn, or patch_transformer"
        )
    if params.d8_substeps_per_day not in {1, 2, 4, 8, 24, 48, 96}:
        raise ValueError("d8_substeps_per_day must be one of 1, 2, 4, 8, 24, 48, 96")
    if params.channel_area_threshold_km2 <= 0:
        raise ValueError("channel_area_threshold_km2 must be positive")
    if params.gauge_snap_max_distance_km <= 0:
        raise ValueError("gauge_snap_max_distance_km must be positive")
    if not 0 <= params.gauge_area_relative_tolerance < 1:
        raise ValueError("gauge_area_relative_tolerance must be in [0,1)")
    if params.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if params.batch_size != 1:
        raise ValueError("Chronological cross-window state carry requires batch_size=1")
    if params.spinup_days < 1 or params.train_window_days < 1:
        raise ValueError("spinup_days and train_window_days must be positive")
    if params.test_window_days < 1:
        raise ValueError("test_window_days must be positive")
    if min(params.hbv_parameter_hidden, params.hbv_parameter_layers,
           params.hbv_parameter_cell_batch_size) < 1:
        raise ValueError("Parameter-encoder dimensions and cell batch size must be positive")
    if params.hbv_parameter_context_block_days < 1:
        raise ValueError("hbv_parameter_context_block_days must be positive")
    if params.down_factor < 1 or not math.isfinite(params.dem_grid_size_km) or params.dem_grid_size_km <= 0:
        raise ValueError("down_factor and dem_grid_size_km must be positive")
    if hasattr(params, "grid_height") and (
        params.grid_height % params.down_factor or params.grid_width % params.down_factor
    ):
        raise ValueError("DEM grid dimensions must be divisible by down_factor")
    if pd.Timestamp(params.train_end) <= pd.Timestamp(params.train_start):
        raise ValueError("train_end must follow train_start")
    if pd.Timestamp(params.test_end) <= pd.Timestamp(params.test_start):
        raise ValueError("test_end must follow test_start")
    if pd.Timestamp(params.test_start) < pd.Timestamp(params.train_end):
        raise ValueError("test_start must not precede the end of the parameter-training context")
    if params.parameter_encoder in {"tcn", "patch_transformer"} and params.hbv_parameter_context_block_days != 1:
        raise ValueError("TCN and Patch Transformer require daily parameter-context steps")


def resolve_experiment_parameters(
    params: Namespace, source_file: str, *, internal_overrides: dict | None = None,
) -> Namespace:
    if getattr(params, "end_dates_inclusive", True):
        params.train_end = (pd.Timestamp(params.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        params.test_end = (pd.Timestamp(params.test_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        params.end_dates_inclusive = False
    project_root = Path(source_file).resolve().parents[2]
    params.project_root = str(project_root)
    params.data_repository = str(project_root / "data_repository")
    settings = internal_defaults(project_root)
    parser_owned = (
        "seed", "basin_name", "temporal_resolution", "train_window_days",
        "test_window_days", "meteo_source_root", "dem_grid_size_km",
        "down_factor", "outlet_id", "channel_area_threshold_km2",
        "gauge_snap_max_distance_km", "gauge_area_relative_tolerance",
        "parameter_encoder", "d8_substeps_per_day",
    )
    settings.update({name: getattr(params, name) for name in parser_owned})
    if internal_overrides:
        unknown = set(internal_overrides).difference(settings)
        if unknown:
            raise KeyError(f"Unknown internal settings: {sorted(unknown)}")
        settings.update(internal_overrides)
    for name, value in settings.items():
        setattr(params, name, value)
    params.basin = normalize_basin_name(params.basin_name)
    params.grid_resolution_m = params.dem_grid_size_km * 1000.0
    params.terrain_channels = 4
    encoder_label = {
        "bigru": "BiGRU",
        "tcn": "TCN",
        "patch_transformer": "PatchTransformer",
    }[params.parameter_encoder]
    params.model_label = f"Grid{encoder_label}-HBV92-DynamicGridRouting"
    validate_experiment_parameters(params)
    return params
