"""Resolve and validate command-line settings for lumped HBV92."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import re
import unicodedata

import pandas as pd


def normalize_basin_name(name: str) -> str:
    display = str(name).strip()
    ascii_name = unicodedata.normalize("NFKD", display).encode(
        "ascii", "ignore"
    ).decode("ascii")
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    if not identifier:
        raise ValueError("basin_name must contain a usable letter or number")
    return identifier


def resolve_experiment_parameters(params: Namespace, source_file: str) -> Namespace:
    project_root = Path(source_file).resolve().parents[2]
    params.project_root = str(project_root)
    params.data_repository = str(project_root / "data_repository")
    params.results_dir = str(project_root / "outputs" / "lumped")
    params.basin = normalize_basin_name(params.basin_name)
    params.model_label = "Lumped-HBV92-Gamma" if params.routing == "gamma" else "Lumped-HBV92"
    if getattr(params, "end_dates_inclusive", True):
        params.calibration_end = (
            pd.Timestamp(params.calibration_end) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        params.validation_end = (
            pd.Timestamp(params.validation_end) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        params.end_dates_inclusive = False
    validate_experiment_parameters(params)
    return params


def validate_experiment_parameters(params: Namespace) -> None:
    if params.mode not in {"prepare", "all"}:
        raise ValueError("mode must be prepare or all")
    calibration_start = pd.Timestamp(params.calibration_start)
    calibration_end = pd.Timestamp(params.calibration_end)
    validation_start = pd.Timestamp(params.validation_start)
    validation_end = pd.Timestamp(params.validation_end)
    if calibration_end <= calibration_start:
        raise ValueError("calibration_end must follow calibration_start")
    if validation_end <= validation_start:
        raise ValueError("validation_end must follow validation_start")
    if validation_start < calibration_end:
        raise ValueError("validation_start must not precede calibration_end")
    if params.spinup_days < 1:
        raise ValueError("spinup_days must be positive")
    if params.repetitions < 1 or params.ngs < 1:
        raise ValueError("repetitions and ngs must be positive")
    if params.routing_length < 1:
        raise ValueError("routing_length must be positive")
    if params.algorithm != "sceua" or params.objective != "nse":
        raise ValueError("The first release supports only SCE-UA with the NSE objective")
