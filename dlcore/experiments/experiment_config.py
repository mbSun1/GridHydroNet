"""Resolve and validate deep-learning experiment settings."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import re
import unicodedata

import pandas as pd

from dlcore.hydromodels.model_factory import available_model_names


def normalize_basin_name(name: str) -> str:
    display = str(name).strip()
    ascii_name = unicodedata.normalize("NFKD", display).encode(
        "ascii", "ignore"
    ).decode("ascii")
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_").lower()
    if not identifier:
        raise ValueError("basin_name must contain a usable letter or number")
    return identifier


def normalize_model_name(name: str) -> str:
    identifier = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    collapsed = identifier.replace("_", "")
    canonical_names = {
        model_name.replace("_", ""): model_name
        for model_name in available_model_names()
    }
    return canonical_names.get(collapsed, identifier)


def convert_inclusive_ends(params: Namespace) -> None:
    if not getattr(params, "end_dates_inclusive", True):
        return
    for name in ("train_end", "validation_end", "test_end"):
        value = pd.Timestamp(getattr(params, name)) + pd.Timedelta(days=1)
        setattr(params, name, value.strftime("%Y-%m-%d"))
    params.end_dates_inclusive = False


def validate_experiment_parameters(params: Namespace) -> None:
    if params.mode not in {"prepare", "all"}:
        raise ValueError("mode must be prepare or all")
    boundaries = [
        pd.Timestamp(params.train_start),
        pd.Timestamp(params.train_end),
        pd.Timestamp(params.validation_start),
        pd.Timestamp(params.validation_end),
        pd.Timestamp(params.test_start),
        pd.Timestamp(params.test_end),
    ]
    if any(value != value.normalize() for value in boundaries):
        raise ValueError("Period boundaries must be midnight day boundaries")
    if not (
        boundaries[0] < boundaries[1]
        <= boundaries[2] < boundaries[3]
        <= boundaries[4] < boundaries[5]
    ):
        raise ValueError("Train, validation, and test periods must be ordered and disjoint")
    if params.lookback < 2 or params.batch_size < 1 or params.max_epochs < 1:
        raise ValueError("lookback, batch_size, and max_epochs must be positive")
    if params.model not in available_model_names():
        raise ValueError(f"Unsupported model: {params.model}")
    if params.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if params.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative")
    if not 0.0 <= params.dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1)")
    if params.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if params.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    if params.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")


def resolve_experiment_parameters(params: Namespace, source_file: str) -> Namespace:
    project_root = Path(source_file).resolve().parents[2]
    params.project_root = str(project_root)
    params.data_repository = str(project_root / "data_repository")
    params.results_dir = str(project_root / "outputs" / "dl")
    params.basin = normalize_basin_name(params.basin_name)
    params.model = normalize_model_name(params.model)
    convert_inclusive_ends(params)
    validate_experiment_parameters(params)
    return params
