"""Validate the input contract without changing data_repository."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dlcore.data_pipeline.data_catalog import load_dl_catalog
from dlcore.data_pipeline.dataset import load_basin_series


def period_summary(series, start, end_exclusive) -> dict:
    mask = series.period_mask(start, end_exclusive)
    values = series.observed_discharge[mask]
    valid = np.isfinite(values) & (values >= 0)
    return {
        "start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_exclusive": pd.Timestamp(end_exclusive).strftime("%Y-%m-%d"),
        "days": int(mask.sum()),
        "valid_outlet_targets": int(valid.sum()),
        "missing_outlet_targets": int((~valid).sum()),
    }


def validate_basin_mean_inputs(project_root, basin: str, params) -> tuple[object, object, dict]:
    catalog = load_dl_catalog(project_root, basin)
    context_start = pd.Timestamp(params.train_start) - pd.Timedelta(
        days=params.lookback - 1
    )
    series = load_basin_series(catalog, context_start, params.test_end)
    contract = {
        "schema": 1,
        "basin": catalog.basin,
        "forcing_file": str(catalog.forcing),
        "discharge_file": str(catalog.discharge),
        "forcing_variables": ["prec", "t", "pet"],
        "forcing_units": ["mm/day", "degree_Celsius", "mm/day"],
        "target": "outlet_discharge",
        "target_units": "m3/s",
        "outlet_id": catalog.outlet_id,
        "outlet_name": catalog.outlet_name,
        "basin_area_km2": catalog.basin_area_km2,
        "input_signature": catalog.input_signature,
        "lookback_days": int(params.lookback),
        "task_definition": "same_day_meteorology_to_same_day_outlet_discharge",
        "input_window_definition": "t-lookback+1 through t",
        "target_definition": "outlet discharge on day t",
        "context_start": context_start.strftime("%Y-%m-%d"),
        "periods": {
            "train": period_summary(series, params.train_start, params.train_end),
            "validation": period_summary(
                series, params.validation_start, params.validation_end
            ),
            "test": period_summary(series, params.test_start, params.test_end),
        },
        "excluded_inputs": [
            "terrain",
            "DEM",
            "river_network",
            "subbasins",
            "non_outlet_discharge",
            "historical_discharge",
            "future_forcing",
        ],
    }
    for label, summary in contract["periods"].items():
        if summary["valid_outlet_targets"] < 2:
            raise ValueError(f"{label} contains fewer than two valid outlet targets")
    return catalog, series, contract
