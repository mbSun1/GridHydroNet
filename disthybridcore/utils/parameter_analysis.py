"""Area-weighted analysis of exported HBV and grid-routing parameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from disthybridcore.utils.parameter_results import effective_parameter_bounds, normalized_parameter_position


def area_weighted_quantiles(values, weights, probabilities):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    probabilities = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.full(probabilities.shape, np.nan)
    order = np.argsort(values[valid], kind="stable")
    ordered_values = values[valid][order]
    cumulative = np.cumsum(weights[valid][order]); cumulative /= cumulative[-1]
    return ordered_values[np.searchsorted(cumulative, probabilities, side="left").clip(0, len(order)-1)]


def area_weighted_statistics(values, weights):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[valid], weights[valid]
    mean = np.average(values, weights=weights)
    result = {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(mean),
        "std": float(np.sqrt(np.average((values - mean) ** 2, weights=weights))),
    }
    for label, value in zip(
        ("q05", "q25", "median", "q75", "q95"),
        area_weighted_quantiles(values, weights, (.05, .25, .5, .75, .95)),
    ):
        result[label] = float(value)
    return result


def weighted_spearman(first, second, weights):
    first, second, weights = (np.asarray(v).ravel() for v in (first, second, weights))
    valid = np.isfinite(first) & np.isfinite(second) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2:
        return np.nan
    first, second, weights = rankdata(first[valid]), rankdata(second[valid]), weights[valid]
    first -= np.average(first, weights=weights); second -= np.average(second, weights=weights)
    denominator = np.sqrt(np.average(first**2, weights=weights) * np.average(second**2, weights=weights))
    return np.nan if denominator == 0 else float(np.average(first*second, weights=weights) / denominator)


def parameter_summary_table(dataset):
    mask = dataset.basin_mask.values.astype(bool)
    weights = dataset.cell_area_km2.values[mask]
    names = json.loads(dataset.attrs["hbv_parameters"]) + json.loads(dataset.attrs["routing_parameters"])
    rows = []
    for name in names:
        variable = dataset[name]
        lower, upper = effective_parameter_bounds(dataset, name)
        row = {
            "parameter": name, "scope": variable.attrs.get("scope", "grid"),
            "family": variable.attrs.get("family", "Routing"),
            "description": variable.attrs.get("long_name", name),
            "description_cn": variable.attrs.get("description_cn", variable.attrs.get("long_name", name)),
            "units": variable.attrs.get("units", "1"),
            "effective_lower": float(np.nanmin(lower)), "effective_upper": float(upper),
        }
        if variable.ndim == 2:
            row.update(area_weighted_statistics(variable.values[mask], weights))
            position = normalized_parameter_position(dataset, name)
            row.update({f"position_{key}": value for key, value in area_weighted_statistics(position[mask], weights).items()})
            row.update(n_cells=int(mask.sum()), area_km2=float(weights.sum()))
        else:
            row.update(value=float(variable), n_cells=0,
                       position_value=float(normalized_parameter_position(dataset, name)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_parameter_tables(dataset, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    mask = dataset.basin_mask.values.astype(bool)
    weights = dataset.cell_area_km2.values[mask]
    hbv = json.loads(dataset.attrs["hbv_parameters"])
    routing = json.loads(dataset.attrs["routing_parameters"])
    rows, cols = np.where(mask)
    cells = pd.DataFrame({
        "row": rows, "col": cols,
        "x_km": dataset.x.values[cols], "y_km": dataset.y.values[rows],
    })
    for name, variable in dataset.data_vars.items():
        if variable.dims == ("y", "x") and name not in {"basin_mask", "crs"}:
            cells[name] = variable.values[mask]
    for name in hbv + [name for name in routing if dataset[name].ndim == 2]:
        cells[f"{name}_range_position"] = normalized_parameter_position(dataset, name)[mask]
    cells.to_csv(directory / "parameter_cells.csv", index=False, encoding="utf-8-sig")
    summary = parameter_summary_table(dataset)
    summary.to_csv(directory / "parameter_summary.csv", index=False, encoding="utf-8-sig")
    correlations = pd.DataFrame(
        [[weighted_spearman(cells[a], cells[b], weights) for b in hbv] for a in hbv],
        index=hbv, columns=hbv,
    )
    correlations.to_csv(directory / "parameter_correlations.csv", encoding="utf-8-sig", index_label="parameter")
    return summary


def analyze_parameter_results(parameter_file, generate_figures=True):
    parameter_file = Path(parameter_file); directory = parameter_file.parent
    with xr.open_dataset(parameter_file, engine="h5netcdf") as source:
        dataset = source.load()
    try:
        summary = save_parameter_tables(dataset, directory)
        figures = []
        if generate_figures:
            from disthybridcore.utils.parameter_plots import generate_parameter_figures
            figures = generate_parameter_figures(dataset, directory)
        manifest = {
            "source": parameter_file.name,
            "metadata": json.loads(dataset.attrs["evaluation_metadata"]),
            "hbv_parameters": json.loads(dataset.attrs["hbv_parameters"]),
            "routing_parameters": json.loads(dataset.attrs["routing_parameters"]),
            "routing": dataset.attrs["routing"],
            "parameter_count": len(summary),
            "active_cells": int(dataset.basin_mask.sum()),
            "basin_area_km2": float(dataset.cell_area_km2.sum()),
            "figures": figures,
        }
        (directory / "parameter_analysis_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"[Parameter analysis] parameters={len(summary)} | figures={len(figures)} | {directory}")
        return directory
    finally:
        dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze saved parameter fields")
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    return analyze_parameter_results(args.result_dir / "model_parameters.nc", args.figures)


if __name__ == "__main__":
    main()
