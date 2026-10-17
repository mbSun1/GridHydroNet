"""Create deterministic basin-mean forcing from the prepared hydro-grid."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def forcing_variable_units() -> dict[str, str]:
    return {
        "prec": "mm/day",
        "t": "degree_Celsius",
        "pet": "mm/day",
    }


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def signature(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def area_weighted_mean(values: np.ndarray, basin_fraction: np.ndarray) -> np.ndarray:
    """Return basin-fraction-weighted means for ``[..., y, x]`` values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(basin_fraction, dtype=np.float64)
    if weights.ndim != 2 or values.shape[-2:] != weights.shape:
        raise ValueError(
            "Forcing and basin_fraction spatial shapes differ: "
            f"{values.shape[-2:]} vs {weights.shape}"
        )
    active = weights > 0
    if not active.any() or not np.isfinite(weights[active]).all():
        raise ValueError("basin_fraction must contain finite positive basin weights")
    if (weights < 0).any() or (weights > 1).any():
        raise ValueError("basin_fraction must lie in [0, 1]")
    active_values = values[..., active]
    if not np.isfinite(active_values).all():
        raise ValueError("Forcing contains NaN/Inf inside the basin")
    return np.sum(active_values * weights[active], axis=-1) / weights[active].sum()


def read_preparation_manifest(project_root: Path, basin: str) -> tuple[Path, dict]:
    basin_root = project_root / "data_repository" / f"basin_{basin}"
    path = basin_root / "preparation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared basin manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != 4:
        raise ValueError(f"Unsupported basin preparation schema: {manifest.get('schema')!r}")
    if manifest.get("basin") != basin:
        raise ValueError(f"Manifest basin {manifest.get('basin')!r} does not match {basin!r}")
    forcing = manifest.get("forcing")
    variable_units = forcing_variable_units()
    if not isinstance(forcing, dict) or set(variable_units).difference(forcing):
        raise ValueError("Prepared basin manifest must contain prec, t, and pet forcing")
    return basin_root, manifest


def grid_metadata(manifest: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    path = Path(manifest["dem"])
    with xr.open_dataset(path, engine="h5netcdf") as grid:
        if "basin_fraction" not in grid:
            raise ValueError(f"Hydro-grid lacks basin_fraction: {path}")
        weights = np.asarray(grid["basin_fraction"], dtype=np.float64)
        x = np.asarray(grid.x, dtype=np.float64)
        y = np.asarray(grid.y, dtype=np.float64)
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Hydro-grid needs at least two x/y coordinates")
        spacing_x = float(np.median(np.diff(x)))
        spacing_y = float(np.median(np.diff(y)))
        if spacing_x <= 0 or spacing_y <= 0 or not np.isclose(spacing_x, spacing_y):
            raise ValueError("Hydro-grid must use increasing square metric cells")
        resolution_m = float(grid.attrs.get("grid_resolution_m", spacing_x))
        crs_wkt = str(grid.attrs.get("crs_wkt", ""))
    area_km2 = float(weights.sum(dtype=np.float64) * resolution_m**2 / 1.0e6)
    return weights, x, y, resolution_m, area_km2


def read_basin_mean_series(
    path: Path,
    variable: str,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_days: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, dict]:
    variable_units = forcing_variable_units()
    with xr.open_dataset(path, engine="h5netcdf") as data:
        if variable not in data:
            raise KeyError(f"Variable {variable!r} not found in {path}")
        array = data[variable].transpose("time", "y", "x")
        if array.attrs.get("units") != variable_units[variable]:
            raise ValueError(
                f"{path.name}:{variable} units must be {variable_units[variable]}"
            )
        if not np.array_equal(np.asarray(data.x), x) or not np.array_equal(
            np.asarray(data.y), y
        ):
            raise ValueError(f"{path.name} grid is not aligned with basin_fraction")
        dates = pd.DatetimeIndex(np.asarray(data.time))
        expected = pd.date_range(dates[0], dates[-1], freq="D")
        if not dates.equals(expected):
            raise ValueError(f"{path.name} does not have a complete daily calendar")
        means = np.empty(len(dates), dtype=np.float64)
        minimum = np.inf
        maximum = -np.inf
        for start in range(0, len(dates), chunk_days):
            stop = min(start + chunk_days, len(dates))
            values = np.asarray(array.isel(time=slice(start, stop)), dtype=np.float64)
            active = values[..., weights > 0]
            if variable != "t" and (active < 0).any():
                raise ValueError(f"{path.name}:{variable} contains negative flux values")
            means[start:stop] = area_weighted_mean(values, weights)
            minimum = min(minimum, float(active.min()))
            maximum = max(maximum, float(active.max()))
    return dates, means, {
        "source_minimum": minimum,
        "source_maximum": maximum,
        "basin_mean_minimum": float(means.min()),
        "basin_mean_maximum": float(means.max()),
        "units": variable_units[variable],
    }


def validate_basin_mean_forcing(path: str | Path) -> dict:
    """Validate the published one-dimensional daily forcing file."""
    path = Path(path)
    variable_units = forcing_variable_units()
    with xr.open_dataset(path, engine="h5netcdf") as data:
        if data.attrs.get("temporal_resolution") != "daily":
            raise ValueError("Basin-mean forcing must declare daily resolution")
        dates = pd.DatetimeIndex(np.asarray(data.time))
        if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
            raise ValueError("Basin-mean forcing has an incomplete daily calendar")
        for variable, units in variable_units.items():
            if variable not in data or data[variable].dims != ("time",):
                raise ValueError(f"Basin-mean forcing lacks one-dimensional {variable}")
            values = np.asarray(data[variable], dtype=np.float64)
            if data[variable].attrs.get("units") != units or not np.isfinite(values).all():
                raise ValueError(f"Invalid {variable} units or values")
            if variable != "t" and (values < 0).any():
                raise ValueError(f"Basin-mean {variable} contains negative values")
        return {
            "start": str(dates[0].date()),
            "end_inclusive": str(dates[-1].date()),
            "days": len(dates),
            "basin_area_km2": float(data.attrs["basin_area_km2"]),
            "input_signature": str(data.attrs["input_signature"]),
        }


def prepare_basin_mean_forcing(
    project_root: str | Path,
    basin: str,
    *,
    force: bool = False,
    chunk_days: int = 365,
) -> Path:
    """Aggregate the prepared grid forcings and publish one basin-mean NetCDF."""
    if chunk_days < 1:
        raise ValueError("chunk_days must be positive")
    project_root = Path(project_root).resolve()
    basin = str(basin).strip().lower()
    basin_root, manifest = read_preparation_manifest(project_root, basin)
    weights, x, y, resolution_m, area_km2 = grid_metadata(manifest)
    variable_units = forcing_variable_units()
    forcing_paths = {
        variable: Path(manifest["forcing"][variable]).resolve()
        for variable in variable_units
    }
    missing = [str(path) for path in forcing_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Grid forcing files are missing: " + ", ".join(missing))

    signature_payload = {
        "schema": 1,
        "aggregation": "basin_fraction_weighted_mean",
        "basin": basin,
        "hydro_grid": file_identity(Path(manifest["dem"])),
        "forcing": {
            variable: file_identity(path)
            for variable, path in forcing_paths.items()
        },
    }
    input_signature = signature(signature_payload)
    output_dir = basin_root / "forcing" / "basin_mean"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(manifest["forcing_start"])
    end = pd.Timestamp(manifest["forcing_end_inclusive"])
    output_path = output_dir / (
        f"basin_{basin}_daily_basin_mean_{start:%Y-%m-%d}_{end:%Y-%m-%d}.nc"
    )
    metadata_path = output_dir / "manifest.json"

    if not force and output_path.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("input_signature") == input_signature:
            validation = validate_basin_mean_forcing(output_path)
            if validation["input_signature"] == input_signature:
                print(f"[流域平均强迫] 已匹配: {output_path}", flush=True)
                return output_path

    dates = None
    series = {}
    summaries = {}
    for variable in ("prec", "t", "pet"):
        current_dates, means, summary = read_basin_mean_series(
            forcing_paths[variable], variable, weights, x, y,
            chunk_days=chunk_days,
        )
        if dates is None:
            dates = current_dates
        elif not current_dates.equals(dates):
            raise ValueError("P, T, and PET daily calendars differ")
        series[variable] = means
        summaries[variable] = summary
    if dates[0] != start or dates[-1] != end:
        raise ValueError("Basin manifest and grid forcing dates disagree")

    dataset = xr.Dataset(
        data_vars={
            variable: xr.DataArray(
                values,
                dims=("time",),
                attrs={"units": variable_units[variable]},
            )
            for variable, values in series.items()
        },
        coords={"time": dates},
        attrs={
            "schema": 1,
            "basin": basin,
            "temporal_resolution": "daily",
            "spatial_aggregation": "basin_fraction_weighted_mean",
            "basin_fraction_source": str(Path(manifest["dem"]).resolve()),
            "basin_area_km2": area_km2,
            "grid_resolution_m": resolution_m,
            "input_signature": input_signature,
            "missing_method": "No imputation; invalid in-basin values raise an error",
        },
    )
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    encoding = {
        variable: {"dtype": "f8", "zlib": True, "complevel": 4}
        for variable in variable_units
    }
    dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
    dataset.close()
    temporary.replace(output_path)
    validation = validate_basin_mean_forcing(output_path)

    basin_mean_manifest = {
        "schema": 1,
        "basin": basin,
        "forcing": str(output_path.resolve()),
        "start": validation["start"],
        "end_inclusive": validation["end_inclusive"],
        "days": validation["days"],
        "variables": variable_units,
        "aggregation": "basin_fraction_weighted_mean",
        "basin_fraction_source": str(Path(manifest["dem"]).resolve()),
        "grid_shape": [int(weights.shape[0]), int(weights.shape[1])],
        "grid_resolution_m": resolution_m,
        "basin_area_km2": area_km2,
        "outlet_id": str(manifest["outlet_id"]),
        "discharge_source": str(Path(manifest["discharge"]).resolve()),
        "source_forcing": {
            variable: str(path) for variable, path in forcing_paths.items()
        },
        "source_summaries": summaries,
        "input_signature": input_signature,
        "missing_method": "No imputation; invalid in-basin values raise an error",
    }
    write_json_atomic(metadata_path, basin_mean_manifest)
    print(
        f"[流域平均强迫] {basin} | {len(dates)} d | {area_km2:.3f} km2 | "
        f"{output_path}",
        flush=True,
    )
    return output_path
