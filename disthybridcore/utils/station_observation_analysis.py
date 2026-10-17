"""Whole-basin observation analysis for a supervised gauge network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from disthybridcore.data_pipeline.basin import load_river_network
from disthybridcore.prepare_data.discharge_preparation import read_station_daily
from disthybridcore.prepare_data.grid_routing import load_grid_gauge_mapping
from disthybridcore.utils.station_observation_figures import (
    plot_all_station_hydrographs,
    plot_station_network_quality,
    plot_station_similarity,
)


def branch_display_name(station_name: str, station_id: str) -> str:
    """Return a concise main-branch name from the downstream branch-root gauge."""
    name = str(station_name).strip()
    upper = name.upper()
    cut = len(name)
    for separator in (" NEAR ", " NR ", " AT ", " ABOVE ", " BELOW "):
        position = upper.find(separator)
        if position >= 0:
            cut = min(cut, position)
    base = name[:cut].strip()
    return f"{base.title()} system" if base else f"Branch {station_id}"


def assign_station_branches(river) -> tuple[dict[str, str], dict[str, str]]:
    """Assign every gauge to the last observed branch root before the outlet."""
    adjacency = river.adjacency.detach().cpu().numpy() > 0
    outlet = int(river.outlet_index)
    branch_ids: dict[str, str] = {}
    branch_names: dict[str, str] = {}
    for source, station_id in enumerate(river.station_ids):
        if source == outlet:
            branch_ids[station_id] = station_id
            branch_names[station_id] = "Basin outlet"
            continue
        current = source
        visited: set[int] = set()
        while current != outlet:
            if current in visited:
                raise ValueError("Gauge adjacency contains a directed cycle")
            visited.add(current)
            targets = np.flatnonzero(adjacency[current])
            if targets.size != 1:
                raise ValueError(
                    f"Gauge {river.station_ids[current]} must have one downstream gauge"
                )
            target = int(targets[0])
            if target == outlet:
                root_id = river.station_ids[current]
                branch_ids[station_id] = root_id
                branch_names[station_id] = branch_display_name(
                    river.station_names[current], root_id
                )
                break
            current = target
    return branch_ids, branch_names


def load_station_observations(params):
    """Load one aligned observation frame and its validated station metadata."""
    river = load_river_network(params.path_discharge)
    mapping = load_grid_gauge_mapping(Path(params.path_grid_gauges))
    if tuple(river.station_ids) != tuple(mapping.gauge_ids):
        raise ValueError("Discharge and grid-gauge station order do not match")

    with xr.open_dataset(params.path_discharge, engine="scipy") as source:
        variable = "discharge_observed" if "discharge_observed" in source else "discharge"
        dates = pd.DatetimeIndex(source.t.values)
        values = np.asarray(source[variable], dtype=np.float64)
        station_ids = [str(value) for value in source.station_id.values.tolist()]
        station_names = [str(value) for value in source.name.values.tolist()]
        official_area = np.asarray(source.official_area_km2, dtype=np.float64)
        nextdown_ids = [str(value) for value in source.nextdown_id.values.tolist()]
        longitude = np.asarray(source.coord.sel(coordinate="longitude"), dtype=np.float64)
        latitude = np.asarray(source.coord.sel(coordinate="latitude"), dtype=np.float64)
        projected_x = np.asarray(
            source.projected_coord.sel(projected_coordinate="x"), dtype=np.float64
        )
        projected_y = np.asarray(
            source.projected_coord.sel(projected_coordinate="y"), dtype=np.float64
        )
        outlet_flags = np.asarray(source.is_outlet, dtype=bool)

    if station_ids != list(river.station_ids):
        raise ValueError("Prepared discharge and river-network station order do not match")
    if values.shape != (len(dates), len(station_ids)):
        raise ValueError("Prepared station discharge has an inconsistent shape")

    start = pd.Timestamp(params.train_start)
    end = pd.Timestamp(params.test_end)
    selection = (dates >= start) & (dates < end)
    observations = pd.DataFrame(
        values[selection], index=dates[selection], columns=station_ids
    )
    observations = observations.where(np.isfinite(observations) & (observations >= 0))

    branch_ids, branch_names = assign_station_branches(river)
    metadata = pd.DataFrame(
        {
            "station_id": station_ids,
            "station_name": station_names,
            "nextdown_id": nextdown_ids,
            "is_outlet": outlet_flags,
            "longitude": longitude,
            "latitude": latitude,
            "projected_x_km": projected_x / 1000.0,
            "projected_y_km": projected_y / 1000.0,
            "official_area_km2": official_area,
            "d8_area_km2": mapping.gauge_grid_area_km2.numpy().astype(np.float64),
            "snap_distance_km": (
                mapping.gauge_snap_distance_m.numpy().astype(np.float64) / 1000.0
            ),
            "branch_id": [branch_ids[value] for value in station_ids],
            "branch_name": [branch_names[value] for value in station_ids],
        }
    ).set_index("station_id", drop=False)
    return observations, metadata, river


def load_preparation_manifest(params) -> dict:
    """Load source-file provenance recorded by basin preparation."""
    manifest_path = (
        Path(params.data_repository)
        / f"basin_{params.basin}"
        / "preparation_manifest.json"
    )
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def compare_raw_observation(
    station_id: str,
    processed: pd.Series,
    source_paths: dict[str, str],
) -> tuple[str, float, bool]:
    """Compare prepared values with the corresponding source CSV over common dates."""
    source_value = source_paths.get(station_id)
    if source_value is None:
        return "", np.nan, False
    source_path = Path(source_value)
    if not source_path.is_file():
        return source_path.name, np.nan, False
    dates, values, _ = read_station_daily(source_path)
    raw = pd.Series(values, index=dates).reindex(processed.index)
    valid = raw.notna() & processed.notna()
    if not valid.any():
        return source_path.name, np.nan, False
    maximum_difference = float((raw[valid] - processed[valid]).abs().max())
    matches = bool(
        np.allclose(
            raw[valid].to_numpy(),
            processed[valid].to_numpy(),
            rtol=1e-6,
            atol=1e-4,
        )
    )
    return source_path.name, maximum_difference, matches


def build_station_summary(
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    params,
    source_paths: dict[str, str],
) -> pd.DataFrame:
    """Build station-scale data quality and hydrological signature statistics."""
    train_start = pd.Timestamp(params.train_start)
    train_end = pd.Timestamp(params.train_end)
    test_start = pd.Timestamp(params.test_start)
    test_end = pd.Timestamp(params.test_end)
    rows = []
    for station_id in observations.columns:
        station = metadata.loc[station_id]
        series = observations[station_id]
        valid = series.dropna()
        train = series[(series.index >= train_start) & (series.index < train_end)]
        test = series[(series.index >= test_start) & (series.index < test_end)]
        source_name, raw_difference, raw_match = compare_raw_observation(
            station_id, series, source_paths
        )
        area = float(station.official_area_km2)
        d8_area = float(station.d8_area_km2)
        mean = float(valid.mean()) if not valid.empty else np.nan
        standard_deviation = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
        maximum_date = valid.idxmax().date().isoformat() if not valid.empty else ""
        rows.append(
            {
                "station_id": station_id,
                "station_name": station.station_name,
                "branch_id": station.branch_id,
                "branch_name": station.branch_name,
                "nextdown_id": station.nextdown_id,
                "is_outlet": bool(station.is_outlet),
                "longitude": float(station.longitude),
                "latitude": float(station.latitude),
                "official_area_km2": area,
                "d8_area_km2": d8_area,
                "d8_area_signed_error_percent": (d8_area - area) / area * 100.0,
                "d8_area_absolute_error_percent": abs(d8_area - area) / area * 100.0,
                "snap_distance_km": float(station.snap_distance_km),
                "analysis_start": observations.index.min().date().isoformat(),
                "analysis_end_inclusive": observations.index.max().date().isoformat(),
                "expected_days": int(len(series)),
                "valid_days": int(valid.size),
                "missing_days": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100.0),
                "train_valid_days": int(train.notna().sum()),
                "test_valid_days": int(test.notna().sum()),
                "mean_discharge_m3_s": mean,
                "median_discharge_m3_s": float(valid.median()),
                "standard_deviation_m3_s": standard_deviation,
                "coefficient_of_variation": (
                    standard_deviation / mean if np.isfinite(mean) and mean > 0 else np.nan
                ),
                "p05_discharge_m3_s": float(valid.quantile(0.05)),
                "p95_discharge_m3_s": float(valid.quantile(0.95)),
                "maximum_discharge_m3_s": float(valid.max()),
                "maximum_discharge_date": maximum_date,
                "mean_runoff_mm_day": mean * 86.4 / area,
                "source_csv": source_name,
                "raw_processed_max_abs_difference_m3_s": raw_difference,
                "raw_processed_match": raw_match,
            }
        )
    return pd.DataFrame(rows)


def station_path_length(adjacency: np.ndarray, source: int, target: int) -> int | None:
    """Return observed-gauge edge count when source drains to target."""
    current = int(source)
    visited: set[int] = set()
    steps = 0
    while current != target:
        if current in visited:
            return None
        visited.add(current)
        following = np.flatnonzero(adjacency[current])
        if following.size != 1:
            return None
        current = int(following[0])
        steps += 1
    return steps


def safe_correlation(first: pd.Series, second: pd.Series, method: str) -> float:
    """Return a finite pairwise correlation or NaN for a degenerate pair."""
    pair = pd.concat([first, second], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def lagged_correlation(
    source: pd.Series,
    target: pd.Series,
    minimum_lag: int = -7,
    maximum_lag: int = 7,
) -> tuple[int, float]:
    """Find the lag maximizing correlation; positive means target peaks later."""
    candidates = []
    for lag in range(minimum_lag, maximum_lag + 1):
        correlation = safe_correlation(source, target.shift(-lag), "pearson")
        if np.isfinite(correlation):
            candidates.append((lag, correlation))
    if not candidates:
        return 0, np.nan
    return max(candidates, key=lambda value: (value[1], -abs(value[0])))


def build_station_pair_similarity(
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    river,
) -> pd.DataFrame:
    """Quantify same-day similarity, duplicate status and downstream lag for all pairs."""
    adjacency = river.adjacency.detach().cpu().numpy() > 0
    ids = list(observations.columns)
    rows = []
    for first in range(len(ids)):
        for second in range(first + 1, len(ids)):
            first_to_second = station_path_length(adjacency, first, second)
            second_to_first = station_path_length(adjacency, second, first)
            if first_to_second is not None:
                source_index, target_index = first, second
                connection_steps = first_to_second
            elif second_to_first is not None:
                source_index, target_index = second, first
                connection_steps = second_to_first
            else:
                first_area = float(metadata.loc[ids[first], "official_area_km2"])
                second_area = float(metadata.loc[ids[second], "official_area_km2"])
                source_index, target_index = (
                    (first, second) if first_area <= second_area else (second, first)
                )
                connection_steps = None
            source_id, target_id = ids[source_index], ids[target_index]
            source = observations[source_id]
            target = observations[target_id]
            pair = pd.concat([source, target], axis=1).dropna()
            best_lag, best_correlation = lagged_correlation(source, target)
            source_mean = float(pair.iloc[:, 0].mean()) if not pair.empty else np.nan
            target_mean = float(pair.iloc[:, 1].mean()) if not pair.empty else np.nan
            source_area = float(metadata.loc[source_id, "official_area_km2"])
            target_area = float(metadata.loc[target_id, "official_area_km2"])
            rows.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_branch": metadata.loc[source_id, "branch_name"],
                    "target_branch": metadata.loc[target_id, "branch_name"],
                    "same_main_branch": bool(
                        metadata.loc[source_id, "branch_id"]
                        == metadata.loc[target_id, "branch_id"]
                    ),
                    "hydrologically_connected": connection_steps is not None,
                    "directly_connected": connection_steps == 1,
                    "connection_steps": (
                        float(connection_steps) if connection_steps is not None else np.nan
                    ),
                    "overlap_days": int(len(pair)),
                    "pearson_discharge": safe_correlation(source, target, "pearson"),
                    "pearson_log1p_discharge": safe_correlation(
                        np.log1p(source), np.log1p(target), "pearson"
                    ),
                    "spearman_discharge": safe_correlation(source, target, "spearman"),
                    "best_target_lag_days": int(best_lag),
                    "best_lag_pearson": best_correlation,
                    "exact_duplicate": bool(
                        not pair.empty
                        and np.array_equal(
                            pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
                        )
                    ),
                    "source_official_area_km2": source_area,
                    "target_official_area_km2": target_area,
                    "area_ratio_target_to_source": target_area / source_area,
                    "mean_flow_ratio_target_to_source": (
                        target_mean / source_mean
                        if np.isfinite(source_mean) and source_mean > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def select_independent_outlet_peaks(
    series: pd.Series,
    count: int = 3,
    separation_days: int = 15,
) -> list[pd.Timestamp]:
    """Select the largest independent outlet peaks without visual cherry-picking."""
    selected: list[pd.Timestamp] = []
    for date in series.dropna().sort_values(ascending=False).index:
        timestamp = pd.Timestamp(date)
        if all(abs((timestamp - other).days) >= separation_days for other in selected):
            selected.append(timestamp)
            if len(selected) == count:
                break
    return sorted(selected)


def build_connected_event_summary(
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    pairs: pd.DataFrame,
    params,
) -> pd.DataFrame:
    """Summarize peak amplification and lag for objective test-period events."""
    test = observations[
        (observations.index >= pd.Timestamp(params.test_start))
        & (observations.index < pd.Timestamp(params.test_end))
    ]
    outlet_id = str(metadata.loc[metadata.is_outlet, "station_id"].iloc[0])
    peak_dates = select_independent_outlet_peaks(test[outlet_id])
    direct_pairs = pairs[pairs.directly_connected].copy()
    rows = []
    for event_rank, outlet_peak_date in enumerate(peak_dates, start=1):
        start = outlet_peak_date - pd.Timedelta(days=7)
        end = outlet_peak_date + pd.Timedelta(days=7)
        window = test.loc[start:end]
        for pair in direct_pairs.itertuples(index=False):
            source = window[pair.source_id].dropna()
            target = window[pair.target_id].dropna()
            if source.empty or target.empty:
                rows.append(
                    {
                        "event_rank": event_rank,
                        "outlet_peak_date": outlet_peak_date.date().isoformat(),
                        "window_start": start.date().isoformat(),
                        "window_end": end.date().isoformat(),
                        "source_id": pair.source_id,
                        "target_id": pair.target_id,
                        "observation_available": False,
                        "source_peak_date": "",
                        "target_peak_date": "",
                        "target_peak_lag_days": np.nan,
                        "source_peak_m3_s": np.nan,
                        "target_peak_m3_s": np.nan,
                        "peak_flow_ratio_target_to_source": np.nan,
                        "source_peak_runoff_mm_day": np.nan,
                        "target_peak_runoff_mm_day": np.nan,
                        "peak_runoff_ratio_target_to_source": np.nan,
                    }
                )
                continue
            source_date = pd.Timestamp(source.idxmax())
            target_date = pd.Timestamp(target.idxmax())
            source_peak = float(source.max())
            target_peak = float(target.max())
            source_area = float(metadata.loc[pair.source_id, "official_area_km2"])
            target_area = float(metadata.loc[pair.target_id, "official_area_km2"])
            rows.append(
                {
                    "event_rank": event_rank,
                    "outlet_peak_date": outlet_peak_date.date().isoformat(),
                    "window_start": start.date().isoformat(),
                    "window_end": end.date().isoformat(),
                    "source_id": pair.source_id,
                    "target_id": pair.target_id,
                    "observation_available": True,
                    "source_peak_date": source_date.date().isoformat(),
                    "target_peak_date": target_date.date().isoformat(),
                    "target_peak_lag_days": int((target_date - source_date).days),
                    "source_peak_m3_s": source_peak,
                    "target_peak_m3_s": target_peak,
                    "peak_flow_ratio_target_to_source": target_peak / source_peak,
                    "source_peak_runoff_mm_day": source_peak * 86.4 / source_area,
                    "target_peak_runoff_mm_day": target_peak * 86.4 / target_area,
                    "peak_runoff_ratio_target_to_source": (
                        (target_peak * 86.4 / target_area)
                        / (source_peak * 86.4 / source_area)
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_area_audit(params) -> pd.DataFrame:
    """Load retained and excluded gauge-area decisions when available."""
    path = Path(params.topology_dir) / "gauge_area_audit.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"gauge_id": str})


def write_analysis_metadata(path: Path, params, summary: pd.DataFrame) -> None:
    """Record definitions required to interpret the manuscript analysis."""
    payload = {
        "analysis_period": {
            "start": str(pd.Timestamp(params.train_start).date()),
            "end_exclusive": str(pd.Timestamp(params.test_end).date()),
        },
        "training_period": {
            "start": str(pd.Timestamp(params.train_start).date()),
            "end_exclusive": str(pd.Timestamp(params.train_end).date()),
        },
        "test_period": {
            "start": str(pd.Timestamp(params.test_start).date()),
            "end_exclusive": str(pd.Timestamp(params.test_end).date()),
        },
        "station_count": int(len(summary)),
        "area_normalization": "runoff_mm_day = discharge_m3_s * 86.4 / official_area_km2",
        "pairwise_lag_definition": (
            "Search -7..+7 days; positive target lag means the target/downstream "
            "series follows the source/upstream series."
        ),
        "event_definition": (
            "Three largest test-period outlet peaks separated by at least 15 days; "
            "station peaks evaluated within plus/minus 7 days."
        ),
        "raw_match_tolerance": "numpy.allclose(rtol=1e-6, atol=1e-4 m3/s)",
        "figure_formats": ["jpg"],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_station_observation_analysis(params, run_directory) -> dict[str, Path]:
    """Create all whole-basin observation tables and JPG manuscript figures."""
    output_dir = Path(run_directory) / "station_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, metadata, river = load_station_observations(params)
    manifest = load_preparation_manifest(params)
    source_paths = {
        str(key): str(value)
        for key, value in manifest.get("discharge_sources", {}).items()
    }
    summary = build_station_summary(observations, metadata, params, source_paths)
    pairs = build_station_pair_similarity(observations, metadata, river)
    events = build_connected_event_summary(
        observations, metadata, pairs, params
    )

    outputs = {
        "summary": output_dir / "station_observation_summary.csv",
        "pairs": output_dir / "station_pair_similarity.csv",
        "events": output_dir / "connected_station_event_summary.csv",
        "metadata": output_dir / "station_analysis_metadata.json",
    }
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")
    pairs.to_csv(outputs["pairs"], index=False, encoding="utf-8-sig")
    events.to_csv(outputs["events"], index=False, encoding="utf-8-sig")
    write_analysis_metadata(outputs["metadata"], params, summary)

    if params.generate_figures:
        area_audit = load_area_audit(params)
        outputs["network_figure"] = plot_station_network_quality(
            params,
            observations,
            metadata,
            river,
            summary,
            area_audit,
            output_dir / "station_network_and_data_quality.jpg",
        )
        outputs["similarity_figure"] = plot_station_similarity(
            observations,
            metadata,
            river,
            summary,
            pairs,
            output_dir / "station_observation_similarity.jpg",
        )
        test = observations[
            (observations.index >= pd.Timestamp(params.test_start))
            & (observations.index < pd.Timestamp(params.test_end))
        ]
        outputs["absolute_hydrographs"] = plot_all_station_hydrographs(
            test,
            metadata,
            output_dir / "all_station_observed_discharge.jpg",
            normalize_by_area=False,
        )
        outputs["normalized_hydrographs"] = plot_all_station_hydrographs(
            test,
            metadata,
            output_dir / "all_station_area_normalized_runoff.jpg",
            normalize_by_area=True,
        )

    raw_matches = int(summary.raw_processed_match.sum())
    maximum_area_error = float(summary.d8_area_absolute_error_percent.max())
    exact_duplicates = int(pairs.exact_duplicate.sum())
    print(
        "[站点观测分析] "
        f"stations={len(summary)} | raw_matches={raw_matches}/{len(summary)} | "
        f"max_area_error={maximum_area_error:.2f}% | "
        f"exact_duplicate_pairs={exact_duplicates} | output={output_dir.resolve()}",
        flush=True,
    )
    return outputs
