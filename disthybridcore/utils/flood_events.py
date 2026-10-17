"""Observation-defined flood-event detection and multi-gauge diagnostics.

Event thresholds are estimated only from a reference training period.  Event
windows are then identified from observations in the evaluated period; model
simulation never influences event selection, boundaries, or identifiers.
"""

from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .hydrology_metrics import station_metrics


@dataclass(frozen=True)
class FloodEventConfig:
    """Configuration for observation-defined variable-duration flood events."""

    peak_quantile: float = 0.95
    recession_quantile: float = 0.75
    maximum_high_flow_gap_days: int = 3
    event_mapping_tolerance_days: int = 2
    maximum_boundary_search_days: int = 15
    minimum_event_duration_days: int = 7
    minimum_valid_pair_days: int = 7
    minimum_pair_coverage: float = 0.80
    minimum_observation_mean_m3_s: float = 1.0e-12
    minimum_observation_std_m3_s: float = 1.0e-12
    maximum_events: int | None = None

    def __post_init__(self):
        if not 0.0 < self.recession_quantile < self.peak_quantile < 1.0:
            raise ValueError(
                "Flood-event quantiles must satisfy 0 < recession < peak < 1"
            )
        integer_fields = (
            "maximum_high_flow_gap_days",
            "event_mapping_tolerance_days",
            "maximum_boundary_search_days",
            "minimum_event_duration_days",
            "minimum_valid_pair_days",
        )
        for name in integer_fields:
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer")
        if self.maximum_high_flow_gap_days < 0:
            raise ValueError("maximum_high_flow_gap_days cannot be negative")
        if self.event_mapping_tolerance_days < 0:
            raise ValueError("event_mapping_tolerance_days cannot be negative")
        if self.maximum_boundary_search_days < 1:
            raise ValueError("maximum_boundary_search_days must be positive")
        if self.minimum_event_duration_days < 1:
            raise ValueError("minimum_event_duration_days must be positive")
        if self.minimum_valid_pair_days < 2:
            raise ValueError("minimum_valid_pair_days must be at least two")
        if not 0.0 < self.minimum_pair_coverage <= 1.0:
            raise ValueError("minimum_pair_coverage must be in (0, 1]")
        if self.minimum_observation_mean_m3_s < 0.0:
            raise ValueError("minimum_observation_mean_m3_s cannot be negative")
        if self.minimum_observation_std_m3_s < 0.0:
            raise ValueError("minimum_observation_std_m3_s cannot be negative")
        if self.maximum_events is not None and self.maximum_events < 1:
            raise ValueError("maximum_events must be positive or None")


@dataclass(frozen=True)
class FloodEventEvaluation:
    """All tabular products from one flood-event evaluation."""

    catalog: pd.DataFrame
    metrics: pd.DataFrame
    by_station: pd.DataFrame
    by_event: pd.DataFrame
    basin_summary: pd.DataFrame
    thresholds: pd.DataFrame
    metadata: dict[str, Any]

    def tables(self) -> dict[str, pd.DataFrame]:
        """Return output tables under their intended CSV stem names."""
        return {
            "flood_event_catalog": self.catalog,
            "flood_event_metrics": self.metrics,
            "flood_event_summary_by_station": self.by_station,
            "flood_event_summary_by_event": self.by_event,
            "flood_event_basin_summary": self.basin_summary,
            "flood_event_thresholds": self.thresholds,
        }


def numpy_matrix(values, name: str, expected_columns: int) -> np.ndarray:
    """Convert a CPU tensor/array to a two-dimensional floating array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and expected_columns == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must have shape [time, {expected_columns}], got {array.shape}"
        )
    return array


def daily_dates(values, name: str, *, require_contiguous: bool) -> pd.DatetimeIndex:
    """Validate a unique chronological date index."""
    dates = pd.DatetimeIndex(values).normalize()
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{name} must be finite, unique, and chronological")
    if require_contiguous and len(dates) > 1:
        differences = np.diff(dates.values).astype("timedelta64[D]").astype(int)
        if not np.all(differences == 1):
            raise ValueError(f"{name} must be a contiguous daily calendar")
    return dates


def compute_flood_thresholds(
    reference_observation,
    station_names: Sequence[str],
    config: FloodEventConfig | None = None,
) -> pd.DataFrame:
    """Compute station-specific high-flow and recession thresholds."""
    config = config or FloodEventConfig()
    names = [str(value) for value in station_names]
    if not names or len(set(names)) != len(names):
        raise ValueError("station_names must be a non-empty unique sequence")
    observation = numpy_matrix(
        reference_observation, "reference_observation", len(names)
    )
    records = []
    for index, station in enumerate(names):
        values = observation[:, index]
        valid = values[np.isfinite(values) & (values >= 0.0)]
        records.append({
            "station": station,
            "reference_valid_days": int(valid.size),
            "peak_quantile": config.peak_quantile,
            "recession_quantile": config.recession_quantile,
            "high_flow_threshold_m3_s": (
                float(np.quantile(valid, config.peak_quantile))
                if valid.size else np.nan
            ),
            "recession_threshold_m3_s": (
                float(np.quantile(valid, config.recession_quantile))
                if valid.size else np.nan
            ),
        })
    return pd.DataFrame.from_records(records)


def pad_event_interval(
    start: int, end: int, total_days: int, minimum_duration: int,
) -> tuple[int, int]:
    """Symmetrically pad an inclusive interval to the configured minimum length."""
    missing = minimum_duration - (end - start + 1)
    if missing <= 0:
        return start, end
    before = min(start, (missing + 1) // 2)
    start -= before
    missing -= before
    after = min(total_days - 1 - end, missing)
    end += after
    missing -= after
    if missing:
        additional_before = min(start, missing)
        start -= additional_before
    return start, end


def detect_station_events(
    dates: pd.DatetimeIndex,
    observation: np.ndarray,
    high_threshold: float,
    recession_threshold: float,
    station: str,
    config: FloodEventConfig,
    *,
    outlet_anchored: bool,
) -> list[dict[str, Any]]:
    """Detect variable-duration events for one station from observations only."""
    if not np.isfinite(high_threshold) or not np.isfinite(recession_threshold):
        return []
    valid = np.isfinite(observation) & (observation >= 0.0)
    above_high = observation >= high_threshold
    if high_threshold <= 0.0:
        # A zero Q95 is possible for intermittent gauges. Treating every
        # zero-flow day as an exceedance would turn the record into a flood.
        above_high = observation > 0.0
    high_positions = np.flatnonzero(valid & above_high)
    if not high_positions.size:
        return []

    clusters: list[list[int]] = [[int(high_positions[0])]]
    for position in high_positions[1:]:
        previous = clusters[-1][-1]
        intervening_days = int((dates[position] - dates[previous]).days) - 1
        if intervening_days <= config.maximum_high_flow_gap_days:
            clusters[-1].append(int(position))
        else:
            clusters.append([int(position)])

    records = []
    for cluster in clusters:
        first_high, last_high = cluster[0], cluster[-1]
        start, end = first_high, last_high
        lower_limit = max(0, first_high - config.maximum_boundary_search_days)
        upper_limit = min(
            len(observation) - 1,
            last_high + config.maximum_boundary_search_days,
        )
        above_recession = observation >= recession_threshold
        if recession_threshold <= 0.0:
            above_recession = observation > 0.0
        while (
            start > lower_limit
            and valid[start - 1]
            and above_recession[start - 1]
        ):
            start -= 1
        while (
            end < upper_limit
            and valid[end + 1]
            and above_recession[end + 1]
        ):
            end += 1
        # Retain the finite crossing day on either side as hydrograph context.
        if start > lower_limit and valid[start - 1]:
            start -= 1
        if end < upper_limit and valid[end + 1]:
            end += 1
        start, end = pad_event_interval(
            start, end, len(observation), config.minimum_event_duration_days
        )
        finite_positions = np.arange(start, end + 1)[valid[start:end + 1]]
        peak_position = int(
            finite_positions[np.argmax(observation[finite_positions])]
        )
        records.append({
            "start": dates[start],
            "end": dates[end],
            "representative_peak_date": dates[peak_position],
            "representative_peak_m3_s": float(observation[peak_position]),
            "representative_peak_ratio": float(
                observation[peak_position] / high_threshold
            ) if high_threshold > 0 else float(observation[peak_position]),
            "outlet_anchored": bool(outlet_anchored),
            "source_stations": {station},
        })

    records = merge_event_intervals(records)
    if config.maximum_events is not None and len(records) > config.maximum_events:
        records = sorted(
            records,
            key=lambda event: event["representative_peak_ratio"],
            reverse=True,
        )[:config.maximum_events]
        records.sort(key=lambda event: event["start"])
    return records


def event_intervals_overlap(
    first: dict[str, Any], second: dict[str, Any], tolerance_days: int = 0,
) -> bool:
    """Return whether two inclusive event intervals overlap within a tolerance."""
    tolerance = pd.Timedelta(days=tolerance_days)
    return (
        first["start"] <= second["end"] + tolerance
        and second["start"] <= first["end"] + tolerance
    )


def merge_two_events(
    first: dict[str, Any], second: dict[str, Any],
) -> dict[str, Any]:
    """Merge two observation-defined interval records without simulation input."""
    anchored = first["outlet_anchored"] or second["outlet_anchored"]
    # Prefer an observed outlet event whenever the component contains one and
    # otherwise use a deterministic peak ranking. This makes event reduction
    # independent of the order of internal gauges.
    representative = max(
        (first, second),
        key=lambda event: (
            int(event["outlet_anchored"]),
            float(event["representative_peak_ratio"]),
            float(event["representative_peak_m3_s"]),
            -pd.Timestamp(event["representative_peak_date"]).value,
        ),
    )
    return {
        "start": min(first["start"], second["start"]),
        "end": max(first["end"], second["end"]),
        "representative_peak_date": representative["representative_peak_date"],
        "representative_peak_m3_s": representative["representative_peak_m3_s"],
        "representative_peak_ratio": representative["representative_peak_ratio"],
        "outlet_anchored": anchored,
        "source_stations": set(first["source_stations"])
        | set(second["source_stations"]),
    }


def merge_event_intervals(
    events: list[dict[str, Any]], tolerance_days: int = 0,
) -> list[dict[str, Any]]:
    """Merge overlapping event windows into chronological non-overlapping records."""
    merged: list[dict[str, Any]] = []
    for event in sorted(
        events,
        key=lambda value: (
            value["start"],
            value["end"],
            not value["outlet_anchored"],
            value["representative_peak_date"],
            tuple(sorted(value["source_stations"])),
        ),
    ):
        event = {**event, "source_stations": set(event["source_stations"])}
        if merged and event_intervals_overlap(
            merged[-1], event, tolerance_days=tolerance_days
        ):
            merged[-1] = merge_two_events(merged[-1], event)
        else:
            merged.append(event)
    return merged


def build_flood_event_catalog(
    dates: pd.DatetimeIndex,
    observation: np.ndarray,
    station_names: Sequence[str],
    outlet_station: str,
    thresholds: pd.DataFrame,
    config: FloodEventConfig,
) -> pd.DataFrame:
    """Build outlet-anchored public events plus mapped local-only events."""
    names = list(station_names)
    threshold_by_station = thresholds.set_index("station")
    outlet_index = names.index(outlet_station)
    outlet_threshold = threshold_by_station.loc[outlet_station]
    events = detect_station_events(
        dates,
        observation[:, outlet_index],
        outlet_threshold["high_flow_threshold_m3_s"],
        outlet_threshold["recession_threshold_m3_s"],
        outlet_station,
        config,
        outlet_anchored=True,
    )

    for station_index, station in enumerate(names):
        if station == outlet_station:
            continue
        threshold = threshold_by_station.loc[station]
        local_events = detect_station_events(
            dates,
            observation[:, station_index],
            threshold["high_flow_threshold_m3_s"],
            threshold["recession_threshold_m3_s"],
            station,
            config,
            outlet_anchored=False,
        )
        events.extend(local_events)

    # Build connected interval components once, after every station has been
    # detected independently. The resulting catalog cannot depend on station
    # iteration order.
    events = merge_event_intervals(
        events,
        tolerance_days=config.event_mapping_tolerance_days,
    )
    records = []
    used_ids: dict[str, int] = {}
    for rank, event in enumerate(events, 1):
        start_position = int(dates.get_indexer([event["start"]])[0])
        end_position = int(dates.get_indexer([event["end"]])[0])
        outlet_values = observation[start_position:end_position + 1, outlet_index]
        outlet_valid = np.isfinite(outlet_values) & (outlet_values >= 0.0)
        if outlet_valid.any():
            local_positions = np.flatnonzero(outlet_valid)
            outlet_peak_offset = int(
                local_positions[np.argmax(outlet_values[local_positions])]
            )
            outlet_peak_date = dates[start_position + outlet_peak_offset]
            outlet_peak_value = float(outlet_values[outlet_peak_offset])
        else:
            outlet_peak_date, outlet_peak_value = pd.NaT, np.nan
        identifier_date = (
            outlet_peak_date
            if event["outlet_anchored"] and not pd.isna(outlet_peak_date)
            else event["representative_peak_date"]
        )
        identifier_base = f"E_{pd.Timestamp(identifier_date):%Y%m%d}"
        used_ids[identifier_base] = used_ids.get(identifier_base, 0) + 1
        event_id = identifier_base
        if used_ids[identifier_base] > 1:
            event_id = f"{identifier_base}_{used_ids[identifier_base]:02d}"
        records.append({
            "event_id": event_id,
            "event_rank": rank,
            "event_type": (
                "outlet_anchored" if event["outlet_anchored"] else "local_only"
            ),
            "event_start": event["start"].date().isoformat(),
            "event_end": event["end"].date().isoformat(),
            "event_duration_days": int((event["end"] - event["start"]).days + 1),
            "representative_peak_date": (
                event["representative_peak_date"].date().isoformat()
            ),
            "representative_peak_m3_s": event["representative_peak_m3_s"],
            "anchor_peak_date": pd.Timestamp(identifier_date).date().isoformat(),
            "outlet_peak_date": (
                outlet_peak_date.date().isoformat()
                if not pd.isna(outlet_peak_date) else None
            ),
            "outlet_peak_m3_s": outlet_peak_value,
            "source_station_count": len(event["source_stations"]),
            "source_stations": json.dumps(
                sorted(event["source_stations"]), ensure_ascii=False
            ),
        })
    return pd.DataFrame.from_records(records, columns=[
        "event_id", "event_rank", "event_type", "event_start", "event_end",
        "event_duration_days", "representative_peak_date",
        "representative_peak_m3_s", "anchor_peak_date", "outlet_peak_date",
        "outlet_peak_m3_s", "source_station_count", "source_stations",
    ])


def calculate_flood_event_metrics(
    catalog: pd.DataFrame,
    dates: pd.DatetimeIndex,
    prediction: np.ndarray,
    observation: np.ndarray,
    station_names: Sequence[str],
    config: FloodEventConfig,
) -> pd.DataFrame:
    """Calculate event-by-station hydrograph, peak, timing, and volume metrics."""
    records = []
    for event in catalog.itertuples(index=False):
        start = int(dates.get_indexer([pd.Timestamp(event.event_start)])[0])
        end = int(dates.get_indexer([pd.Timestamp(event.event_end)])[0]) + 1
        event_dates = dates[start:end]
        source_stations = set(json.loads(str(event.source_stations)))
        for station_index, station in enumerate(station_names):
            obs = observation[start:end, station_index]
            sim = prediction[start:end, station_index]
            observation_valid = np.isfinite(obs) & (obs >= 0.0)
            simulation_valid = np.isfinite(sim)
            paired = observation_valid & simulation_valid
            valid_days = int(paired.sum())
            duration = len(event_dates)
            coverage = valid_days / duration if duration else 0.0
            reasons = []
            if valid_days < config.minimum_valid_pair_days:
                reasons.append("insufficient_valid_pairs")
            if coverage < config.minimum_pair_coverage:
                reasons.append("coverage_below_threshold")
            paired_obs = obs[paired]
            if (
                valid_days
                and np.std(paired_obs) <= config.minimum_observation_std_m3_s
            ):
                reasons.append("constant_observation")
            if (
                valid_days
                and np.mean(paired_obs) <= config.minimum_observation_mean_m3_s
            ):
                reasons.append("zero_observation_mean")
            score = station_metrics(sim, obs)
            metrics_valid = not reasons and np.isfinite(score["NSE"]) and np.isfinite(score["KGE"])
            if not metrics_valid and not reasons:
                reasons.append("nonfinite_efficiency_metric")

            observed_positions = np.flatnonzero(observation_valid)
            simulated_positions = np.flatnonzero(simulation_valid)
            if observed_positions.size:
                observed_peak_position = int(
                    observed_positions[np.argmax(obs[observed_positions])]
                )
                observed_peak = float(obs[observed_peak_position])
                observed_volume = float(
                    obs[observed_positions].sum() * 86400.0
                )
                observed_peak_date = event_dates[observed_peak_position]
            else:
                observed_peak = observed_volume = np.nan
                observed_peak_date = pd.NaT

            if simulated_positions.size:
                simulated_peak_position = int(
                    simulated_positions[np.argmax(sim[simulated_positions])]
                )
                simulated_peak = float(sim[simulated_peak_position])
                simulated_peak_date = event_dates[simulated_peak_position]
            else:
                simulated_peak = np.nan
                simulated_peak_date = pd.NaT

            # Compare volumes over all observation-valid days. If simulation
            # is missing on any such day, keep the observation-only volume but
            # mark the simulated volume and its error unavailable.
            volume_comparable = bool(
                observed_positions.size
                and simulation_valid[observed_positions].all()
            )
            simulated_volume = (
                float(sim[observed_positions].sum() * 86400.0)
                if volume_comparable else np.nan
            )

            records.append({
                "event_id": event.event_id,
                "event_rank": event.event_rank,
                "event_type": event.event_type,
                "event_start": event.event_start,
                "event_end": event.event_end,
                "event_duration_days": duration,
                "station": station,
                "station_high_flow_detected": station in source_stations,
                "observation_valid_days": int(observation_valid.sum()),
                "simulation_valid_days": int(simulation_valid.sum()),
                "valid_pair_days": valid_days,
                "pair_coverage": coverage,
                "metrics_valid": bool(metrics_valid),
                "invalid_reason": "|".join(reasons),
                "volume_comparable": volume_comparable,
                "event_nse": score["NSE"] if metrics_valid else np.nan,
                "event_kge": score["KGE"] if metrics_valid else np.nan,
                "event_correlation": score["Correlation"] if metrics_valid else np.nan,
                "event_std_ratio": score["StdRatio"] if metrics_valid else np.nan,
                "event_mean_ratio": score["MeanRatio"] if metrics_valid else np.nan,
                "event_rmse_m3_s": score["RMSE"] if valid_days else np.nan,
                "event_bias_m3_s": score["Bias"] if valid_days else np.nan,
                "observed_peak_date": (
                    observed_peak_date.date().isoformat()
                    if not pd.isna(observed_peak_date) else None
                ),
                "simulated_peak_date": (
                    simulated_peak_date.date().isoformat()
                    if not pd.isna(simulated_peak_date) else None
                ),
                "observed_peak_m3_s": observed_peak,
                "simulated_peak_m3_s": simulated_peak,
                "peak_error_m3_s": simulated_peak - observed_peak,
                "peak_relative_error": (
                    (simulated_peak - observed_peak) / observed_peak
                    if observed_peak > 0 else np.nan
                ),
                "peak_timing_error_days": (
                    int((simulated_peak_date - observed_peak_date).days)
                    if not pd.isna(observed_peak_date)
                    and not pd.isna(simulated_peak_date) else np.nan
                ),
                "observed_event_volume_m3": observed_volume,
                "simulated_event_volume_m3": simulated_volume,
                "event_volume_error_m3": simulated_volume - observed_volume,
                "event_volume_relative_error": (
                    (simulated_volume - observed_volume) / observed_volume
                    if observed_volume > 0 else np.nan
                ),
            })
    return pd.DataFrame.from_records(records, columns=[
        "event_id", "event_rank", "event_type", "event_start", "event_end",
        "event_duration_days", "station", "observation_valid_days",
        "simulation_valid_days", "valid_pair_days", "pair_coverage",
        "station_high_flow_detected", "metrics_valid", "invalid_reason",
        "volume_comparable",
        "event_nse", "event_kge",
        "event_correlation", "event_std_ratio", "event_mean_ratio",
        "event_rmse_m3_s", "event_bias_m3_s", "observed_peak_date",
        "simulated_peak_date", "observed_peak_m3_s", "simulated_peak_m3_s",
        "peak_error_m3_s", "peak_relative_error", "peak_timing_error_days",
        "observed_event_volume_m3", "simulated_event_volume_m3",
        "event_volume_error_m3", "event_volume_relative_error",
    ])


def summarize_values(records: pd.DataFrame, column: str) -> dict[str, float]:
    """Return finite mean, median, and interquartile statistics."""
    values = pd.to_numeric(records[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            f"{column}_mean": np.nan,
            f"{column}_median": np.nan,
            f"{column}_q25": np.nan,
            f"{column}_q75": np.nan,
        }
    return {
        f"{column}_mean": float(np.mean(values)),
        f"{column}_median": float(np.median(values)),
        f"{column}_q25": float(np.quantile(values, 0.25)),
        f"{column}_q75": float(np.quantile(values, 0.75)),
    }


def summarize_flood_events(
    catalog: pd.DataFrame,
    metrics: pd.DataFrame,
    station_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create station-, event-, and equal-station-weight basin summaries."""
    summary_metrics = (
        "event_nse", "event_kge", "event_rmse_m3_s", "event_bias_m3_s",
        "peak_relative_error", "peak_timing_error_days",
        "event_volume_relative_error",
    )
    primary_metrics = metrics[
        metrics["event_type"] == "outlet_anchored"
    ]
    station_records = []
    for station in station_names:
        group = primary_metrics[primary_metrics["station"] == station]
        row = {
            "station": station,
            "event_scope": "outlet_anchored",
            "events_total": int(len(group)),
            "events_valid": int(group["metrics_valid"].sum()) if len(group) else 0,
            "valid_event_fraction": (
                float(group["metrics_valid"].mean()) if len(group) else np.nan
            ),
        }
        for column in summary_metrics:
            row.update(summarize_values(group, column))
        station_records.append(row)
    by_station = pd.DataFrame.from_records(station_records)

    event_records = []
    for event in catalog.itertuples(index=False):
        group = metrics[metrics["event_id"] == event.event_id]
        row = {
            "event_id": event.event_id,
            "event_rank": event.event_rank,
            "event_type": event.event_type,
            "event_start": event.event_start,
            "event_end": event.event_end,
            "stations_total": int(len(group)),
            "stations_valid": int(group["metrics_valid"].sum()) if len(group) else 0,
            "valid_station_fraction": (
                float(group["metrics_valid"].mean()) if len(group) else np.nan
            ),
        }
        for column in summary_metrics:
            row.update(summarize_values(group, column))
        event_records.append(row)
    if event_records:
        by_event = pd.DataFrame.from_records(event_records)
    else:
        statistic_columns = [
            f"{metric}_{statistic}"
            for metric in summary_metrics
            for statistic in ("mean", "median", "q25", "q75")
        ]
        by_event = pd.DataFrame(columns=[
            "event_id", "event_rank", "event_type", "event_start", "event_end",
            "stations_total", "stations_valid", "valid_station_fraction",
            *statistic_columns,
        ])

    basin = {
        "events_total": int(len(catalog)),
        "outlet_anchored_events": int(
            (catalog["event_type"] == "outlet_anchored").sum()
        ) if len(catalog) else 0,
        "local_only_events": int(
            (catalog["event_type"] == "local_only").sum()
        ) if len(catalog) else 0,
        "stations_total": int(len(station_names)),
        "aggregation": (
            "outlet-anchored events only; station median across events, then "
            "equal weight across stations"
        ),
    }
    for column in summary_metrics:
        station_values = pd.to_numeric(
            by_station.get(f"{column}_median", pd.Series(dtype=float)),
            errors="coerce",
        ).to_numpy(dtype=float)
        station_values = station_values[np.isfinite(station_values)]
        basin[f"{column}_station_equal_mean"] = (
            float(np.mean(station_values)) if station_values.size else np.nan
        )
        basin[f"{column}_station_equal_median"] = (
            float(np.median(station_values)) if station_values.size else np.nan
        )
        basin[f"{column}_stations_available"] = int(station_values.size)
    return by_station, by_event, pd.DataFrame([basin])


def evaluate_flood_events(
    dates,
    prediction,
    observation,
    station_names: Sequence[str],
    *,
    reference_dates,
    reference_observation,
    outlet_station: str,
    config: FloodEventConfig | None = None,
) -> FloodEventEvaluation:
    """Evaluate observation-defined flood events for one continuous period.

    ``reference_observation`` must be the training-period observations even
    when ``dates`` describes the test period.  This prevents test-period
    information from changing the high-flow thresholds.
    """
    config = config or FloodEventConfig()
    names = [str(value) for value in station_names]
    if not names or len(set(names)) != len(names):
        raise ValueError("station_names must be a non-empty unique sequence")
    outlet_station = str(outlet_station)
    if outlet_station not in names:
        raise ValueError("outlet_station must be present in station_names")
    period_dates = daily_dates(dates, "dates", require_contiguous=True)
    training_dates = daily_dates(
        reference_dates, "reference_dates", require_contiguous=False
    )
    prediction_array = numpy_matrix(prediction, "prediction", len(names))
    observation_array = numpy_matrix(observation, "observation", len(names))
    reference_array = numpy_matrix(
        reference_observation, "reference_observation", len(names)
    )
    if len(period_dates) != len(prediction_array) or len(period_dates) != len(observation_array):
        raise ValueError("dates, prediction, and observation must share time length")
    if len(training_dates) != len(reference_array):
        raise ValueError("reference_dates and reference_observation must share time length")

    thresholds = compute_flood_thresholds(reference_array, names, config)
    outlet_threshold = thresholds.loc[
        thresholds["station"] == outlet_station, "high_flow_threshold_m3_s"
    ].iloc[0]
    if not np.isfinite(outlet_threshold):
        raise ValueError("Outlet training observations contain no valid discharge values")
    catalog = build_flood_event_catalog(
        period_dates,
        observation_array,
        names,
        outlet_station,
        thresholds,
        config,
    )
    metrics = calculate_flood_event_metrics(
        catalog,
        period_dates,
        prediction_array,
        observation_array,
        names,
        config,
    )
    by_station, by_event, basin_summary = summarize_flood_events(
        catalog, metrics, names
    )
    metadata = {
        "selection": "observation-only variable-duration threshold events",
        "threshold_reference": "training-period station observations only",
        "public_event_anchor": outlet_station,
        "local_event_policy": (
            "internal-station events overlapping a catalog event extend that event; "
            "non-overlapping events remain local_only"
        ),
        "event_id_policy": "deterministic observation-defined representative peak date",
        "efficiency_validity": (
            "minimum paired days and coverage, nonconstant positive-mean observation"
        ),
        "peak_and_volume_policy": (
            "observed peak/date/volume use observation-valid days only; "
            "simulated volume is reported only when simulation is finite on "
            "every observation-valid day"
        ),
        "basin_aggregation": (
            "median across events within each station, then equal station weighting"
        ),
        "affects_training_or_checkpoint_selection": False,
        "period_start": period_dates[0].date().isoformat() if len(period_dates) else None,
        "period_end": period_dates[-1].date().isoformat() if len(period_dates) else None,
        "reference_start": (
            training_dates[0].date().isoformat() if len(training_dates) else None
        ),
        "reference_end": (
            training_dates[-1].date().isoformat() if len(training_dates) else None
        ),
        "config": asdict(config),
        "thresholds_m3_s": {
            row.station: {
                "high_flow": (
                    float(row.high_flow_threshold_m3_s)
                    if np.isfinite(row.high_flow_threshold_m3_s) else None
                ),
                "recession": (
                    float(row.recession_threshold_m3_s)
                    if np.isfinite(row.recession_threshold_m3_s) else None
                ),
                "reference_valid_days": int(row.reference_valid_days),
            }
            for row in thresholds.itertuples(index=False)
        },
    }
    return FloodEventEvaluation(
        catalog=catalog,
        metrics=metrics,
        by_station=by_station,
        by_event=by_event,
        basin_summary=basin_summary,
        thresholds=thresholds,
        metadata=metadata,
    )
