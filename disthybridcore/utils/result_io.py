"""Save consolidated daily discharge series and model metrics."""

import json
from pathlib import Path

import pandas as pd
import torch

from .hydrology_metrics import build_flow_regime_metrics, build_metrics_table
from .flood_events import FloodEventConfig, evaluate_flood_events


def save_flood_event_results(
    result_dir,
    dates,
    prediction,
    observation,
    station_names,
    *,
    reference_dates,
    reference_observation,
    outlet_station,
    period_label,
    config=None,
):
    """Evaluate variable-duration events and save all event tables."""
    evaluation = evaluate_flood_events(
        dates,
        prediction,
        observation,
        station_names,
        reference_dates=reference_dates,
        reference_observation=reference_observation,
        outlet_station=outlet_station,
        config=config or FloodEventConfig(),
    )
    result_dir = Path(result_dir)
    for stem, table in evaluation.tables().items():
        exported = table.copy()
        exported.insert(0, "period", str(period_label))
        exported.to_csv(
            result_dir / f"{stem}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return evaluation.metadata


def save_daily_results(
    result_dir,
    dates,
    prediction,
    observation,
    station_names,
    model_name,
    primary_station=None,
    metadata=None,
    training_target=None,
    event_reference_dates=None,
    event_reference_observation=None,
):
    """保存日流量张量、可直接打开的CSV、指标和回放身份，不复制原始网格文件。"""
    result_dir = Path(result_dir)
    series_dir = result_dir / "discharge_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dates": [pd.Timestamp(value).isoformat() for value in dates],
        "prediction": prediction.detach().cpu(),
        "observation": observation.detach().cpu(),
        "station_names": list(station_names),
        "units": "m3 s-1",
        "frequency": "daily",
        "metadata": dict(metadata or {}),
    }
    prediction, observation = payload["prediction"], payload["observation"]
    expected = (len(dates), len(station_names))
    if any(tuple(values.shape) != expected for values in (prediction, observation)):
        raise ValueError("Saved discharge dates, stations, and tensor shapes must match")
    date_index = pd.DatetimeIndex(dates)
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise ValueError("Saved daily discharge dates must be unique and chronological")
    if training_target is not None:
        training_target = training_target.detach().cpu()
        if tuple(training_target.shape) != expected:
            raise ValueError("Training targets must match saved discharge dates and stations")
        payload["training_target"] = training_target
        if not (torch.eq(training_target, observation) | (torch.isnan(training_target) & torch.isnan(observation))).all():
            raise ValueError("Imputation is disabled: training targets must be original observations")
        payload["metadata"].update({
            "discharge_missing_method": "no_imputation; missing observations retained as NaN",
            "missing_observed_values": int((~torch.isfinite(observation)).sum()),
            "metric_reference": "original finite nonnegative observations only",
        })
    if primary_station is None:
        raise ValueError("Flood-event evaluation requires primary_station")
    if event_reference_dates is None or event_reference_observation is None:
        raise ValueError(
            "Flood-event evaluation requires training-period reference observations"
        )
    payload["observation_valid"] = torch.isfinite(observation) & (observation >= 0)
    rows = [
        pd.DataFrame({
            "date": payload["dates"], "station": station,
            "observed_m3_s": observation[:, index].numpy(),
            "simulated_m3_s": prediction[:, index].numpy(),
        })
        for index, station in enumerate(station_names)
    ]
    if training_target is not None:
        for index, frame in enumerate(rows):
            frame["training_target_m3_s"] = training_target[:, index].numpy()
            frame["observation_valid"] = payload["observation_valid"][:, index].numpy()
    pd.concat(rows, ignore_index=True).to_csv(
        series_dir / "daily_discharge.csv", index=False, encoding="utf-8-sig"
    )
    payload["metadata"]["flood_event_diagnostics"] = save_flood_event_results(
        result_dir,
        date_index,
        prediction.numpy(),
        observation.numpy(),
        station_names,
        reference_dates=event_reference_dates,
        reference_observation=event_reference_observation,
        outlet_station=primary_station,
        period_label=payload["metadata"].get("period", "Unspecified"),
    )
    flow_regime_metrics = build_flow_regime_metrics(
        prediction.numpy(),
        observation.numpy(),
        event_reference_observation,
        station_names,
    )
    flow_regime_metrics.to_csv(
        result_dir / "flow_regime_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload["metadata"]["flow_regime_diagnostics"] = {
        "output": "flow_regime_metrics.csv",
        "threshold_reference": "training-period station observations only",
        "low_flow_definition": "Qobs <= station training Q75",
        "high_flow_definition": "Qobs >= station training Q95",
        "daily_change_definition": (
            "difference of log1p discharge between consecutive low-flow days"
        ),
        "affects_training_or_checkpoint_selection": False,
    }
    payload["metadata"]["internal_process_diagnostics"] = {
        "daily_basin_processes": "daily_hydrology.csv",
        "daily_gauge_upstream_processes": "gauge_process_daily.csv",
        "gauge_period_summary": "gauge_process_diagnostics.csv",
        "gauge_routing_water_balance": "gauge_routing_water_balance.csv",
        "console_output": False,
        "affects_training_or_checkpoint_selection": False,
    }
    torch.save(payload, series_dir / "daily_discharge.pt")
    with (result_dir / "evaluation_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(payload["metadata"], stream, ensure_ascii=False, indent=2, allow_nan=False)

    metrics = build_metrics_table(
        prediction.numpy(),
        observation.numpy(),
        station_names,
        model_name,
    )
    metrics.to_csv(result_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    summary_station = "Mean" if len(station_names) > 1 else station_names[0]
    summary = (
        metrics[metrics["Station"] == summary_station]
        .set_index("Model")
        .to_dict("index")
    )
    with (result_dir / "metrics_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=True)
    if primary_station is not None:
        outlet_summary = (
            metrics[metrics["Station"] == primary_station]
            .set_index("Model")
            .to_dict("index")
        )
        with (result_dir / "outlet_metrics_summary.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(
                outlet_summary, stream, ensure_ascii=False, indent=2, allow_nan=True
            )
    return metrics


def daily_hydrology_frame(diagnostics, start):
    """导出面积加权的逐日强迫与产流；全部保持物理单位。"""
    fields = {
        "prec_mm_day": "prec", "temp_c": "temp", "pet_mm_day": "ep",
        "precipitation_mm_day": "precipitation", "actual_et_mm_day": "actual_et",
        "q0_mm_day": "q0", "q1_mm_day": "q1", "q2_mm_day": "q2",
        "runoff_mm_day": "total_runoff",
        "generated_q0_m3_s": "generated_q0_m3_s",
        "generated_q1_m3_s": "generated_q1_m3_s",
        "generated_q2_m3_s": "generated_q2_m3_s",
        "generated_total_m3_s": "generated_total_m3_s",
        "routing_outlet_m3_s": "routing_outlet_daily_m3_s",
        "snowpack_storage_mm": "snowpack_storage",
        "meltwater_storage_mm": "meltwater_storage",
        "soil_moisture_storage_mm": "soil_moisture_storage",
        "upper_zone_storage_mm": "upper_zone_storage",
        "lower_zone_storage_mm": "lower_zone_storage",
        "routing_storage_m3": "routing_storage_daily_m3",
    }
    return pd.DataFrame({
        "date": pd.date_range(start, periods=diagnostics["prec"].shape[1], freq="D"),
        **{column: diagnostics[key][0].detach().cpu().numpy()
           for column, key in fields.items()},
    })


def gauge_process_frame(
    diagnostics,
    start,
    station_names,
    *,
    simulation=None,
    observation=None,
):
    """Build long-form daily upstream-production and routed-flow diagnostics."""
    fields = {
        "precipitation_mm_day": "gauge_precipitation",
        "actual_et_mm_day": "gauge_actual_et",
        "recharge_mm_day": "gauge_recharge",
        "q0_mm_day": "gauge_q0",
        "q1_mm_day": "gauge_q1",
        "q2_mm_day": "gauge_q2",
        "generated_runoff_mm_day": "gauge_total_runoff",
        "generated_runoff_m3_s": "gauge_generated_total_m3_s",
        "routed_discharge_m3_s": "routing_gauge_daily_m3_s",
        "routing_storage_m3": "routing_gauge_storage_daily_m3",
    }
    days = diagnostics["gauge_precipitation"].shape[1]
    gauges = diagnostics["gauge_precipitation"].shape[2]
    if gauges != len(station_names):
        raise ValueError("Gauge diagnostics and station names do not match")
    dates = pd.date_range(start, periods=days, freq="D")
    area = diagnostics["gauge_upstream_area_km2"][0].detach().cpu().numpy()
    arrays = {
        column: diagnostics[key][0].detach().cpu().numpy()
        for column, key in fields.items()
    }
    if simulation is not None:
        arrays["simulated_m3_s"] = simulation.detach().cpu().numpy()
    if observation is not None:
        arrays["observed_m3_s"] = observation.detach().cpu().numpy()
    frames = []
    for gauge_index, station in enumerate(station_names):
        frames.append(pd.DataFrame({
            "date": dates,
            "station": station,
            "upstream_area_km2": area[gauge_index],
            **{
                column: values[:, gauge_index]
                for column, values in arrays.items()
            },
        }))
    return pd.concat(frames, ignore_index=True)


def gauge_routing_budget_frame(diagnostics, start, station_names):
    """Build one exact D8 storage budget row per gauge and window."""
    start = pd.Timestamp(start)
    days = diagnostics["routing_gauge_daily_m3_s"].shape[1]
    fields = {
        "routing_storage_initial_m3": "routing_gauge_storage_initial_m3",
        "routing_storage_final_m3": "routing_gauge_storage_final_m3",
        "routing_local_inflow_volume_m3": "routing_gauge_local_inflow_volume_m3",
        "routing_outflow_volume_m3": "routing_gauge_outflow_volume_m3",
        "routing_budget_residual_m3": "routing_gauge_budget_residual_m3",
    }
    arrays = {
        column: diagnostics[key][0].detach().cpu().numpy()
        for column, key in fields.items()
    }
    return pd.DataFrame({
        "target_start": start.isoformat(),
        "target_end_exclusive": (start + pd.Timedelta(days=days)).isoformat(),
        "target_days": days,
        "station": list(station_names),
        **arrays,
    })


def save_gauge_process_diagnostics(result_dir, daily_frames, budget_frames):
    """Save daily, period-summary, and exact routing-budget gauge diagnostics."""
    result_dir = Path(result_dir)
    daily = pd.concat(daily_frames, ignore_index=True)
    budgets = pd.concat(budget_frames, ignore_index=True)
    daily.to_csv(
        result_dir / "gauge_process_daily.csv", index=False, encoding="utf-8-sig"
    )
    budgets.to_csv(
        result_dir / "gauge_routing_water_balance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rows = []
    for station, group in daily.groupby("station", sort=False):
        valid = group["observed_m3_s"].notna() & (group["observed_m3_s"] >= 0)
        observed_mean = group.loc[valid, "observed_m3_s"].mean()
        simulated_mean = group.loc[valid, "simulated_m3_s"].mean()
        generated_mean = group.loc[valid, "generated_runoff_m3_s"].mean()
        runoff_mean = group["generated_runoff_mm_day"].mean()
        rows.append({
            "station": station,
            "upstream_area_km2": group["upstream_area_km2"].iloc[0],
            "valid_observation_days": int(valid.sum()),
            "precipitation_mean_mm_day": group["precipitation_mm_day"].mean(),
            "actual_et_mean_mm_day": group["actual_et_mm_day"].mean(),
            "recharge_mean_mm_day": group["recharge_mm_day"].mean(),
            "q0_mean_mm_day": group["q0_mm_day"].mean(),
            "q1_mean_mm_day": group["q1_mm_day"].mean(),
            "q2_mean_mm_day": group["q2_mm_day"].mean(),
            "generated_runoff_mean_mm_day": runoff_mean,
            "generated_runoff_mean_m3_s": generated_mean,
            "routed_discharge_mean_m3_s": simulated_mean,
            "observed_discharge_mean_m3_s": observed_mean,
            "generated_to_observed_ratio": (
                generated_mean / observed_mean if observed_mean > 0 else float("nan")
            ),
            "routed_to_generated_ratio": (
                simulated_mean / generated_mean if generated_mean > 0 else float("nan")
            ),
            "quickflow_fraction": (
                group["q0_mm_day"].mean() / runoff_mean
                if runoff_mean > 0 else float("nan")
            ),
            "interflow_fraction": (
                group["q1_mm_day"].mean() / runoff_mean
                if runoff_mean > 0 else float("nan")
            ),
            "baseflow_fraction": (
                group["q2_mm_day"].mean() / runoff_mean
                if runoff_mean > 0 else float("nan")
            ),
            "routing_budget_mean_abs_residual_m3": budgets.loc[
                budgets["station"] == station, "routing_budget_residual_m3"
            ].abs().mean(),
        })
    pd.DataFrame(rows).to_csv(
        result_dir / "gauge_process_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )


def window_water_balance(diagnostics, start):
    """把一个窗口的逐日水文诊断汇总为P-AET-R-ΔS及数值闭合误差。"""
    start = pd.Timestamp(start)
    days = diagnostics["precipitation"].shape[1]
    precipitation = diagnostics["precipitation"].sum(dim=1).mean()
    actual_et = diagnostics["actual_et"].sum(dim=1).mean()
    runoff = diagnostics["total_runoff"].sum(dim=1).mean()
    storage_change = (
        diagnostics["runoff_storage_final"] - diagnostics["runoff_storage_initial"]
    ).mean()
    values = {
        "precipitation_total_mm": precipitation,
        "actual_et_total_mm": actual_et,
        "runoff_generation_total_mm": runoff,
        "runoff_storage_initial_mm": diagnostics["runoff_storage_initial"].mean(),
        "runoff_storage_final_mm": diagnostics["runoff_storage_final"].mean(),
        "runoff_storage_change_mm": storage_change,
        "runoff_budget_closure_mm": precipitation - actual_et - runoff - storage_change,
        "actual_et_mean_mm_day": diagnostics["actual_et"].mean(),
        "quickflow_mean_mm_day": diagnostics["quickflow"].mean(),
        "interflow_mean_mm_day": diagnostics["interflow"].mean(),
        "baseflow_mean_mm_day": diagnostics["baseflow"].mean(),
        "runoff_balance_mean_abs_mm_day": diagnostics["runoff_water_balance"].abs().mean(),
        "runoff_balance_max_abs_mm_day": diagnostics["runoff_water_balance"].abs().max(),
    }
    # 汇流体积预算将窗口末边内储水计入最终存储量。
    for name in (
        "routing_storage_initial_m3", "routing_storage_final_m3",
        "routing_local_inflow_volume_m3", "routing_generated_runoff_volume_m3",
        "routing_outlet_volume_m3", "routing_budget_residual_m3",
    ):
        if name in diagnostics:
            values[name] = diagnostics[name].mean()
    return {
        "target_start": start.isoformat(),
        "target_end_exclusive": (start + pd.Timedelta(days=days)).isoformat(),
        "target_days": days,
        **{name: float(value.detach().cpu()) for name, value in values.items()},
    }


def save_water_balance_diagnostics(result_dir, records):
    """Persist per-segment flux and numerical closure diagnostics."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame.from_records(records)
    table.to_csv(
        result_dir / "water_balance_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    numeric = table.select_dtypes(include="number")
    summary = {
        column: {
            "mean": float(numeric[column].mean()),
            "maximum": float(numeric[column].max()),
        }
        for column in numeric.columns
        if column != "target_days"
    }
    with (result_dir / "water_balance_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return table
