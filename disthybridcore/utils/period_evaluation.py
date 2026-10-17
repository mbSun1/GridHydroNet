"""Replay a fixed checkpoint continuously and persist hydrological diagnostics."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from disthybridcore.hydromodels.losses import daily_flow_values
from .result_io import (
    daily_hydrology_frame, gauge_process_frame, gauge_routing_budget_frame,
    save_daily_results, save_gauge_process_diagnostics,
    save_water_balance_diagnostics, window_water_balance,
)
from .result_plots import plot_daily_results
from .state_initialization import detach_state, warmup_to_date


def move_batch_to_device(value, device):
    """Move nested tensors to the model device while retaining date strings."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {name: move_batch_to_device(item, device) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_batch_to_device(item, device) for item in value)
    return value


@torch.no_grad()
def evaluate_period(module, dataset, output_dir, *, metadata=None):
    """Replay one continuous period with fixed weights and cross-window state."""
    if not len(dataset):
        raise ValueError("Period evaluation requires at least one target window")
    device = next(module.parameters()).device
    was_training = module.training
    module.eval()
    first_start = pd.Timestamp(dataset.windows[0].target_start)
    initialization_start = min(
        pd.Timestamp(module.params.train_start),
        first_start - pd.Timedelta(days=dataset.spinup_days),
    )
    state, initialization_record = warmup_to_date(
        module.model, dataset, initialization_start, first_start, device,
        chunk_days=module.params.train_window_days,
    )
    previous_end = None
    predictions, observations, dates = [], [], []
    training_targets, process_frames, budgets = [], [], []
    gauge_process_frames, gauge_budget_frames = [], []
    names = list(module.params.observed_station_names)
    try:
        for batch in DataLoader(dataset, batch_size=1, shuffle=False):
            start = pd.Timestamp(batch["target_start"][0])
            if previous_end is not None and start != previous_end:
                raise ValueError(
                    "Continuous evaluation windows must be chronological and contiguous"
                )
            batch = move_batch_to_device(batch, device)
            output = module.model(
                **module.model_arguments(batch, "forcing"), state=state
            )
            state = detach_state(output.final_state)
            prediction = daily_flow_values(output.discharge_daily)[0].cpu()
            count = prediction.shape[0]
            current_dates = pd.date_range(start, periods=count, freq="D")
            previous_end = start + pd.Timedelta(days=count)
            predictions.append(prediction)
            observed = batch["discharge_daily"][0, :, 0].cpu()
            observations.append(observed)
            training_targets.append(
                batch.get("discharge_training_target", batch["discharge_daily"])[
                    0, :, 0
                ].cpu()
            )
            dates.extend(current_dates.tolist())
            budgets.append(window_water_balance(output.diagnostics, start))
            process_frames.append(daily_hydrology_frame(output.diagnostics, start))
            gauge_process_frames.append(gauge_process_frame(
                output.diagnostics,
                start,
                names,
                simulation=prediction,
                observation=observed,
            ))
            gauge_budget_frames.append(gauge_routing_budget_frame(
                output.diagnostics, start, names
            ))
    finally:
        module.train(was_training)

    prediction, observation = (
        torch.cat(values, dim=0) for values in (predictions, observations)
    )
    details = {
        **dict(metadata or {}),
        "evaluation_mode": "continuous",
        "state_initialization": initialization_record,
        "initialization_policy": (
            "fixed checkpoint; one historical warm-up followed by chronological "
            "cross-window state carry"
        ),
        "spinup_days": dataset.spinup_days,
        "target_start": dates[0].isoformat(),
        "target_end_exclusive": previous_end.isoformat(),
        "days": len(dates),
        "forcing_export": "basin-area-weighted physical daily means, not full grid copies",
    }
    reference_dates = pd.date_range(
        pd.Timestamp(module.params.train_start),
        pd.Timestamp(module.params.train_end),
        freq="D",
        inclusive="left",
    )
    reference_observation = dataset.discharge.read(
        reference_dates[0], len(reference_dates)
    )[:, 0]
    metrics = save_daily_results(
        output_dir, dates, prediction, observation, names, module.params.model_label,
        primary_station=module.params.outlet_station_name, metadata=details,
        training_target=torch.cat(training_targets, dim=0),
        event_reference_dates=reference_dates,
        event_reference_observation=reference_observation,
    )
    pd.concat(process_frames, ignore_index=True).to_csv(
        Path(output_dir) / "daily_hydrology.csv", index=False, encoding="utf-8-sig"
    )
    save_water_balance_diagnostics(output_dir, budgets)
    save_gauge_process_diagnostics(
        output_dir, gauge_process_frames, gauge_budget_frames
    )
    if module.params.generate_figures:
        plot_daily_results(
            output_dir, dates, prediction, observation, names, metrics,
            module.params.model_label, period_label=details.get("period", "Training"),
            outlet_station=module.params.outlet_station_name,
        )
    return metrics


def save_period_diagnostics(
    seed_dir, evaluations, model_name, basin_area_km2, *, primary_station,
):
    """Compare fixed-checkpoint forcing, runoff, and discharge by period."""
    records = []
    for label, directory in evaluations:
        directory = Path(directory)
        row = pd.read_csv(directory / "metrics.csv")
        row = row[
            (row["Model"] == model_name) & (row["Station"] == primary_station)
        ].iloc[0]
        process = pd.read_csv(directory / "daily_hydrology.csv").drop(columns="date").mean()
        generated_runoff = process.get("runoff_mm_day", float("nan"))
        generated_total = process.get("generated_total_m3_s", float("nan"))
        routing_outlet = process.get("routing_outlet_m3_s", float("nan"))
        observed_std = row["ObservedStd"]
        bias_penalty = (
            (row["Bias"] / observed_std) ** 2 if observed_std > 0 else float("nan")
        )
        alpha, correlation = row["StdRatio"], row["Correlation"]
        records.append({
            "period": label, "days": int(row["Days"]),
            "KGE": row["KGE"], "NSE": row["NSE"],
            "observed_mean_m3_s": row["ObservedMean"],
            "simulated_mean_m3_s": row["SimulatedMean"],
            "observed_std_m3_s": observed_std, "RMSE_m3_s": row["RMSE"],
            "mean_ratio": row["MeanRatio"], "std_ratio": alpha,
            "correlation": correlation, "nse_bias_penalty": bias_penalty,
            "nse_centered_error_penalty": alpha ** 2 + 1 - 2 * alpha * correlation,
            "observed_runoff_mean_mm_day": row["ObservedMean"] * 86.4 / basin_area_km2,
            "simulated_outflow_mean_mm_day": row["SimulatedMean"] * 86.4 / basin_area_km2,
            "generated_runoff_mean_mm_day": generated_runoff,
            "generated_to_observed_runoff_ratio": (
                generated_runoff
                / (row["ObservedMean"] * 86.4 / basin_area_km2)
                if row["ObservedMean"] > 0 and pd.notna(generated_runoff)
                else float("nan")
            ),
            "routing_outflow_to_generated_ratio": (
                routing_outlet / generated_total
                if pd.notna(routing_outlet) and pd.notna(generated_total)
                and generated_total > 0 else float("nan")
            ),
            "simulated_minus_observed_runoff_mm_day": (
                row["SimulatedMean"] * 86.4 / basin_area_km2
                - row["ObservedMean"] * 86.4 / basin_area_km2
            ),
            **process.to_dict(),
            "generated_baseflow_fraction": (
                process["q2_mm_day"] / process["runoff_mm_day"]
                if process["runoff_mm_day"] > 0 else float("nan")
            ),
        })
    table = pd.DataFrame(records)
    table.to_csv(
        Path(seed_dir) / "period_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    return table
