"""Prepare, calibrate, and validate a continuous lumped HBV92 simulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lumpedcore.calibration import HBV92SpotpySetup, calibrate_sceua
from lumpedcore.data_pipeline import load_lumped_catalog, load_lumped_dataset
from lumpedcore.hydromodels import HBV92Parameters, LumpedHBV92
from lumpedcore.hydromodels.routing import (
    route_runoff_gamma,
    runoff_mm_day_to_discharge,
)
from lumpedcore.prepare_data import prepare_basin_mean_forcing
from lumpedcore.utils.hydrology_metrics import hydrology_metrics
from lumpedcore.utils.result_io import write_csv, write_json, write_yaml
from lumpedcore.utils.result_plots import (
    plot_discharge_scatter,
    plot_flow_duration,
    plot_hydrograph,
)


def create_run_directory(params) -> Path:
    root = Path(params.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    candidate = root / f"{params.basin}_hbv92_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{params.basin}_hbv92_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def serializable_parameters(params) -> dict:
    return {
        key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        for key, value in vars(params).items()
    }


def simulate(dataset, parameter_values: dict, routing: str, routing_length: int):
    hbv_parameters = HBV92Parameters.from_mapping(parameter_values)
    hbv_result = LumpedHBV92().simulate(
        dataset.precipitation,
        dataset.temperature,
        dataset.pet,
        hbv_parameters,
    )
    generated = hbv_result.runoff_mm_day
    if routing == "gamma":
        routing_result = route_runoff_gamma(
            generated,
            float(parameter_values["routing_shape"]),
            float(parameter_values["routing_scale"]),
            routing_length,
        )
        routed = routing_result.routed_runoff_mm_day
        routing_diagnostics = {
            "unit_hydrograph": routing_result.unit_hydrograph.tolist(),
            "tail_runoff_mm": routing_result.tail_runoff_mm.tolist(),
            "tail_storage_mm": float(routing_result.tail_runoff_mm.sum()),
        }
    elif routing == "none":
        routed = generated
        routing_diagnostics = {
            "unit_hydrograph": [1.0],
            "tail_runoff_mm": [],
            "tail_storage_mm": 0.0,
        }
    else:
        raise ValueError(f"Unsupported routing mode: {routing}")
    return (
        hbv_result,
        runoff_mm_day_to_discharge(generated, dataset.basin_area_km2),
        runoff_mm_day_to_discharge(routed, dataset.basin_area_km2),
        routing_diagnostics,
    )


def daily_frame(dataset, hbv_result, generated_q, simulated_q, mask) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dataset.dates[mask],
        "precipitation_mm_day": dataset.precipitation[mask],
        "temperature_degree_Celsius": dataset.temperature[mask],
        "pet_mm_day": dataset.pet[mask],
        "observed_m3_s": dataset.observed_discharge[mask],
        "generated_m3_s": generated_q[mask],
        "simulated_m3_s": simulated_q[mask],
        "q0_mm_day": hbv_result.fluxes["q0"][mask],
        "q1_mm_day": hbv_result.fluxes["q1"][mask],
        "q2_mm_day": hbv_result.fluxes["q2"][mask],
        "actual_et_mm_day": hbv_result.fluxes["actual_et"][mask],
        "snowpack_mm": hbv_result.states["snowpack"][mask],
        "meltwater_mm": hbv_result.states["meltwater"][mask],
        "soil_moisture_mm": hbv_result.states["soil_moisture"][mask],
        "upper_zone_mm": hbv_result.states["upper_zone"][mask],
        "lower_zone_mm": hbv_result.states["lower_zone"][mask],
        "water_balance_mm": hbv_result.fluxes["water_balance"][mask],
    })


def save_period(
    run_dir: Path,
    period_name: str,
    label: str,
    frame: pd.DataFrame,
    model_label: str,
    outlet_name: str,
    *,
    generate_figures: bool,
) -> dict:
    directory = run_dir / period_name
    directory.mkdir(parents=True, exist_ok=True)
    discharge = frame[["date", "observed_m3_s", "simulated_m3_s"]].copy()
    write_csv(directory / "daily_discharge.csv", discharge)
    metrics = hydrology_metrics(frame["simulated_m3_s"], frame["observed_m3_s"])
    metric_row = {
        "Model": model_label,
        "Station": outlet_name,
        "Period": label,
        **metrics,
    }
    write_csv(directory / "metrics.csv", pd.DataFrame([metric_row]))
    if generate_figures:
        plot_hydrograph(
            frame,
            run_dir / "figures" / f"{period_name}_hydrograph.jpg",
            (
                f"{label} | {outlet_name}\n"
                f"NSE = {metrics['NSE']:.3f} | KGE = {metrics['KGE']:.3f} | "
                f"RMSE = {metrics['RMSE']:.3f} m³/s | "
                f"PBIAS = {metrics['PBIAS']:.2f}%"
            ),
        )
    return metric_row


def build_parameter_payload(params, catalog, calibration_result) -> dict:
    return {
        "schema": 1,
        "model": params.model_label,
        "basin": params.basin,
        "outlet_id": catalog.outlet_id,
        "outlet_name": catalog.outlet_name,
        "basin_area_km2": catalog.basin_area_km2,
        "routing": params.routing,
        "routing_length": params.routing_length,
        "objective": "1-NSE",
        "objective_value": calibration_result.objective,
        "evaluations": calibration_result.evaluations,
        "seed": params.seed,
        "periods": {
            "calibration_start": params.calibration_start,
            "calibration_end_exclusive": params.calibration_end,
            "validation_start": params.validation_start,
            "validation_end_exclusive": params.validation_end,
            "spinup_days": params.spinup_days,
        },
        "parameters": calibration_result.parameters,
    }


def run_experiment(params) -> Path:
    run_dir = create_run_directory(params)
    print(f"[本次输出] {run_dir}", flush=True)
    write_yaml(run_dir / "run_parameters.yaml", serializable_parameters(params))

    if params.mode in {"prepare", "all"}:
        prepare_basin_mean_forcing(
            params.project_root,
            params.basin,
            force=params.force_prepare,
        )
    catalog = load_lumped_catalog(params.project_root, params.basin)
    input_summary = {
        "basin": catalog.basin,
        "forcing": str(catalog.forcing),
        "discharge": str(catalog.discharge),
        "outlet_id": catalog.outlet_id,
        "outlet_name": catalog.outlet_name,
        "basin_area_km2": catalog.basin_area_km2,
        "official_area_km2": catalog.official_area_km2,
        "forcing_start": catalog.forcing_start,
        "forcing_end_inclusive": catalog.forcing_end_inclusive,
        "input_signature": catalog.input_signature,
    }
    write_json(run_dir / "preprocessing" / "input_summary.json", input_summary)
    print(
        f"[输入] basin={catalog.basin} | outlet={catalog.outlet_id} "
        f"{catalog.outlet_name} | area={catalog.basin_area_km2:.3f} km2",
        flush=True,
    )
    if params.mode == "prepare":
        return run_dir

    calibration_start = pd.Timestamp(params.calibration_start)
    calibration_end = pd.Timestamp(params.calibration_end)
    validation_start = pd.Timestamp(params.validation_start)
    validation_end = pd.Timestamp(params.validation_end)
    initialization_start = calibration_start - pd.Timedelta(days=params.spinup_days)
    simulation_end = validation_end
    dataset = load_lumped_dataset(catalog, initialization_start, simulation_end)
    dataset.require_observations(calibration_start, calibration_end, "Calibration")
    if params.mode == "all":
        dataset.require_observations(validation_start, validation_end, "Validation")

    if params.mode == "all":
        setup = HBV92SpotpySetup(
            dataset,
            calibration_start,
            calibration_end,
            routing=params.routing,
            routing_length=params.routing_length,
        )
        print(
            f"[率定] SCE-UA | evaluations={params.repetitions} | "
            f"parameters={len(setup.parameter_names)} | objective=1-NSE",
            flush=True,
        )
        calibration_result = calibrate_sceua(
            setup,
            params.repetitions,
            seed=params.seed,
            ngs=params.ngs,
        )
        parameter_payload = build_parameter_payload(params, catalog, calibration_result)
        write_yaml(run_dir / "calibration" / "best_parameters.yaml", parameter_payload)
        write_csv(
            run_dir / "calibration" / "calibration_history.csv",
            calibration_result.history,
        )
        parameter_values = calibration_result.parameters
        routing = params.routing
        routing_length = params.routing_length
        print(
            f"[率定完成] evaluations={calibration_result.evaluations} | "
            f"objective={calibration_result.objective:.6f}",
            flush=True,
        )
    hbv_result, generated_q, simulated_q, routing_diagnostics = simulate(
        dataset, parameter_values, routing, routing_length
    )
    target_mask = dataset.dates >= calibration_start
    diagnostic_frame = daily_frame(
        dataset, hbv_result, generated_q, simulated_q, target_mask
    )
    write_csv(run_dir / "diagnostics" / "daily_fluxes_states.csv", diagnostic_frame)
    water_balance = diagnostic_frame["water_balance_mm"].to_numpy()
    write_json(
        run_dir / "diagnostics" / "water_balance.json",
        {
            "maximum_absolute_daily_residual_mm": float(np.max(np.abs(water_balance))),
            "mean_daily_residual_mm": float(np.mean(water_balance)),
            "cumulative_residual_mm": float(np.sum(water_balance)),
            "routing": routing,
            **routing_diagnostics,
        },
    )

    rows = []
    calibration_mask = dataset.period_mask(calibration_start, calibration_end)
    calibration_frame = daily_frame(
        dataset, hbv_result, generated_q, simulated_q, calibration_mask
    )
    if params.mode == "all":
        rows.append(save_period(
            run_dir,
            "calibration",
            "Calibration",
            calibration_frame,
            params.model_label,
            catalog.outlet_name,
            generate_figures=params.generate_figures,
        ))

    comparison_frame = calibration_frame
    if params.mode == "all":
        validation_mask = dataset.period_mask(validation_start, validation_end)
        validation_frame = daily_frame(
            dataset, hbv_result, generated_q, simulated_q, validation_mask
        )
        rows.append(save_period(
            run_dir,
            "validation",
            "Validation",
            validation_frame,
            params.model_label,
            catalog.outlet_name,
            generate_figures=params.generate_figures,
        ))
        comparison_frame = pd.concat(
            [calibration_frame, validation_frame], ignore_index=True
        )
    write_csv(run_dir / "period_metrics_comparison.csv", pd.DataFrame(rows))

    if params.generate_figures:
        metrics_text = "\n".join(
            f"{row['Period']}: NSE={row['NSE']:.3f}, KGE={row['KGE']:.3f}, "
            f"RMSE={row['RMSE']:.3f} m³/s, PBIAS={row['PBIAS']:.2f}%"
            for row in rows
        )
        plot_flow_duration(
            comparison_frame,
            run_dir / "figures" / "flow_duration_curve.jpg",
            metrics_text,
        )
        plot_discharge_scatter(
            comparison_frame,
            run_dir / "figures" / "discharge_scatter.jpg",
            metrics_text,
        )
    for row in rows:
        print(
            f"[{row['Period']}] NSE={row['NSE']:.3f} | KGE={row['KGE']:.3f} | "
            f"RMSE={row['RMSE']:.3f} m3/s",
            flush=True,
        )
    return run_dir
