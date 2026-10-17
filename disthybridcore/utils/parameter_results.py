"""Export HBV and D8 kinematic-routing parameters from one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import xarray as xr

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def effective_parameter_bounds(dataset, name):
    variable = dataset[name]
    return float(variable.attrs["effective_lower"]), float(variable.attrs["effective_upper"])


def normalized_parameter_position(dataset, name):
    lower, upper = effective_parameter_bounds(dataset, name)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(upper > lower, (dataset[name].values - lower) / (upper - lower), np.nan)


def add_grid_variable(dataset, name, values, mask, **attrs):
    array = np.asarray(values)
    if array.shape != mask.shape:
        raise ValueError(f"Grid field {name} has shape {array.shape}, expected {mask.shape}")
    dataset[name] = (("y", "x"), np.where(mask, array, np.nan))
    dataset[name].attrs.update(attrs)


def save_model_parameter_results(module, datamodule, directory, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = module.model
    topology = datamodule.routing_topology
    device = next(model.parameters()).device
    if datamodule.parameter_climate_sequence is None:
        raise ValueError("Parameter export requires the prepared training context")
    hbv_descriptions = {
        "beta": ("Soil", "Recharge exponent", "1"),
        "fc": ("Soil", "Soil storage capacity", "mm"),
        "k0": ("Response", "Fast runoff coefficient", "day-1"),
        "k1": ("Response", "Upper-zone slow runoff coefficient", "day-1"),
        "k2": ("Response", "Lower-zone baseflow coefficient", "day-1"),
        "lp": ("Evaporation", "Evaporation limitation fraction", "1"),
        "perc": ("Response", "Maximum percolation flux", "mm day-1"),
        "uzl": ("Response", "Fast runoff threshold", "mm"),
        "tt": ("Rain/snow", "Rain/snow temperature threshold", "degC"),
        "cfmax": ("Snow", "Degree-day melt factor", "mm day-1 degC-1"),
        "cfr": ("Snow", "Refreezing factor", "1"),
        "cwh": ("Snow", "Snow liquid-water capacity fraction", "1"),
    }
    modes = [(child, child.training) for child in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            fields = model.runoff_generator.generate_parameters(
                datamodule.terrain[None].to(device),
                datamodule.parameter_climate_sequence[None].to(device),
                topology.basin_mask[None].to(device),
            )
            fields = {name: value[0, 0].cpu().numpy() for name, value in fields.items()}
            a_active, b_active = model.grid_router.parameter_fields()
            a_active, b_active = a_active.cpu(), b_active.cpu()
    finally:
        for child, training in modes:
            child.training = training

    x, y = topology.x.cpu().numpy(), topology.y.cpu().numpy()
    mask = topology.basin_mask.cpu().numpy().astype(bool)
    dataset = xr.Dataset(coords={"x": x, "y": y})
    dataset.x.attrs.update(units="km", standard_name="projection_x_coordinate")
    dataset.y.attrs.update(units="km", standard_name="projection_y_coordinate")
    dataset["basin_mask"] = (("y", "x"), mask.astype(np.int8))
    for name, values in fields.items():
        bound = next(item for item in model.runoff_generator.parameterization.parameter_bounds if item[0] == name)
        family, description, unit = hbv_descriptions[name]
        add_grid_variable(
            dataset, name, values, mask, scope="grid", family=family,
            long_name=description, description_cn=description, units=unit,
            configured_lower=float(bound[1]), configured_upper=float(bound[2]),
            effective_lower=float(bound[1]), effective_upper=float(bound[2]),
        )
    for name, tensor, unit, long_name in (
        ("cell_area_km2", datamodule.cell_area_km2, "km2", "Effective within-basin cell area"),
        ("elevation_m", topology.original_dem, "m", "Original terrain elevation"),
        ("slope", topology.slope, "m m-1", "D8 edge slope"),
        ("flow_accumulation_km2", topology.flow_accumulation, "km2", "D8 upstream area"),
        ("flow_distance_km", topology.flow_distance_km, "km", "D8 downstream edge length"),
        ("channel_mask", topology.channel_mask, "1", "Preprocessed channel mask"),
    ):
        add_grid_variable(dataset, name, tensor.cpu().numpy(), mask, units=unit, long_name=long_name)

    active = model.grid_router.active_global_index.cpu()
    a_map = torch.full((mask.size,), float("nan"))
    b_map = a_map.clone()
    a_map[active] = a_active
    b_map[active] = b_active
    a_map = a_map.reshape(mask.shape).numpy()
    b_map = b_map.reshape(mask.shape).numpy()
    add_grid_variable(
        dataset, "kinematic_a", a_map, mask, scope="grid", family="Routing",
        long_name="Kinematic storage coefficient a", description_cn="运动波蓄量系数a",
        units="(s m-1) (m3 s-1)^(1-b)", effective_lower=model.grid_router.a_min,
        effective_upper=model.grid_router.a_max,
    )
    add_grid_variable(
        dataset, "kinematic_b", b_map, mask, scope="grid", family="Routing",
        long_name="Kinematic storage exponent b", description_cn="运动波蓄量指数b",
        units="1", effective_lower=model.grid_router.b_min,
        effective_upper=model.grid_router.b_max,
    )
    _, _, hillslope_a, channel_a, hillslope_b, channel_b = (
        model.grid_router.parameter_fields(include_classes=True)
    )
    for name, values, unit, long_name, lower, upper in (
        ("kinematic_hillslope_a", hillslope_a, "routing units", "Hillslope storage coefficient a", model.grid_router.a_min, model.grid_router.a_max),
        ("kinematic_channel_a", channel_a, "routing units", "Channel storage coefficient a", model.grid_router.a_min, model.grid_router.a_max),
        ("kinematic_hillslope_b", hillslope_b, "1", "Hillslope storage exponent b", model.grid_router.b_min, model.grid_router.b_max),
        ("kinematic_channel_b", channel_b, "1", "Channel storage exponent b", model.grid_router.b_min, model.grid_router.b_max),
    ):
        class_map = torch.full((mask.size,), float("nan"))
        class_map[active] = values.detach().cpu()
        add_grid_variable(
            dataset, name, class_map.reshape(mask.shape).numpy(), mask,
            scope="grid", family="Routing", units=unit, long_name=long_name,
            effective_lower=lower, effective_upper=upper,
        )
    routing_names = [
        "kinematic_a", "kinematic_b", "kinematic_hillslope_a",
        "kinematic_channel_a", "kinematic_hillslope_b", "kinematic_channel_b",
    ]

    for channel, values in enumerate(datamodule.terrain.cpu().numpy()):
        add_grid_variable(dataset, f"static_feature_{channel}_z", values, mask,
                           units="1", long_name="Standardized static network input")
    climate = datamodule.parameter_climate_sequence
    for channel, variable in enumerate(("prec", "t", "pet")):
        add_grid_variable(dataset, f"climate_{variable}_mean_z",
                           climate[:, channel].mean(dim=0).cpu().numpy(), mask,
                           units="1", long_name="Mean standardized parameter-encoder climate input")
        add_grid_variable(dataset, f"climate_{variable}_std_z",
                           climate[:, channel].std(dim=0, correction=0).cpu().numpy(), mask,
                           units="1", long_name="Std. dev. of standardized parameter-encoder climate input")
    dataset.attrs.update(
        schema="physical_parameter_results_hbv92_v1", temporal_resolution="daily",
        routing="d8_kinematic_hillslope_channel",
        routing_time_step_days=1.0 / model.grid_router.routing_substeps_per_day,
        routing_step_hours=24.0 / model.grid_router.routing_substeps_per_day,
        runoff_model="single_component_hbv92",
        parameter_encoder=model.runoff_generator.parameter_encoder,
        parameter_encoder_configuration=json.dumps(
            model.runoff_generator.parameterization.checkpoint_configuration()
        ),
        routing_input="q0_plus_q1_plus_q2",
        hbv_parameters=json.dumps([item[0] for item in model.runoff_generator.parameterization.parameter_bounds]),
        routing_parameters=json.dumps(routing_names),
        evaluation_metadata=json.dumps(metadata or {}, ensure_ascii=False),
        outlet_row=int(topology.outlet_row), outlet_col=int(topology.outlet_col),
    )
    target = directory / "model_parameters.nc"
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, value in dataset.data_vars.items() if value.ndim == 2
    }
    dataset.to_netcdf(target, engine="h5netcdf", encoding=encoding)
    dataset.close()
    print(f"[Parameter export] 12 HBV92 fields + hillslope/channel D8 fields: {target}")
    return target


def export_checkpoint_parameters(seed_directory, checkpoint=None, generate_figures=True):
    import yaml
    from disthybridcore.experments.run import build_parser
    from disthybridcore.experments.experiment_config import resolve_experiment_parameters, internal_defaults
    from disthybridcore.data_pipeline.data_catalog import prepare_data_catalog
    from disthybridcore.data_pipeline.datamodule import DailyHydroDataModule
    from disthybridcore.hydromodels.training_module import DistributedHBVTrainingModule
    from disthybridcore.utils.parameter_analysis import analyze_parameter_results

    seed_directory = Path(seed_directory).resolve()
    with (seed_directory / "train" / "parameters.yaml").open(encoding="utf-8") as stream:
        saved = yaml.safe_load(stream)
    source = str(Path(__file__).resolve().parents[1] / "experments" / "run.py")
    params = build_parser().parse_args([])
    for name in vars(params):
        if name in saved:
            setattr(params, name, saved[name])
    params.end_dates_inclusive = saved["end_dates_inclusive"]
    defaults = internal_defaults(Path(source).resolve().parents[2])
    overrides = {name: saved[name] for name in defaults if name in saved and name != "results_dir"}
    params = resolve_experiment_parameters(params, source, internal_overrides=overrides)
    prepare_data_catalog(Path(params.project_root), params)
    checkpoint = Path(checkpoint) if checkpoint else seed_directory / "train" / "checkpoints" / "last.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    datamodule = DailyHydroDataModule(params)
    try:
        datamodule.setup_static(); datamodule.prepare_training_background()
        module = DistributedHBVTrainingModule(
            params, parameter_context=datamodule.parameter_context_metadata,
            routing_topology=datamodule.routing_topology,
            grid_gauges=datamodule.grid_gauges,
        )
        module.restore_checkpoint(payload)
        device = "cuda" if params.accelerator != "cpu" and torch.cuda.is_available() else "cpu"
        module.to(device)
        metadata = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_epoch": int(payload["epoch"]) + 1,
            "checkpoint_global_step": int(payload["global_step"]),
            "parameter_context": datamodule.parameter_context_metadata,
        }
        target = save_model_parameter_results(
            module, datamodule, seed_directory / "train" / "parameter_analysis", metadata
        )
        return analyze_parameter_results(target, generate_figures=generate_figures)
    finally:
        datamodule.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export and analyze checkpoint parameters")
    parser.add_argument("--run_dir", "--seed_dir", dest="run_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    return export_checkpoint_parameters(args.run_dir, args.checkpoint, args.figures)


if __name__ == "__main__":
    main()
