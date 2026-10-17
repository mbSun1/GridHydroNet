"""Train/test workflow for the differentiable grid-to-river model."""

import math
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import torch
import yaml
import pandas as pd
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from disthybridcore.data_pipeline.datamodule import DailyHydroDataModule
from disthybridcore.experments.experiment_config import validate_experiment_parameters
from disthybridcore.hydromodels.training_module import DistributedHBVTrainingModule
from disthybridcore.utils.period_evaluation import evaluate_period, save_period_diagnostics
from disthybridcore.utils.parameter_results import save_model_parameter_results
from disthybridcore.utils.parameter_analysis import analyze_parameter_results
from disthybridcore.utils.station_observation_analysis import (
    run_station_observation_analysis,
)
from disthybridcore.prepare_data.basin_workspace import (
    create_basin_workspace, check_required_inputs, write_json,
)
from disthybridcore.data_pipeline.data_catalog import prepare_data_catalog


def create_run_directory(params) -> Path:
    """按流域和启动分钟创建输出目录，同分钟重复运行时添加序号。"""
    root = Path(params.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"{params.basin}_{datetime.now():%Y_%m_%d_%H_%M}"
    for number in range(1000):
        path = root / (prefix if number == 0 else f"{prefix}_{number+1:02d}")
        try:
            path.mkdir()
            params.run_directory = str(path.resolve())
            return path
        except FileExistsError:
            continue
    raise FileExistsError("本分钟运行目录太多，请稍后再运行")


def adopt_checkpoint_configuration(params):
    """Restore the model/data contract saved beside a standalone-test checkpoint."""
    checkpoint = Path(params.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(params.project_root) / checkpoint
    checkpoint = checkpoint.resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "train" / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    run_parameters_path = checkpoint.parent.parent / "run_parameters.json"
    if not run_parameters_path.is_file():
        raise FileNotFoundError(
            "Standalone test requires train/run_parameters.json beside the checkpoint: "
            f"{run_parameters_path}"
        )
    saved = json.loads(run_parameters_path.read_text(encoding="utf-8"))
    locked_fields = (
        "basin_name", "meteo_source_root", "dem_grid_size_km", "down_factor",
        "outlet_id", "channel_area_threshold_km2",
        "gauge_snap_max_distance_km", "gauge_area_relative_tolerance",
        "temporal_resolution", "train_start", "train_end", "test_start",
        "test_end", "spinup_days", "parameter_encoder", "train_window_days",
        "test_window_days", "d8_substeps_per_day", "hbv_parameter_hidden",
        "hbv_parameter_layers", "hbv_parameter_cell_batch_size",
        "hbv_parameter_context_block_days", "weight_decay", "gradient_clip_val",
        "batch_size",
    )
    missing = [name for name in locked_fields if name not in saved]
    if missing:
        raise ValueError(
            "Saved checkpoint configuration is incomplete: " + ", ".join(missing)
        )

    explicit_fields = getattr(params, "_explicit_cli_fields", set())
    comparable_fields = (
        "basin_name", "meteo_source_root", "dem_grid_size_km", "down_factor",
        "outlet_id", "channel_area_threshold_km2",
        "gauge_snap_max_distance_km", "gauge_area_relative_tolerance",
        "temporal_resolution", "train_start", "train_end", "test_start",
        "test_end", "spinup_days", "parameter_encoder", "train_window_days",
        "test_window_days", "d8_substeps_per_day",
    )
    mismatches = []
    for name in comparable_fields:
        if name not in explicit_fields or name not in saved:
            continue
        current = getattr(params, name)
        expected = saved[name]
        if name == "meteo_source_root":
            current = str(Path(current).expanduser().resolve()).casefold()
            expected = str(Path(expected).expanduser().resolve()).casefold()
        elif name == "outlet_id":
            current = None if current is None else int(current)
            expected = None if expected is None else int(expected)
        elif name in {"train_start", "train_end", "test_start", "test_end"}:
            current = pd.Timestamp(current).strftime("%Y-%m-%d")
            expected = pd.Timestamp(expected).strftime("%Y-%m-%d")
        if current != expected:
            mismatches.append(f"{name}: 命令行={current!r}, checkpoint={expected!r}")
    if mismatches:
        raise ValueError(
            "测试参数与 checkpoint 的训练配置不一致，已停止测试。\n"
            + "\n".join(f"- {item}" for item in mismatches)
            + "\n请重新训练匹配当前参数的模型，或删除这些命令行参数，"
            "让测试自动采用 checkpoint 保存的配置。"
        )
    for name in locked_fields:
        setattr(params, name, saved[name])
    params.checkpoint = checkpoint
    params.meteo_source_root = Path(params.meteo_source_root)
    params.end_dates_inclusive = False
    params.basin = saved.get("basin", params.basin_name.lower())
    encoder_label = {
        "bigru": "BiGRU",
        "tcn": "TCN",
        "patch_transformer": "PatchTransformer",
    }[params.parameter_encoder]
    params.model_label = f"Grid{encoder_label}-HBV92-DynamicGridRouting"
    params.grid_resolution_m = params.dem_grid_size_km * 1000.0
    validate_experiment_parameters(params)
    return params


def required_forcing_period(params):
    """Return the inclusive forcing range needed by training and testing."""
    start = min(
        pd.Timestamp(params.train_start) - pd.Timedelta(days=params.spinup_days),
        pd.Timestamp(params.test_start) - pd.Timedelta(days=params.spinup_days),
    )
    end_exclusive = max(
        pd.Timestamp(params.train_end),
        pd.Timestamp(params.test_end),
    )
    return start.normalize(), (end_exclusive - pd.Timedelta(days=1)).normalize()


def forcing_cache_covers(manifest, start, end) -> bool:
    """Return whether a prepared manifest covers the requested forcing range."""
    if not manifest:
        return False
    try:
        available_start = pd.Timestamp(manifest["forcing_start"])
        available_end = pd.Timestamp(manifest["forcing_end_inclusive"])
    except (KeyError, TypeError, ValueError):
        return False
    return available_start <= start and available_end >= end


def prepare_basin_inputs(params, preprocessing_dir):
    """准备/校验流域输入，并返回本次是否实际执行了预处理。"""
    from disthybridcore.prepare_data.dem_download import download_dem_tiles
    from disthybridcore.prepare_data.hydro_grid import prepare_hydro_grid
    from disthybridcore.prepare_data.terrain_hydrology import analyze_terrain
    from disthybridcore.prepare_data.terrain_static_attributes import prepare_terrain_static_attributes
    from disthybridcore.prepare_data.grid_routing import prepare_grid_gauge_cache
    from disthybridcore.prepare_data.daily_forcing import align_daily_forcing
    from disthybridcore.prepare_data.discharge_preparation import prepare_discharge_observations
    from disthybridcore.prepare_data.basin_validation import validate_basin_data
    from disthybridcore.utils.basin_preparation_plots import plot_basin_forcing_checks

    workspace = create_basin_workspace(Path(params.data_repository), params.basin)
    report_dir = preprocessing_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = workspace.manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    forcing_start, forcing_end = required_forcing_period(params)
    required_paths = [] if manifest is None else [
        manifest["dem"], manifest["topology"], manifest["discharge"],
        manifest.get("grid_gauges", ""), *manifest["forcing"].values(),
        manifest.get("static_attributes", ""),
    ]
    cache_matches = prepared_cache_matches(
        manifest, params.channel_area_threshold_km2
    )
    if manifest is not None and not cache_matches and params.mode != "prepare":
        print(
            "[预处理缓存] 清单、D8拓扑或测站栅格缓存不一致，"
            "将按当前 channel_area_threshold_km2 重新构建。",
            flush=True,
        )
    needs_preparation = (
        params.mode == "prepare" or manifest is None
        or manifest.get("schema") != 4
        or any(not Path(path).is_file() for path in required_paths)
        or manifest.get("grid_resolution_m") != params.grid_resolution_m
        or manifest.get("down_factor") != params.down_factor
        or manifest.get("channel_area_threshold_km2") != params.channel_area_threshold_km2
        or manifest.get("gauge_snap_max_distance_km") != params.gauge_snap_max_distance_km
        or manifest.get("gauge_area_relative_tolerance") != params.gauge_area_relative_tolerance
        or not forcing_cache_covers(manifest, forcing_start, forcing_end)
        or not cache_matches
        or (
            params.outlet_id is not None
            and str(params.outlet_id) != manifest.get("outlet_id")
        )
    )
    if needs_preparation:
        try:
            vectors, catalogs, discharge_sources = check_required_inputs(workspace, Path(params.meteo_source_root), params.outlet_id)
        except (FileNotFoundError, ValueError) as error:
            write_json(report_dir / "missing_inputs.json", {"basin": params.basin, "error": str(error), "no_download_or_training": True})
            raise
        # 粗网格的整格边界与down_factor补边可能超过固定3km缓冲。
        # 瓦片覆盖包含粗网格整格边界与down_factor补边空间。
        tiles = download_dem_tiles(workspace, vectors, buffer_m=3000.0 + 2*params.grid_resolution_m*params.down_factor)
        dem = prepare_hydro_grid(workspace, params.basin, vectors, tiles, params.grid_resolution_m, params.down_factor)
        topology = analyze_terrain(workspace, params.basin, dem, vectors, params.down_factor, params.channel_area_threshold_km2)
        static_attributes = prepare_terrain_static_attributes(workspace, params.basin, topology)
        grid_gauges = workspace.terrain / (
            f"basin_{params.basin}_grid_gauges_"
            f"channel{params.channel_area_threshold_km2:g}km2.pt"
        )
        prepare_grid_gauge_cache(
            topology, vectors["gauge_path"], vectors["outlet_id"],
            grid_gauges,
            params.gauge_snap_max_distance_km,
            params.gauge_area_relative_tolerance,
            observed_station_ids=list(discharge_sources),
        )
        # 地形图应在漫长强迫处理/训练之前可见。
        plot_prepared_terrain(
            params, workspace, dem, topology, vectors,
            preprocessing_dir / "figures", static_attributes, grid_gauges,
        )
        forcing = align_daily_forcing(
            workspace,
            params.basin,
            catalogs,
            dem,
            forcing_start,
            forcing_end,
        )
        discharge = prepare_discharge_observations(
            workspace, params.basin, vectors, discharge_sources
        )
        start, end = forcing_start, forcing_end
        import xarray as xr
        with xr.open_dataset(dem, engine="h5netcdf") as grid:
            grid_shape = [grid.sizes["y"], grid.sizes["x"]]
        manifest = {"schema": 4,
                    "basin": params.basin, "outlet_id": vectors["outlet_id"],
                    "dem": str(dem.resolve()), "topology": str(topology.resolve()),
                    "static_attributes": str(static_attributes.resolve()),
                    "grid_gauges": str(grid_gauges.resolve()),
                    "channel_area_threshold_km2": params.channel_area_threshold_km2,
                    "gauge_snap_max_distance_km": params.gauge_snap_max_distance_km,
                    "gauge_area_relative_tolerance": params.gauge_area_relative_tolerance,
                    "discharge": str(discharge.resolve()),
                    "forcing": {key: str(path.resolve()) for key, path in forcing.items()},
                    "forcing_start": str(start.date()), "forcing_end_inclusive": str(end.date()),
                    "grid_resolution_m": params.grid_resolution_m, "down_factor": params.down_factor,
                    "grid_crs": "EPSG:3035", "grid_shape": grid_shape,
                    "sources": {key: {"dataset": catalog.dataset, "variable": catalog.variable,
                                      "start": str(catalog.start.date()), "end_inclusive": str(catalog.end.date()),
                                      "directory": str(next(iter(catalog.files_by_day.values())).parent),
                                      "available_days": len(catalog.files_by_day), "native_crs": "EPSG:4326"}
                                for key, catalog in catalogs.items()},
                    "prepared_units": {"prec": "mm/day", "pet": "mm/day", "t": "degree_Celsius", "discharge": "m3/s"},
                    "forcing_days": (end-start).days+1,
                    "missing_method": "No forcing imputation; missing source dates/pixels raise an error",
                    "discharge_sources": {
                        station_id: str(path.resolve())
                        for station_id, path in discharge_sources.items()
                    },
                    "station_ids": [
                        station_id for station_id in discharge_sources
                    ],
                    "discharge_metric_policy": (
                        "All observed gauges (one or more); union daily calendar; "
                        "no imputation; "
                        "station-balanced physDL-HBV combined RMSE after "
                        "D8 upstream-area runoff-depth conversion"
                    ),
                    "boundary": str(vectors["boundary_path"]), "gauges": str(vectors["gauge_path"]),
                    "river_lines": str(vectors["river_path"]) if vectors["river_path"] else None}
        # 完整核验通过后以原子替换方式发布清单。
        validate_basin_data(manifest, report_dir)
        write_json(manifest_path, manifest)
        if params.generate_figures:
            plot_basin_forcing_checks(report_dir, preprocessing_dir / "figures")
    else:
        validate_basin_data(manifest, report_dir, inspect_values=False)
    write_json(report_dir / "preparation_manifest.json", manifest)
    prepare_data_catalog(Path(params.project_root), params)
    return needs_preparation


def prepared_cache_matches(manifest, channel_area_threshold_km2: float) -> bool:
    """Validate that the manifest and both preprocessing caches describe one graph.

    The topology signature, threshold and gauge order must match before the
    prepared files are used together.
    """
    if not manifest:
        return False
    topology_path = Path(manifest.get("topology", ""))
    gauge_path = Path(manifest.get("grid_gauges", ""))
    if not topology_path.is_file() or not gauge_path.is_file():
        return False
    try:
        import xarray as xr

        with xr.open_dataset(topology_path, engine="h5netcdf") as topology:
            actual_threshold = topology.attrs.get("channel_area_threshold_km2")
            topology_signature = topology.attrs.get("input_signature")
            if actual_threshold is None or topology_signature is None:
                return False
            if not math.isclose(
                float(actual_threshold), float(channel_area_threshold_km2),
                rel_tol=0.0, abs_tol=1e-9,
            ):
                return False
        payload = torch.load(gauge_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != 1:
            return False
        if tuple(str(value) for value in payload.get("gauge_ids", ())) != tuple(
            str(value) for value in manifest.get("station_ids", ())
        ):
            return False
        if payload.get("topology_input_signature") != str(topology_signature):
            return False
        return True
    except (OSError, ValueError, KeyError, RuntimeError, TypeError):
        return False


def plot_prepared_terrain(
    params, workspace, dem, topology_path, vectors, figure_dir,
    static_attributes=None, grid_gauge_path=None,
):
    """无需训练/观测读取器就可绘制地形；只使用真实SHP出口坐标。"""
    if not params.generate_figures:
        return
    from types import SimpleNamespace
    import numpy as np
    import xarray as xr
    from pyproj import Transformer
    from disthybridcore.data_pipeline.routing_topology import read_routing_topology
    from disthybridcore.data_pipeline.basin import RiverNetworkData
    from disthybridcore.utils.terrain_preparation_plots import plot_terrain_preparation_maps
    from disthybridcore.utils.terrain_preparation_plots import plot_terrain_static_attributes
    from disthybridcore.prepare_data.grid_routing import (
        load_grid_gauge_mapping, write_gauge_table,
    )

    if grid_gauge_path is None:
        raise ValueError("grid_gauge_path is required for preprocessing figures")
    gauge_mapping = load_grid_gauge_mapping(grid_gauge_path)

    with xr.open_dataset(dem, engine="h5netcdf") as data:
        local = SimpleNamespace(**vars(params))
        local.path_dem = local.path_catchments = str(dem)
        local.dem_variable, local.catchment_variable = "dem", "basin_fraction"
        local.grid_crs_wkt = data.attrs["crs_wkt"]
        local.grid_height, local.grid_width = data.sizes["y"], data.sizes["x"]
        local.grid_size = local.grid_width
        local.native_cell_size_km = data.attrs["grid_resolution_m"] / 1000
        local.path_basin_boundary = str(vectors["boundary_path"])
        local.path_river_lines = str(vectors["river_path"]) if vectors["river_path"] else None
    feature_by_id = {
        str(feature["properties"]["ID"]): feature
        for feature in vectors["gauges"]
    }
    observed_features = [feature_by_id[station_id] for station_id in gauge_mapping.gauge_ids]
    projected = np.asarray(
        [feature["geometry"]["coordinates"] for feature in observed_features],
        dtype=np.float64,
    )
    lon, lat = Transformer.from_crs(3035, 4326, always_xy=True).transform(
        projected[:, 0], projected[:, 1]
    )
    station_ids = list(gauge_mapping.gauge_ids)
    station_names = [
        str(feature["properties"].get("name") or station_id)
        for feature, station_id in zip(observed_features, station_ids)
    ]
    station_count = len(station_ids)
    river = RiverNetworkData(
        torch.zeros((station_count, station_count)),
        torch.zeros((station_count, station_count)),
        torch.zeros((station_count, station_count)),
        torch.arange(station_count, dtype=torch.long),
        station_ids,
        station_names,
        torch.tensor(np.column_stack((lon, lat)), dtype=torch.float64),
        torch.zeros(station_count, dtype=torch.bool),
        station_ids.index(str(vectors["outlet_id"])),
    )
    topology = read_routing_topology(topology_path)
    plot_terrain_preparation_maps(local, topology, figure_dir, river)
    if static_attributes is not None:
        plot_terrain_static_attributes(local, static_attributes, figure_dir)
    table = write_gauge_table(gauge_mapping, Path(figure_dir).parent)
    print(f"[预处理图件] {Path(figure_dir).resolve()} | 站点表={table.name}", flush=True)


def save_parameters(params, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        key: value
        if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        else str(value)
        for key, value in vars(params).items()
    }
    with (directory / "parameters.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(values, stream, allow_unicode=True, sort_keys=False)
    write_json(directory / "run_parameters.json", values)


def trainer_device(params) -> tuple[str, str]:
    accelerator = params.accelerator
    if accelerator == "auto":
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = params.precision if accelerator == "gpu" else "32-true"
    return accelerator, precision


def print_configuration(params, datamodule, stage: str) -> None:
    """打印一次模型/空间摘要和当前阶段的运行参数。"""
    model_shape = tuple(datamodule.routing_topology.basin_mask.shape)
    topology = datamodule.routing_topology
    show_model_summary = stage == "训练" or params.mode == "test"
    print(f"\n[运行] {params.model_label} | basin={params.basin} | {stage}")
    if show_model_summary:
        print(
            f"[模型] {params.parameter_encoder}参数 -> grid HBV92 -> Q0+Q1+Q2 -> "
            "dynamic full-grid routing -> observed gauges"
        )
        print(
            f"[空间] grid={model_shape[0]}x{model_shape[1]} | "
            f"active/channel={int(topology.basin_mask.sum()):,}/"
            f"{int(topology.channel_mask.sum()):,} | "
            f"gauges={len(datamodule.gauge_ids)} | "
            f"outlet=({topology.outlet_row},{topology.outlet_col}) | "
            f"max_snap={float(datamodule.grid_gauges.gauge_snap_distance_m.max()) / 1000.0:.2f} km"
        )
        print(
            f"[汇流] mode=d8_kinematic | "
            f"d8_substeps/day={params.d8_substeps_per_day}"
        )
    if stage == "训练":
        steps_per_epoch = math.ceil(len(datamodule.train_dataset) / params.batch_size)
        print(
            f"[训练] samples={len(datamodule.train_dataset)} | "
            f"continuous_window={params.train_window_days} d | "
            f"spinup={params.spinup_days} d | epochs={params.epochs} | "
            f"batch={params.batch_size} | lr={params.learning_rate:g} | "
            f"updates={steps_per_epoch * params.epochs}"
        )
        updates = steps_per_epoch * params.epochs
        if updates < 100:
            print(
                "[训练提示] 当前少于100次参数更新，只适合检查流程和数值稳定性；"
                "不能据此评价最终KGE/NSE。"
            )
    else:
        print(
            f"[测试] {params.test_start}.."
            f"{(pd.Timestamp(params.test_end) - pd.Timedelta(days=1)):%Y-%m-%d} | "
            f"windows={len(datamodule.test_dataset)} | 状态跨窗口继承"
        )


def resolve_checkpoint(params, train_dir: Path) -> Path:
    if params.mode == "test":
        checkpoint = Path(params.checkpoint)
    else:
        checkpoint = train_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"未找到测试检查点：{checkpoint}。请先运行 --mode train 或 --mode all。"
        )
    return checkpoint


def generate_static_figures(params, datamodule, directory: Path) -> None:
    if not params.generate_figures:
        return
    from disthybridcore.utils.terrain_preparation_plots import (
        plot_terrain_preparation_maps, plot_terrain_static_attributes,
    )
    from disthybridcore.prepare_data.grid_routing import write_gauge_table
    figure_dir = directory / "figures"
    plot_terrain_preparation_maps(
        params, datamodule.routing_topology, figure_dir, datamodule.river
    )
    plot_terrain_static_attributes(params, params.path_terrain_attributes, figure_dir)
    write_gauge_table(datamodule.grid_gauges, directory)
    print(f"[预处理产物] 图件={figure_dir.resolve()} | 站点表=gauge_to_grid.csv")


def run_experiment(params):
    # Lightning的INFO模板会在fit/test各打印一次；保留WARNING/ERROR，
    # 并由本工程自己报告一次实际设备。Windows上单进程DataLoader
    # 是当前明确配置，不重复输出Lightning的通用worker性能建议。
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The '.*_dataloader' does not have many workers.*",
        module=r"lightning\.pytorch\.trainer\.connectors\.data_connector",
    )
    if params.mode == "test" and params.checkpoint is None:
        raise ValueError("独立test模式需指定已有运行的 train/checkpoints/last.ckpt")
    if params.mode == "test":
        params = adopt_checkpoint_configuration(params)
    if params.mode == "test":
        seed_dir = Path(params.checkpoint).resolve().parents[2]
        params.run_directory = str(seed_dir)
    else:
        seed_dir = create_run_directory(params)
    train_dir = seed_dir / "train"
    test_dir = seed_dir / "test"
    preprocessing_dir = seed_dir / "preprocessing"
    print(f"[本次输出] {seed_dir}", flush=True)
    prepared_now = prepare_basin_inputs(params, preprocessing_dir)
    if params.mode == "prepare":
        # 完整数据准备已完成；不启动训练背景、优化器或训练回放。
        return seed_dir
    seed_everything(params.seed, workers=True)
    accelerator, precision = trainer_device(params)
    print(f"[设备] accelerator={accelerator} | precision={precision}")
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
    datamodule = DailyHydroDataModule(params)

    try:
        # Terrain/topology is prepared, validated and visualized before any
        # optimizer or training DataLoader is created.
        datamodule.setup_static()
        topology = datamodule.routing_topology
        params.outlet_station_name = datamodule.outlet_station_name
        params.observed_station_names = list(datamodule.gauge_names)
        params.observed_station_ids = list(datamodule.gauge_ids)
        save_parameters(params, preprocessing_dir)
        # 新准备流域在强迫处理前生成图件；已有缓存则在本次输出目录重绘。
        if not prepared_now:
            generate_static_figures(params, datamodule, preprocessing_dir)
        print(
            "[地形预处理完成] "
            f"模型网格={tuple(topology.basin_mask.shape)} | "
            f"流域面积={float(datamodule.cell_area_km2.sum()):,.1f} km2 | "
            f"河道网格={int(topology.channel_mask.sum()):,}/"
            f"{int(topology.basin_mask.sum()):,} | "
            f"Strahler最高等级={int(topology.stream_order.max())} | "
            f"出口=({topology.outlet_row}, {topology.outlet_col})"
        )
        # 独立测试准备训练窗口，用于同一检查点的无梯度回放。
        datamodule.setup("fit")
        module = DistributedHBVTrainingModule(
            params,
            result_dir=str(test_dir),
            parameter_context=datamodule.parameter_context_metadata,
            train_result_dir=str(train_dir),
            training_dataset=datamodule.train_dataset,
            routing_topology=datamodule.routing_topology,
            grid_gauges=datamodule.grid_gauges,
        )
        if params.mode in {"train", "all"}:
            save_parameters(params, train_dir)
            print_configuration(params, datamodule, "训练")
            checkpoint_callback = ModelCheckpoint(
                dirpath=train_dir / "checkpoints",
                save_last=True,
                save_top_k=0,
                enable_version_counter=False,
            )
            trainer = Trainer(
                accelerator=accelerator,
                devices=1,
                precision=precision,
                max_epochs=params.epochs,
                default_root_dir=str(train_dir),
                callbacks=[checkpoint_callback],
                logger=False,
                deterministic=params.deterministic,
                gradient_clip_val=params.gradient_clip_val,
                enable_checkpointing=True,
                enable_model_summary=False,
                enable_progress_bar=False,
                num_sanity_val_steps=0,
                limit_val_batches=0,
                log_every_n_steps=1,
            )
            trainer.fit(module, datamodule=datamodule)

        # 正式结果只使用预先规定训练轮数结束时的last.ckpt。
        # 测试指标不参与检查点或训练轮次选择。
        checkpoint = resolve_checkpoint(params, train_dir)
        print(f"[检查点] {checkpoint}")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        try:
            module.restore_checkpoint(checkpoint_payload)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                "检查点与当前单组件HBV92/模型结构不兼容。不能继续"
                "使用旧权重，请在新的运行目录中重新训练。"
            ) from error
        module.to("cuda" if accelerator == "gpu" else "cpu")
        module.evaluation_metadata = {
            "checkpoint": checkpoint.name,
            "checkpoint_epoch": int(checkpoint_payload["epoch"]) + 1,
            "checkpoint_global_step": int(checkpoint_payload["global_step"]),
            "loss_configuration": module.loss_configuration,
            "optimization_configuration": module.optimization_configuration,
            "parameter_context": datamodule.parameter_context_metadata,
            "fitted_state_initialization": checkpoint_payload[
                "training_state_initialization"
            ],
        }
        comparisons = []
        evaluations = []
        if params.mode in {"train", "all"}:
            label = "Training (continuous)"
            metrics = evaluate_period(
                module, datamodule.training_evaluation_dataset(), train_dir,
                metadata={
                    **module.evaluation_metadata, "period": label,
                    "train_window_days": params.train_window_days,
                    "scope": "continuous training target span",
                },
            )
            comparisons.append(metrics.assign(Period=label))
            evaluations.append((label, train_dir))
            model_metrics = metrics[metrics["Model"] == params.model_label]
            summary_name = "Mean" if len(datamodule.gauge_names) > 1 else datamodule.gauge_names[0]
            mean_row = model_metrics[model_metrics["Station"] == summary_name].iloc[0]
            outlet_row = model_metrics[
                model_metrics["Station"] == datamodule.outlet_station_name
            ].iloc[0]
            print(
                f"[固定权重回放] {label} | multi-station mean "
                f"KGE={mean_row['KGE']:.3f}, NSE={mean_row['NSE']:.3f}, "
                f"RMSE={mean_row['RMSE']:.3f} m3/s | outlet "
                f"KGE={outlet_row['KGE']:.3f}, NSE={outlet_row['NSE']:.3f}, "
                f"RMSE={outlet_row['RMSE']:.3f} m3/s"
            )

        if params.mode in {"test", "all"}:
            datamodule.setup("test")
            module.prepare_test_state(datamodule.test_dataset)
            save_parameters(params, test_dir)
            print_configuration(params, datamodule, "测试")
            test_trainer = Trainer(
                accelerator=accelerator,
                devices=1,
                precision=precision,
                default_root_dir=str(test_dir),
                logger=False,
                deterministic=params.deterministic,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
            )
            test_trainer.test(module, datamodule=datamodule)
            comparisons.append(pd.read_csv(test_dir / "metrics.csv").assign(Period="Test"))
            evaluations.append(("Test", test_dir))
        pd.concat(comparisons, ignore_index=True).to_csv(
            seed_dir / "period_metrics_comparison.csv", index=False, encoding="utf-8-sig"
        )
        save_period_diagnostics(
            seed_dir, evaluations, params.model_label,
            float(datamodule.cell_area_km2.sum()),
            primary_station=datamodule.outlet_station_name,
        )
        # 固定检查点的结果保存完成后，独立导出全部物理参数并调用文件驱动分析。
        # Standalone test writes these diagnostics under its own test directory.
        parameter_dir = train_dir if params.mode in {"train", "all"} else test_dir
        parameter_file = save_model_parameter_results(
            module, datamodule, parameter_dir / "parameter_analysis", module.evaluation_metadata
        )
        analyze_parameter_results(parameter_file, generate_figures=params.generate_figures)
        if params.mode == "all":
            run_station_observation_analysis(params, seed_dir)
    finally:
        datamodule.close()
    return seed_dir
