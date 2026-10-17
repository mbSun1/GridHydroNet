"""Command-line entry point for the distributed grid-routing experiment."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from disthybridcore.experments.experiment_config import resolve_experiment_parameters
from disthybridcore.experments.experiment_workflow import run_experiment


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI for daily grid HBV92 and dynamic grid routing."""
    parser = argparse.ArgumentParser(
        description=(
            "Neural parameterization -> grid HBV92 -> dynamic grid routing -> gauges"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 1. Experiment identity and lifecycle.
    experiment = parser.add_argument_group("1. experiment")
    experiment.add_argument("--mode", choices=["prepare", "train", "test", "all"], default="all", help="Prepare, train, test, or run all stages.")
    experiment.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file or run directory for standalone test mode.",
    )
    experiment.add_argument("--seed", type=int, default=64, help="Random seed.")

    # 2. Basin data and spatial discretization.
    data = parser.add_argument_group("2. basin data and grid")
    meteo_root = Path(
        r"D:\\分布式\\全球气象数据集整理_降雨_潜在蒸发_2m气温\\天尺度驱动数据"
    )
    data.add_argument("--basin_name", default="pascagoula", help="Basin workspace name under data_repository.")
    data.add_argument("--meteo_source_root", type=Path, default=meteo_root, help="Prepared daily MSWEP/GLEAM/ERA5 root.")
    data.add_argument("--dem_grid_size_km", type=float, default=2.0, help="Target DEM grid spacing (km).")
    data.add_argument("--down_factor", type=int, default=1, help="Additional integer terrain coarsening factor.")
    data.add_argument("--outlet_id", type=int, default=None, help="Outlet gauge ID; inferred from prepared basin inputs when omitted.")
    data.add_argument("--channel_area_threshold_km2", type=float, default=20.0, help="D8 upstream-area threshold (km^2).")
    data.add_argument("--gauge_snap_max_distance_km", type=float, default=4.0, help="Maximum gauge-to-D8-channel snap distance (km).")
    data.add_argument("--gauge_area_relative_tolerance", type=float, default=0.15, help="Preferred absolute relative mismatch between official and D8 drainage area.")

    # 3. Daily simulation period. CLI end dates include the stated final day.
    dates = parser.add_argument_group("3. daily period")
    dates.add_argument("--temporal_resolution", choices=["daily"], default="daily", help="HBV forcing and supervision resolution.")
    dates.add_argument("--train_start", default="2010-01-01")
    dates.add_argument("--train_end", default="2017-12-31")
    dates.add_argument("--test_start", default="2018-01-01")
    dates.add_argument("--test_end", default="2019-12-31")
    dates.add_argument("--spinup_days", type=int, default=365, help="Historical forcing days for window warm-up.")

    # 4. Window construction and optimizer-facing training settings.
    training = parser.add_argument_group("4. training and windows")
    training.add_argument("--epochs", type=int, default=5,help="Training epochs.")
    training.add_argument("--learning_rate", type=float, default=1e-3)
    training.add_argument("--parameter_encoder", choices=["bigru", "tcn", "patch_transformer"], default="bigru", help="Shared per-grid-cell encoder used to generate static HBV92 parameters.")
    training.add_argument("--train_window_days", type=int, default=30, help="Target days per training sample.")
    training.add_argument("--test_window_days", type=int, default=30, help="Target days per test sample.")

    # 5. D8 grid-routing configuration.
    routing = parser.add_argument_group("5. D8 kinematic routing")
    routing.add_argument("--d8_substeps_per_day", type=int, choices=[1, 2, 4, 8, 24, 48, 96], default=1, help="D8 kinematic-wave steps per daily HBV interval.")

    # 6. Execution backend.
    runtime = parser.add_argument_group("6. execution")
    runtime.add_argument("--accelerator", choices=["auto", "gpu", "cpu"], default="auto", help="Lightning execution accelerator.")
    return parser


def main(argv=None, internal_overrides=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    params = build_parser().parse_args(raw_args)
    params._explicit_cli_fields = {
        token[2:].split("=", 1)[0]
        for token in raw_args
        if token.startswith("--") and token != "--"
    }
    params = resolve_experiment_parameters(params, __file__, internal_overrides=internal_overrides)
    return run_experiment(params)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
