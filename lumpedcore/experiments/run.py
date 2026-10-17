"""Command-line entry point for the daily lumped HBV92 experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lumpedcore.experiments.experiment_config import resolve_experiment_parameters
from lumpedcore.experiments.experiment_workflow import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basin-mean forcing -> lumped HBV92 -> outlet discharge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experiment = parser.add_argument_group("1. experiment")
    experiment.add_argument(
        "--mode",
        choices=["prepare", "all"],
        default="all",
    )
    experiment.add_argument("--basin_name", default="pascagoula")
    experiment.add_argument("--seed", type=int, default=42)

    dates = parser.add_argument_group("2. daily periods")
    dates.add_argument("--calibration_start", default="2010-01-01")
    dates.add_argument("--calibration_end", default="2017-12-31")
    dates.add_argument("--validation_start", default="2018-01-01")
    dates.add_argument("--validation_end", default="2019-12-31")
    dates.add_argument("--spinup_days", type=int, default=365)

    calibration = parser.add_argument_group("3. calibration")
    calibration.add_argument("--algorithm", choices=["sceua"], default="sceua")
    calibration.add_argument("--objective", choices=["nse"], default="nse")
    calibration.add_argument("--repetitions", type=int, default=1000)
    calibration.add_argument("--ngs", type=int, default=5)

    model = parser.add_argument_group("4. model")
    model.add_argument("--routing", choices=["none", "gamma"], default="gamma")
    model.add_argument("--routing_length", type=int, default=15)

    output = parser.add_argument_group("5. output")
    output.add_argument(
        "--generate_figures",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    output.add_argument(
        "--force_prepare",
        action="store_true",
        help="Regenerate basin-mean forcing even when the source signature matches.",
    )
    return parser


def main(argv=None):
    params = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    params = resolve_experiment_parameters(params, __file__)
    return run_experiment(params)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
