"""Command-line entry point for lumped deep-learning rainfall-runoff models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dlcore.experiments.experiment_config import resolve_experiment_parameters
from dlcore.experiments.experiment_workflow import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basin-mean P/T/PET -> deep-learning model -> same-day outlet discharge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    experiment = parser.add_argument_group("1. experiment")
    experiment.add_argument("--mode", choices=["prepare", "all"], default="all")
    experiment.add_argument("--basin_name", default="pascagoula")
    # 可选择的深度学习模型包括：LSTM、BiLSTM、GRU、BiGRU、RNN、TCN、
    # Informer、Reformer、Transformer、ETSformer、iTransformer、
    # Pyraformer、Crossformer、CARDformer、DLinear。
    experiment.add_argument("--model", default="TCN")
    experiment.add_argument("--seed", type=int, default=42)

    periods = parser.add_argument_group("2. data periods")
    periods.add_argument("--train_start", default="2010-01-01")
    periods.add_argument("--train_end", default="2016-12-31")
    periods.add_argument("--validation_start", default="2017-01-01")
    periods.add_argument("--validation_end", default="2017-12-31")
    periods.add_argument("--test_start", default="2018-01-01")
    periods.add_argument("--test_end", default="2019-12-31")

    input_window = parser.add_argument_group("3. input window")
    input_window.add_argument("--lookback", type=int, default=365, help="Number of meteorological days ending on the runoff target date")

    optimization = parser.add_argument_group("4. training and optimization")
    optimization.add_argument("--max_epochs", type=int, default=50)
    optimization.add_argument("--batch_size", type=int, default=64)
    optimization.add_argument("--learning_rate", type=float, default=1e-4)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--dropout", type=float, default=0.1)
    optimization.add_argument("--early_stopping_patience", type=int, default=10)

    runtime = parser.add_argument_group("5. runtime")
    runtime.add_argument("--num_workers", type=int, default=0)
    runtime.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    output = parser.add_argument_group("6. output")
    output.add_argument("--generate_figures", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--clip_negative_discharge", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv=None):
    params = build_parser().parse_args(argv)
    params.end_dates_inclusive = True
    params = resolve_experiment_parameters(params, __file__)
    run_directory = run_experiment(params)
    print(f"Results: {run_directory}")
    return run_directory


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
