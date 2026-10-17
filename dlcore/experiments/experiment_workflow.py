"""End-to-end training, validation, and testing for all supported models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import pandas as pd
import torch

from dlcore.data_pipeline.datamodule import DLDataModule
from dlcore.data_pipeline.window_dataset import fit_scaler
from dlcore.hydromodels.model_config import ModelConfig
from dlcore.hydromodels.model_factory import count_trainable_parameters
from dlcore.hydromodels.training_module import RunoffTrainingModule
from dlcore.prepare_data.basin_mean_validation import validate_basin_mean_inputs
from dlcore.utils.hydrology_metrics import hydrology_metrics
from dlcore.utils.reproducibility import seed_everything
from dlcore.utils.result_io import write_csv, write_json, write_yaml
from dlcore.utils.result_plots import plot_discharge_results


def serializable_parameters(params) -> dict:
    payload = {}
    for name, value in vars(params).items():
        payload[name] = str(value) if isinstance(value, Path) else value
    return payload


def create_run_directory(params) -> Path:
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = Path(params.results_dir) / f"{params.basin}_{params.model}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def build_model_config(params, model_name: str) -> ModelConfig:
    return ModelConfig(
        model_name=model_name,
        lookback=params.lookback,
        dropout=params.dropout,
        learning_rate=float(params.learning_rate),
        weight_decay=float(params.weight_decay),
    )


def load_checkpoint_module(checkpoint_path: str | Path) -> RunoffTrainingModule:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if "model_config" not in hyperparameters:
        raise ValueError("Checkpoint does not contain dlcore model_config")
    module = RunoffTrainingModule(hyperparameters["model_config"])
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module


def inference_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def predict_dataset(module, loader, series, scaler, params) -> pd.DataFrame:
    device = inference_device(params.device)
    module = module.to(device)
    module.eval()
    predictions = []
    observations = []
    target_indices = []
    with torch.inference_mode():
        for batch in loader:
            forcing = batch["forcing"].to(device)
            time_features = batch["time_features"].to(device)
            normalized = module(forcing, time_features)
            raw = normalized * scaler.discharge_std + scaler.discharge_mean
            predictions.append(raw.squeeze(-1).cpu().numpy())
            observations.append(batch["observed_discharge"].numpy())
            target_indices.append(batch["target_index"].numpy())
    simulation = np.concatenate(predictions, axis=0).astype(np.float64)
    observation = np.concatenate(observations, axis=0).astype(np.float64)
    indices = np.concatenate(target_indices, axis=0).astype(np.int64)
    if simulation.ndim != 1:
        raise RuntimeError("Model must produce one discharge value per sample")
    if observation.shape != simulation.shape or indices.shape != simulation.shape:
        raise RuntimeError("Prediction, observation, and target-index shapes differ")
    negative_before_clipping = simulation < 0
    if params.clip_negative_discharge:
        simulation = np.maximum(simulation, 0.0)
    frame = pd.DataFrame(
        {
            "date": series.dates[indices],
            "observed_discharge_m3_s": observation,
            "simulated_discharge_m3_s": simulation,
            "negative_prediction_before_clipping": negative_before_clipping,
        }
    )
    return frame


def save_period_results(
    module,
    loader,
    series,
    scaler,
    params,
    model_directory: Path,
    period: str,
    model_name: str,
) -> dict:
    frame = predict_dataset(module, loader, series, scaler, params)
    period_directory = model_directory / period
    write_csv(period_directory / "daily_discharge.csv", frame)
    metrics = hydrology_metrics(
        frame["simulated_discharge_m3_s"],
        frame["observed_discharge_m3_s"],
    )
    metrics.update(
        {
            "Model": model_name,
            "Period": period,
            "NegativePredictionRatioBeforeClipping": float(
                frame["negative_prediction_before_clipping"].mean()
            ),
            "NegativeDischargeClipped": bool(params.clip_negative_discharge),
        }
    )
    if params.generate_figures:
        plot_discharge_results(
            frame,
            model_directory / "figures" / f"{period}_discharge.png",
            f"{params.basin.upper()} {model_name} {period} | NSE = {metrics['NSE']:.3f}",
        )
    write_csv(period_directory / "metrics.csv", metrics_summary_table([metrics]))
    write_json(period_directory / "metrics.json", metrics)
    return metrics


def internal_training_settings() -> dict:
    return {
        "gradient_clip": 1.0,
        "devices": 1,
        "precision": "32-true",
        "progress_bar": True,
        "pin_memory": "auto_cuda",
    }


def trainer_for_model(params, model_directory: Path):
    settings = internal_training_settings()
    accelerator = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[params.device]
    if params.device != "cpu" and torch.cuda.is_available():
        torch.set_float32_matmul_precision("highest")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_directory / "checkpoints",
        filename="best",
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=params.early_stopping_patience,
    )
    logger = CSVLogger(
        save_dir=str(model_directory), name="training_history", version=""
    )
    trainer = L.Trainer(
        default_root_dir=str(model_directory),
        max_epochs=params.max_epochs,
        accelerator=accelerator,
        devices=settings["devices"],
        precision=settings["precision"],
        gradient_clip_val=settings["gradient_clip"],
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        deterministic=True,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_progress_bar=settings["progress_bar"],
    )
    return trainer, checkpoint_callback, logger


def train_model(params, model_name: str, data_module, model_directory: Path):
    config = build_model_config(params, model_name)
    module = RunoffTrainingModule(config)
    trainer, checkpoint_callback, logger = trainer_for_model(params, model_directory)
    started = time.perf_counter()
    trainer.fit(module, datamodule=data_module)
    elapsed = time.perf_counter() - started
    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError(f"{model_name} training did not produce best.ckpt")
    best_module = load_checkpoint_module(best_path)
    history_path = Path(logger.log_dir) / "metrics.csv"
    metadata = {
        "model": model_name,
        "model_config": config.to_dict(),
        "architecture_parameters": dict(
            getattr(best_module.model, "architecture_parameters", {})
        ),
        "trainable_parameters": count_trainable_parameters(best_module.model),
        "training_seconds": float(elapsed),
        "best_checkpoint": str(Path(best_path).resolve()),
        "last_checkpoint": str((model_directory / "checkpoints" / "last.ckpt").resolve()),
        "training_history": str(history_path.resolve()),
        "best_validation_loss": float(checkpoint_callback.best_model_score),
    }
    write_json(model_directory / "model_metadata.json", metadata)
    return best_module, metadata


def evaluate_model_periods(
    module,
    params,
    data_module,
    series,
    scaler,
    model_directory: Path,
    model_name: str,
    periods: tuple[str, ...],
) -> list[dict]:
    loaders = {
        "train": data_module.train_evaluation_dataloader,
        "validation": data_module.val_dataloader,
        "test": data_module.test_dataloader,
    }
    metrics = []
    for period in periods:
        metrics.append(
            save_period_results(
                module,
                loaders[period](),
                series,
                scaler,
                params,
                model_directory,
                period,
                model_name,
            )
        )
    return metrics


def print_nse_summary(metrics: list[dict]) -> None:
    print("\nFinal NSE metrics:")
    for period_metrics in metrics:
        nse = float(period_metrics["NSE"])
        nse_text = f"{nse:.6f}" if np.isfinite(nse) else "nan"
        print(f"  {period_metrics['Period']:<10} NSE = {nse_text}")


def metrics_summary_table(metrics: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics)
    preferred_columns = [
        "Model",
        "Period",
        "NSE",
        "KGE",
        "RMSE",
        "MAE",
        "Bias",
        "PBIAS",
        "Correlation",
        "Days",
    ]
    remaining_columns = [
        column for column in frame.columns if column not in preferred_columns
    ]
    return frame[preferred_columns + remaining_columns]


def run_single_model(params, model_name, data_module, series, scaler, run_directory):
    model_directory = run_directory
    module, metadata = train_model(params, model_name, data_module, model_directory)
    periods = ("train", "validation", "test")
    metrics = evaluate_model_periods(
        module,
        params,
        data_module,
        series,
        scaler,
        model_directory,
        model_name,
        periods,
    )
    return metadata, metrics


def run_experiment(params) -> Path:
    seed_everything(params.seed, deterministic=True)
    run_directory = create_run_directory(params)
    run_config = serializable_parameters(params)
    run_config["resolved_model_config"] = build_model_config(
        params, params.model
    ).to_dict()
    run_config["internal_training_settings"] = internal_training_settings()
    write_yaml(run_directory / "run_config.yaml", run_config)
    catalog, series, contract = validate_basin_mean_inputs(
        params.project_root, params.basin, params
    )
    scaler = fit_scaler(series, params.train_start, params.train_end)
    write_json(run_directory / "preprocessing" / "data_contract.json", contract)
    write_json(run_directory / "preprocessing" / "scaler.json", scaler.to_dict())
    if params.mode == "prepare":
        return run_directory
    data_module = DLDataModule(series, scaler, params)
    data_module.setup()
    seed_everything(params.seed, deterministic=True)
    metadata, metrics = run_single_model(
        params, params.model, data_module, series, scaler, run_directory
    )
    print_nse_summary(metrics)
    write_csv(
        run_directory / "period_metrics_comparison.csv",
        metrics_summary_table(metrics),
    )
    write_json(
        run_directory / "completion.json",
        {"metadata": metadata, "metrics": metrics},
    )
    return run_directory
