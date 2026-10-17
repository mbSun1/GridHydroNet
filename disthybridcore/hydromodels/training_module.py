"""Lightning interface for grid HBV and dynamic grid routing."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import pandas as pd
import torch
from lightning.pytorch import LightningModule

from .distributed_hydro_model import build_distributed_hbv_model
from .losses import HydroLoss, daily_flow_values
from disthybridcore.utils.state_initialization import (
    detach_state, warmup_to_date,
)


class DistributedHBVTrainingModule(LightningModule):
    def __init__(
        self, params, result_dir: str | None = None, *, parameter_context=None,
        train_result_dir: str | None = None, training_dataset=None,
        routing_topology, grid_gauges,
    ):
        super().__init__()
        self.checkpoint_model_schema = "grid_hbv92_neural_parameterizer_d8_hillslope_channel_v10"
        self.loss_configuration = {
            "objective": "station_balanced_physdl_hbv_combined_rmse_in_runoff_depth",
            "scale": "m3_s_to_mm_day_by_fixed_d8_upstream_area",
            "aggregation": (
                "mean_over_valid_dates_then_equal_mean_over_valid_stations"
            ),
            "raw_rmse_weight": 0.75,
            "log_sqrt_rmse_weight": 0.25,
            "flow_transform": "log10(sqrt(Q_mm_day + 1e-6) + 0.1)",
            "reference": "referenceCode/physDL_HBV RmseLossComb(alpha=0.25)",
            "raw_rmse_units": "mm/day",
            "metrics_only": [
                "kge", "nse", "rmse", "flood_event_diagnostics"
            ],
        }
        self.optimization_configuration = {
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "monitor": "train_loss",
            "mode": "min",
            "factor": 0.5,
            "patience": 1,
            "relative_threshold": 1.0e-3,
            "minimum_learning_rate_fraction": 0.05,
        }
        self.params = params
        self.training_dataset = training_dataset
        self.model = build_distributed_hbv_model(
            params, routing_topology, grid_gauges
        )
        self.loss_function = HydroLoss()
        self.result_dir = Path(result_dir) if result_dir else None
        self.train_result_dir = Path(train_result_dir) if train_result_dir else None
        self.parameter_context_metadata = deepcopy(parameter_context)
        self.evaluation_metadata = {}
        self.train_history = []
        self.gradient_diagnostic_history = []
        self.parameter_diagnostic_history = []
        self.gradient_diagnostic_sums = {}
        self.gradient_diagnostic_maxima = {}
        self.gradient_diagnostic_steps = 0
        self.gradient_clip_exceeded_steps = 0
        self.parameter_diagnostic_sums = {}
        self.parameter_diagnostic_steps = 0
        self.train_loss_sum = 0.0
        self.train_raw_rmse_sum = 0.0
        self.train_log_sqrt_rmse_sum = 0.0
        self.train_sample_count = 0
        self._test_predictions, self._test_observations = [], []
        self.test_training_targets, self._test_dates = [], []
        self._test_diagnostics, self.test_hydrology = [], []
        self.test_gauge_hydrology, self.test_gauge_budgets = [], []
        self._test_state = None
        self.initial_test_state = None
        self._train_state = None
        self._next_train_start = None
        self._training_warmup_record = None
        self._epoch_training_started = None

    def on_fit_start(self):
        diagnostics = self.model.grid_router.diagnostics()
        if self.train_result_dir is not None:
            self.train_result_dir.mkdir(parents=True, exist_ok=True)
            (self.train_result_dir / "routing_diagnostics_initial.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.print(
            f"[路由初值] mode={diagnostics['mode']} | "
            f"active_cells={diagnostics['active_cells']} | "
            f"substeps/day={diagnostics['routing_substeps_per_day']}"
        )

    def on_save_checkpoint(self, checkpoint):
        if not self.parameter_context_metadata:
            raise RuntimeError("Saving requires the parameter-context record")
        if self.training_dataset is None:
            raise RuntimeError("Saving requires the training dataset")
        checkpoint["parameter_context"] = deepcopy(self.parameter_context_metadata)
        checkpoint["loss_configuration"] = self.loss_configuration
        checkpoint["optimization_configuration"] = self.optimization_configuration
        checkpoint["model_schema"] = self.checkpoint_model_schema
        checkpoint["hbv_parameter_bounds"] = [
            list(bound)
            for bound in self.model.runoff_generator.parameterization.parameter_bounds
        ]
        checkpoint["parameter_encoder_configuration"] = (
            self.model.runoff_generator.parameterization.checkpoint_configuration()
        )
        checkpoint["grid_routing_parameterization"] = (
            self.model.grid_router.checkpoint_configuration()
        )
        checkpoint["training_state_initialization"] = {
            "method": "epoch_warmup_then_chronological_state_carry",
            "anchor": self.params.train_start,
            "warmup_days": self.params.spinup_days,
            "window_days": self.params.train_window_days,
            "batch_size": 1,
            "shuffle": False,
            "state_detach": "each_window",
            "history_gradients": False,
        }
        checkpoint["training_diagnostics"] = {
            "gradient_file": "training_gradient_diagnostics.csv",
            "parameter_file": "training_parameter_diagnostics.csv",
            "console_output": False,
            "affects_optimization": False,
        }

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("model_schema") != self.checkpoint_model_schema:
            raise ValueError("Checkpoint uses a removed routing architecture; retraining is required")
        bounds = [
            list(bound)
            for bound in self.model.runoff_generator.parameterization.parameter_bounds
        ]
        if checkpoint.get("hbv_parameter_bounds") != bounds:
            raise ValueError("Checkpoint HBV parameter bounds differ from the current model")
        encoder_configuration = (
            self.model.runoff_generator.parameterization.checkpoint_configuration()
        )
        if checkpoint.get("parameter_encoder_configuration") != encoder_configuration:
            raise ValueError("Checkpoint parameter encoder differs from the current model")
        if checkpoint.get("grid_routing_parameterization") != self.model.grid_router.checkpoint_configuration():
            raise ValueError("Checkpoint grid-routing configuration differs from the current model")
        if checkpoint.get("loss_configuration") != self.loss_configuration:
            raise ValueError("Checkpoint loss configuration differs from the current model")
        if checkpoint.get("optimization_configuration") != self.optimization_configuration:
            raise ValueError("Checkpoint optimization configuration differs from the current model")
        expected = {
            "method": "epoch_warmup_then_chronological_state_carry",
            "anchor": self.params.train_start,
            "warmup_days": self.params.spinup_days,
            "window_days": self.params.train_window_days,
            "batch_size": 1,
            "shuffle": False,
            "state_detach": "each_window",
            "history_gradients": False,
        }
        if checkpoint.get("training_state_initialization") != expected:
            raise ValueError("Checkpoint state-initialization contract differs")
        if not checkpoint.get("parameter_context") or not self.parameter_context_metadata:
            raise ValueError("Parameter-context metadata is missing")
        if checkpoint["parameter_context"] != self.parameter_context_metadata:
            raise ValueError("Checkpoint training context, grid, or gauge mapping differs")

    def restore_checkpoint(self, checkpoint):
        self.on_load_checkpoint(checkpoint)
        self.load_state_dict(checkpoint["state_dict"], strict=True)

    @staticmethod
    def model_arguments(batch, forcing_name):
        forcing = batch[forcing_name]
        return {
            "precipitation": forcing["precipitation"],
            "temperature": forcing["temperature"],
            "pet": forcing["pet"],
            "terrain": batch["terrain"],
            "parameter_climate_sequence": batch["parameter_climate_sequence"],
            "basin_mask": batch["basin_mask"],
            "cell_area_km2": batch["cell_area_km2"],
        }

    @staticmethod
    def supervised_gauges(discharge_daily, batch):
        if discharge_daily.ndim != 4 or discharge_daily.shape[-1] != 1:
            raise ValueError("Grid router must return [B,T,G,1]")
        observed = batch["discharge_daily"]
        if observed.ndim != 4 or observed.shape[2] != 1:
            raise ValueError(f"Gauge observations must be [B,T,1,G], got {tuple(observed.shape)}")
        if discharge_daily.shape[:2] != observed.shape[:2] or discharge_daily.shape[2] != observed.shape[3]:
            raise ValueError("Simulated and observed dates/stations do not match")
        return discharge_daily, observed

    def forward_window(self, batch):
        if self._train_state is None or self._next_train_start is None:
            raise RuntimeError("Training state was not initialized at epoch start")
        if batch["terrain"].shape[0] != 1 or len(batch["target_start"]) != 1:
            raise RuntimeError("Chronological state carry requires batch_size=1")
        start = pd.Timestamp(batch["target_start"][0])
        if start != self._next_train_start:
            raise RuntimeError(
                "Training windows must be chronological and contiguous: "
                f"expected {self._next_train_start.date()}, got {start.date()}"
            )
        arguments = self.model_arguments(batch, "forcing")
        arguments["state"] = self._train_state
        output = self.model(**arguments)
        days = int(arguments["precipitation"].shape[1])
        if days < 1:
            raise RuntimeError("Training window must contain at least one day")
        self._train_state = detach_state(output.final_state)
        self._next_train_start = start + pd.Timedelta(days=days)
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_window(batch)
        simulated, observed = self.supervised_gauges(output.discharge_daily, batch)
        target = batch.get("discharge_training_target", observed)
        if target.shape != observed.shape:
            raise ValueError("Training targets must match observation shapes")
        losses = self.loss_function(
            simulated,
            target,
            batch["gauge_area_km2"],
        )
        batch_size = batch["terrain"].shape[0]
        self.train_loss_sum += float(losses["loss"].detach().cpu()) * batch_size
        self.train_raw_rmse_sum += float(
            losses["raw_rmse"].detach().cpu()
        ) * batch_size
        self.train_log_sqrt_rmse_sum += float(
            losses["log_sqrt_rmse"].detach().cpu()
        ) * batch_size
        self.train_sample_count += batch_size
        parameter_items = [
            (name, value)
            for name, value in output.diagnostics.items()
            if (
                name.startswith("hbv_parameter_")
                or name.startswith("routing_parameter_")
            ) and torch.is_tensor(value) and value.numel() == 1
        ]
        parameter_values = {}
        if parameter_items:
            detached_values = torch.stack([
                value.detach().float().reshape(())
                for _, value in parameter_items
            ]).cpu().tolist()
            parameter_values = {
                name: value
                for (name, _), value in zip(parameter_items, detached_values)
            }
        for name, value in parameter_values.items():
            self.parameter_diagnostic_sums[name] = (
                self.parameter_diagnostic_sums.get(name, 0.0) + value
            )
        self.parameter_diagnostic_steps += 1
        self.log("train_loss", losses["loss"], on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        self.log("train_raw_rmse", losses["raw_rmse"], on_step=False,
                 on_epoch=True, batch_size=batch_size)
        self.log("train_log_sqrt_rmse", losses["log_sqrt_rmse"],
                 on_step=False, on_epoch=True, batch_size=batch_size)
        for name in ("runoff_water_balance", "routing_budget_residual_m3"):
            self.log(f"train_{name}", output.diagnostics[name].mean(),
                     on_step=False, on_epoch=True, batch_size=batch_size)
        a, b = self.model.grid_router.parameter_fields()
        self.log("train_kinematic_a", a.mean(), on_epoch=True, batch_size=batch_size)
        self.log("train_kinematic_b", b.mean(), on_epoch=True, batch_size=batch_size)
        return losses["loss"]

    def on_train_epoch_start(self):
        self.train_loss_sum = 0.0
        self.train_raw_rmse_sum = 0.0
        self.train_log_sqrt_rmse_sum = 0.0
        self.train_sample_count = 0
        self.gradient_diagnostic_sums = {}
        self.gradient_diagnostic_maxima = {}
        self.gradient_diagnostic_steps = 0
        self.gradient_clip_exceeded_steps = 0
        self.parameter_diagnostic_sums = {}
        self.parameter_diagnostic_steps = 0
        if self.training_dataset is None:
            raise RuntimeError("Training requires a chronological dataset")
        if self.params.batch_size != 1:
            raise RuntimeError("Chronological state carry requires batch_size=1")
        windows = self.training_dataset.windows
        if not windows:
            raise RuntimeError("Training dataset has no windows")
        first = pd.Timestamp(windows[0].target_start)
        expected_first = (
            pd.Timestamp(self.params.train_start)
            + pd.Timedelta(days=self.params.spinup_days)
        )
        if first != expected_first:
            raise RuntimeError(
                f"First training window must start after exactly {self.params.spinup_days} "
                f"warm-up days: expected {expected_first.date()}, got {first.date()}"
            )
        expected = first
        for window in windows:
            actual = pd.Timestamp(window.target_start)
            if actual != expected:
                raise RuntimeError(
                    "Training windows must be ordered, non-overlapping and contiguous: "
                    f"expected {expected.date()}, got {actual.date()}"
                )
            expected = actual + pd.Timedelta(days=window.target_days)
        self.print(
            f"[第{self.current_epoch + 1}轮预热] {self.params.train_start} 至 "
            f"{(first - pd.Timedelta(days=1)).date()} | {self.params.spinup_days}天"
        )
        started = time.perf_counter()
        state, record = warmup_to_date(
            self.model, self.training_dataset, self.params.train_start, first,
            next(self.parameters()).device,
            chunk_days=self.params.train_window_days,
        )
        elapsed = time.perf_counter() - started
        self._train_state = state
        self._next_train_start = first
        self._training_warmup_record = {
            **record,
            "epoch": self.current_epoch + 1,
            "elapsed_seconds": elapsed,
            "weights": "epoch_start",
        }
        self._epoch_training_started = time.perf_counter()
        self.print(
            f"[第{self.current_epoch + 1}轮预热完成] {elapsed:.1f}s | "
            f"随后连续训练{len(windows)}个窗口"
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update one in-place progress line without Lightning/tqdm duplication."""
        if not self.trainer.is_global_zero:
            return
        total = int(self.trainer.num_training_batches)
        completed = batch_idx + 1
        elapsed = time.perf_counter() - self._epoch_training_started
        seconds_per_window = elapsed / completed
        eta = seconds_per_window * max(total - completed, 0)
        sys.stdout.write(
            f"\r[第{self.current_epoch + 1}轮训练] "
            f"{completed}/{total}窗口 ({completed / total:.0%}) | "
            f"已用时 {elapsed / 60.0:.1f} min | ETA {eta / 60.0:.1f} min"
        )
        if completed == total:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def on_after_backward(self):
        nonfinite = [
            name for name, parameter in self.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if nonfinite:
            raise FloatingPointError(f"Non-finite gradients: {', '.join(nonfinite[:8])}")

    def on_before_optimizer_step(self, optimizer):
        """Accumulate detached gradient summaries without changing gradients."""
        del optimizer
        device = next(self.model.parameters()).device
        group_squared = {
            name: torch.zeros((), dtype=torch.float32, device=device)
            for name in ("total", "encoder", "parameter_head", "routing")
        }
        gradient_elements = 0
        zero_elements = torch.zeros((), dtype=torch.float32, device=device)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                squared = gradient.square().sum()
                group_squared["total"] += squared
                gradient_elements += gradient.numel()
                zero_elements += (gradient == 0).sum()
                if ".parameter_head." in name:
                    group_squared["parameter_head"] += squared
                elif name.startswith("runoff_generator.parameterization."):
                    group_squared["encoder"] += squared
                elif name.startswith("grid_router."):
                    group_squared["routing"] += squared

            current_tensors = {
                f"gradient_{name}_l2": torch.sqrt(value)
                for name, value in group_squared.items()
            }
            current_tensors["gradient_zero_fraction"] = (
                zero_elements / gradient_elements if gradient_elements else 1.0
            )

            parameterization = self.model.runoff_generator.parameterization
            output_layer = parameterization.parameter_head.network[-1]
            if output_layer.weight.grad is not None:
                weight_gradient = output_layer.weight.grad.detach().float()
                bias_gradient = (
                    output_layer.bias.grad.detach().float()
                    if output_layer.bias is not None
                    and output_layer.bias.grad is not None
                    else torch.zeros(weight_gradient.shape[0], device=weight_gradient.device)
                )
                row_norm = torch.sqrt(
                    weight_gradient.square().sum(dim=1) + bias_gradient.square()
                )
                for index, (name, _, _) in enumerate(
                    parameterization.parameter_bounds
                ):
                    current_tensors[f"gradient_hbv_output_{name}_l2"] = (
                        row_norm[index]
                    )

            for name, parameter in self.model.grid_router.named_parameters():
                if parameter.grad is not None:
                    safe_name = name.replace(".", "_")
                    current_tensors[f"gradient_routing_{safe_name}_l2"] = (
                        parameter.grad.detach().float().norm()
                    )

            diagnostic_names = list(current_tensors)
            diagnostic_values = torch.stack([
                torch.as_tensor(
                    current_tensors[name], dtype=torch.float32, device=device
                ).reshape(())
                for name in diagnostic_names
            ]).cpu().tolist()
            current = dict(zip(diagnostic_names, diagnostic_values))

        for name, value in current.items():
            self.gradient_diagnostic_sums[name] = (
                self.gradient_diagnostic_sums.get(name, 0.0) + value
            )
            self.gradient_diagnostic_maxima[name] = max(
                self.gradient_diagnostic_maxima.get(name, float("-inf")), value
            )
        self.gradient_diagnostic_steps += 1
        if current["gradient_total_l2"] > float(self.params.gradient_clip_val):
            self.gradient_clip_exceeded_steps += 1

    def on_train_epoch_end(self):
        if not self.train_sample_count:
            self._train_state = None
            self._next_train_start = None
            self._training_warmup_record = None
            self._epoch_training_started = None
            return
        final_window = self.training_dataset.windows[-1]
        expected_end = (
            pd.Timestamp(final_window.target_start)
            + pd.Timedelta(days=final_window.target_days)
        )
        if self._next_train_start != expected_end:
            raise RuntimeError(
                f"Training state ended at {self._next_train_start}, expected {expected_end}"
            )
        loss = self.train_loss_sum / self.train_sample_count
        raw_rmse = self.train_raw_rmse_sum / self.train_sample_count
        log_sqrt_rmse = self.train_log_sqrt_rmse_sum / self.train_sample_count
        optimizer = (
            self._trainer.optimizers[0]
            if self._trainer and self._trainer.optimizers
            else None
        )
        record = {
            "epoch": self.current_epoch + 1,
            "composite_training_loss": loss,
            "raw_runoff_depth_rmse_mm_day": raw_rmse,
            "log_sqrt_runoff_depth_rmse": log_sqrt_rmse,
            "samples": self.train_sample_count,
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ) if optimizer is not None else float(self.params.learning_rate),
            "routing": "d8_kinematic",
            "record_type": "chronological_windows_with_state_carry",
            "warmup_days": self._training_warmup_record["initialization_days"],
            "warmup_seconds": self._training_warmup_record["elapsed_seconds"],
            "state_detach": "each_window",
        }
        a, b = self.model.grid_router.parameter_fields()
        record.update(
            kinematic_a_mean=float(a.mean().detach()),
            kinematic_b_mean=float(b.mean().detach()),
        )
        self.train_history.append(record)
        if self.gradient_diagnostic_steps:
            gradient_record = {
                "epoch": self.current_epoch + 1,
                "optimizer_steps": self.gradient_diagnostic_steps,
                "clip_threshold_l2": float(self.params.gradient_clip_val),
                "clip_exceeded_fraction": (
                    self.gradient_clip_exceeded_steps
                    / self.gradient_diagnostic_steps
                ),
            }
            for name in sorted(self.gradient_diagnostic_sums):
                gradient_record[f"{name}_mean"] = (
                    self.gradient_diagnostic_sums[name]
                    / self.gradient_diagnostic_steps
                )
                gradient_record[f"{name}_max"] = (
                    self.gradient_diagnostic_maxima[name]
                )
            self.gradient_diagnostic_history.append(gradient_record)
        if self.parameter_diagnostic_steps:
            self.parameter_diagnostic_history.append({
                "epoch": self.current_epoch + 1,
                "windows": self.parameter_diagnostic_steps,
                **{
                    name: value / self.parameter_diagnostic_steps
                    for name, value in sorted(
                        self.parameter_diagnostic_sums.items()
                    )
                },
            })
        self.print(
            f"[第{self.current_epoch + 1}轮优化] composite={loss:.4f} | "
            f"raw-RMSE={raw_rmse:.4f} | log-sqrt-RMSE={log_sqrt_rmse:.4f}"
        )
        if self.train_result_dir is not None:
            self.train_result_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.train_history).to_csv(
                self.train_result_dir / "training_history.csv", index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(self.gradient_diagnostic_history).to_csv(
                self.train_result_dir / "training_gradient_diagnostics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(self.parameter_diagnostic_history).to_csv(
                self.train_result_dir / "training_parameter_diagnostics.csv",
                index=False,
                encoding="utf-8-sig",
            )
        self._train_state = None
        self._next_train_start = None
        self._training_warmup_record = None
        self._epoch_training_started = None

    def on_fit_end(self):
        if self.train_result_dir is not None:
            (self.train_result_dir / "routing_diagnostics_final_epoch.json").write_text(
                json.dumps(self.model.grid_router.diagnostics(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if self.train_result_dir is not None and self.params.generate_figures:
            from disthybridcore.utils.result_plots import plot_training_history
            plot_training_history(self.train_result_dir, self.train_history)

    def on_test_epoch_start(self):
        self._test_predictions.clear(); self._test_observations.clear()
        self.test_training_targets.clear(); self._test_dates.clear()
        self._test_diagnostics.clear(); self.test_hydrology.clear()
        self.test_gauge_hydrology.clear(); self.test_gauge_budgets.clear()
        self._test_state = self.initial_test_state

    def prepare_test_state(self, dataset):
        start = min(
            pd.Timestamp(self.params.train_start),
            pd.Timestamp(self.params.test_start) - pd.Timedelta(days=self.params.spinup_days),
        )
        self.initial_test_state, record = warmup_to_date(
            self.model, dataset, start, self.params.test_start,
            next(self.parameters()).device, chunk_days=self.params.test_window_days,
        )
        self.evaluation_metadata["state_initialization"] = record
        print(
            f"[测试历史状态] {start.date()} 至 "
            f"{(pd.Timestamp(self.params.test_start) - pd.Timedelta(days=1)).date()} | "
            f"连续推进 {record['initialization_days']} 天"
        )

    def test_step(self, batch, batch_idx):
        if self._test_state is None:
            raise RuntimeError("prepare_test_state must run before continuous testing")
        arguments = self.model_arguments(batch, "forcing")
        arguments["state"] = self._test_state
        output = self.model(**arguments)
        self._test_state = detach_state(output.final_state)
        prediction = daily_flow_values(output.discharge_daily)
        simulated, observed = self.supervised_gauges(output.discharge_daily, batch)
        losses = self.loss_function(
            simulated,
            observed,
            batch["gauge_area_km2"],
        )
        self.log("test_loss", losses["loss"], prog_bar=True, batch_size=prediction.shape[1])
        self._test_predictions.append(prediction[0].detach().cpu())
        self._test_observations.append(batch["discharge_daily"][0, :, 0].detach().cpu())
        self.test_training_targets.append(
            batch.get("discharge_training_target", batch["discharge_daily"])[0, :, 0].detach().cpu()
        )
        start = pd.Timestamp(batch["target_start"][0])
        from disthybridcore.utils.result_io import (
            daily_hydrology_frame, gauge_process_frame,
            gauge_routing_budget_frame, window_water_balance,
        )
        self.test_hydrology.append(daily_hydrology_frame(output.diagnostics, start))
        self.test_gauge_hydrology.append(gauge_process_frame(
            output.diagnostics,
            start,
            self.params.observed_station_names,
            simulation=prediction[0].detach().cpu(),
            observation=batch["discharge_daily"][0, :, 0].detach().cpu(),
        ))
        self.test_gauge_budgets.append(gauge_routing_budget_frame(
            output.diagnostics, start, self.params.observed_station_names
        ))
        self._test_dates.extend(pd.date_range(start, periods=prediction.shape[1], freq="D").tolist())
        self._test_diagnostics.append(window_water_balance(output.diagnostics, start))
        return losses

    def on_test_epoch_end(self):
        if self.result_dir is None:
            return
        from disthybridcore.utils.result_io import (
            save_daily_results, save_gauge_process_diagnostics,
            save_water_balance_diagnostics,
        )
        from disthybridcore.utils.result_plots import plot_daily_results
        prediction = torch.cat(self._test_predictions, dim=0)
        observation = torch.cat(self._test_observations, dim=0)
        names = list(self.params.observed_station_names)
        reference_dates = pd.date_range(
            pd.Timestamp(self.params.train_start),
            pd.Timestamp(self.params.train_end),
            freq="D",
            inclusive="left",
        )
        reference_observation = self.training_dataset.discharge.read(
            reference_dates[0], len(reference_dates)
        )[:, 0]
        metrics = save_daily_results(
            result_dir=self.result_dir, dates=self._test_dates,
            prediction=prediction, observation=observation, station_names=names,
            model_name=self.params.model_label,
            primary_station=self.params.outlet_station_name,
            training_target=torch.cat(self.test_training_targets, dim=0),
            metadata={
                **self.evaluation_metadata, "period": "Test",
                "evaluation_mode": "continuous", "spinup_days": self.params.spinup_days,
                "target_start": self._test_dates[0].isoformat(),
                "target_end_exclusive": (self._test_dates[-1] + pd.Timedelta(days=1)).isoformat(),
                "days": len(self._test_dates),
            },
            event_reference_dates=reference_dates,
            event_reference_observation=reference_observation,
        )
        pd.concat(self.test_hydrology, ignore_index=True).to_csv(
            self.result_dir / "daily_hydrology.csv", index=False, encoding="utf-8-sig"
        )
        save_water_balance_diagnostics(self.result_dir, self._test_diagnostics)
        save_gauge_process_diagnostics(
            self.result_dir, self.test_gauge_hydrology, self.test_gauge_budgets
        )
        summary_station = "Mean" if len(names) > 1 else names[0]
        summary = metrics[(metrics["Model"] == self.params.model_label) & (metrics["Station"] == summary_station)].iloc[0]
        outlet = metrics[(metrics["Model"] == self.params.model_label) & (metrics["Station"] == self.params.outlet_station_name)].iloc[0]
        self.log("test_kge", float(summary["KGE"]), prog_bar=True)
        self.log("test_nse", float(summary["NSE"]), prog_bar=True)
        self.log("test_rmse", float(summary["RMSE"]), prog_bar=True)
        self.log("test_outlet_kge", float(outlet["KGE"]))
        self.log("test_outlet_nse", float(outlet["NSE"]))
        self.log("test_outlet_rmse", float(outlet["RMSE"]))
        if self.params.generate_figures:
            plot_daily_results(
                self.result_dir, self._test_dates, prediction, observation, names,
                metrics, self.params.model_label, period_label="Test",
                outlet_station=self.params.outlet_station_name,
            )
        print(f"[测试结果] 已保存到: {self.result_dir}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            threshold=1.0e-3,
            threshold_mode="rel",
            min_lr=max(self.params.learning_rate * 0.05, 1.0e-6),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
