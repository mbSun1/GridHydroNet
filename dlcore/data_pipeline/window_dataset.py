"""Leakage-safe normalization and same-day rainfall-runoff windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dlcore.data_pipeline.dataset import DLBasinSeries


@dataclass(frozen=True)
class ScalerStatistics:
    forcing_mean: tuple[float, float, float]
    forcing_std: tuple[float, float, float]
    discharge_mean: float
    discharge_std: float
    fitted_start: str
    fitted_end_exclusive: str

    def to_dict(self) -> dict:
        return {
            "forcing_variables": ["prec", "t", "pet"],
            "forcing_mean": list(self.forcing_mean),
            "forcing_std": list(self.forcing_std),
            "discharge_mean": self.discharge_mean,
            "discharge_std": self.discharge_std,
            "fitted_start": self.fitted_start,
            "fitted_end_exclusive": self.fitted_end_exclusive,
            "fit_scope": "training period only",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScalerStatistics":
        return cls(
            forcing_mean=tuple(float(value) for value in payload["forcing_mean"]),
            forcing_std=tuple(float(value) for value in payload["forcing_std"]),
            discharge_mean=float(payload["discharge_mean"]),
            discharge_std=float(payload["discharge_std"]),
            fitted_start=str(payload["fitted_start"]),
            fitted_end_exclusive=str(payload["fitted_end_exclusive"]),
        )


def fit_scaler(series: DLBasinSeries, train_start, train_end_exclusive) -> ScalerStatistics:
    mask = series.period_mask(train_start, train_end_exclusive)
    if mask.sum() < 2:
        raise ValueError("Training period contains fewer than two days")
    forcing = series.forcing[mask]
    discharge = series.observed_discharge[mask]
    valid_discharge = discharge[np.isfinite(discharge) & (discharge >= 0)]
    if valid_discharge.size < 2:
        raise ValueError("Training period contains fewer than two valid outlet observations")
    forcing_mean = forcing.mean(axis=0)
    forcing_std = forcing.std(axis=0)
    forcing_std = np.where(forcing_std > 1e-12, forcing_std, 1.0)
    discharge_std = float(valid_discharge.std())
    if discharge_std <= 1e-12:
        discharge_std = 1.0
    return ScalerStatistics(
        forcing_mean=tuple(float(value) for value in forcing_mean),
        forcing_std=tuple(float(value) for value in forcing_std),
        discharge_mean=float(valid_discharge.mean()),
        discharge_std=discharge_std,
        fitted_start=pd.Timestamp(train_start).strftime("%Y-%m-%d"),
        fitted_end_exclusive=pd.Timestamp(train_end_exclusive).strftime("%Y-%m-%d"),
    )


def calendar_features(dates: pd.DatetimeIndex) -> np.ndarray:
    day_of_year = dates.dayofyear.to_numpy(dtype=np.float64)
    day_of_week = dates.dayofweek.to_numpy(dtype=np.float64)
    annual_angle = 2.0 * np.pi * (day_of_year - 1.0) / 365.2425
    weekly_angle = 2.0 * np.pi * day_of_week / 7.0
    return np.column_stack(
        [
            np.sin(annual_angle),
            np.cos(annual_angle),
            np.sin(weekly_angle),
            np.cos(weekly_angle),
        ]
    ).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        series: DLBasinSeries,
        scaler: ScalerStatistics,
        target_start,
        target_end_exclusive,
        lookback: int = 365,
    ):
        if lookback < 1:
            raise ValueError("lookback must be positive")
        forcing_mean = np.asarray(scaler.forcing_mean, dtype=np.float64)
        forcing_std = np.asarray(scaler.forcing_std, dtype=np.float64)
        normalized_forcing = (series.forcing - forcing_mean) / forcing_std
        target_mask = series.period_mask(target_start, target_end_exclusive)
        valid_target = np.isfinite(series.observed_discharge) & (
            series.observed_discharge >= 0
        )
        target_indices = np.flatnonzero(target_mask & valid_target)
        target_indices = target_indices[target_indices >= lookback - 1]
        if target_indices.size == 0:
            raise ValueError("Period contains no valid targets with a complete lookback window")
        self.forcing = torch.as_tensor(normalized_forcing, dtype=torch.float32)
        self.time_features = torch.as_tensor(calendar_features(series.dates))
        self.targets = torch.as_tensor(series.observed_discharge, dtype=torch.float32)
        self.target_indices = torch.as_tensor(target_indices, dtype=torch.long)
        self.lookback = int(lookback)
        self.discharge_mean = float(scaler.discharge_mean)
        self.discharge_std = float(scaler.discharge_std)

    def __len__(self) -> int:
        return int(self.target_indices.numel())

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        target_index = int(self.target_indices[item])
        input_start = target_index - self.lookback + 1
        raw_target = self.targets[target_index]
        normalized_target = (raw_target - self.discharge_mean) / self.discharge_std
        return {
            "forcing": self.forcing[input_start : target_index + 1],
            "time_features": self.time_features[input_start : target_index + 1],
            "target": normalized_target.reshape(1),
            "observed_discharge": raw_target,
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }
