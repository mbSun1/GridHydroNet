"""Daily multi-gauge discharge loss and differentiable hydrology metrics."""

import torch
from torch import nn


def daily_flow_values(discharge_daily: torch.Tensor) -> torch.Tensor:
    """Convert daily discharge from [B,D,S,1] to [B,D,S] in m3/s."""
    if discharge_daily.ndim != 4 or discharge_daily.shape[-1] != 1:
        raise ValueError("Daily discharge must have shape [B,D,S,1]")
    if discharge_daily.shape[1] < 1:
        raise ValueError("Daily discharge must contain at least one day")
    return discharge_daily[..., 0].float()


def differentiable_kge(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean KGE across batch-station series with valid observations."""
    prediction = prediction.float()
    target = target.float()
    valid = torch.isfinite(target) & (target >= 0) & torch.isfinite(prediction)
    target = torch.where(valid, target, torch.zeros_like(target))
    prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    raw_count = valid.sum(dim=1)
    eligible = raw_count >= 2
    if not eligible.any():
        return prediction.sum() * 0.0 + torch.full(
            (), torch.nan, device=prediction.device, dtype=prediction.dtype
        )
    count = raw_count.clamp_min(2)
    pred_mean = prediction.sum(dim=1) / count
    obs_mean = target.sum(dim=1) / count
    pred_centered = prediction - pred_mean[:, None]
    obs_centered = target - obs_mean[:, None]
    covariance = (pred_centered * obs_centered * valid).sum(dim=1) / count
    pred_std = torch.sqrt(
        (pred_centered.square() * valid).sum(dim=1) / count + 1.0e-8
    )
    obs_std = torch.sqrt(
        (obs_centered.square() * valid).sum(dim=1) / count + 1.0e-8
    )
    eligible = (
        eligible
        & (obs_mean > 0)
        & ((obs_centered.square() * valid).sum(dim=1) > 1.0e-8)
    )
    if not eligible.any():
        return prediction.sum() * 0.0 + torch.full(
            (), torch.nan, device=prediction.device, dtype=prediction.dtype
        )
    correlation = covariance / (pred_std * obs_std + 1.0e-8)
    alpha = pred_std / (obs_std + 1.0e-8)
    beta = pred_mean / (obs_mean + 1.0e-8)
    station_kge = 1.0 - torch.sqrt(
        (correlation - 1.0).square()
        + (alpha - 1.0).square()
        + (beta - 1.0).square()
    )
    return station_kge[eligible].mean()


def differentiable_nse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean NSE across batch-station series with valid observations."""
    prediction = prediction.float()
    target = target.float()
    valid = torch.isfinite(target) & (target >= 0) & torch.isfinite(prediction)
    target = torch.where(valid, target, torch.zeros_like(target))
    prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
    raw_count = valid.sum(dim=1)
    count = raw_count.clamp_min(1)
    target_mean = target.sum(dim=1) / count
    target_anomaly = target - target_mean[:, None]
    squared_error = ((prediction - target).square() * valid).sum(dim=1)
    target_variance = (target_anomaly.square() * valid).sum(dim=1)
    eligible = (raw_count >= 2) & (target_variance > 1.0e-8)
    if not eligible.any():
        return prediction.sum() * 0.0 + torch.full(
            (), torch.nan, device=prediction.device, dtype=prediction.dtype
        )
    station_nse = 1.0 - squared_error / target_variance.clamp_min(1.0e-8)
    return station_nse[eligible].mean()


class HydroLoss(nn.Module):
    """Station-balanced physDL-HBV combined RMSE objective.

    The reference objective is 0.75 raw-flow RMSE plus 0.25 transformed-flow
    RMSE, where the transform is log10(sqrt(Q + 1e-6) + 0.1). Discharge is
    converted from m3/s to runoff depth in mm/day with each station's fixed D8
    upstream area before both terms. Station losses are then averaged equally.
    """

    def forward(
        self,
        discharge_daily,
        discharge_daily_observed,
        gauge_area_km2,
    ):
        prediction = daily_flow_values(discharge_daily)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(
                "Simulated daily discharge contains NaN or Inf"
            )
        if (
            discharge_daily_observed.ndim != 4
            or discharge_daily_observed.shape[2] != 1
        ):
            raise ValueError("Daily observations must have shape [B,D,1,S]")
        observation = discharge_daily_observed[:, :, 0].float()
        if observation.shape != prediction.shape:
            raise ValueError(
                "Daily simulated and observed dates/stations must match exactly"
            )

        area = gauge_area_km2.to(
            device=prediction.device, dtype=prediction.dtype
        )
        if area.ndim == 1:
            area = area[None, None, :]
        elif area.ndim == 2:
            area = area[:, None, :]
        else:
            raise ValueError("gauge_area_km2 must have shape [S] or [B,S]")
        if area.shape[-1] != prediction.shape[-1] or area.shape[0] not in {
            1, prediction.shape[0]
        }:
            raise ValueError("gauge_area_km2 does not match batch/station dimensions")
        if not torch.isfinite(area).all() or torch.any(area <= 0):
            raise ValueError("gauge_area_km2 must be finite and positive")
        runoff_depth_factor = (
            86400.0 / (area * 1000.0)
        ).expand_as(prediction)

        valid = (
            torch.isfinite(observation)
            & (observation >= 0)
            & torch.isfinite(prediction)
        )
        if not valid.any():
            zero = prediction.sum() * 0.0
            return {
                "loss": zero,
                "raw_rmse": zero,
                "log_sqrt_rmse": zero,
                "kge": differentiable_kge(prediction, observation),
                "nse": differentiable_nse(prediction, observation),
            }

        safe_prediction = torch.where(
            valid, prediction, torch.zeros_like(prediction)
        ).clamp_min(0.0) * runoff_depth_factor
        safe_observation = torch.where(
            valid, observation, torch.zeros_like(observation)
        ).clamp_min(0.0) * runoff_depth_factor
        count = valid.sum(dim=(0, 1))
        eligible = count > 0

        raw_squared_error = (safe_prediction - safe_observation).square()
        raw_rmse_by_station = torch.sqrt(
            (raw_squared_error * valid).sum(dim=(0, 1))
            / count.clamp_min(1)
        )
        raw_rmse = raw_rmse_by_station[eligible].mean()

        transformed_prediction = torch.log10(
            torch.sqrt(safe_prediction + 1.0e-6) + 0.1
        )
        transformed_observation = torch.log10(
            torch.sqrt(safe_observation + 1.0e-6) + 0.1
        )
        transformed_squared_error = (
            transformed_prediction - transformed_observation
        ).square()
        log_sqrt_rmse_by_station = torch.sqrt(
            (transformed_squared_error * valid).sum(dim=(0, 1))
            / count.clamp_min(1)
        )
        log_sqrt_rmse = log_sqrt_rmse_by_station[eligible].mean()

        objective = 0.75 * raw_rmse + 0.25 * log_sqrt_rmse
        return {
            "loss": objective,
            "raw_rmse": raw_rmse,
            "log_sqrt_rmse": log_sqrt_rmse,
            "kge": differentiable_kge(prediction, observation),
            "nse": differentiable_nse(prediction, observation),
        }
