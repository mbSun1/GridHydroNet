"""Hydrological metrics used by the new daily model result tools."""

import numpy as np
import pandas as pd


def station_metrics(simulation: np.ndarray, observation: np.ndarray) -> dict:
    """原始流量NSE/KGE及其分解，用于区分均值偏差、振幅和相关性问题。"""
    simulation = np.asarray(simulation, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    valid = np.isfinite(simulation) & np.isfinite(observation) & (observation >= 0)
    if valid.sum() < 2:
        return {
            **{name: np.nan for name in (
                "KGE", "NSE", "RMSE", "Bias", "Correlation", "StdRatio", "MeanRatio",
                "ObservedMean", "SimulatedMean", "ObservedStd", "SimulatedStd", "SSE", "SST",
            )},
            "Days": int(valid.sum()),
        }
    sim = simulation[valid]
    obs = observation[valid]
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    if obs_std <= 0:
        correlation = np.nan
    elif sim_std > 0:
        correlation = np.corrcoef(sim, obs)[0, 1]
    else:
        # A constant simulation has no linear association with a varying
        # observation. Use r=0 so KGE remains a finite poor score, matching
        # the differentiable training metric instead of silently dropping it.
        correlation = 0.0
    alpha = sim_std / obs_std if obs_std > 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (alpha - 1.0) ** 2
        + (beta - 1.0) ** 2
    )
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    nse = (
        1.0 - np.sum((sim - obs) ** 2) / denominator
        if denominator > 0
        else np.nan
    )
    return {
        "KGE": float(kge),
        "NSE": float(nse),
        "RMSE": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "Bias": float(np.mean(sim - obs)),
        "Days": int(valid.sum()),
        "Correlation": float(correlation),
        "StdRatio": float(alpha),
        "MeanRatio": float(beta),
        "ObservedMean": float(np.mean(obs)),
        "SimulatedMean": float(np.mean(sim)),
        "ObservedStd": float(obs_std),
        "SimulatedStd": float(sim_std),
        "SSE": float(np.sum((sim - obs) ** 2)),
        "SST": float(denominator),
    }


def build_metrics_table(prediction, observation, station_names, model_name):
    """按站点计算模拟与实测的指标，多站时追加站点平均指标。"""
    rows = []
    for index, station in enumerate(station_names):
        row = {"Model": model_name, "Station": station}
        row.update(station_metrics(prediction[:, index], observation[:, index]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(station_names) == 1:
        return frame
    means = (
        frame.groupby("Model", as_index=False)[["KGE", "NSE", "RMSE", "Bias"]]
        .mean(numeric_only=True)
        .assign(Station="Mean")
    )
    return pd.concat([frame, means], ignore_index=True)


def build_flow_regime_metrics(
    prediction,
    observation,
    reference_observation,
    station_names,
):
    """Summarize high-flow levels and low-flow levels/daily changes by station."""
    prediction = np.asarray(prediction, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    reference = np.asarray(reference_observation, dtype=np.float64)
    expected = (prediction.shape[0], len(station_names))
    if prediction.shape != expected or observation.shape != expected:
        raise ValueError("Flow-regime series must have matching [days, stations] shapes")
    if reference.ndim != 2 or reference.shape[1] != len(station_names):
        raise ValueError("Reference observations must have shape [days, stations]")

    rows = []
    for index, station in enumerate(station_names):
        reference_valid = reference[:, index]
        reference_valid = reference_valid[
            np.isfinite(reference_valid) & (reference_valid >= 0)
        ]
        if reference_valid.size == 0:
            raise ValueError(f"Training observations are empty for station {station}")
        q75, q95 = np.quantile(reference_valid, [0.75, 0.95])
        sim = prediction[:, index]
        obs = observation[:, index]
        finite = np.isfinite(sim) & np.isfinite(obs) & (obs >= 0)
        low = finite & (obs <= q75)
        high = finite & (obs >= q95)
        low_obs = obs[low]
        low_sim = sim[low]
        high_obs = obs[high]
        high_sim = sim[high]
        pair = low[1:] & low[:-1]
        obs_delta = np.diff(np.log1p(np.clip(obs, 0, None)))[pair]
        sim_delta = np.diff(np.log1p(np.clip(sim, 0, None)))[pair]
        low_obs_mean = np.mean(low_obs) if low_obs.size else np.nan
        high_obs_mean = np.mean(high_obs) if high_obs.size else np.nan
        obs_delta_std = np.std(obs_delta) if obs_delta.size else np.nan
        sim_delta_std = np.std(sim_delta) if sim_delta.size else np.nan
        rows.append({
            "Station": station,
            "Q75Training_m3_s": float(q75),
            "Q95Training_m3_s": float(q95),
            "LowFlowDays": int(low.sum()),
            "LowFlowObservedMean_m3_s": float(low_obs_mean),
            "LowFlowSimulatedMean_m3_s": (
                float(np.mean(low_sim)) if low_sim.size else np.nan
            ),
            "LowFlowMeanRatio": (
                float(np.mean(low_sim) / low_obs_mean)
                if low_sim.size and low_obs_mean > 0 else np.nan
            ),
            "LowFlowLogRMSE": (
                float(np.sqrt(np.mean((
                    np.log1p(np.clip(low_sim, 0, None))
                    - np.log1p(low_obs)
                ) ** 2))) if low_sim.size else np.nan
            ),
            "LowFlowConsecutivePairs": int(pair.sum()),
            "LowFlowDeltaCorrelation": (
                float(np.corrcoef(sim_delta, obs_delta)[0, 1])
                if obs_delta.size >= 2
                and np.std(obs_delta) > 0 and np.std(sim_delta) > 0
                else np.nan
            ),
            "LowFlowDeltaStdRatio": (
                float(sim_delta_std / obs_delta_std)
                if obs_delta_std > 0 else np.nan
            ),
            "HighFlowDays": int(high.sum()),
            "HighFlowObservedMean_m3_s": float(high_obs_mean),
            "HighFlowSimulatedMean_m3_s": (
                float(np.mean(high_sim)) if high_sim.size else np.nan
            ),
            "HighFlowMeanRatio": (
                float(np.mean(high_sim) / high_obs_mean)
                if high_sim.size and high_obs_mean > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)
