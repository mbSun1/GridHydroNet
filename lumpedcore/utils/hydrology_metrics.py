"""Full-precision hydrological metrics for one outlet station."""

from __future__ import annotations

import numpy as np


def hydrology_metrics(simulation, observation) -> dict[str, float | int]:
    simulation = np.asarray(simulation, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if simulation.shape != observation.shape:
        raise ValueError("Simulation and observation shapes differ")
    valid = np.isfinite(simulation) & np.isfinite(observation) & (observation >= 0)
    if valid.sum() < 2:
        return {
            "NSE": np.nan,
            "KGE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "PBIAS": np.nan,
            "Correlation": np.nan,
            "StdRatio": np.nan,
            "MeanRatio": np.nan,
            "ObservedMean": np.nan,
            "SimulatedMean": np.nan,
            "Days": int(valid.sum()),
        }
    sim = simulation[valid]
    obs = observation[valid]
    obs_mean = float(obs.mean())
    sim_mean = float(sim.mean())
    obs_std = float(obs.std())
    sim_std = float(sim.std())
    denominator = float(np.sum((obs - obs_mean) ** 2))
    nse = 1.0 - float(np.sum((sim - obs) ** 2)) / denominator if denominator > 0 else np.nan
    correlation = (
        float(np.corrcoef(sim, obs)[0, 1])
        if obs_std > 0 and sim_std > 0
        else 0.0 if obs_std > 0 else np.nan
    )
    alpha = sim_std / obs_std if obs_std > 0 else np.nan
    beta = sim_mean / obs_mean if obs_mean != 0 else np.nan
    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
    )
    bias = float(np.mean(sim - obs))
    pbias = 100.0 * float(np.sum(sim - obs)) / float(np.sum(obs)) if np.sum(obs) != 0 else np.nan
    return {
        "NSE": float(nse),
        "KGE": float(kge),
        "RMSE": float(np.sqrt(np.mean((sim - obs) ** 2))),
        "MAE": float(np.mean(np.abs(sim - obs))),
        "Bias": bias,
        "PBIAS": float(pbias),
        "Correlation": float(correlation),
        "StdRatio": float(alpha),
        "MeanRatio": float(beta),
        "ObservedMean": obs_mean,
        "SimulatedMean": sim_mean,
        "Days": int(valid.sum()),
    }
