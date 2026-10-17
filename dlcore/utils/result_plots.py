"""Compact hydrological evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_discharge_results(frame, output_path: str | Path, title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed = frame["observed_discharge_m3_s"].to_numpy(dtype=float)
    simulated = frame["simulated_discharge_m3_s"].to_numpy(dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(frame["date"], observed, color="black", linewidth=0.8, label="Observed")
    axes[0].plot(frame["date"], simulated, color="#0072B2", linewidth=0.8, label="Simulated")
    axes[0].set(title=title, ylabel="Discharge (m³/s)")
    axes[0].legend(frameon=False)
    if valid.any():
        obs_sorted = np.sort(observed[valid])[::-1]
        sim_sorted = np.sort(simulated[valid])[::-1]
        exceedance = 100.0 * (np.arange(obs_sorted.size) + 0.5) / obs_sorted.size
        axes[1].plot(exceedance, obs_sorted, color="black", linewidth=0.8)
        axes[1].plot(exceedance, sim_sorted, color="#0072B2", linewidth=0.8)
        axes[1].set(xlabel="Exceedance probability (%)", ylabel="Discharge (m³/s)")
        axes[2].scatter(observed[valid], simulated[valid], s=8, alpha=0.5, color="#0072B2")
        lower = float(min(observed[valid].min(), simulated[valid].min()))
        upper = float(max(observed[valid].max(), simulated[valid].max()))
        axes[2].plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axes[2].set(xlabel="Observed (m³/s)", ylabel="Simulated (m³/s)")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path
