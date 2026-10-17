"""Compact outlet hydrograph and diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def plot_hydrograph(frame: pd.DataFrame, output_path: str | Path, title: str) -> Path:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(frame["date"])
    fig, (rain_axis, flow_axis) = plt.subplots(
        2, 1, figsize=(10.5, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08},
    )
    rain_axis.bar(dates, frame["precipitation_mm_day"], width=1.0, color="#4C78A8")
    rain_axis.invert_yaxis()
    rain_axis.set_ylabel("P (mm/d)")
    rain_axis.spines[["top", "right"]].set_visible(False)
    flow_axis.plot(dates, frame["observed_m3_s"], color="black", lw=0.9, label="Observed")
    flow_axis.plot(dates, frame["simulated_m3_s"], color="#C62828", lw=0.9, label="HBV92")
    flow_axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)")
    flow_axis.set_xlabel("Date")
    flow_axis.set_ylim(bottom=0)
    flow_axis.grid(axis="y", color="0.9", lw=0.5)
    flow_axis.spines[["top", "right"]].set_visible(False)
    flow_axis.legend(frameon=False, ncols=2)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    flow_axis.xaxis.set_major_locator(locator)
    flow_axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle(title, x=0.08, ha="left", fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_flow_duration(
    frame: pd.DataFrame,
    output_path: str | Path,
    metrics_text: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    for column, color, label in (
        ("observed_m3_s", "black", "Observed"),
        ("simulated_m3_s", "#C62828", "HBV92"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        values = np.sort(values[np.isfinite(values) & (values >= 0)])[::-1]
        probability = 100.0 * np.arange(1, len(values) + 1) / (len(values) + 1)
        axis.plot(probability, values, color=color, lw=1.1, label=label)
    axis.set_yscale("symlog", linthresh=0.1)
    axis.set_xlabel("Exceedance probability (%)")
    axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)")
    axis.grid(color="0.9", lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    if metrics_text:
        axis.set_title(metrics_text, loc="left", fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_discharge_scatter(
    frame: pd.DataFrame,
    output_path: str | Path,
    metrics_text: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observed = pd.to_numeric(frame["observed_m3_s"], errors="coerce").to_numpy()
    simulated = pd.to_numeric(frame["simulated_m3_s"], errors="coerce").to_numpy()
    valid = np.isfinite(observed) & np.isfinite(simulated) & (observed >= 0)
    limit = float(max(observed[valid].max(), simulated[valid].max())) if valid.any() else 1.0
    fig, axis = plt.subplots(figsize=(4.8, 4.5))
    axis.scatter(observed[valid], simulated[valid], s=8, alpha=0.35, color="#C62828")
    axis.plot([0, limit], [0, limit], color="black", lw=0.8, ls="--")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_xlabel(r"Observed Q (m$^3$ s$^{-1}$)")
    axis.set_ylabel(r"Simulated Q (m$^3$ s$^{-1}$)")
    axis.grid(color="0.9", lw=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    if metrics_text:
        axis.set_title(metrics_text, loc="left", fontsize=8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
