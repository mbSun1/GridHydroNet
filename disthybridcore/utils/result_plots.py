"""训练与测试使用同一科研图样式；只导出JPG，源数据另存CSV。"""

from pathlib import Path
import re

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from disthybridcore.utils.hydrology_metrics import station_metrics


def plot_station_hydrographs(
    output_dir,
    dates,
    prediction,
    observation,
    station_names,
    metrics,
    model_name,
    period_label=None,
    outlet_station=None,
):
    """Save one observed-versus-simulated hydrograph per supervised gauge."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "station_discharge_series.jpg").unlink(missing_ok=True)
    for obsolete_path in output_dir.glob("*_outlet_discharge_series.jpg"):
        obsolete_path.unlink()
    # NaN和负值缺测编码统一显示为时间轴断线。
    observation = np.where(np.isfinite(observation) & (observation >= 0), observation, np.nan)
    model_metrics = metrics[metrics["Model"] == model_name].set_index(
        "Station"
    )
    paths = []
    for index, name in enumerate(station_names):
        station_label = str(name)
        station_parts = [part.strip() for part in station_label.split("|", 1)]
        station_id = station_parts[0]
        display_name = station_parts[-1]
        filename_id = re.sub(r"[^A-Za-z0-9_-]+", "_", station_id).strip("_")
        if not filename_id:
            filename_id = f"station_{index + 1:02d}"
        is_outlet = (
            outlet_station is not None
            and station_label == str(outlet_station)
        )
        legacy_output_path = output_dir / f"{filename_id}_discharge_series.jpg"
        if is_outlet:
            legacy_output_path.unlink(missing_ok=True)
            output_path = (
                output_dir / f"{filename_id}_outlet_discharge_series.jpg"
            )
        else:
            output_path = legacy_output_path

        with plt.rc_context({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
        }):
            fig, axis = plt.subplots(figsize=(7.2, 3.2))
            axis.plot(
                dates,
                observation[:, index],
                color="black",
                lw=1.05,
                label="Observed",
                zorder=2,
            )
            axis.plot(
                dates,
                prediction[:, index],
                color="#C62828",
                lw=0.9,
                label="Simulated",
                zorder=3,
            )
            values = model_metrics.loc[name]
            metric_label = (
                f"KGE = {values['KGE']:.2f}\n"
                f"NSE = {values['NSE']:.2f}"
            )
            axis.text(
                0.98,
                0.96,
                metric_label,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                    "boxstyle": "square,pad=0.35",
                },
            )
            if is_outlet:
                axis.text(
                    0.02,
                    0.96,
                    "OUTLET GAUGE",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.2,
                    fontweight="bold",
                    color="white",
                    bbox={
                        "facecolor": "#B2182B",
                        "edgecolor": "#B2182B",
                        "boxstyle": "square,pad=0.3",
                    },
                    zorder=5,
                )
            title_parts = [station_id, display_name]
            if is_outlet:
                title_parts.append("OUTLET GAUGE")
            if period_label is not None:
                title_parts.append(str(period_label))
            title = " | ".join(title_parts)
            axis.set_title(
                title,
                fontsize=10,
                loc="left",
                fontweight="bold",
                color="#B2182B" if is_outlet else "#111111",
            )
            axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)", fontsize=9)
            axis.tick_params(labelsize=8)
            date_locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            axis.xaxis.set_major_locator(date_locator)
            axis.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(date_locator)
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="0.9", linewidth=0.5, zorder=0)
            axis.set_ylim(bottom=0)
            axis.margins(x=0.01)
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncols=2,
                frameon=False,
                fontsize=9,
            )
            fig.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.80)
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
        paths.append(output_path)
    return paths


def plot_daily_results(
    result_dir,
    dates,
    prediction,
    observation,
    station_names,
    metrics,
    model_name,
    period_label=None,
    outlet_station=None,
):
    figures_dir = Path(result_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    date_index = pd.DatetimeIndex(dates)
    prediction = prediction.numpy()
    observation = observation.numpy()
    plot_station_hydrographs(
        figures_dir,
        date_index,
        prediction,
        observation,
        station_names,
        metrics,
        model_name,
        period_label=period_label,
        outlet_station=outlet_station,
    )
    if (Path(result_dir) / "flood_event_catalog.csv").is_file():
        plot_flood_event_results(
            result_dir,
            include_station_figures=True,
            period_label=period_label,
        )
    return figures_dir


def plot_training_history(result_dir, history):
    """显示各epoch的复合训练目标；KGE/NSE只来自固定权重回放。"""
    table = pd.DataFrame(history)
    if table.empty:
        return
    plt.switch_backend("Agg")
    figure_dir = Path(result_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "font.size": 8,
    }):
        fig, axis = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
        axis.plot(
            table["epoch"], table["composite_training_loss"],
            "o-", color="#2166AC", lw=1, ms=2.5,
        )
        axis.set_title("Optimization loss", loc="left", fontsize=9)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Composite training loss")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", lw=0.4)
        fig.suptitle("Weights update between training batches", fontsize=8)
        fig.savefig(figure_dir / "training_history.jpg", dpi=400)
        plt.close(fig)


def station_numeric_label(station: str) -> str:
    """Return only the numeric gauge identifier for compact figure labels."""
    identifier = str(station).split("|", 1)[0].strip()
    identifier = re.sub(r"^ID_", "", identifier, flags=re.IGNORECASE)
    match = re.search(r"\d+", identifier)
    return match.group(0) if match else identifier


def plot_flood_event_performance_overview(catalog, metrics, output_dir):
    """Save each station-by-event performance metric as an individual heat map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flood_event_performance_overview.jpg").unlink(
        missing_ok=True
    )

    basin_catalog = catalog[
        catalog["event_type"].str.lower().isin(
            ["outlet_anchored", "basin", "basin_event", "common"]
        )
    ].copy()
    basin_catalog = basin_catalog.sort_values(
        ["event_start", "event_id"], kind="stable"
    )
    event_ids = basin_catalog["event_id"].drop_duplicates().tolist()
    selected = metrics[metrics["event_id"].isin(event_ids)].copy()
    stations = selected["station"].drop_duplicates().tolist()
    station_labels = [station_numeric_label(station) for station in stations]

    label_lookup = basin_catalog.set_index("event_id")
    event_labels = []
    for event_id in event_ids:
        row = label_lookup.loc[event_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        peak_date = row.get("anchor_peak_date", pd.NaT)
        if pd.notna(peak_date):
            event_labels.append(
                f"{event_id}\n{pd.Timestamp(peak_date):%Y-%m-%d}"
            )
        else:
            event_labels.append(str(event_id))

    specifications = [
        (
            "event_nse", "Flood-event NSE across gauges", "Event NSE",
            "RdYlGn", (-1.0, 1.0), ".2f", "flood_event_nse_heatmap.jpg",
        ),
        (
            "event_kge", "Flood-event KGE across gauges", "Event KGE",
            "RdYlGn", (-1.0, 1.0), ".2f", "flood_event_kge_heatmap.jpg",
        ),
        (
            "peak_relative_error", "Flood-peak relative error across gauges",
            "Peak relative error (%)", "RdBu_r", None, "+.0f",
            "flood_peak_relative_error_heatmap.jpg",
        ),
        (
            "peak_timing_error_days", "Flood-peak timing error across gauges",
            "Peak timing error (days)", "coolwarm", None, "+.0f",
            "flood_peak_timing_error_heatmap.jpg",
        ),
    ]
    figure_width = min(10.0, max(6.3, 2.4 + 0.82 * len(event_ids)))
    figure_height = min(11.0, max(5.2, 1.8 + 0.34 * len(stations)))
    paths = []

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
    }):
        for (
            column,
            title,
            colorbar_label,
            color_map,
            fixed_limits,
            value_format,
            filename,
        ) in specifications:
            output_path = output_dir / filename
            if not event_ids or not stations:
                fig, axis = plt.subplots(figsize=(6.0, 2.2))
                axis.axis("off")
                axis.text(
                    0.5,
                    0.5,
                    "No basin flood events available",
                    ha="center",
                    va="center",
                    color="0.35",
                )
                fig.savefig(output_path, dpi=400, bbox_inches="tight")
                plt.close(fig)
                paths.append(output_path)
                continue

            table = selected.pivot_table(
                index="station",
                columns="event_id",
                values=column,
                aggfunc="first",
                dropna=False,
            ).reindex(index=stations, columns=event_ids)
            values = table.to_numpy(dtype=np.float64)
            display_values = (
                values * 100.0 if column == "peak_relative_error" else values
            )
            masked = np.ma.masked_invalid(display_values)
            cmap = plt.get_cmap(color_map).with_extremes(bad="#E3E3E3")
            finite = display_values[np.isfinite(display_values)]
            if fixed_limits is not None:
                norm = colors.Normalize(
                    vmin=fixed_limits[0], vmax=fixed_limits[1]
                )
            else:
                default_limit = 50.0 if column == "peak_relative_error" else 1.0
                limit = max(
                    default_limit,
                    float(np.nanpercentile(np.abs(finite), 95))
                    if finite.size else default_limit,
                )
                norm = colors.TwoSlopeNorm(
                    vmin=-limit, vcenter=0.0, vmax=limit
                )

            fig, axis = plt.subplots(
                figsize=(figure_width, figure_height), layout="constrained"
            )
            image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
            axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
            axis.set_xlabel("Flood event")
            axis.set_ylabel("Gauge ID")
            axis.set_xticks(np.arange(len(event_ids)), event_labels)
            axis.set_yticks(np.arange(len(stations)), station_labels)
            axis.tick_params(axis="x", rotation=35, labelsize=7.2)
            axis.tick_params(axis="y", labelsize=7.2)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
            axis.set_xticks(np.arange(-0.5, len(event_ids), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(stations), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.8)
            axis.tick_params(which="minor", bottom=False, left=False)

            for row_index in range(len(stations)):
                for column_index in range(len(event_ids)):
                    value = display_values[row_index, column_index]
                    if not np.isfinite(value):
                        axis.text(
                            column_index,
                            row_index,
                            "NA",
                            ha="center",
                            va="center",
                            fontsize=6.2,
                            color="0.4",
                        )
                        continue
                    red, green, blue, _ = cmap(norm(value))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    axis.text(
                        column_index,
                        row_index,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if luminance < 0.48 else "black",
                    )
            colorbar = fig.colorbar(image, ax=axis, shrink=0.84, pad=0.025)
            colorbar.set_label(colorbar_label)
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)
    return paths


def plot_flood_peak_scatter(metrics, output_dir):
    """Plot observed and simulated event peaks on logarithmic 1:1 axes."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flood_peak_scatter.jpg"
    selected = metrics[
        np.isfinite(metrics["observed_peak_m3_s"])
        & np.isfinite(metrics["simulated_peak_m3_s"])
        & (metrics["observed_peak_m3_s"] > 0)
        & (metrics["simulated_peak_m3_s"] > 0)
    ].copy()

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
    }):
        if selected.empty:
            fig, axis = plt.subplots(figsize=(4.8, 2.2))
            axis.axis("off")
            axis.text(
                0.5, 0.5, "No valid flood peaks available",
                ha="center", va="center", color="0.35",
            )
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            return output_path

        fig, axis = plt.subplots(figsize=(5.3, 4.7), layout="constrained")
        all_peaks = np.concatenate([
            selected["observed_peak_m3_s"].to_numpy(),
            selected["simulated_peak_m3_s"].to_numpy(),
        ])
        lower = float(np.nanmin(all_peaks))
        upper = float(np.nanmax(all_peaks))
        lower = max(np.finfo(float).tiny, lower / 1.25)
        upper = upper * 1.25
        reference = np.geomspace(lower, upper, 300)
        axis.fill_between(
            reference, 0.5 * reference, 1.5 * reference,
            color="#ECEFF4", label=r"$\pm$50%", zorder=0,
        )
        axis.fill_between(
            reference, 0.8 * reference, 1.2 * reference,
            color="#D5EAF4", label=r"$\pm$20%", zorder=1,
        )
        axis.plot(
            reference, reference, color="black", lw=0.9,
            label="1:1", zorder=2,
        )
        stations = selected["station"].drop_duplicates().tolist()
        color_values = plt.get_cmap("tab20")(
            np.linspace(0.0, 1.0, max(1, len(stations)))
        )
        for color, station in zip(color_values, stations):
            group = selected[selected["station"] == station]
            axis.scatter(
                group["observed_peak_m3_s"],
                group["simulated_peak_m3_s"],
                s=24, color=color, edgecolor="white", linewidth=0.35,
                alpha=0.9, label=station, zorder=3,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"Observed peak Q (m$^3$ s$^{-1}$)")
        axis.set_ylabel(r"Simulated peak Q (m$^3$ s$^{-1}$)")
        axis.set_title(
            f"Flood-peak reproduction (n = {len(selected)})",
            loc="left", fontsize=9, fontweight="bold",
        )
        axis.grid(which="both", color="0.9", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            frameon=False, fontsize=6.5, borderaxespad=0.0,
        )
        fig.savefig(output_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
    return output_path


def plot_flood_event_station_summary(metrics, output_dir):
    """Plot event values and station-wise median/interquartile ranges."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "flood_event_station_summary.jpg"
    stations = metrics["station"].drop_duplicates().tolist()

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
    }):
        if not stations:
            fig, axis = plt.subplots(figsize=(6.0, 2.2))
            axis.axis("off")
            axis.text(
                0.5, 0.5, "No flood-event station metrics available",
                ha="center", va="center", color="0.35",
            )
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            return output_path

        specifications = [
            ("event_nse", "Event NSE", 1.0, 1.0),
            ("event_kge", "Event KGE", 1.0, 1.0),
            ("peak_relative_error", "Peak error (%)", 100.0, 0.0),
            ("peak_timing_error_days", "Peak timing error (days)", 1.0, 0.0),
        ]
        fig, axes = plt.subplots(
            1, 4,
            figsize=(13.2, min(10.0, max(3.8, 1.8 + 0.36 * len(stations)))),
            sharey=True, layout="constrained",
        )
        station_positions = np.arange(len(stations), dtype=float)
        for axis, (column, title, scale, target) in zip(axes, specifications):
            for station_index, station in enumerate(stations):
                values = pd.to_numeric(
                    metrics.loc[metrics["station"] == station, column],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)] * scale
                if not values.size:
                    continue
                offsets = (
                    np.linspace(-0.16, 0.16, len(values))
                    if len(values) > 1 else np.array([0.0])
                )
                axis.scatter(
                    values, station_index + offsets,
                    s=11, color="#7AA6C2", alpha=0.6,
                    edgecolors="none", zorder=2,
                )
                lower_quartile, median, upper_quartile = np.nanpercentile(
                    values, [25.0, 50.0, 75.0]
                )
                axis.plot(
                    [lower_quartile, upper_quartile],
                    [station_index, station_index],
                    color="#1D3557", lw=2.2, zorder=3,
                )
                axis.scatter(
                    median, station_index, marker="D", s=22,
                    color="#D1495B", edgecolor="white", linewidth=0.35,
                    zorder=4,
                )
            axis.axvline(target, color="0.35", lw=0.75, ls="--", zorder=1)
            axis.set_title(title, loc="left", fontsize=9, fontweight="bold")
            axis.grid(axis="x", color="0.9", linewidth=0.45)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
        axes[0].set_yticks(station_positions, stations)
        axes[0].invert_yaxis()
        axes[0].tick_params(axis="y", labelsize=7)
        legend = [
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#7AA6C2",
                markeredgecolor="none", markersize=4.5, label="Individual event",
            ),
            Line2D(
                [0], [0], color="#1D3557", lw=2.2,
                label="Interquartile range",
            ),
            Line2D(
                [0], [0], marker="D", color="none", markerfacecolor="#D1495B",
                markeredgecolor="white", markersize=5, label="Median",
            ),
        ]
        fig.legend(
            handles=legend, loc="upper center", ncols=3,
            frameon=False, bbox_to_anchor=(0.5, 1.03),
        )
        fig.savefig(output_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
    return output_path


def plot_flood_event_hydrographs(catalog, metrics, daily_discharge, output_dir):
    """Plot one multi-gauge hydrograph figure for each basin flood event."""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_path in output_dir.glob("*_hydrographs.jpg"):
        obsolete_path.unlink()
    basin_catalog = catalog[
        catalog["event_type"].str.lower().isin(
            ["outlet_anchored", "basin", "basin_event", "common"]
        )
    ].sort_values(["event_start", "event_id"], kind="stable")
    paths = []
    used_filenames = set()

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.75,
    }):
        for event_number, event in enumerate(
            basin_catalog.drop_duplicates("event_id").itertuples(index=False), 1
        ):
            event_id = str(event.event_id)
            start = pd.Timestamp(event.event_start)
            end = pd.Timestamp(event.event_end)
            event_series = daily_discharge[
                daily_discharge["date"].between(start, end, inclusive="both")
            ]
            event_metrics = metrics[metrics["event_id"] == event_id]
            stations = event_metrics["station"].drop_duplicates().tolist()
            if not stations:
                stations = event_series["station"].drop_duplicates().tolist()

            safe_event_id = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", event_id
            ).strip("._")
            if not safe_event_id:
                safe_event_id = f"event_{event_number:03d}"
            base_name = safe_event_id
            suffix = 2
            while safe_event_id.lower() in used_filenames:
                safe_event_id = f"{base_name}_{suffix}"
                suffix += 1
            used_filenames.add(safe_event_id.lower())
            output_path = output_dir / f"{safe_event_id}_hydrographs.jpg"

            if not stations or event_series.empty:
                fig, axis = plt.subplots(figsize=(6.4, 2.2))
                axis.axis("off")
                axis.text(
                    0.5, 0.5, f"{event_id}: no daily discharge available",
                    ha="center", va="center", color="0.35",
                )
                fig.savefig(output_path, dpi=400, bbox_inches="tight")
                plt.close(fig)
                paths.append(output_path)
                continue

            fig, axes = plt.subplots(
                len(stations), 1,
                figsize=(8.0, max(3.0, 1.85 * len(stations) + 1.0)),
                sharex=True, squeeze=False,
            )
            axes = axes[:, 0]
            for axis, station in zip(axes, stations):
                station_series = event_series[
                    event_series["station"] == station
                ].sort_values("date")
                dates = station_series["date"]
                observed = station_series["observed_m3_s"].where(
                    np.isfinite(station_series["observed_m3_s"])
                    & (station_series["observed_m3_s"] >= 0)
                )
                simulated = station_series["simulated_m3_s"].where(
                    np.isfinite(station_series["simulated_m3_s"])
                )
                axis.plot(
                    dates, observed, color="black", lw=1.05,
                    label="Observed", zorder=3,
                )
                axis.plot(
                    dates, simulated, color="#C62828", lw=0.95,
                    label="Simulated", zorder=3,
                )
                if observed.notna().any():
                    peak_index = observed.idxmax()
                    axis.scatter(
                        station_series.loc[peak_index, "date"], observed.loc[peak_index],
                        marker="o", s=20, color="black", edgecolor="white",
                        linewidth=0.4, zorder=4,
                    )
                if simulated.notna().any():
                    peak_index = simulated.idxmax()
                    axis.scatter(
                        station_series.loc[peak_index, "date"], simulated.loc[peak_index],
                        marker="^", s=24, color="#C62828", edgecolor="white",
                        linewidth=0.4, zorder=4,
                    )
                row = event_metrics[event_metrics["station"] == station]
                nse = row["event_nse"].iloc[0] if not row.empty else np.nan
                kge = row["event_kge"].iloc[0] if not row.empty else np.nan
                peak_error = (
                    row["peak_relative_error"].iloc[0]
                    if not row.empty else np.nan
                )
                timing_error = (
                    row["peak_timing_error_days"].iloc[0]
                    if not row.empty else np.nan
                )
                labels = [
                    f"NSE={nse:.2f}" if np.isfinite(nse) else "NSE=NA",
                    f"KGE={kge:.2f}" if np.isfinite(kge) else "KGE=NA",
                    (
                        f"Peak={100.0 * peak_error:+.0f}%"
                        if np.isfinite(peak_error) else "Peak=NA"
                    ),
                    (
                        f"Timing={timing_error:+.0f} d"
                        if np.isfinite(timing_error) else "Timing=NA"
                    ),
                ]
                axis.text(
                    0.99, 0.94, " | ".join(labels),
                    transform=axis.transAxes, ha="right", va="top",
                    fontsize=6.8,
                    bbox={
                        "facecolor": "white", "edgecolor": "0.82",
                        "alpha": 0.9, "boxstyle": "square,pad=0.25",
                    },
                )
                axis.set_title(str(station), loc="left", fontsize=8, fontweight="bold")
                axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)")
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", color="0.9", linewidth=0.45)
                axis.spines[["top", "right"]].set_visible(False)
                axis.margins(x=0.01)
            date_locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            axes[-1].xaxis.set_major_locator(date_locator)
            axes[-1].xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(date_locator)
            )
            axes[-1].set_xlabel("Date")
            handles, labels = axes[0].get_legend_handles_labels()
            peak_date = getattr(event, "anchor_peak_date", pd.NaT)
            peak_text = (
                f" | anchor peak {pd.Timestamp(peak_date):%Y-%m-%d}"
                if pd.notna(peak_date) else ""
            )
            fig.suptitle(
                f"{event_id} | {start:%Y-%m-%d} to {end:%Y-%m-%d}{peak_text}",
                x=0.08, ha="left", fontsize=10, fontweight="bold",
            )
            fig.legend(
                handles, labels, loc="upper right", ncols=2,
                frameon=False, bbox_to_anchor=(0.98, 0.995),
            )
            fig.subplots_adjust(
                left=0.09, right=0.98, bottom=0.08,
                top=max(0.78, 0.91 - 0.01 * len(stations)), hspace=0.34,
            )
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)
    return paths


def station_figure_filename(station: str, station_index: int) -> str:
    """Return a stable filename stem led by the gauge identifier."""
    station_id = str(station).split("|", 1)[0].strip()
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", station_id).strip("._")
    return safe_name or f"station_{station_index + 1:02d}"


def metric_annotation(event_row: pd.Series) -> str:
    """Format the four primary event metrics for one hydrograph panel."""
    nse = pd.to_numeric(event_row.get("event_nse"), errors="coerce")
    kge = pd.to_numeric(event_row.get("event_kge"), errors="coerce")
    peak_error = pd.to_numeric(
        event_row.get("peak_relative_error"), errors="coerce"
    )
    timing_error = pd.to_numeric(
        event_row.get("peak_timing_error_days"), errors="coerce"
    )
    labels = [
        f"NSE={nse:.2f}" if np.isfinite(nse) else "NSE=NA",
        f"KGE={kge:.2f}" if np.isfinite(kge) else "KGE=NA",
        (
            f"Peak={100.0 * peak_error:+.0f}%"
            if np.isfinite(peak_error) else "Peak=NA"
        ),
        (
            f"Timing={timing_error:+.0f} d"
            if np.isfinite(timing_error) else "Timing=NA"
        ),
    ]
    return " | ".join(labels)


def plot_station_flood_event_hydrographs(
    catalog,
    metrics,
    daily_discharge,
    thresholds,
    output_dir,
    period_label="Period",
    maximum_events_per_figure=12,
):
    """Plot every detected flood event separately for each gauge."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_path in output_dir.glob("*.jpg"):
        obsolete_path.unlink()

    if maximum_events_per_figure < 1:
        raise ValueError("maximum_events_per_figure must be at least one")

    catalog_ids = set(catalog["event_id"].astype(str))
    station_names = metrics["station"].drop_duplicates().tolist()
    threshold_lookup = (
        thresholds.drop_duplicates("station").set_index("station")
        if thresholds is not None and not thresholds.empty
        else pd.DataFrame()
    )
    paths = []
    figure_records = []

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.7,
    }):
        for station_index, station in enumerate(station_names):
            station_rows = metrics[
                (metrics["station"] == station)
                & metrics["station_high_flow_detected"]
                & metrics["event_id"].isin(catalog_ids)
            ].copy()
            station_rows = station_rows.sort_values(
                ["event_start", "event_id"], kind="stable"
            )
            event_count = len(station_rows)
            if event_count:
                chunks = [
                    station_rows.iloc[start:start + maximum_events_per_figure]
                    for start in range(0, event_count, maximum_events_per_figure)
                ]
            else:
                chunks = [station_rows]

            station_text = str(station)
            station_parts = [
                value.strip() for value in station_text.split("|", 1)
            ]
            station_id = station_parts[0]
            station_name = station_parts[-1]
            filename_stem = station_figure_filename(station_text, station_index)
            high_threshold = np.nan
            recession_threshold = np.nan
            if not threshold_lookup.empty and station in threshold_lookup.index:
                threshold_row = threshold_lookup.loc[station]
                if isinstance(threshold_row, pd.DataFrame):
                    threshold_row = threshold_row.iloc[0]
                high_threshold = pd.to_numeric(
                    threshold_row.get("high_flow_threshold_m3_s"), errors="coerce"
                )
                recession_threshold = pd.to_numeric(
                    threshold_row.get("recession_threshold_m3_s"), errors="coerce"
                )

            for chunk_index, chunk in enumerate(chunks, start=1):
                part_suffix = (
                    f"_part{chunk_index:02d}" if len(chunks) > 1 else ""
                )
                output_path = output_dir / (
                    f"{filename_stem}_flood_events{part_suffix}.jpg"
                )
                if chunk.empty:
                    fig, axis = plt.subplots(figsize=(7.2, 2.4))
                    axis.axis("off")
                    axis.text(
                        0.5,
                        0.5,
                        "No station-specific high-flow event was detected",
                        ha="center",
                        va="center",
                        color="0.35",
                    )
                    fig.suptitle(
                        f"{station_id} | {station_name} | {period_label}",
                        x=0.08,
                        ha="left",
                        fontsize=10,
                        fontweight="bold",
                    )
                    fig.savefig(output_path, dpi=400, bbox_inches="tight")
                    plt.close(fig)
                    paths.append(output_path)
                    figure_records.append({
                        "period": str(period_label),
                        "station": station,
                        "detected_event_count": 0,
                        "figure_part": chunk_index,
                        "events_in_figure": 0,
                        "figure_file": output_path.name,
                    })
                    continue

                column_count = 2 if len(chunk) > 1 else 1
                row_count = int(np.ceil(len(chunk) / column_count))
                fig, axes = plt.subplots(
                    row_count,
                    column_count,
                    figsize=(8.2, 2.35 * row_count + 0.75),
                    squeeze=False,
                )
                axes_flat = axes.ravel()
                for axis, (_, event_row) in zip(axes_flat, chunk.iterrows()):
                    event_id = str(event_row["event_id"])
                    start = pd.Timestamp(event_row["event_start"])
                    end = pd.Timestamp(event_row["event_end"])
                    event_series = daily_discharge[
                        (daily_discharge["station"] == station)
                        & daily_discharge["date"].between(
                            start, end, inclusive="both"
                        )
                    ].sort_values("date")
                    observed = event_series["observed_m3_s"].where(
                        np.isfinite(event_series["observed_m3_s"])
                        & (event_series["observed_m3_s"] >= 0)
                    )
                    simulated = event_series["simulated_m3_s"].where(
                        np.isfinite(event_series["simulated_m3_s"])
                    )
                    axis.plot(
                        event_series["date"],
                        observed,
                        color="black",
                        lw=1.05,
                        zorder=3,
                    )
                    axis.plot(
                        event_series["date"],
                        simulated,
                        color="#C62828",
                        lw=0.95,
                        zorder=3,
                    )
                    if observed.notna().any():
                        peak_index = observed.idxmax()
                        axis.scatter(
                            event_series.loc[peak_index, "date"],
                            observed.loc[peak_index],
                            marker="o",
                            s=19,
                            color="black",
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=4,
                        )
                    if simulated.notna().any():
                        peak_index = simulated.idxmax()
                        axis.scatter(
                            event_series.loc[peak_index, "date"],
                            simulated.loc[peak_index],
                            marker="^",
                            s=22,
                            color="#C62828",
                            edgecolor="white",
                            linewidth=0.4,
                            zorder=4,
                        )
                    if np.isfinite(high_threshold):
                        axis.axhline(
                            high_threshold,
                            color="#2F6690",
                            lw=0.75,
                            ls="--",
                            zorder=1,
                        )
                    if np.isfinite(recession_threshold):
                        axis.axhline(
                            recession_threshold,
                            color="#7A7A7A",
                            lw=0.65,
                            ls=":",
                            zorder=1,
                        )
                    event_type = str(event_row["event_type"]).replace("_", " ")
                    axis.set_title(
                        f"{event_id} | {event_type} | "
                        f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
                        loc="left",
                        fontsize=7.4,
                        fontweight="bold",
                    )
                    axis.text(
                        0.99,
                        0.94,
                        metric_annotation(event_row),
                        transform=axis.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6.2,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "0.82",
                            "alpha": 0.9,
                            "boxstyle": "square,pad=0.22",
                        },
                    )
                    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                    axis.xaxis.set_major_locator(locator)
                    axis.xaxis.set_major_formatter(
                        mdates.ConciseDateFormatter(locator)
                    )
                    axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)")
                    axis.set_ylim(bottom=0)
                    axis.grid(axis="y", color="0.9", linewidth=0.45)
                    axis.spines[["top", "right"]].set_visible(False)
                    axis.margins(x=0.015)
                for axis in axes_flat[len(chunk):]:
                    axis.axis("off")

                part_text = (
                    f" | part {chunk_index}/{len(chunks)}"
                    if len(chunks) > 1 else ""
                )
                fig.suptitle(
                    f"{station_id} | {station_name} | {period_label} flood events "
                    f"(n={event_count}){part_text}",
                    x=0.07,
                    y=0.995,
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                )
                legend_handles = [
                    Line2D([0], [0], color="black", lw=1.05, label="Observed"),
                    Line2D(
                        [0], [0], color="#C62828", lw=0.95,
                        label="Simulated",
                    ),
                    Line2D(
                        [0], [0], color="#2F6690", lw=0.75, ls="--",
                        label=(
                            f"Training Q95 = {high_threshold:.2f} m3/s"
                            if np.isfinite(high_threshold)
                            else "Training Q95 threshold"
                        ),
                    ),
                    Line2D(
                        [0], [0], color="#7A7A7A", lw=0.65, ls=":",
                        label=(
                            f"Training Q75 = {recession_threshold:.2f} m3/s"
                            if np.isfinite(recession_threshold)
                            else "Training Q75 recession"
                        ),
                    ),
                ]
                fig.legend(
                    handles=legend_handles,
                    loc="upper right",
                    ncols=4,
                    frameon=False,
                    bbox_to_anchor=(0.99, 0.965),
                    fontsize=6.8,
                )
                fig.subplots_adjust(
                    left=0.08,
                    right=0.985,
                    bottom=0.07,
                    top=0.88,
                    hspace=0.48,
                    wspace=0.24,
                )
                fig.savefig(output_path, dpi=400, bbox_inches="tight")
                plt.close(fig)
                paths.append(output_path)
                figure_records.append({
                    "period": str(period_label),
                    "station": station,
                    "detected_event_count": event_count,
                    "figure_part": chunk_index,
                    "events_in_figure": len(chunk),
                    "figure_file": output_path.name,
                })

    pd.DataFrame(figure_records).to_csv(
        output_dir / "station_flood_event_figure_index.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return paths


def select_top_station_peaks(
    daily_discharge,
    thresholds,
    period_label="Period",
    peak_count=5,
    minimum_separation_days=15,
    half_window_days=7,
):
    """Extract ranked independent observed peaks and fixed comparison windows."""
    if peak_count < 1:
        raise ValueError("peak_count must be at least one")
    if minimum_separation_days < 1:
        raise ValueError("minimum_separation_days must be at least one")
    if half_window_days < 1:
        raise ValueError("half_window_days must be at least one")

    threshold_lookup = (
        thresholds.drop_duplicates("station").set_index("station")
        if thresholds is not None and not thresholds.empty
        else pd.DataFrame()
    )
    records = []
    station_names = daily_discharge["station"].drop_duplicates().tolist()
    for station in station_names:
        station_series = daily_discharge[
            daily_discharge["station"] == station
        ].sort_values("date").reset_index(drop=True)
        dates = pd.DatetimeIndex(station_series["date"])
        observed = station_series["observed_m3_s"].to_numpy(dtype=np.float64)
        valid = np.isfinite(observed) & (observed >= 0)
        candidate_positions = []
        for position, value in enumerate(observed):
            if not valid[position]:
                continue
            previous = (
                observed[position - 1]
                if position > 0 and valid[position - 1]
                else -np.inf
            )
            following = (
                observed[position + 1]
                if position + 1 < len(observed) and valid[position + 1]
                else -np.inf
            )
            if value >= previous and value > following:
                candidate_positions.append(position)

        ranked_candidates = sorted(
            candidate_positions,
            key=lambda position: (-observed[position], dates[position]),
        )
        selected_positions = []
        for position in ranked_candidates:
            peak_date = dates[position]
            if all(
                abs((peak_date - dates[other]).days) >= minimum_separation_days
                for other in selected_positions
            ):
                selected_positions.append(position)
                if len(selected_positions) == peak_count:
                    break

        if len(selected_positions) < peak_count:
            fallback_positions = sorted(
                np.flatnonzero(valid),
                key=lambda position: (-observed[position], dates[position]),
            )
            for position in fallback_positions:
                if position in selected_positions:
                    continue
                peak_date = dates[position]
                if all(
                    abs((peak_date - dates[other]).days)
                    >= minimum_separation_days
                    for other in selected_positions
                ):
                    selected_positions.append(position)
                    if len(selected_positions) == peak_count:
                        break

        selected_positions = sorted(
            selected_positions,
            key=lambda position: (-observed[position], dates[position]),
        )
        high_threshold = np.nan
        recession_threshold = np.nan
        if not threshold_lookup.empty and station in threshold_lookup.index:
            threshold_row = threshold_lookup.loc[station]
            if isinstance(threshold_row, pd.DataFrame):
                threshold_row = threshold_row.iloc[0]
            high_threshold = pd.to_numeric(
                threshold_row.get("high_flow_threshold_m3_s"), errors="coerce"
            )
            recession_threshold = pd.to_numeric(
                threshold_row.get("recession_threshold_m3_s"), errors="coerce"
            )

        station_parts = [value.strip() for value in str(station).split("|", 1)]
        station_id = station_parts[0]
        station_name = station_parts[-1]
        first_date = dates.min()
        last_date = dates.max()
        for peak_rank, position in enumerate(selected_positions, start=1):
            observed_peak_date = dates[position]
            observed_peak = float(observed[position])
            window_start = max(
                first_date,
                observed_peak_date - pd.Timedelta(days=half_window_days),
            )
            window_end = min(
                last_date,
                observed_peak_date + pd.Timedelta(days=half_window_days),
            )
            window = station_series[
                station_series["date"].between(
                    window_start, window_end, inclusive="both"
                )
            ].copy()
            window_score = station_metrics(
                window["simulated_m3_s"].to_numpy(dtype=np.float64),
                window["observed_m3_s"].to_numpy(dtype=np.float64),
            )
            simulation_at_observed_peak = pd.to_numeric(
                station_series.loc[position, "simulated_m3_s"], errors="coerce"
            )
            simulated_values = window["simulated_m3_s"].to_numpy(
                dtype=np.float64
            )
            simulated_valid = np.isfinite(simulated_values)
            if simulated_valid.any():
                valid_positions = np.flatnonzero(simulated_valid)
                simulated_peak_offset = int(
                    valid_positions[np.argmax(simulated_values[valid_positions])]
                )
                simulated_peak = float(simulated_values[simulated_peak_offset])
                simulated_peak_date = pd.Timestamp(
                    window.iloc[simulated_peak_offset]["date"]
                )
            else:
                simulated_peak = np.nan
                simulated_peak_date = pd.NaT
            peak_relative_error = (
                (simulated_peak - observed_peak) / observed_peak
                if observed_peak > 0 and np.isfinite(simulated_peak)
                else np.nan
            )
            peak_timing_error = (
                int((simulated_peak_date - observed_peak_date).days)
                if pd.notna(simulated_peak_date)
                else np.nan
            )
            records.append({
                "period": str(period_label),
                "station": station,
                "station_id": station_id,
                "station_name": station_name,
                "peak_rank": peak_rank,
                "observed_peak_date": observed_peak_date.date().isoformat(),
                "observed_peak_m3_s": observed_peak,
                "simulation_on_observed_peak_date_m3_s": (
                    float(simulation_at_observed_peak)
                    if np.isfinite(simulation_at_observed_peak) else np.nan
                ),
                "simulated_window_peak_date": (
                    simulated_peak_date.date().isoformat()
                    if pd.notna(simulated_peak_date) else ""
                ),
                "simulated_window_peak_m3_s": simulated_peak,
                "peak_relative_error": peak_relative_error,
                "peak_timing_error_days": peak_timing_error,
                "window_start": window_start.date().isoformat(),
                "window_end": window_end.date().isoformat(),
                "window_valid_pair_days": int(window_score["Days"]),
                "window_nse": window_score["NSE"],
                "window_kge": window_score["KGE"],
                "training_q95_m3_s": high_threshold,
                "training_q75_m3_s": recession_threshold,
                "observed_peak_to_training_q95_ratio": (
                    observed_peak / high_threshold
                    if np.isfinite(high_threshold) and high_threshold > 0
                    else np.nan
                ),
                "above_training_q95": bool(
                    np.isfinite(high_threshold) and observed_peak >= high_threshold
                ),
                "minimum_peak_separation_days": minimum_separation_days,
                "half_window_days": half_window_days,
                "selection_method": (
                    "ranked observed local maxima with greedy minimum-date separation"
                ),
            })
    return pd.DataFrame.from_records(records)


def top_peak_metric_annotation(peak_row: pd.Series) -> str:
    """Format fixed-window performance for a ranked station peak."""
    nse = pd.to_numeric(peak_row.get("window_nse"), errors="coerce")
    kge = pd.to_numeric(peak_row.get("window_kge"), errors="coerce")
    peak_error = pd.to_numeric(
        peak_row.get("peak_relative_error"), errors="coerce"
    )
    timing_error = pd.to_numeric(
        peak_row.get("peak_timing_error_days"), errors="coerce"
    )
    labels = [
        f"NSE={nse:.2f}" if np.isfinite(nse) else "NSE=NA",
        f"KGE={kge:.2f}" if np.isfinite(kge) else "KGE=NA",
        (
            f"Peak={100.0 * peak_error:+.0f}%"
            if np.isfinite(peak_error) else "Peak=NA"
        ),
        (
            f"Timing={timing_error:+.0f} d"
            if np.isfinite(timing_error) else "Timing=NA"
        ),
    ]
    return " | ".join(labels)


def plot_top_station_peak_hydrographs(
    peaks,
    daily_discharge,
    output_dir,
    period_label="Period",
):
    """Plot the five largest independent peaks for every gauge."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_path in output_dir.glob("*.jpg"):
        obsolete_path.unlink()

    paths = []
    station_names = peaks["station"].drop_duplicates().tolist()
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.7,
    }):
        for station_index, station in enumerate(station_names):
            station_peaks = peaks[peaks["station"] == station].sort_values(
                "peak_rank", kind="stable"
            )
            station_parts = [value.strip() for value in str(station).split("|", 1)]
            station_id = station_parts[0]
            station_name = station_parts[-1]
            output_path = output_dir / (
                f"{station_figure_filename(station, station_index)}"
                "_top5_flood_peaks.jpg"
            )
            column_count = 2
            row_count = max(1, int(np.ceil(len(station_peaks) / column_count)))
            fig, axes = plt.subplots(
                row_count,
                column_count,
                figsize=(8.2, 2.45 * row_count + 0.75),
                squeeze=False,
            )
            axes_flat = axes.ravel()
            for axis, (_, peak_row) in zip(axes_flat, station_peaks.iterrows()):
                window_start = pd.Timestamp(peak_row["window_start"])
                window_end = pd.Timestamp(peak_row["window_end"])
                observed_peak_date = pd.Timestamp(
                    peak_row["observed_peak_date"]
                )
                window = daily_discharge[
                    (daily_discharge["station"] == station)
                    & daily_discharge["date"].between(
                        window_start, window_end, inclusive="both"
                    )
                ].sort_values("date")
                observed = window["observed_m3_s"].where(
                    np.isfinite(window["observed_m3_s"])
                    & (window["observed_m3_s"] >= 0)
                )
                simulated = window["simulated_m3_s"].where(
                    np.isfinite(window["simulated_m3_s"])
                )
                axis.plot(
                    window["date"], observed,
                    color="black", lw=1.1, label="Observed", zorder=3,
                )
                axis.plot(
                    window["date"], simulated,
                    color="#C62828", lw=0.95, label="Simulated", zorder=3,
                )
                axis.axvline(
                    observed_peak_date,
                    color="#5B8FA8",
                    lw=0.65,
                    ls="--",
                    zorder=1,
                )
                high_threshold = pd.to_numeric(
                    peak_row.get("training_q95_m3_s"), errors="coerce"
                )
                recession_threshold = pd.to_numeric(
                    peak_row.get("training_q75_m3_s"), errors="coerce"
                )
                if np.isfinite(high_threshold):
                    axis.axhline(
                        high_threshold,
                        color="#2F6690",
                        lw=0.7,
                        ls="--",
                        zorder=1,
                    )
                if np.isfinite(recession_threshold):
                    axis.axhline(
                        recession_threshold,
                        color="#7A7A7A",
                        lw=0.6,
                        ls=":",
                        zorder=1,
                    )
                axis.scatter(
                    observed_peak_date,
                    float(peak_row["observed_peak_m3_s"]),
                    marker="o",
                    s=22,
                    color="black",
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=4,
                )
                simulated_peak_date = pd.to_datetime(
                    peak_row.get("simulated_window_peak_date"), errors="coerce"
                )
                simulated_peak = pd.to_numeric(
                    peak_row.get("simulated_window_peak_m3_s"), errors="coerce"
                )
                if pd.notna(simulated_peak_date) and np.isfinite(simulated_peak):
                    axis.scatter(
                        simulated_peak_date,
                        simulated_peak,
                        marker="^",
                        s=24,
                        color="#C62828",
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=4,
                    )
                axis.set_title(
                    f"Rank {int(peak_row['peak_rank'])} | "
                    f"{observed_peak_date:%Y-%m-%d} | "
                    f"Q={float(peak_row['observed_peak_m3_s']):.2f} m3/s",
                    loc="left",
                    fontsize=7.5,
                    fontweight="bold",
                )
                axis.text(
                    0.99,
                    0.94,
                    top_peak_metric_annotation(peak_row),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.2,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "0.82",
                        "alpha": 0.9,
                        "boxstyle": "square,pad=0.22",
                    },
                )
                locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator)
                )
                axis.set_ylabel(r"Q (m$^3$ s$^{-1}$)")
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", color="0.9", linewidth=0.45)
                axis.spines[["top", "right"]].set_visible(False)
                axis.margins(x=0.015)
            for axis in axes_flat[len(station_peaks):]:
                axis.axis("off")

            fig.suptitle(
                f"{station_id} | {station_name} | five largest independent "
                f"{period_label} peaks",
                x=0.07,
                y=0.995,
                ha="left",
                fontsize=10,
                fontweight="bold",
            )
            high_threshold = pd.to_numeric(
                station_peaks.iloc[0].get("training_q95_m3_s"), errors="coerce"
            )
            recession_threshold = pd.to_numeric(
                station_peaks.iloc[0].get("training_q75_m3_s"), errors="coerce"
            )
            legend_handles = [
                Line2D([0], [0], color="black", lw=1.1, label="Observed"),
                Line2D([0], [0], color="#C62828", lw=0.95, label="Simulated"),
                Line2D(
                    [0], [0], color="#5B8FA8", lw=0.65, ls="--",
                    label="Observed peak date",
                ),
                Line2D(
                    [0], [0], color="#2F6690", lw=0.7, ls="--",
                    label=(
                        f"Training Q95 = {high_threshold:.2f} m3/s"
                        if np.isfinite(high_threshold)
                        else "Training Q95 threshold"
                    ),
                ),
                Line2D(
                    [0], [0], color="#7A7A7A", lw=0.6, ls=":",
                    label=(
                        f"Training Q75 = {recession_threshold:.2f} m3/s"
                        if np.isfinite(recession_threshold)
                        else "Training Q75 recession"
                    ),
                ),
            ]
            fig.legend(
                handles=legend_handles,
                loc="upper right",
                ncols=5,
                frameon=False,
                bbox_to_anchor=(0.99, 0.96),
                fontsize=6.3,
            )
            fig.subplots_adjust(
                left=0.08,
                right=0.985,
                bottom=0.075,
                top=0.87,
                hspace=0.52,
                wspace=0.24,
            )
            fig.savefig(output_path, dpi=400, bbox_inches="tight")
            plt.close(fig)
            paths.append(output_path)
    return paths


def plot_flood_event_results(
    result_dir,
    include_station_figures=True,
    period_label=None,
):
    """Read saved event tables and create the complete flood-event figure set."""
    result_dir = Path(result_dir)
    period_label = str(period_label or result_dir.name.title())
    input_paths = {
        "catalog": result_dir / "flood_event_catalog.csv",
        "metrics": result_dir / "flood_event_metrics.csv",
        "daily": result_dir / "discharge_series" / "daily_discharge.csv",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Flood-event plotting inputs are missing: " + ", ".join(missing)
        )

    catalog = pd.read_csv(input_paths["catalog"], encoding="utf-8-sig")
    metrics = pd.read_csv(input_paths["metrics"], encoding="utf-8-sig")
    daily_discharge = pd.read_csv(input_paths["daily"], encoding="utf-8-sig")
    threshold_path = result_dir / "flood_event_thresholds.csv"
    thresholds = (
        pd.read_csv(threshold_path, encoding="utf-8-sig")
        if threshold_path.is_file()
        else pd.DataFrame()
    )
    required_columns = {
        "catalog": {"event_id", "event_type", "event_start", "event_end"},
        "metrics": {
            "event_id", "event_type", "station", "event_nse", "event_kge",
            "peak_relative_error", "peak_timing_error_days",
            "observed_peak_m3_s", "simulated_peak_m3_s", "metrics_valid",
            "station_high_flow_detected", "event_start", "event_end",
        },
        "daily": {
            "date", "station", "observed_m3_s", "simulated_m3_s",
        },
    }
    tables = {
        "catalog": catalog,
        "metrics": metrics,
        "daily": daily_discharge,
    }
    for name, required in required_columns.items():
        absent = sorted(required.difference(tables[name].columns))
        if absent:
            raise ValueError(
                f"{input_paths[name]} is missing columns: {', '.join(absent)}"
            )

    for table in (catalog, metrics):
        table["event_id"] = table["event_id"].astype(str)
        table["event_type"] = table["event_type"].fillna("").astype(str)
    metrics["station"] = metrics["station"].astype(str)
    daily_discharge["station"] = daily_discharge["station"].astype(str)
    if not thresholds.empty:
        required_threshold_columns = {
            "station", "high_flow_threshold_m3_s",
            "recession_threshold_m3_s",
        }
        absent = sorted(required_threshold_columns.difference(thresholds.columns))
        if absent:
            raise ValueError(
                f"{threshold_path} is missing columns: {', '.join(absent)}"
            )
        thresholds["station"] = thresholds["station"].astype(str)
    for column in ["event_start", "event_end", "anchor_peak_date"]:
        if column in catalog:
            catalog[column] = pd.to_datetime(catalog[column], errors="coerce")
    daily_discharge["date"] = pd.to_datetime(
        daily_discharge["date"], errors="coerce"
    )
    numeric_metric_columns = [
        "event_nse", "event_kge", "peak_relative_error",
        "peak_timing_error_days", "observed_peak_m3_s",
        "simulated_peak_m3_s",
    ]
    for column in numeric_metric_columns:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics["event_start"] = pd.to_datetime(
        metrics["event_start"], errors="coerce"
    )
    metrics["event_end"] = pd.to_datetime(
        metrics["event_end"], errors="coerce"
    )
    for column in ["observed_m3_s", "simulated_m3_s"]:
        daily_discharge[column] = pd.to_numeric(
            daily_discharge[column], errors="coerce"
        )
    valid_metrics = (
        metrics["metrics_valid"].eq(True)
        | metrics["metrics_valid"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    )
    metrics["station_high_flow_detected"] = (
        metrics["station_high_flow_detected"].eq(True)
        | metrics["station_high_flow_detected"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    )
    metrics.loc[~valid_metrics, ["event_nse", "event_kge"]] = np.nan
    basin_event_ids = catalog.loc[
        catalog["event_type"].str.lower().isin(
            ["outlet_anchored", "basin", "basin_event", "common"]
        ),
        "event_id",
    ]
    basin_metrics = metrics[metrics["event_id"].isin(basin_event_ids)].copy()

    output_dir = result_dir / "figures" / "flood_events"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = plot_flood_event_performance_overview(
        catalog, metrics, output_dir
    )
    paths.extend([
        plot_flood_peak_scatter(basin_metrics, output_dir),
        plot_flood_event_station_summary(basin_metrics, output_dir),
    ])
    paths.extend(
        plot_flood_event_hydrographs(
            catalog, metrics, daily_discharge, output_dir
        )
    )
    if include_station_figures:
        paths.extend(
            plot_station_flood_event_hydrographs(
                catalog,
                metrics,
                daily_discharge,
                thresholds,
                output_dir / "by_station",
                period_label=period_label,
            )
        )
        top_peaks = select_top_station_peaks(
            daily_discharge,
            thresholds,
            period_label=period_label,
            peak_count=5,
            minimum_separation_days=15,
            half_window_days=7,
        )
        top_peaks.to_csv(
            result_dir / "station_top5_flood_peaks.csv",
            index=False,
            encoding="utf-8-sig",
        )
        paths.extend(
            plot_top_station_peak_hydrographs(
                top_peaks,
                daily_discharge,
                output_dir / "top5_by_station",
                period_label=period_label,
            )
        )
    return paths
