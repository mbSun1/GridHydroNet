"""Publication-oriented JPG figures for whole-basin gauge observations."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from disthybridcore.prepare_data.basin_workspace import read_vector
from disthybridcore.utils.spatial_maps import (
    load_map_arrays,
    load_reference_river_lines,
    map_extent,
)


def manuscript_style() -> dict:
    """Return the shared, editable-text-compatible manuscript plot style."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }


def branch_colors(metadata: pd.DataFrame) -> dict[str, str]:
    """Assign a restrained and deterministic color to every major branch."""
    palette = [
        "#4C78A8",
        "#E39C37",
        "#5A9A78",
        "#9C78B4",
        "#6F8A91",
        "#C7776B",
    ]
    branch_rows = metadata.loc[~metadata.is_outlet].drop_duplicates("branch_id")
    branch_rows = branch_rows.sort_values("branch_id")
    colors = {
        str(row.branch_id): palette[index % len(palette)]
        for index, row in enumerate(branch_rows.itertuples(index=False))
    }
    outlet_rows = metadata.loc[metadata.is_outlet]
    if not outlet_rows.empty:
        colors[str(outlet_rows.iloc[0].branch_id)] = "#B33B32"
    return colors


def boundary_line_segments(path: str | Path) -> list[np.ndarray]:
    """Convert the basin polygon rings to projected-kilometre line segments."""
    segments = []
    for feature in read_vector(Path(path)):
        geometry = feature["geometry"]
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon in polygons:
            segments.extend(
                np.asarray(ring, dtype=np.float64) / 1000.0 for ring in polygon
            )
    return segments


def add_panel_label(axis, label: str) -> None:
    """Place a consistent panel letter just outside the upper-left corner."""
    axis.text(
        -0.12,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_jpg(fig, path: str | Path) -> Path:
    """Save one high-resolution JPG and close its Matplotlib figure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    return output


def plot_station_map(
    axis,
    params,
    metadata: pd.DataFrame,
    river,
    colors: dict[str, str],
) -> None:
    """Draw reference rivers, station topology, drainage-area symbols and IDs."""
    x, y, _, mask = load_map_arrays(params)
    extent = map_extent(x, y)
    axis.imshow(
        np.ma.masked_where(~mask, np.ones(mask.shape)),
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap=ListedColormap(["#F4F4F1"]),
        zorder=0,
    )
    river_lines, _ = load_reference_river_lines(
        params.path_river_lines, x, y, params=params
    )
    if river_lines:
        axis.add_collection(
            LineCollection(
                river_lines,
                colors="#8BA9B7",
                linewidths=0.45,
                alpha=0.8,
                zorder=2,
            )
        )
    boundary = boundary_line_segments(params.path_basin_boundary)
    axis.add_collection(
        LineCollection(boundary, colors="#222222", linewidths=0.85, zorder=5)
    )

    ids = list(river.station_ids)
    adjacency = river.adjacency.detach().cpu().numpy() > 0
    for source, target in np.argwhere(adjacency):
        source_id, target_id = ids[int(source)], ids[int(target)]
        source_row, target_row = metadata.loc[source_id], metadata.loc[target_id]
        axis.annotate(
            "",
            xy=(target_row.projected_x_km, target_row.projected_y_km),
            xytext=(source_row.projected_x_km, source_row.projected_y_km),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#555555",
                "linewidth": 0.65,
                "linestyle": "--",
                "alpha": 0.8,
                "mutation_scale": 6,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=4,
        )

    area = metadata.official_area_km2.to_numpy(dtype=float)
    sizes = 18.0 + 70.0 * np.sqrt(area / np.nanmax(area))
    label_offsets = ((4, 5), (4, -7), (-4, 8), (-4, -10))
    for position, (station_id, row) in enumerate(metadata.iterrows()):
        color = colors[str(row.branch_id)]
        marker = "D" if bool(row.is_outlet) else "o"
        axis.scatter(
            row.projected_x_km,
            row.projected_y_km,
            s=sizes[position],
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=7,
        )
        x_offset, y_offset = label_offsets[position % len(label_offsets)]
        if row.projected_x_km > np.mean(extent[:2]):
            x_offset *= -1
        axis.annotate(
            station_id,
            (row.projected_x_km, row.projected_y_km),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom" if y_offset > 0 else "top",
            fontsize=4.9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=8,
        )

    legend_handles = []
    for branch_id, branch_frame in metadata.groupby("branch_id", sort=True):
        label = str(branch_frame.branch_name.iloc[0])
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="D" if bool(branch_frame.is_outlet.all()) else "o",
                color="none",
                markerfacecolor=colors[str(branch_id)],
                markeredgecolor="white",
                markersize=5,
                label=label,
            )
        )
    legend_handles.append(
        Line2D(
            [],
            [],
            color="#555555",
            linestyle="--",
            linewidth=0.7,
            marker=">",
            markersize=3,
            label="Observed-gauge routing",
        )
    )
    axis.legend(handles=legend_handles, loc="upper right", fontsize=5.2)
    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])
    axis.set_aspect("equal")
    axis.set_xlabel("Projected X (km)")
    axis.set_ylabel("Projected Y (km)")
    axis.set_title("Gauge network and downstream topology", loc="left", fontweight="bold")
    axis.ticklabel_format(axis="both", style="plain", useOffset=False)


def plot_area_validation(
    axis,
    summary: pd.DataFrame,
    area_audit: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    """Compare official and D8 contributing areas against the acceptance band."""
    minimum = float(
        min(summary.official_area_km2.min(), summary.d8_area_km2.min()) * 0.75
    )
    maximum = float(
        max(summary.official_area_km2.max(), summary.d8_area_km2.max()) * 1.3
    )
    reference = np.geomspace(minimum, maximum, 200)
    axis.fill_between(
        reference,
        reference * 0.85,
        reference * 1.15,
        color="#DDE6E1",
        alpha=0.85,
        label="±15% acceptance band",
        zorder=0,
    )
    axis.plot(reference, reference, color="#333333", linewidth=0.8, label="1:1")
    for row in summary.itertuples(index=False):
        axis.scatter(
            row.official_area_km2,
            row.d8_area_km2,
            s=20,
            color=colors[str(row.branch_id)],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    if not area_audit.empty and "status" in area_audit:
        excluded = area_audit[area_audit.status.astype(str).str.startswith("excluded")]
        for excluded_index, row in enumerate(excluded.itertuples(index=False)):
            axis.scatter(
                row.official_area_km2,
                row.d8_area_km2,
                marker="x",
                s=30,
                color="#B33B32",
                linewidth=1.0,
                zorder=4,
                label="Excluded gauge" if excluded_index == 0 else None,
            )
            axis.annotate(
                str(row.gauge_id),
                (row.official_area_km2, row.d8_area_km2),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=5.0,
                color="#8F2D28",
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_xlabel(r"Official drainage area (km$^2$)")
    axis.set_ylabel(r"D8 contributing area (km$^2$)")
    axis.set_title("Drainage-area consistency", loc="left", fontweight="bold")
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="lower right", fontsize=5.0)
    axis.grid(which="both", color="#E6E6E6", linewidth=0.4, zorder=-1)


def plot_observation_coverage(axis, observations: pd.DataFrame, params) -> None:
    """Show daily station availability and the fixed train-test boundary."""
    validity = observations.notna().to_numpy(dtype=np.uint8).T
    start_number = mdates.date2num(observations.index[0])
    end_number = mdates.date2num(observations.index[-1] + pd.Timedelta(days=1))
    axis.imshow(
        validity,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(start_number, end_number, len(observations.columns) - 0.5, -0.5),
        cmap=ListedColormap(["#E6E6E6", "#4C78A8"]),
        vmin=0,
        vmax=1,
    )
    test_start = pd.Timestamp(params.test_start)
    axis.axvline(test_start, color="#B33B32", linewidth=0.9, linestyle="--")
    axis.text(
        test_start,
        -0.8,
        "Test",
        color="#B33B32",
        fontsize=5.3,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    axis.set_yticks(np.arange(len(observations.columns)))
    axis.set_yticklabels(observations.columns)
    axis.xaxis_date()
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlabel("Date")
    axis.set_ylabel("Gauge ID")
    axis.set_title("Daily observation availability", loc="left", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(True)


def plot_station_network_quality(
    params,
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    river,
    summary: pd.DataFrame,
    area_audit: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create the study-network and observation-quality manuscript figure."""
    colors = branch_colors(metadata)
    with mpl.rc_context(manuscript_style()):
        fig = plt.figure(figsize=(7.2, 5.5), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.4, 1.0), height_ratios=(1, 1))
        map_axis = fig.add_subplot(grid[:, 0])
        area_axis = fig.add_subplot(grid[0, 1])
        coverage_axis = fig.add_subplot(grid[1, 1])
        plot_station_map(map_axis, params, metadata, river, colors)
        plot_area_validation(area_axis, summary, area_audit, colors)
        plot_observation_coverage(coverage_axis, observations, params)
        add_panel_label(map_axis, "a")
        add_panel_label(area_axis, "b")
        add_panel_label(coverage_axis, "c")
        return save_jpg(fig, output_path)


def station_plot_order(metadata: pd.DataFrame) -> list[str]:
    """Order gauges by main branch and then increasing drainage area."""
    table = metadata.reset_index(drop=True)
    internal = table.loc[~table.is_outlet].sort_values(
        ["branch_name", "official_area_km2", "station_id"]
    )
    outlet = table.loc[table.is_outlet].sort_values("station_id")
    return internal.station_id.tolist() + outlet.station_id.tolist()


def plot_correlation_matrix(
    axis,
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    river,
) -> None:
    """Plot topologically grouped Spearman correlations and direct connections."""
    order = station_plot_order(metadata)
    correlations = observations[order].corr(method="spearman").to_numpy()
    image = axis.imshow(correlations, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    axis.set_xticks(np.arange(len(order)))
    axis.set_yticks(np.arange(len(order)))
    axis.set_xticklabels(order, rotation=90)
    axis.set_yticklabels(order)
    axis.tick_params(length=0, labelsize=4.6)
    for row in range(len(order)):
        for column in range(len(order)):
            value = correlations[row, column]
            color = "white" if abs(value) >= 0.65 else "#222222"
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=3.5,
                color=color,
            )
    index_by_id = {station_id: index for index, station_id in enumerate(order)}
    adjacency = river.adjacency.detach().cpu().numpy() > 0
    for source, target in np.argwhere(adjacency):
        source_id = river.station_ids[int(source)]
        target_id = river.station_ids[int(target)]
        row, column = index_by_id[source_id], index_by_id[target_id]
        for first, second in ((row, column), (column, row)):
            axis.add_patch(
                mpl.patches.Rectangle(
                    (second - 0.49, first - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=0.65,
                )
            )
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("Spearman correlation")
    axis.set_title(
        "Pairwise hydrograph similarity\n(boxes: directly connected gauges)",
        loc="left",
        fontweight="bold",
    )


def log_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit log10(y) against log10(x) and return predictions, slope and R squared."""
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    coefficients = np.polyfit(np.log10(x[valid]), np.log10(y[valid]), 1)
    prediction = np.polyval(coefficients, np.log10(x[valid]))
    residual = np.log10(y[valid]) - prediction
    denominator = np.square(np.log10(y[valid]) - np.log10(y[valid]).mean()).sum()
    r_squared = 1.0 - np.square(residual).sum() / denominator if denominator > 0 else np.nan
    return coefficients, float(coefficients[0]), float(r_squared)


def plot_flow_area_scaling(
    axis,
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    """Show how mean and high flow scale with official drainage area."""
    area = summary.official_area_km2.to_numpy(dtype=float)
    mean = summary.mean_discharge_m3_s.to_numpy(dtype=float)
    high = summary.p95_discharge_m3_s.to_numpy(dtype=float)
    for row in summary.itertuples(index=False):
        branch_id = str(metadata.loc[row.station_id, "branch_id"])
        color = colors[branch_id]
        axis.scatter(
            row.official_area_km2,
            row.mean_discharge_m3_s,
            marker="o",
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axis.scatter(
            row.official_area_km2,
            row.p95_discharge_m3_s,
            marker="^",
            s=20,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    x_line = np.geomspace(area.min(), area.max(), 100)
    for values, color, label, style in (
        (mean, "#333333", "Mean flow", "-"),
        (high, "#777777", "95th percentile", "--"),
    ):
        coefficients, slope, r_squared = log_regression(area, values)
        y_line = 10 ** np.polyval(coefficients, np.log10(x_line))
        axis.plot(
            x_line,
            y_line,
            color=color,
            linestyle=style,
            linewidth=0.9,
            label=f"{label}: slope={slope:.2f}, $R^2$={r_squared:.2f}",
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Official drainage area (km$^2$)")
    axis.set_ylabel(r"Observed discharge (m$^3$ s$^{-1}$)")
    axis.set_title("Flow magnitude increases with drainage area", loc="left", fontweight="bold")
    axis.grid(which="both", color="#E8E8E8", linewidth=0.4)
    shape_handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor="#777777", markersize=4, label="Mean"),
        Line2D([], [], marker="^", color="none", markerfacecolor="#777777", markersize=4, label="95th percentile"),
    ]
    fit_handles, fit_labels = axis.get_legend_handles_labels()
    axis.legend(shape_handles + fit_handles, [value.get_label() for value in shape_handles] + fit_labels, loc="upper left", fontsize=4.8)


def plot_connected_lags(
    axis,
    pairs: pd.DataFrame,
    metadata: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    """Show daily downstream lags and their maximum correlations for direct edges."""
    direct = pairs[pairs.directly_connected].copy()
    direct["source_area"] = direct.source_id.map(metadata.official_area_km2)
    direct = direct.sort_values(["source_branch", "source_area", "source_id"])
    positions = np.arange(len(direct))
    labels = [
        f"{row.source_id}→{row.target_id}"
        for row in direct.itertuples(index=False)
    ]
    axis.axvline(0, color="#777777", linewidth=0.7, linestyle="--")
    for position, row in zip(positions, direct.itertuples(index=False)):
        branch_id = str(metadata.loc[row.source_id, "branch_id"])
        color = colors[branch_id]
        axis.plot(
            [0, row.best_target_lag_days],
            [position, position],
            color=color,
            linewidth=0.8,
            alpha=0.75,
        )
        axis.scatter(
            row.best_target_lag_days,
            position,
            s=12 + 28 * max(float(row.best_lag_pearson), 0.0) ** 2,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        axis.text(
            row.best_target_lag_days + (0.18 if row.best_target_lag_days >= 0 else -0.18),
            position,
            f"r={row.best_lag_pearson:.2f}",
            ha="left" if row.best_target_lag_days >= 0 else "right",
            va="center",
            fontsize=4.3,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=4.5)
    axis.set_ylim(len(direct) - 0.3, -0.7)
    axis.set_xlim(-7.5, 7.5)
    axis.set_xticks(np.arange(-6, 7, 2))
    axis.set_xlabel("Best target/downstream lag (days)")
    axis.set_title("Directly connected gauge propagation", loc="left", fontweight="bold")
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.4)


def plot_station_similarity(
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    river,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Create the whole-basin observation-similarity manuscript figure."""
    colors = branch_colors(metadata)
    with mpl.rc_context(manuscript_style()):
        fig = plt.figure(figsize=(7.2, 5.2), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), height_ratios=(1, 1))
        correlation_axis = fig.add_subplot(grid[:, 0])
        scaling_axis = fig.add_subplot(grid[0, 1])
        lag_axis = fig.add_subplot(grid[1, 1])
        plot_correlation_matrix(correlation_axis, observations, metadata, river)
        plot_flow_area_scaling(scaling_axis, summary, metadata, colors)
        plot_connected_lags(lag_axis, pairs, metadata, colors)
        add_panel_label(correlation_axis, "a")
        add_panel_label(scaling_axis, "b")
        add_panel_label(lag_axis, "c")
        return save_jpg(fig, output_path)


def plot_all_station_hydrographs(
    observations: pd.DataFrame,
    metadata: pd.DataFrame,
    output_path: str | Path,
    *,
    normalize_by_area: bool,
) -> Path:
    """Create traceable small multiples for every retained gauge."""
    order = station_plot_order(metadata)
    colors = branch_colors(metadata)
    columns = 3
    rows = int(np.ceil(len(order) / columns))
    with mpl.rc_context(manuscript_style()):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(7.2, 1.42 * rows),
            sharex=True,
            sharey=normalize_by_area,
            layout="constrained",
        )
        axes = np.asarray(axes).reshape(-1)
        plotted_values = []
        for station_id in order:
            values = observations[station_id].to_numpy(dtype=float)
            if normalize_by_area:
                values = values * 86.4 / float(metadata.loc[station_id, "official_area_km2"])
            plotted_values.append(values)
        if normalize_by_area:
            finite = np.concatenate([value[np.isfinite(value)] for value in plotted_values])
            shared_maximum = float(np.nanpercentile(finite, 99.9) * 1.08)
        else:
            shared_maximum = np.nan

        for axis, station_id, values in zip(axes, order, plotted_values):
            branch_id = str(metadata.loc[station_id, "branch_id"])
            axis.plot(
                observations.index,
                values,
                color=colors[branch_id],
                linewidth=0.65,
            )
            axis.set_title(
                f"{station_id} | {metadata.loc[station_id, 'station_name']}",
                loc="left",
                fontsize=5.6,
                fontweight="bold",
                pad=2,
            )
            axis.grid(axis="y", color="#E8E8E8", linewidth=0.35)
            axis.set_ylim(bottom=0, top=shared_maximum if normalize_by_area else None)
            axis.xaxis.set_major_locator(mdates.YearLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axis.tick_params(axis="x", rotation=0)
            axis.margins(x=0.005)
        for axis in axes[len(order):]:
            axis.set_visible(False)
        unit = (
            r"Area-normalized runoff (mm day$^{-1}$)"
            if normalize_by_area
            else r"Observed discharge (m$^3$ s$^{-1}$)"
        )
        fig.supylabel(unit, fontsize=6.8)
        fig.supxlabel("Date", fontsize=6.8)
        title = (
            "All retained gauges: area-normalized observed runoff"
            if normalize_by_area
            else "All retained gauges: observed discharge"
        )
        fig.suptitle(title, fontsize=8.2, fontweight="bold")
        return save_jpg(fig, output_path)
