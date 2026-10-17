"""采用参考地形检查图的直观布局：真实栅格、黑色边界、坐标、右侧色棒。

沿用参考程序的分图、row/col定位和静态叠加思想，但不复制全局路径、
错误投影或吞掉异常的代码。分析网格仍为EPSG:3035，不为绘图改成经纬度；
图中的X/Y为投影距离km。DEM、D8、上游面积和河网分别展示，只输出JPG。
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr

from disthybridcore.data_pipeline.spatial import hydro_grid_crs
from disthybridcore.prepare_data.basin_workspace import read_vector
from disthybridcore.utils.spatial_maps import (
    load_map_arrays,
    load_reference_river_lines,
    map_extent,
    station_xy,
    terrain_colormap,
)


def boundary_segments(params):
    """优先描绘真实SHP边界，而非把粗网格mask当作精确矢量边界。"""
    path = getattr(params, "path_basin_boundary", None)
    if not path:
        return []
    target = hydro_grid_crs(params)
    factor = target.axis_info[0].unit_conversion_factor / 1000
    segments = []
    for feature in read_vector(Path(path), target):
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            segments.extend(np.asarray(ring) * factor for ring in polygon)
    return segments


def generate_dark_station_colors(station_count):
    """Generate deterministic, distinct dark colors for any number of stations."""
    if station_count < 0:
        raise ValueError("station_count cannot be negative")
    if station_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    station_numbers = np.arange(station_count, dtype=np.float64)
    golden_angle_fraction = 0.6180339887498949
    hues = np.mod(0.57 + station_numbers * golden_angle_fraction, 1.0)
    saturations = 0.72 + 0.08 * (station_numbers % 3.0) / 2.0
    values = 0.48 + 0.10 * (station_numbers % 2.0)
    return mpl.colors.hsv_to_rgb(
        np.column_stack((hues, saturations, values))
    )


def plot_terrain_preparation_maps(params, topology, output_dir, river=None):
    """以白底、离散网格、分层要素和明确单位绘制地形预处理图。"""
    plt.switch_backend("Agg")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = topology.x.numpy(), topology.y.numpy()
    mask = topology.basin_mask.numpy().astype(bool)
    channel = topology.channel_mask.numpy().astype(bool)
    _, w = mask.shape
    boundary = boundary_segments(params)
    reference_lines, _ = load_reference_river_lines(
        params.path_river_lines, x, y, params=params,
    )
    dx = float(np.diff(x).mean())
    outlet_x, outlet_y = x[topology.outlet_col], y[topology.outlet_row]
    output = {}

    def context(axis, title):
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
        if boundary:
            axis.add_collection(LineCollection(boundary, colors="#181818", linewidths=1.0, zorder=7))
        else:
            axis.contour(x, y, mask.astype(float), [0.5], colors="black", linewidths=1)
        axis.scatter(outlet_x, outlet_y, marker="*", s=95, c="#b14c38", edgecolors="white", linewidths=0.6, zorder=10)
        axis.set(xlabel="Projected X (km)", ylabel="Projected Y (km)", aspect="equal")
        extent = map_extent(x, y)
        axis.set_xlim(extent[:2])
        axis.set_ylim(extent[2:])
        axis.ticklabel_format(axis="both", style="plain", useOffset=False)
        axis.tick_params(direction="out", width=0.65, length=3)

    def save(fig, name, key):
        path = output_dir / f"{name}.jpg"
        fig.savefig(path, dpi=450)
        plt.close(fig)
        output[key] = path

    with mpl.rc_context({"font.family": "Times New Roman", "font.size": 9,
                         "axes.linewidth": 0.7, "legend.frameon": False}):
        # 1. 高程独立一图，连续色棒不覆盖河网。显示原基准网格，不制造细尺度纹理。
        native_x, native_y, dem, _ = load_map_arrays(params)
        fig, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
        image = axis.imshow(dem, extent=map_extent(native_x, native_y), origin="lower", interpolation="nearest", cmap=terrain_colormap())
        context(axis, f"{params.basin.title()} basin · DEM on the hydro-grid")
        fig.colorbar(image, ax=axis, shrink=0.83, pad=0.025).set_label("Surface elevation (m)")
        axis.legend(handles=[Line2D([], [], color="#181818", label="Basin boundary"),
                             Line2D([], [], marker="*", color="none", markerfacecolor="#b14c38", markersize=9, label="D8 outlet")], loc="upper right", fontsize=8)
        save(fig, "terrain_elevation_map", "terrain")

        # 2. 每个有效格都有唯一D8方向；出口为自链接，故不画虚假方向箭头。
        fig, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
        palette = ListedColormap(["#a8b8c7", "#7e9dba", "#729e98", "#adc2a4", "#cbb984", "#bd986c", "#b394ac", "#aaa4c3"])
        directions = np.ma.masked_where(~mask, topology.flow_direction.numpy())
        image = axis.imshow(directions, extent=map_extent(x, y), origin="lower", interpolation="nearest", cmap=palette, norm=BoundaryNorm(np.arange(-0.5, 8.5), 8))
        rows, cols = np.where(mask)
        sources = rows*w+cols
        targets = topology.downstream_index.numpy()[rows, cols]
        active = targets != sources
        rows, cols, targets = rows[active], cols[active], targets[active]
        dr, dc = np.divmod(targets, w)
        u, v = x[dc]-x[cols], y[dr]-y[rows]
        norm = np.hypot(u, v)
        u, v = u/norm*dx*0.32, v/norm*dx*0.32
        axis.quiver(x[cols]-u/2, y[rows]-v/2, u, v, angles="xy", scale_units="xy", scale=1,
                    color="#303535", width=0.0015, headwidth=3.2, zorder=8)
        context(axis, "D8 flow direction · one arrow per basin cell")
        colorbar = fig.colorbar(image, ax=axis, shrink=0.83, pad=0.025, ticks=range(8))
        colorbar.ax.set_yticklabels(["S", "SE", "E", "NE", "N", "NW", "W", "SW"])
        colorbar.set_label("Downstream direction")
        save(fig, "d8_flow_direction_map", "d8")

        # 3. 上游贡献面积采用真实边界面积，并以km²对数色棒显示。
        fig, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
        accumulation = np.ma.masked_where(~mask, topology.flow_accumulation.numpy())
        image = axis.imshow(accumulation, extent=map_extent(x, y), origin="lower", interpolation="nearest", cmap="cividis", norm=LogNorm(vmin=float(accumulation.min()), vmax=float(accumulation.max())))
        context(axis, "Upstream contributing area")
        fig.colorbar(image, ax=axis, shrink=0.83, pad=0.025).set_label("Contributing area (km²; logarithmic scale)")
        save(fig, "d8_flow_accumulation_map", "flow_accumulation")

        # 4. 真正参与计算的D8河道边，用连续线连接中心；河道已经在
        #    terrain preprocessing 中按 channel_area_threshold_km2 固定。
        fig, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
        axis.imshow(np.ma.masked_where(~mask, np.ones(mask.shape)), extent=map_extent(x, y), origin="lower", cmap=ListedColormap(["#eff0ed"]), interpolation="nearest")
        rows, cols = np.where(channel)
        targets = topology.downstream_index.numpy()[rows, cols]
        source_ids = rows * w + cols
        keep = targets != source_ids
        rows, cols, targets = rows[keep], cols[keep], targets[keep]
        dr, dc = np.divmod(targets, w)
        d8_segments = np.stack((
            np.column_stack((x[cols], y[rows])),
            np.column_stack((x[dc], y[dr])),
        ), axis=1)
        orders = topology.stream_order.numpy()[rows, cols]
        # Keep first-order channel cells visibly blue.  With ``vmin=1`` the
        # lowest order was mapped to the white end of ``Blues`` and appeared
        # to be missing when compared with the orange D8 overlay below.
        maximum_order = max(2, int(orders.max()))
        order_colors = ListedColormap(
            mpl.colormaps["Blues"](
                np.linspace(0.45, 0.95, maximum_order)
            )
        )
        line_image = LineCollection(
            d8_segments,
            cmap=order_colors,
            norm=mpl.colors.BoundaryNorm(
                np.arange(0.5, maximum_order + 1.5), order_colors.N,
            ),
            linewidths=1.4,
            zorder=5,
        )
        line_image.set_array(orders)
        axis.add_collection(line_image)
        threshold = float(topology.channel_area_threshold_km2)
        context(axis, f"D8 channel network · Aₑᴾᵖ ≥ {threshold:g} km²")
        fig.colorbar(line_image, ax=axis, shrink=0.83, pad=0.025, ticks=range(1, int(orders.max())+1)).set_label("Strahler order")
        save(fig, "classified_river_network_map", "classified_river")

        # 5. 同时显示外部参考河线和由固定汇水面积阈值得到的D8河网。
        fig, axis = plt.subplots(figsize=(10.2, 7.2), layout="constrained")
        axis.imshow(np.ma.masked_where(~mask, np.ones(mask.shape)), extent=map_extent(x, y), origin="lower", interpolation="nearest", cmap=ListedColormap(["#eff0ed"]))
        if reference_lines:
            axis.add_collection(LineCollection(reference_lines, colors="#3e789e", linewidths=0.85, zorder=5))
        axis.add_collection(LineCollection(d8_segments, colors="#d95f02", linewidths=1.5, zorder=6))
        context(axis, f"Reference and D8 river networks · Aₑᴾᵖ ≥ {threshold:g} km²")
        handles = [Line2D([], [], color="#3e789e", linewidth=1.2, label="Reference SHP river"),
                   Line2D([], [], color="#d95f02", linewidth=1.5, label=f"D8 channel (Aₑᴾᵖ ≥ {threshold:g} km²)"),
                   Line2D([], [], marker="*", color="none", markerfacecolor="#b14c38", markersize=9, label="D8 outlet")]
        if river is not None:
            sx, sy = station_xy(river.station_coordinates, params)
            outlet = int(river.outlet_index)
            internal = np.asarray([index for index in range(len(sx)) if index != outlet])
            station_colors = [None] * len(sx)
            if internal.size:
                internal_colors = generate_dark_station_colors(internal.size)
                for index, color in zip(internal, internal_colors):
                    station_colors[index] = color
                axis.scatter(
                    sx[internal], sy[internal], marker="^", c=internal_colors,
                    edgecolors="white", linewidths=1.25, s=430, zorder=10,
                )
            outlet_color = "#b2182b"
            station_colors[outlet] = outlet_color
            axis.scatter(
                [sx[outlet]], [sy[outlet]], marker="D", c=outlet_color,
                edgecolors="white", linewidths=1.25, s=320, zorder=11,
            )
            for index, station_id in enumerate(river.station_ids):
                station_number = index + 1
                marker = "D" if index == outlet else "^"
                marker_size = 11.5 if index == outlet else 13.0
                text_offset = 0.0 if index == outlet else -1.5
                label = f"{station_number}  {station_id}"
                if index == outlet:
                    label += "  (Outlet)"
                axis.annotate(
                    str(station_number),
                    (sx[index], sy[index]),
                    xytext=(0, text_offset),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=max(6.0, 9.2 - 0.8 * len(str(station_number))),
                    fontweight="bold",
                    color="white",
                    annotation_clip=False,
                    zorder=12,
                )
                handles.append(Line2D(
                    [], [], marker=marker, color="none",
                    markerfacecolor=station_colors[index],
                    markeredgecolor="#303030", markeredgewidth=0.55,
                    markersize=marker_size, label=label,
                ))
        axis.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=7.8,
            title="Network and gauge key",
            title_fontsize=8.5,
            labelspacing=0.88,
        )
        save(fig, "gauge_river_network_map", "gauge_river")

        # 6. 静态叠加帮助检查分类阈值与边界，不用于证明模型预测精度。
        fig, axis = plt.subplots(figsize=(7.2, 5.7), layout="constrained")
        axis.imshow(np.ma.masked_where(~mask, topology.basin_fraction.numpy()), extent=map_extent(x, y), origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.35, interpolation="nearest")
        axis.imshow(np.ma.masked_where(~channel, np.ones(mask.shape)), extent=map_extent(x, y), origin="lower", cmap=ListedColormap(["#3e789e"]), interpolation="nearest")
        context(axis, "Hydro static overview · basin coverage and channel cells")
        save(fig, "hydro_static_overview", "overview")
    return output


def plot_terrain_static_attributes(params, attributes_path, output_dir):
    """按原始像元展示NC中保存的高程、坡度和两个有符号梯度。"""
    plt.switch_backend("Agg")
    with xr.open_dataset(attributes_path, engine="h5netcdf") as source:
        data = source.load()
    x, y = data.x.values / 1000.0, data.y.values / 1000.0
    mask = data.basin_fraction.values > 0
    boundary = boundary_segments(params)
    panels = (("elevation", "(a) Surface elevation", "m", terrain_colormap()),
              ("slope_degrees", "(b) Terrain slope", "degrees", "YlGn"),
              ("dzdx", "(c) West–east elevation gradient", "m/m", "RdBu_r"),
              ("dzdy", "(d) South–north elevation gradient", "m/m", "RdBu_r"))
    with mpl.rc_context({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout="constrained")
        for axis, (variable, title, unit, palette) in zip(axes.ravel(), panels):
            values = np.ma.masked_where(~mask, data[variable].values)
            limits = {}
            if variable in {"dzdx", "dzdy"}:
                maximum = max(float(np.abs(values).max()), 1e-6)
                limits = {"vmin": -maximum, "vmax": maximum}
            image = axis.imshow(values, extent=map_extent(x, y), origin="lower", interpolation="nearest",
                                cmap=palette, **limits)
            if boundary:
                axis.add_collection(LineCollection(boundary, colors="black", linewidths=0.7))
            axis.set(title=title, xlabel="Projected X (km)", ylabel="Projected Y (km)", aspect="equal")
            axis.ticklabel_format(style="plain", useOffset=False)
            axis.tick_params(labelsize=7, direction="out", length=2)
            fig.colorbar(image, ax=axis, shrink=0.8, pad=0.025).set_label(unit)
        target = Path(output_dir) / "terrain_static_attributes_map.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=450)
        plt.close(fig)
    data.close()
    return target
