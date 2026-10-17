"""在SHP控制范围和指定出口约束下计算填洼、D8、面积累积、河道和层级。

实现使用优先队列填洼和D8算法，并记录填洼改动、出口吸附距离与面积偏差。
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
import xarray as xr

from disthybridcore.data_pipeline.routing_topology import (
    priority_flood, d8_from_conditioned, compute_flow_accumulation,
    routing_levels, compute_stream_order,
)
from .basin_workspace import file_identity, input_signature


def analyze_terrain(workspace, basin_name, dem_path, vectors, down_factor, channel_threshold_km2):
    """每次分辨率改变重新推导 D8，不对方向编码或下游索引求平均。"""
    with xr.open_dataset(dem_path, engine="h5netcdf") as source:
        dem = source.dem.values.copy()
        fraction = source.basin_fraction.values.copy()
        x, y = source.x.values.copy(), source.y.values.copy()
        crs_wkt = source.attrs["crs_wkt"]
    factor = int(down_factor)
    signature = input_signature({"dem": file_identity(dem_path), "outlet": vectors["outlet"],
                                 "factor": factor, "threshold": channel_threshold_km2, "schema": 1})
    shape = (dem.shape[0] // factor, dem.shape[1] // factor)
    path = workspace.terrain / f"basin_{basin_name}_d8_routing_topology_{shape[0]}x{shape[1]}.nc"
    if path.exists():
        with xr.open_dataset(path, engine="h5netcdf") as cached:
            if cached.attrs.get("input_signature") == signature:
                print(f"[水文分析] 已匹配：{path.name}", flush=True)
                return path
    if factor > 1:
        h, w = dem.shape
        dem = dem.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        fraction = fraction.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
        x, y = x.reshape(-1, factor).mean(axis=1), y.reshape(-1, factor).mean(axis=1)
    spacing_km = float(np.diff(x).mean()) / 1000.0
    mask = fraction > 0
    # 出口测站吸附到距离最近的流域边界格。
    boundary_cells = mask & ~binary_erosion(mask, structure=np.ones((3, 3)))
    rows, cols = np.where(boundary_cells)
    station_x, station_y = vectors["outlet"]["geometry"]["coordinates"]
    distances_m = np.hypot(x[cols] - station_x, y[rows] - station_y)
    selected = int(np.argmin(distances_m))
    if distances_m[selected] > 2 * np.sqrt(2) * spacing_km * 1000:
        raise ValueError("出口距流域边界超过两个对角格距：请核对边界与出口是否对应")
    outlet_row, outlet_col = int(rows[selected]), int(cols[selected])
    seeds = np.zeros(mask.shape, dtype=bool)
    seeds[outlet_row, outlet_col] = True
    conditioned, parent = priority_flood(dem, mask, seeds)
    downstream, direction, slope, length = d8_from_conditioned(conditioned, mask, parent, seeds, spacing_km)
    accumulation = compute_flow_accumulation(downstream, mask, fraction * spacing_km**2)
    levels = routing_levels(downstream, mask)
    channel = mask & (accumulation >= channel_threshold_km2)
    channel[outlet_row, outlet_col] = True
    order = compute_stream_order(downstream, channel)
    outlet_index = outlet_row * mask.shape[1] + outlet_col
    grid_values = {
        "dem_original": dem, "dem_conditioned": conditioned,
        "basin_mask": mask.astype(np.int8), "basin_fraction": fraction,
        "downstream_index": downstream, "flow_direction": direction,
        "flow_accumulation": accumulation, "stream_order": order,
        "slope": slope, "flow_distance_km": length,
        "channel_mask": channel.astype(np.int8), "routing_level": levels,
    }
    data = xr.Dataset({key: (("y", "x"), value) for key, value in grid_values.items()},
                      coords={"x": x / 1000, "y": y / 1000, "station": [0]})
    data["station_index"] = ("station", np.asarray([outlet_index], dtype=np.int64))
    data["station_snap_distance_km"] = ("station", np.asarray([distances_m[selected] / 1000], dtype=np.float32))
    data["station_area_relative_error"] = ("station", np.asarray([0], dtype=np.float32))
    area_km2 = float(fraction.sum()) * spacing_km**2
    official_area = float(vectors["outlet"]["properties"].get("area_gov", area_km2))
    adjustment = conditioned[mask].astype(np.float64) - dem[mask]
    data.x.attrs.update(units="km", standard_name="projection_x_coordinate")
    data.y.attrs.update(units="km", standard_name="projection_y_coordinate")
    data.attrs.update(
        input_signature=signature, crs_wkt=crs_wkt, basin=basin_name, outlet_id=vectors["outlet_id"],
        outlet_row=outlet_row, outlet_col=outlet_col,
        channel_area_threshold_km2=float(channel_threshold_km2),
        basin_area_km2=area_km2, official_area_km2=official_area,
        official_area_relative_error=(area_km2 - official_area) / official_area,
        maximum_fill_depth_m=float(adjustment.max()), median_fill_depth_m=float(np.median(adjustment)),
        terrain_method="Boundary-constrained priority flood to the specified gauge; steepest D8",
        flow_direction_codes="0:S,1:SE,2:E,3:NE,4:N,5:NW,6:W,7:SW (y increases north); outlet uses self-link",
    )
    temporary = path.with_suffix(".nc.part")
    data.to_netcdf(temporary, engine="h5netcdf")
    temporary.replace(path)
    data.close()
    print(f"[水文分析] 出口格=({outlet_row},{outlet_col}) | 面积={area_km2:.2f} km2 | 层级={levels.max()+1}", flush=True)
    return path
