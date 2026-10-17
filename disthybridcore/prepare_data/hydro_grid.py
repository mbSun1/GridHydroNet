"""将 Copernicus DEM 投影到 EPSG:3035 方形像元，建立唯一空间模板。"""

from contextlib import ExitStack
from pathlib import Path
import math

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr

from .basin_workspace import geometry_points, file_identity, input_signature


def polygon_parts(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError("边界必须是 Polygon 或 MultiPolygon")


def ring_area(ring):
    points = np.asarray(ring, dtype=np.float64)
    # 使用局地原点提高百万米级投影坐标的面积计算精度。
    points = points - points[0]
    return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))) / 2


def polygon_area(geometry):
    return sum(ring_area(rings[0]) - sum(ring_area(hole) for hole in rings[1:]) for rings in polygon_parts(geometry))


def clip_ring(ring, left, bottom, right, top):
    """Sutherland–Hodgman：将环精确裁剪到矩形像元，用于边界面积比例。"""
    points = [tuple(point[:2]) for point in ring]
    for axis, limit, greater in ((0, left, True), (0, right, False), (1, bottom, True), (1, top, False)):
        if not points:
            break
        output = []
        previous = points[-1]
        previous_inside = previous[axis] >= limit if greater else previous[axis] <= limit
        for current in points:
            inside = current[axis] >= limit if greater else current[axis] <= limit
            if inside != previous_inside:
                ratio = (limit - previous[axis]) / (current[axis] - previous[axis])
                cross = tuple(previous[i] + ratio * (current[i] - previous[i]) for i in range(2))
                output.append(cross)
            if inside:
                output.append(current)
            previous, previous_inside = current, inside
        points = output
    return points


def exact_basin_fraction(geometry, transform, shape):
    """方形米制格网的精确几何相交面积，不用格点计数替代控制面积。"""
    touched = rasterize([(geometry, 1)], out_shape=shape, transform=transform, all_touched=True, dtype="uint8")
    fraction = np.zeros(shape, dtype=np.float64)
    spacing = transform.a
    parts = polygon_parts(geometry)
    for row, col in zip(*np.where(touched)):
        left, top = transform * (col, row)
        area = 0.0
        for rings in parts:
            for index, ring in enumerate(rings):
                clipped = clip_ring(ring, left, top - spacing, left + spacing, top)
                if len(clipped) >= 3:
                    area += ring_area(clipped) * (1 if index == 0 else -1)
        fraction[row, col] = np.clip(area / spacing**2, 0.0, 1.0)
    represented = fraction.sum() * spacing**2
    official = polygon_area(geometry)
    if not np.isclose(represented, official, rtol=1e-5):
        raise ValueError(f"SHP 面积与网格相交面积不闭合：{represented:g}/{official:g} m2")
    return fraction.astype(np.float32)


def prepare_hydro_grid(workspace, basin_name, vectors, tiles, resolution_m, down_factor=1):
    """DEM平均重采样；输出坐标为米且 y 递增；分析窗口可为矩形。"""
    if resolution_m <= 0 or down_factor < 1:
        raise ValueError("grid_resolution_m 和 down_factor 必须为正")
    geometry = vectors["boundary"]["geometry"]
    signature = input_signature({"tiles": [file_identity(path) for path in tiles], "geometry": geometry,
                                 "resolution_m": resolution_m, "down_factor": down_factor, "schema": 2})
    path = workspace.hydro_grid / f"basin_{basin_name}_hydro_grid_epsg3035_{resolution_m:g}m.nc"
    if path.exists():
        with xr.open_dataset(path, engine="h5netcdf") as cached:
            if cached.attrs.get("input_signature") == signature:
                print(f"[DEM基准网格] 已匹配：{path.name}", flush=True)
                return path
    points = geometry_points(geometry)
    left, bottom = np.floor(points.min(axis=0) / resolution_m) * resolution_m - resolution_m
    right, top = np.ceil(points.max(axis=0) / resolution_m) * resolution_m + resolution_m
    width = math.ceil((right - left) / resolution_m / down_factor) * down_factor
    height = math.ceil((top - bottom) / resolution_m / down_factor) * down_factor
    right, bottom = left + width*resolution_m, top - height*resolution_m
    transform = from_origin(left, top, resolution_m, resolution_m)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in tiles]
        # 合并范围限定为流域缓冲窗口。
        from rasterio.warp import transform_bounds
        bounds = transform_bounds("EPSG:3035", sources[0].crs, left - 3000, bottom - 3000, right + 3000, top + 3000, densify_pts=21)
        mosaic, mosaic_transform = merge(sources, bounds=bounds, nodata=-9999.0, dtype="float32")
        elevation = np.full((height, width), np.nan, dtype=np.float32)
        reproject(mosaic[0], elevation, src_transform=mosaic_transform, src_crs=sources[0].crs,
                  src_nodata=-9999.0, dst_transform=transform, dst_crs="EPSG:3035",
                  dst_nodata=np.nan, resampling=Resampling.average)
        regional_mosaic = workspace.dem_processed / f"basin_{basin_name}_copernicus_dem_mosaic.tif"
        with rasterio.open(regional_mosaic.with_suffix(".tif.part"), "w", driver="GTiff", height=mosaic.shape[1],
                           width=mosaic.shape[2], count=1, dtype="float32", crs=sources[0].crs,
                           transform=mosaic_transform, nodata=-9999.0, compress="deflate") as output:
            output.write(mosaic)
        regional_mosaic.with_suffix(".tif.part").replace(regional_mosaic)
    if not np.isfinite(elevation).all():
        raise ValueError("DEM 缓冲窗口覆盖不完整；不能用虚拟高程补齐")
    projected_dem = workspace.dem_processed / f"basin_{basin_name}_dem_epsg3035_{resolution_m:g}m.tif"
    with rasterio.open(projected_dem.with_suffix(".tif.part"), "w", driver="GTiff", height=height, width=width,
                       count=1, dtype="float32", crs="EPSG:3035", transform=transform, nodata=-9999.0, compress="deflate") as output:
        output.write(elevation, 1)
    projected_dem.with_suffix(".tif.part").replace(projected_dem)
    fraction = exact_basin_fraction(geometry, transform, elevation.shape)
    x = left + (np.arange(width) + 0.5) * resolution_m
    y = top - (np.arange(height)[::-1] + 0.5) * resolution_m
    data = xr.Dataset({"dem": (("y", "x"), elevation[::-1]),
                       "basin_fraction": (("y", "x"), fraction[::-1]),
                       "crs": ((), 0)}, coords={"x": x, "y": y})
    data.x.attrs.update(units="m", standard_name="projection_x_coordinate")
    data.y.attrs.update(units="m", standard_name="projection_y_coordinate")
    data.dem.attrs.update(units="m", grid_mapping="crs", long_name="Copernicus GLO-30 surface elevation, averaged onto analysis grid")
    data.basin_fraction.attrs.update(units="1", long_name="Exact polygon/cell intersection area fraction")
    from pyproj import CRS
    data.crs.attrs.update(CRS.from_epsg(3035).to_cf())
    data.attrs.update(input_signature=signature, crs_wkt=data.crs.attrs["crs_wkt"], grid_resolution_m=float(resolution_m),
                      polygon_area_km2=polygon_area(geometry) / 1e6, source="Copernicus DEM GLO-30 (DSM, not guaranteed bare-earth DTM)")
    temporary = path.with_suffix(".nc.part")
    data.to_netcdf(temporary, engine="h5netcdf")
    temporary.replace(path)
    data.close()
    print(f"[DEM基准网格] {height}x{width} | {resolution_m:g} m | EPSG:3035", flush=True)
    return path
