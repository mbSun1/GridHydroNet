"""地形图共享的地图读取、矢量投影、地形色阶和像元边界工具。"""

from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from disthybridcore.data_pipeline.spatial import hydro_grid_crs, coordinate_factor_to_km


def load_map_arrays(params):
    """读取原始分辨率DEM、汇水区和严格一致的投影坐标。"""
    with xr.open_dataset(params.path_dem, engine="h5netcdf") as data:
        if "x" not in data.coords or "y" not in data.coords:
            raise ValueError("Prepared DEM must contain x/y coordinates")
        x = np.asarray(data.x, dtype=np.float64)
        y = np.asarray(data.y, dtype=np.float64)
        x = x * coordinate_factor_to_km(data.x)
        y = y * coordinate_factor_to_km(data.y)
        elevation = np.asarray(data[params.dem_variable], dtype=np.float32).squeeze()
    with xr.open_dataset(params.path_catchments, engine="h5netcdf") as data:
        catchments = np.asarray(data[params.catchment_variable]) > 0
        catchment_x = np.asarray(data.x, dtype=np.float64)
        catchment_y = np.asarray(data.y, dtype=np.float64)
        catchment_x = catchment_x * coordinate_factor_to_km(data.x)
        catchment_y = catchment_y * coordinate_factor_to_km(data.y)
    if not (
        np.allclose(catchment_x, x, rtol=0.0, atol=1e-4)
        and np.allclose(catchment_y, y, rtol=0.0, atol=1e-4)
    ):
        raise ValueError("Catchments are not co-registered with the DEM")
    basin_mask = catchments.any(axis=0) if catchments.ndim == 3 else catchments
    expected = (len(y), len(x))
    if elevation.shape != expected or basin_mask.shape != expected:
        raise ValueError(
            "Map inputs are not co-registered: "
            f"DEM={elevation.shape}, catchments={basin_mask.shape}, grid={expected}"
        )
    return x, y, elevation, basin_mask


def station_xy(station_coordinates, params):
    """将站点WGS84经纬度转换到准备清单声明的DEM投影。"""
    coordinates = station_coordinates.detach().cpu().numpy()
    target_crs = hydro_grid_crs(params)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    factor = target_crs.axis_info[0].unit_conversion_factor / 1000
    x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return np.asarray(x) * factor, np.asarray(y) * factor


def load_reference_river_lines(path, x, y, params):
    """读取参考河线并重投影到准备清单声明的DEM坐标系。"""
    import shapefile

    if path is None:
        return [], np.empty(0)
    shape_path = Path(path)
    source_crs = CRS.from_wkt(
        shape_path.with_suffix(".prj").read_text(
            encoding="utf-8", errors="replace"
        )
    )
    target_crs = hydro_grid_crs(params)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    factor = target_crs.axis_info[0].unit_conversion_factor / 1000
    margin = 2.0
    bounds = (
        float(x.min() - margin),
        float(x.max() + margin),
        float(y.min() - margin),
        float(y.max() + margin),
    )
    lines = []
    lengths = []
    encoding_path = shape_path.with_suffix(".cpg")
    encoding = encoding_path.read_text(encoding="utf-8").strip() if encoding_path.exists() else "latin1"
    if encoding == "65001":
        encoding = "utf-8"
    reader = shapefile.Reader(str(shape_path), encoding=encoding)
    for item in reader.iterShapeRecords():
        attributes = item.record.as_dict()
        length = float(attributes.get("LAENGE", attributes.get("LENGTH_KM", 0)) or 0)
        if len(item.shape.points) < 2:
            continue
        points = np.asarray(item.shape.points, dtype=np.float64)
        px, py = transformer.transform(points[:, 0], points[:, 1])
        px, py = np.asarray(px) * factor, np.asarray(py) * factor
        if length <= 0:
            length = float(np.hypot(np.diff(px), np.diff(py)).sum())
        threshold = float(params.native_cell_size_km)
        if length < threshold:
            continue
        if (
            px.max() < bounds[0]
            or px.min() > bounds[1]
            or py.max() < bounds[2]
            or py.min() > bounds[3]
        ):
            continue
        parts = list(item.shape.parts) + [len(points)]
        for begin, end in zip(parts[:-1], parts[1:]):
            line = np.column_stack([px[begin:end], py[begin:end]])
            if len(line) >= 2:
                lines.append(line)
                lengths.append(length)
    reader.close()
    return lines, np.asarray(lengths, dtype=np.float64)


def terrain_colormap():
    """返回低饱和度、感知上单调的高程色阶。"""
    import matplotlib as mpl

    return mpl.colors.LinearSegmentedColormap.from_list(
        "disthybridcore_hypsometric_elevation",
        [
            (0.00, "#f2f3c6"),
            (0.18, "#dceba6"),
            (0.38, "#a7d676"),
            (0.56, "#5caf5d"),
            (0.69, "#d2bd5b"),
            (0.82, "#b37736"),
            (0.93, "#7b3e20"),
            (1.00, "#4e2618"),
        ],
    )


def map_extent(x, y):
    """由规则网格中心坐标得到包含完整边缘像元的地图范围。"""
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    return (
        float(x.min() - abs(dx) / 2),
        float(x.max() + abs(dx) / 2),
        float(y.min() - abs(dy) / 2),
        float(y.max() + abs(dy) / 2),
    )
