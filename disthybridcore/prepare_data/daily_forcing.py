"""已下载日气象数据的流域裁剪、单位转换和DEM网格对齐。

不下载气象数据，不执行全球合并，也不将日数据伪装成小时数据。
先在原网格裁剪流域缓冲窗口，转换单位，最后双线性重投影到DEM。
发现缺日或缺测格点就报告并停止，不自动补值或借用未来资料。
逐日写入压缩NetCDF，每个强迫变量只有一份文件，原始文件保持只读。
"""

from collections import OrderedDict
from pathlib import Path
import time

import h5netcdf
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr

from .basin_workspace import common_period, file_identity, input_signature, open_source_dataset


def physical_daily_values(values, units, variable):
    """只依据明确单位换算；未知单位报错，不凭量级猜测是否开尔文。"""
    unit = str(units).lower().replace(" ", "").replace("**", "^")
    values = np.asarray(values, dtype=np.float32).copy()
    if not np.isfinite(values).all():
        raise ValueError(f"{variable} 有 {int((~np.isfinite(values)).sum())} 个缺测格点；请检查源数据，不自动补值")
    if variable == "t":
        if unit in {"k", "kelvin"}:
            values -= np.float32(273.15)
        elif unit not in {"°c", "degc", "celsius", "degree_celsius", "degrees_celsius"}:
            raise ValueError(f"气温单位未知：{units!r}")
    elif variable in {"prec", "pet"}:
        if unit not in {"mmd-1", "mmday-1", "mm.day-1", "mm/day", "mm/d", "mmday^-1", "mmd^-1", "mm"}:
            raise ValueError(f"{variable} 需要日累计 mm 或 mm/day，实际单位：{units!r}")
        if variable == "pet":
            # HBV接收非负蒸散需求；按数据处理约定将负PET置零，而非补缺测。
            values = np.maximum(values, np.float32(0.0))
        elif (values < 0).any():
            raise ValueError(f"{variable} 有 {int((values < 0).sum())} 个非法负通量，不静默截断或补值")
    return values


class RegionalDailySourceReader:
    """读取日单文件、GLEAM年文件或ERA5标量time；最多同时打开两份。"""

    def __init__(self, catalog, bounds_lonlat):
        self.catalog = catalog
        self.bounds = bounds_lonlat
        self.datasets = OrderedDict()
        self.source_grid_shape = None
        self.pet_negative_pixels = 0
        self.pet_negative_days = 0
        self.pet_negative_min = None

    def read(self, date):
        path = self.catalog.files_by_day.get(pd.Timestamp(date).normalize())
        if path is None:
            raise FileNotFoundError(f"{self.catalog.dataset} 缺少 {pd.Timestamp(date).date()} 的日数据；不自动补值")
        if path not in self.datasets:
            if len(self.datasets) >= 2:
                _, old = self.datasets.popitem(last=False)
                old.close()
            self.datasets[path] = open_source_dataset(path)
        dataset = self.datasets[path]
        if self.catalog.variable not in dataset:
            raise KeyError(f"{path.name} 缺少变量 {self.catalog.variable}")
        data = dataset[self.catalog.variable]
        time_name = next((name for name in ("time", "valid_time") if name in dataset.variables), None)
        if time_name:
            dates = pd.DatetimeIndex(np.asarray(dataset[time_name]).reshape(-1)).normalize()
            positions = np.flatnonzero(dates == pd.Timestamp(date).normalize())
            if len(positions) != 1:
                raise ValueError(f"{path.name} 时间轴不对应文件日期 {date}")
            if time_name in data.dims:
                data = data.isel({time_name: int(positions[0])})
        else:
            raise ValueError(f"{path.name} 缺少可核对的时间坐标")
        lat_name = next((name for name in ("lat", "latitude") if name in data.dims), None)
        lon_name = next((name for name in ("lon", "longitude") if name in data.dims), None)
        if not lat_name or not lon_name:
            raise ValueError(f"{path.name} 必须有一维经纬度坐标")
        lat = np.asarray(data[lat_name], dtype=np.float64)
        lon = np.asarray(data[lon_name], dtype=np.float64)
        self.source_grid_shape = (len(lat), len(lon))
        west, south, east, north = self.bounds
        if lon.min() >= 0:
            west, east = west % 360, east % 360
            if west > east:
                raise ValueError("当前裁剪不支持跨0°/360°的范围，请先在工程外整理经度轴")
        dy, dx = abs(float(np.median(np.diff(lat)))), abs(float(np.median(np.diff(lon))))
        rows = np.flatnonzero((lat >= south - 2 * dy) & (lat <= north + 2 * dy))
        cols = np.flatnonzero((lon >= west - 2 * dx) & (lon <= east + 2 * dx))
        if len(rows) < 2 or len(cols) < 2:
            raise ValueError(f"{path.name} 不覆盖流域缓冲范围")
        selected = data.isel({lat_name: slice(rows[0], rows[-1]+1), lon_name: slice(cols[0], cols[-1]+1)}).squeeze(drop=True)
        selected = selected.transpose(lat_name, lon_name)
        raw_values = selected.values
        if self.catalog.dataset == "gleam":
            negative = np.isfinite(raw_values) & (raw_values < 0)
            if negative.any():
                self.pet_negative_pixels += int(negative.sum())
                self.pet_negative_days += 1
                minimum = float(raw_values[negative].min())
                self.pet_negative_min = minimum if self.pet_negative_min is None else min(self.pet_negative_min, minimum)
        try:
            values = physical_daily_values(raw_values, data.attrs.get("units", ""),
                                           "prec" if self.catalog.dataset == "mswep" else "pet" if self.catalog.dataset == "gleam" else "t")
        except ValueError as error:
            raise ValueError(f"{path.name} | {pd.Timestamp(date).date()} | {error}") from error
        lat, lon = lat[rows], lon[cols]
        if lat[0] < lat[-1]:
            lat, values = lat[::-1], values[::-1]
        if lon[0] > lon[-1]:
            lon, values = lon[::-1], values[:, ::-1]
        if not np.allclose(abs(np.diff(lat)), dy, atol=1e-5) or not np.allclose(np.diff(lon), dx, atol=1e-5):
            raise ValueError("原始经纬度不是规则网格")
        return values, from_origin(lon[0]-dx/2, lat[0]+dy/2, dx, dy)

    def close(self):
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()


def align_one_forcing(workspace, basin_name, variable, catalog, dem_path, start, end):
    """日块流式写入；保存物理单位与源文件身份，不生成虚假的补值字段。"""
    target = workspace.forcing_grid / f"basin_{basin_name}_{catalog.dataset}_{variable}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.nc"
    with xr.open_dataset(dem_path, engine="h5netcdf") as grid:
        x, y = grid.x.values.copy(), grid.y.values.copy()
        crs_wkt = grid.attrs["crs_wkt"]
        resolution_m = float(grid.attrs["grid_resolution_m"])
        active_cells = int((grid.basin_fraction.values > 0).sum())
    dates = pd.date_range(start, end, freq="D")
    identities = [file_identity(path) for path in sorted(set(catalog.files_by_day.values()))]
    signature = input_signature({"version": 3, "sources": identities, "dem": file_identity(dem_path), "start": start, "end": end})
    if target.exists():
        with xr.open_dataset(target, engine="h5netcdf") as data:
            if data.attrs.get("input_signature") == signature and data.sizes.get("time") == len(dates):
                print(
                    f"[强迫对齐] {variable}: 已匹配 {target.name} | "
                    f"{len(dates)} d | {len(y)}x{len(x)}",
                    flush=True,
                )
                return target
    lonlat = Transformer.from_crs(crs_wkt, 4326, always_xy=True).transform_bounds(x[0]-resolution_m/2, y[0]-resolution_m/2, x[-1]+resolution_m/2, y[-1]+resolution_m/2, densify_pts=21)
    destination_transform = from_origin(x[0]-resolution_m/2, y[-1]+resolution_m/2, resolution_m, resolution_m)
    reader = RegionalDailySourceReader(catalog, lonlat)
    temporary = target.with_suffix(".nc.part")
    clock_start = time.monotonic()
    previous_source_transform = None
    try:
        with h5netcdf.File(temporary, "w") as output:
            output.dimensions = {"time": len(dates), "y": len(y), "x": len(x)}
            for name, values in (("x", x), ("y", y)):
                coordinate = output.create_variable(name, (name,), dtype="f8")
                coordinate[:] = values
                coordinate.attrs.update(units="m", standard_name=f"projection_{name}_coordinate")
            coordinate = output.create_variable("time", ("time",), dtype="i4")
            coordinate[:] = (dates - pd.Timestamp("1970-01-01")).days.to_numpy()
            coordinate.attrs.update(units="days since 1970-01-01", calendar="proleptic_gregorian")
            raster = output.create_variable(variable, ("time", "y", "x"), dtype="f4", compression="gzip", compression_opts=4, chunks=(1, len(y), len(x)), fillvalue=np.nan)
            raster.attrs.update(units="degree_Celsius" if variable == "t" else "mm/day", scale_to_physical_units=1.0, grid_mapping="crs")
            crs = output.create_variable("crs", (), dtype="i4")
            crs[()] = 0
            crs.attrs["crs_wkt"] = crs_wkt
            for index, date in enumerate(dates):
                values, source_transform = reader.read(date)
                if index == 0:
                    source_rows, source_cols = reader.source_grid_shape
                    print(f"[网格 {variable}] 原始={source_rows}x{source_cols} ({source_rows*source_cols:,}格)"
                          f" -> 区域裁剪={values.shape[0]}x{values.shape[1]} ({values.size:,}格)"
                          f" -> 对齐={len(y)}x{len(x)} ({len(y)*len(x):,}格)"
                          f" | 流域有效={active_cells:,}格；均为单日空间网格数", flush=True)
                if previous_source_transform is not None and source_transform != previous_source_transform:
                    raise ValueError(f"{variable} 原始网格随日期变化：{date.date()}，请在工程外统一源网格")
                previous_source_transform = source_transform
                destination = np.full((len(y), len(x)), np.nan, dtype=np.float32)
                reproject(values, destination, src_transform=source_transform, src_crs="EPSG:4326",
                          dst_transform=destination_transform, dst_crs=crs_wkt, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
                if not np.isfinite(destination).all():
                    raise ValueError(f"{variable} 对齐后存在未覆盖格点：{date.date()}")
                raster[index] = destination[::-1]
                if index == 0 or (index+1) % 365 == 0 or index+1 == len(dates):
                    elapsed = time.monotonic() - clock_start
                    eta = elapsed / (index+1) * (len(dates)-index-1)
                    print(f"[对齐 {variable}] {index+1}/{len(dates)} 天 | 已用 {elapsed/60:.1f} min | 剩余约 {eta/60:.1f} min", flush=True)
            output.attrs.update(input_signature=signature, temporal_resolution="daily", crs_wkt=crs_wkt,
                                source_dataset=catalog.dataset, source_variable=catalog.variable,
                                time_start=str(start.date()), time_end_inclusive=str(end.date()),
                                missing_method="No imputation; missing source dates or pixels raise an error",
                                spatial_method="Bilinear reprojection EPSG:4326 to DEM EPSG:3035; no temporal upsampling")
            output.attrs.update(source_grid_rows=reader.source_grid_shape[0], source_grid_cols=reader.source_grid_shape[1],
                                regional_grid_rows=values.shape[0], regional_grid_cols=values.shape[1],
                                aligned_grid_rows=len(y), aligned_grid_cols=len(x), basin_active_cells=active_cells)
            if variable == "pet":
                output.attrs.update(pet_negative_policy="Negative PET set to zero on the cropped source grid before alignment",
                                    pet_negative_pixels_zeroed=reader.pet_negative_pixels,
                                    pet_negative_days=reader.pet_negative_days,
                                    pet_negative_min_before_processing=float(reader.pet_negative_min or 0.0))
        temporary.replace(target)
    finally:
        reader.close()
    return target


def align_daily_forcing(workspace, basin_name, catalogs, dem_path, start=None, end=None):
    available_start, available_end = common_period(catalogs)
    start = available_start if start is None else pd.Timestamp(start).normalize()
    end = available_end if end is None else pd.Timestamp(end).normalize()
    if end < start:
        raise ValueError("Forcing end date must not precede its start date")
    if start < available_start or end > available_end:
        raise ValueError(
            "Requested forcing period "
            f"[{start.date()}, {end.date()}] exceeds the common source coverage "
            f"[{available_start.date()}, {available_end.date()}]"
        )
    print(
        f"[强迫时段] {start.date()} 至 {end.date()}（包含末日）| "
        f"{(end-start).days+1} 天",
        flush=True,
    )
    return {
        variable: align_one_forcing(
            workspace, basin_name, variable, catalog, dem_path, start, end
        )
        for variable, catalog in catalogs.items()
    }
