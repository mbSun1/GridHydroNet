"""从最终模型DEM计算并保存四个地形静态属性及其神经网络输入。

Copernicus原始瓦片先整理到run.py指定的米制分析网格；设置down_factor时，
DEM先执行块平均，再在最终网格计算梯度。文件同时保存物理属性和标准化属性，
供参数编码器的区域化输入与绘图使用。
"""

from pathlib import Path

import numpy as np
from pyproj import CRS
import torch
import xarray as xr

from disthybridcore.data_pipeline.spatial import build_terrain_channels
from .basin_workspace import file_identity, input_signature


def prepare_terrain_static_attributes(workspace, basin_name, topology_path):
    """在与D8完全相同的网格上计算高程、坡度模长、东西/南北梯度。

高程为m，梯度与坡度模长为m/m；slope_degrees仅供解释与地图展示。
模型仍用原来的四通道中位数/MAD标准化，数学含义不改变。
所有坐标保存为m，读取时才转换为路由/绘图使用的km。
"""
    with xr.open_dataset(topology_path, engine="h5netcdf") as topology:
        # 静态高程特征采用当前D8网格上的原始地形dem_original。
        dem = torch.from_numpy(topology.dem_original.values.astype(np.float32, copy=True))
        x = topology.x.values.astype(np.float64) * 1000.0
        y = topology.y.values.astype(np.float64) * 1000.0
        fraction = topology.basin_fraction.values.copy()
        crs_wkt = topology.attrs["crs_wkt"]
    dx_m, dy_m = float(np.median(np.diff(x))), float(np.median(np.diff(y)))
    if not np.isfinite(dem.numpy()).all() or dx_m <= 0 or dy_m <= 0:
        raise ValueError("静态属性需要有限高程与递增的米制坐标")
    if not np.isclose(dx_m, dy_m, atol=1e-6, rtol=1e-6):
        raise ValueError("当前地形特征/D8需要正方形像元")
    signature = input_signature({"version": 1, "topology": file_identity(Path(topology_path)),
                                 "features": ["elevation", "slope_gradient", "dzdx", "dzdy"]})
    target = workspace.terrain / f"basin_{basin_name}_terrain_attributes_{dx_m:g}m_{len(y)}x{len(x)}.nc"
    if target.exists():
        with xr.open_dataset(target, engine="h5netcdf") as saved:
            if saved.attrs.get("input_signature") == signature:
                print(f"[地形静态属性] 已匹配：{target.name}", flush=True)
                return target

    # y由南向北递增，x由西向东递增，因此梯度正负具有明确方向意义。
    dzdy = torch.gradient(dem, spacing=(dy_m,), dim=(0,))[0]
    dzdx = torch.gradient(dem, spacing=(dx_m,), dim=(1,))[0]
    slope = torch.sqrt(dzdx.square() + dzdy.square())
    physical = torch.stack([dem, slope, dzdx, dzdy])
    centers = torch.stack([values.median() for values in physical])
    scales = torch.stack([(values-values.median()).abs().median().clamp_min(1e-6)*1.4826
                          for values in physical])
    features = build_terrain_channels(dem, dx_m / 1000.0)
    data = xr.Dataset({
        "elevation": (("y", "x"), dem.numpy()),
        "slope_gradient": (("y", "x"), slope.numpy()),
        "slope_degrees": (("y", "x"), np.rad2deg(np.arctan(slope.numpy()))),
        "dzdx": (("y", "x"), dzdx.numpy()), "dzdy": (("y", "x"), dzdy.numpy()),
        "basin_fraction": (("y", "x"), fraction),
        "terrain_features": (("feature", "y", "x"), features.numpy()),
        "normalization_center": ("feature", centers.numpy()),
        "normalization_scale": ("feature", scales.numpy()),
        "crs": xr.DataArray(0, attrs=CRS.from_wkt(crs_wkt).to_cf()),
    }, coords={"x": x, "y": y,
               "feature": ["elevation", "slope_gradient", "dzdx", "dzdy"]})
    for name in ("x", "y"):
        data[name].attrs.update(units="m", standard_name=f"projection_{name}_coordinate")
    for name in ("elevation", "slope_gradient", "dzdx", "dzdy", "slope_degrees"):
        data[name].attrs.update(units="m" if name == "elevation" else "degrees" if name == "slope_degrees" else "m/m",
                                grid_mapping="crs")
    data.terrain_features.attrs.update(units="1", grid_mapping="crs",
                                      normalization="(value-median)/(1.4826*max(MAD,1e-6)), clipped to [-8,8]")
    data.attrs.update(input_signature=signature, crs_wkt=crs_wkt,
                      model_cell_size_m=dx_m, source_topology=str(Path(topology_path).resolve()),
                      note="Features computed on unconditioned model-grid DEM; not on original 30m DSM")
    temporary = target.with_suffix(".nc.part")
    data.to_netcdf(temporary, engine="h5netcdf",
                   encoding={name: {"zlib": True, "complevel": 4} for name in data.data_vars if data[name].ndim >= 2})
    data.close()
    temporary.replace(target)
    print(f"[地形静态属性] 已保存：{target.name}", flush=True)
    return target
