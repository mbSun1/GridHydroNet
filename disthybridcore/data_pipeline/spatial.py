"""DEM基准网格、空间降尺度和静态地形特征构建。

整理后的流域DEM定义空间数据的投影、范围、分辨率和``x/y``坐标。
降水、气温、PET、汇水区和河网在数据准备阶段对齐该DEM，训练阶段直接
读取对齐结果。

本模块负责：

1. 检查DEM是规则、方形像元且物理距离明确的投影网格；
2. 按 ``down_factor`` 对连续变量进行守恒的块平均降尺度；
3. 读取准备阶段保存的高程、坡度和两个方向梯度；
4. 对不同量纲的静态属性进行稳健标准化；
5. 根据DEM真实格距生成模型单元面积。
"""

import numpy as np
import torch
import xarray as xr
from pyproj import CRS
from torch.nn import functional as F


def average_downsample(data: torch.Tensor, factor: int) -> torch.Tensor:
    """对最后两个空间维进行不重叠块平均降尺度。

    Parameters
    ----------
    data
        任意前导维度、最后两维为 ``[height, width]`` 的张量。
    factor
        每个模型栅格在单方向上包含的原始栅格数。例如原始网格为
        200×200，``factor=5`` 时输出为 40×40。

    Returns
    -------
    torch.Tensor
        前导维度保持不变、空间边长缩小 ``factor`` 倍的张量。

    对降水、PET和高程等连续场采用面积块平均。对于子流域0/1掩膜，块平均
    的结果表示粗网格被该流域覆盖的面积比例。
    """
    if factor == 1:
        return data
    if data.shape[-2] % factor or data.shape[-1] % factor:
        raise ValueError(f"Spatial shape {data.shape[-2:]} is not divisible by {factor}")
    # avg_pool2d 只接受批次和通道形式。先把全部前导维展平为一个批次维，
    # 完成降尺度后再恢复原有前导维度。
    leading = data.shape[:-2]
    flat = data.reshape(-1, 1, *data.shape[-2:])
    pooled = F.avg_pool2d(flat, factor, factor)
    return pooled.reshape(*leading, *pooled.shape[-2:])


def robust_standardize(values: torch.Tensor) -> torch.Tensor:
    """使用中位数和 MAD 对一个静态栅格进行稳健标准化。

    中心取所有有限值的中位数，尺度取
    ``1.4826 * median(abs(x - median(x)))``。系数 1.4826 使正态分布下的
    MAD 与标准差具有相同量级。输出截断到 [-8, 8]，避免局部异常高程或
    坡度向神经网络提供过大的输入。

    缺测位置由有效值中位数填充，因此标准化后对应0。
    """
    finite = torch.isfinite(values)
    if not finite.any():
        raise ValueError("Static grid contains no finite values")
    valid = values[finite]
    center = valid.median()
    scale = (valid - center).abs().median().clamp_min(1e-6) * 1.4826
    return ((values.nan_to_num(center.item()) - center) / scale).clamp(-8.0, 8.0)


def hydro_grid_crs(params) -> CRS:
    """读取准备清单记录的权威网格投影，不猜测缺失的CRS。"""

    value = getattr(params, "grid_crs_wkt", None)
    if not value:
        raise ValueError("Prepared basin manifest must provide grid_crs_wkt")
    return CRS.from_wkt(value)


def coordinate_factor_to_km(coordinate) -> float:
    """文件坐标可为 m 或 km；模型路由/绘图内部距离统一为 km。"""
    units = str(coordinate.attrs.get("units", "km")).lower()
    if units in {"m", "metre", "meter", "metres", "meters"}:
        return 0.001
    if units in {"km", "kilometre", "kilometer"}:
        return 1.0
    raise ValueError(f"网格投影坐标单位必须为m或km，而不是 {units!r}")


def regular_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    """验证一维投影坐标规则递增，并返回绝对格距(km)。"""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"DEM {name} coordinate must be a finite one-dimensional axis")
    differences = np.diff(values)
    if np.all(differences < 0):
        values = values[::-1]
        differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError(f"DEM {name} coordinate must be strictly monotonic")
    spacing = float(np.median(differences))
    # 部分NetCDF坐标以float32存储；转换为float64后仍保留毫米级量化误差。
    # 1e-5 km即1 cm，远小于1-km像元，同时足以区分真正的不规则格点。
    tolerance_km = max(1e-5, abs(spacing) * 1e-5)
    if not np.allclose(differences, spacing, rtol=0.0, atol=tolerance_km):
        raise ValueError(f"DEM {name} coordinate is not regularly spaced")
    return values, spacing


def build_terrain_channels(dem_m: torch.Tensor, cell_size_km: float) -> torch.Tensor:
    """从 DEM 构造四个逐栅格静态特征通道。

    通道依次为：标准化高程、标准化坡度模长、标准化 x 方向高程梯度、
    标准化 y 方向高程梯度。梯度以实际栅格间距（m）计算，因此坡度及两个
    方向梯度均为无量纲高程变化率。
    """
    spacing = cell_size_km * 1000.0
    # torch.gradient 分别沿行方向(y)和列方向(x)计算中心差分梯度。
    dy = torch.gradient(dem_m, spacing=(spacing,), dim=(0,))[0]
    dx = torch.gradient(dem_m, spacing=(spacing,), dim=(1,))[0]
    slope = torch.sqrt(dx.square() + dy.square())
    return torch.stack([
        robust_standardize(dem_m), robust_standardize(slope),
        robust_standardize(dx), robust_standardize(dy),
    ])


def load_dem_grid(params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取已整理的流域DEM并降尺度，同时返回模型格点中心坐标。

    DEM文件是最终流域空间窗口。若文件的y轴为递减顺序，则连同高程一起
    翻转为南到北递增。

    Returns
    -------
    dem
        ``[model_height, model_width]`` 的高程，单位 m。
    x, y
        准备清单声明的投影坐标系下的模型网格中心坐标。
    """
    with xr.open_dataset(params.path_dem, engine="h5netcdf") as data:
        dem = data[params.dem_variable]
        if dem.dims[-2:] != ("y", "x"):
            dem = dem.transpose(..., "y", "x")
        values = np.asarray(dem).squeeze()
        x = np.asarray(data.x, dtype=np.float64)
        y = np.asarray(data.y, dtype=np.float64)
        x = x * coordinate_factor_to_km(data.x)
        y = y * coordinate_factor_to_km(data.y)
    if x[-1] < x[0]:
        x = x[::-1].copy()
        values = values[:, ::-1].copy()
    if y[-1] < y[0]:
        y = y[::-1].copy()
        values = values[::-1].copy()
    x, _ = regular_axis(x, "x")
    y, _ = regular_axis(y, "y")
    expected = (getattr(params, "grid_height", params.grid_size), getattr(params, "grid_width", params.grid_size))
    if values.shape != expected:
        raise ValueError(
            f"Prepared DEM grid={values.shape}; expected={expected}. "
            "Spatial cropping belongs in one-time preprocessing, not model loading."
        )
    dem_m = average_downsample(
        torch.from_numpy(values.astype(np.float32, copy=True)), params.down_factor
    )
    if params.down_factor > 1:
        x = x.reshape(-1, params.down_factor).mean(axis=1)
        y = y.reshape(-1, params.down_factor).mean(axis=1)
    return dem_m.numpy(), x, y


def load_terrain(params) -> tuple[torch.Tensor, torch.Tensor]:
    """组装模型静态输入和每个粗网格的平面面积。

    ``terrain`` 形状为 ``[channels, height, width]``，前四个通道来自 DEM，
    后续通道来自可选静态属性。单元边长和面积由DEM坐标实际格距推导。
    """
    dem_values, x, y = load_dem_grid(params)
    dem_m = torch.from_numpy(dem_values.astype(np.float32, copy=True))
    spacing_x_km = float(np.median(np.diff(x)))
    spacing_y_km = float(np.median(np.diff(y)))
    if not np.isclose(spacing_x_km, spacing_y_km, rtol=0.01, atol=1e-6):
        raise ValueError(
            f"D8 currently requires square projected cells; "
            f"dx={spacing_x_km:g} km, dy={spacing_y_km:g} km"
        )
    if not getattr(params, "prepared_daily", False):
        raise ValueError("Current model requires a validated prepared-daily basin manifest")
    with xr.open_dataset(params.path_terrain_attributes, engine="h5netcdf") as attributes:
        if attributes.attrs.get("crs_wkt") != params.grid_crs_wkt:
            raise ValueError("静态地形属性与DEM投影不一致")
        expected_features = ["elevation", "slope_gradient", "dzdx", "dzdy"]
        if attributes.feature.values.tolist() != expected_features:
            raise ValueError("静态地形属性通道顺序错误")
        if not (np.allclose(attributes.x.values / 1000.0, x, rtol=0, atol=1e-9)
                and np.allclose(attributes.y.values / 1000.0, y, rtol=0, atol=1e-9)):
            raise ValueError("静态地形属性没有逐像元对齐当前模型DEM")
        values = attributes.terrain_features.values.astype(np.float32, copy=True)
        if values.shape != (4, *dem_values.shape) or not np.isfinite(values).all():
            raise ValueError("静态地形属性尺寸错误或存在缺测")
        terrain = torch.from_numpy(values)
    cell_area_km2 = torch.full_like(dem_m, spacing_x_km * spacing_y_km)
    return terrain, cell_area_km2
