"""读取已准备的日尺度物理强迫，并构建参数编码器训练期气候背景。"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask import compute


class PreparedDailyRasterReader:
    """直接读取已准备的物理日数据；没有24倍放大、重新插值或隐式补值。"""

    def __init__(self, path, variable, start, end, down_factor=1):
        self.files = [Path(path)]
        self.variable = variable
        self.start, self.end_exclusive = pd.Timestamp(start), pd.Timestamp(end)
        self.down_factor = int(down_factor)
        self.scale, self.offset = 1.0, 0.0
        self.clip_min = self.fill_value = None
        self.fill_from_previous_day = False
        self.dataset = xr.open_dataset(path, engine="h5netcdf", chunks={"time": 84})
        if self.dataset.attrs.get("temporal_resolution") != "daily":
            raise ValueError(f"{path} 不是准备好的日尺度强迫")
        expected_units = "degree_Celsius" if variable == "t" else "mm/day"
        if self.dataset[variable].attrs.get("units") != expected_units:
            raise ValueError(f"{variable} 文件单位必须为 {expected_units}")
        self.source_data = self.dataset[variable].transpose("time", "y", "x")
        self.data = self.source_data.sel(time=slice(self.start, self.end_exclusive-pd.Timedelta(days=1)))
        self.available_time = pd.DatetimeIndex(self.data.time.values)
        shape = self.source_data.shape[-2:]
        if any(size % self.down_factor for size in shape):
            raise ValueError("DEM/强迫窗口不能整除 down_factor，请重新运行数据准备")
        self.model_grid_shape = tuple(size // self.down_factor for size in shape)

    def covers(self, start, days):
        start = pd.Timestamp(start)
        return start == start.normalize() and pd.date_range(start, periods=int(days), freq="D").isin(self.available_time).all()

    def model_grid_data(self, data=None):
        values = self.data if data is None else data
        if self.down_factor > 1:
            values = values.coarsen(y=self.down_factor, x=self.down_factor, boundary="exact").mean(skipna=False)
        return values

    def read(self, start, days):
        if not self.covers(start, days):
            raise IndexError(f"{self.variable} 日窗口缺少日期或超出允许阶段：{start}, {days}天")
        dates = pd.date_range(start, periods=int(days), freq="D")
        values = self.model_grid_data(self.data.sel(time=dates)).compute().values
        if not np.isfinite(values).all():
            raise ValueError(f"{self.variable} 文件含NaN/Inf，请检查源数据和对齐覆盖；不自动补值")
        return torch.from_numpy(values.astype(np.float32, copy=True)).unsqueeze(1)

    def load_or_create_normalization(self, directory):
        """只保存训练期统计，HBV方程仍接收真实 mm/day、°C。"""
        values = self.model_grid_data()
        results = compute(values.mean(), values.std(), values.min(), values.max())
        path = Path(directory) / f"{self.files[0].stem}__daily_normalization_{self.start:%Y-%m-%d}_{self.end_exclusive:%Y-%m-%d}_{self.model_grid_shape[0]}x{self.model_grid_shape[1]}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(output, mean=float(results[0]), std=float(results[1]),
                                minimum=float(results[2]), maximum=float(results[3]),
                                temporal_resolution="daily", applied_to_hbv=False)
        return path

    def close(self):
        self.dataset.close()


def build_forcing_readers(
    params, start, end,
) -> tuple[PreparedDailyRasterReader, PreparedDailyRasterReader, PreparedDailyRasterReader]:
    """建立严格对应准备清单的P、T、PET日尺度物理强迫读取器。"""

    if not getattr(params, "prepared_daily", False):
        raise ValueError("Current model requires prepared daily forcing files")
    readers = []
    try:
        sources = (
            (params.path_precipitation, "prec"),
            (params.path_temperature, "t"),
            (params.path_pet, "pet"),
        )
        for path, variable in sources:
            readers.append(
                PreparedDailyRasterReader(
                    path, variable, start, end, params.down_factor
                )
            )
    except Exception:
        for reader in readers:
            reader.close()
        raise
    return tuple(readers)


def build_parameter_climate_sequence(
    readers: tuple[PreparedDailyRasterReader, ...],
    start,
    end,
    block_days: int,
    basin_mask: torch.Tensor,
    *,
    cache_dir: str | Path | None = None,
) -> torch.Tensor:
    """构造只供HBV静态参数区域化使用的训练期气候序列。

    参数编码器不读取测试期。训练期日尺度强迫按连续``block_days``时间块汇总：
    降水和PET转为该块内的平均日累计量，气温取块平均值。所得三个通道
    分别标准化，并把流域外像元置零。

    该序列只描述每个栅格的训练期气候背景，不进入HBV水量方程。HBV本身仍
    使用未经归一化的日物理量``P(mm/day)、T(°C)、PET(mm/day)``。

    默认block_days=1，即日降水总量、日均温和日PET总量；读取按连续日块
    进行。归一化的均值、标准差只来自固定训练背景，在所有格之间共用，
    保留格间气候差异。
    可将背景张量和统计保存在forcing/grid/normalization；这里只缓存原始输入
    特征，编码器及参数映射仍在每次训练前向计算图内。

    返回形状为``[context_steps,3,H,W]``，通道顺序固定为P、T、PET。
    """
    # Match dXAJ's static parameter-inversion preprocessing.  This function
    # returns encoder features only; physical HBV forcing is never transformed.
    if len(readers) != 3:
        raise ValueError("HBV parameter context requires P, T, and PET readers")
    if block_days < 1:
        raise ValueError("block_days must be a positive integer")
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError("Parameter climate context end must follow start")
    if start != start.normalize() or end != end.normalize():
        raise ValueError("Parameter climate period must use midnight day boundaries")
    mask = basin_mask.to(torch.bool).cpu()
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Parameter context requires a nonempty 2D basin mask")
    total_days = (end - start).days
    if not all(reader.covers(start, total_days) for reader in readers):
        raise ValueError("Parameter background must contain every complete training-period day")
    expected_steps = ((end - start).days + block_days - 1) // block_days
    expected_shape = (expected_steps, 3, *mask.shape)
    path = None
    if cache_dir is not None:
        sources = []
        for reader in readers:
            source = reader.files[0]
            stat = source.stat()
            sources.append({
                "file": source.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "variable": reader.variable, "scale": reader.scale, "offset": reader.offset,
                "clip_min": reader.clip_min, "fill_value": reader.fill_value,
                "fill_previous_day": reader.fill_from_previous_day,
                "down_factor": reader.down_factor,
            })
        identity = {
            "version": 3, "temporal_resolution": "daily", "sources": sources, "start": start.isoformat(),
            "end_exclusive": end.isoformat(), "block_days": block_days,
            "mask": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
            "context_transforms": ["log10_sqrt_plus_0.1", "identity", "identity"],
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = Path(cache_dir) / f"hbv_parameter_context_{start:%Y-%m-%d}_{end:%Y-%m-%d}_{digest[:16]}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as cached:
                sequence = cached["sequence"]
                if sequence.shape == expected_shape and np.isfinite(sequence).all():
                    print(f"[参数背景] 已匹配: {path}")
                    return torch.from_numpy(sequence.copy())

    steps = []
    current = start
    read_days = max(block_days, (7 // block_days) * block_days)
    print(f"[参数背景] {start.date()} 至 {end.date()}（右端不含）| {expected_steps}步")
    while current < end:
        read_end = min(current + pd.Timedelta(days=read_days), end)
        days = (read_end - current).days
        prec, temp, pet = (reader.read(current, days) for reader in readers)
        for offset in range(0, days, block_days):
            stop = min(offset + block_days, days)
            duration_days = stop - offset
            steps.append(
                torch.cat((
                    prec[offset:stop].sum(dim=0) / duration_days,
                    temp[offset:stop].mean(dim=0),
                    pet[offset:stop].sum(dim=0) / duration_days,
                ), dim=0)
            )
        current = read_end
        # 多年背景约每年报告一次进度。
        if (current - start).days % 365 < read_days or current == end:
            print(f"[参数背景] 已读取 {(current-start).days}/{(end-start).days} 天")
    if not steps:
        raise ValueError("Parameter climate context contains no complete days")

    sequence = torch.stack(steps, dim=0).to(torch.float32)
    if mask.ndim != 2 or sequence.shape[-2:] != mask.shape:
        raise ValueError(
            "Parameter context and basin mask grids differ: "
            f"{tuple(sequence.shape[-2:])} vs {tuple(mask.shape)}"
        )
    # dXAJ 的参数反演输入对降水采用 log10(sqrt(P)+0.1)，压缩极端降水
    # 的长尾并保留小降水事件。该变换用于参数背景特征；HBV过程接收
    # 原始物理单位的P/T/PET。
    transformed = sequence.clone()
    transformed[:, 0] = torch.log10(
        transformed[:, 0].clamp_min(0.0).sqrt() + 0.1
    )

    normalized = torch.zeros_like(transformed)
    means, stds = [], []
    for channel in range(transformed.shape[1]):
        values = transformed[:, channel][:, mask]
        mean = values.mean()
        std = values.std(unbiased=False).clamp_min(1e-6)
        normalized[:, channel, mask] = (values - mean) / std
        means.append(mean)
        stds.append(std)
    if not torch.isfinite(normalized).all():
        raise ValueError("Non-finite values in HBV parameter climate sequence")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, sequence=normalized.numpy(), mean=torch.stack(means).numpy(),
                std=torch.stack(stds).numpy(), identity=json.dumps(identity, sort_keys=True),
                channels=np.asarray(["prec_log_sqrt_zscore", "t_zscore", "pet_zscore"]),
                context_transforms=np.asarray(
                    ["log10_sqrt_plus_0.1", "identity", "identity"]
                ),
            )
        temporary.replace(path)
        print(f"[参数背景] 已保存: {path}")
    return normalized
