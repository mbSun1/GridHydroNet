"""构建连续时间窗口，并组装端到端水文模型需要的完整样本。"""

from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HydroWindow:
    """一个目标模拟窗口的起始日期和持续天数。"""
    target_start: pd.Timestamp
    target_days: int


def build_training_windows(
    start_date: str,
    end_date: str,
    spinup_days: int,
    target_days: int,
) -> list[HydroWindow]:
    """在训练时段内构建连续、非重叠的固定长度目标窗口。

    第一个目标窗口从 ``start_date + spinup_days`` 开始，确保前方存在完整
    预热资料。每次向前移动 ``target_days``；只有完整包含 ``target_days``
    的窗口才会保留，不重复填补末尾不足一个窗口的日期。
    """
    current = pd.Timestamp(start_date) + pd.Timedelta(days=spinup_days)
    end = pd.Timestamp(end_date)
    windows = []
    while current + pd.Timedelta(days=target_days) <= end:
        windows.append(HydroWindow(current, target_days))
        current += pd.Timedelta(days=target_days)
    if not windows:
        raise ValueError("Training period is shorter than spin-up plus target window")
    # 训练窗口限定为完整目标时段，末尾短片段直接舍弃。
    return windows


def build_test_windows(start_date: str, end_date: str, window_days: int) -> list[HydroWindow]:
    """将整个测试期连续切分，最后一个窗口可以短于标准窗口长度。

    测试窗口首尾严格连续且不重叠，因此拼接各窗口结果后，每个测试日期
    恰好出现一次。
    """
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end <= current:
        raise ValueError("test_end must follow test_start")
    windows = []
    while current < end:
        length = min(window_days, (end - current).days)
        windows.append(HydroWindow(current, length))
        current += pd.Timedelta(days=length)
    return windows


class DailyHydroDataset(Dataset):
    """按需读取一个预热期和目标期的气象、流量及静态空间数据。

    气象强迫和观测流量保持逐日分辨率。静态地形和面积由同一流域的
    所有样本共享；固定D8图保存在模型路由器中，不随样本重复传输。
    """
    def __init__(
        self,
        windows,
        spinup_days,
        precipitation,
        temperature,
        pet,
        parameter_climate_sequence,
        discharge,
        terrain,
        basin_mask,
        cell_area_km2,
        gauge_area_km2,
    ):
        self.windows = list(windows)
        self.spinup_days = int(spinup_days)
        self.precipitation = precipitation
        self.temperature = temperature
        self.pet = pet
        self.parameter_climate_sequence = parameter_climate_sequence
        self.discharge = discharge
        self.terrain = terrain
        self.basin_mask = basin_mask
        self.cell_area_km2 = cell_area_km2
        self.gauge_area_km2 = gauge_area_km2

    def __len__(self):
        """返回当前训练或测试阶段的目标窗口数量。"""
        return len(self.windows)

    def read_forcing(self, start, days):
        """读取同一时段、同一模型网格上的逐日 P、T 和 PET。"""
        return {
            "precipitation": self.precipitation.read(start, days),
            "temperature": self.temperature.read(start, days),
            "pet": self.pet.read(start, days),
        }

    def __getitem__(self, index):
        """组装一个完整水文样本。

        目标期强迫和观测按窗口读取。训练每轮只在首个窗口前预热，
        随后的HBV与汇流状态按时间顺序跨窗口传递。
        """
        window = self.windows[index]
        target_start = window.target_start
        observed = self.discharge.read(target_start, window.target_days)
        return {
            "forcing": self.read_forcing(target_start, window.target_days),
            "terrain": self.terrain,
            # 同一实验的所有窗口共享训练期气候背景，用于生成一套固定的
            # 静态HBV参数场。
            "parameter_climate_sequence": self.parameter_climate_sequence,
            "basin_mask": self.basin_mask,
            "cell_area_km2": self.cell_area_km2,
            "gauge_area_km2": self.gauge_area_km2,
            # 训练目标、评价和观测曲线使用同一份原始实测。
            "discharge_daily": observed,
            "discharge_training_target": observed,
            "target_start": target_start.isoformat(),
        }
