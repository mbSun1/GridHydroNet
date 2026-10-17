"""组织流域静态数据、逐日强迫和训练/测试DataLoader。"""

import hashlib
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .basin import (
    load_outlet_catchment,
    load_river_network,
)
from .dataset import DailyHydroDataset, build_test_windows, build_training_windows
from .forcing import build_forcing_readers, build_parameter_climate_sequence
from .observations import DailyDischargeReader
from .routing_topology import read_routing_topology
from disthybridcore.prepare_data.grid_routing import load_grid_gauge_mapping
from .spatial import load_terrain


class DailyHydroDataModule(LightningDataModule):
    """确保所有数据源在相同时间窗口、空间网格和站点顺序下进入模型。"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_dataset = None
        self.test_dataset = None
        self.parameter_climate_sequence = None
        self.parameter_context_metadata = None
        self.grid_gauges = None
        self.gauge_ids = []
        self.gauge_names = []
        self._static_ready = False

    def setup_static(self):
        """读取并缓存整个实验共享的地形、D8、汇水区和河网静态数据。"""
        if self._static_ready:
            return
        p = self.params
        # 流量观测与栅格测站映射按站点ID显式对齐。
        self.river = load_river_network(p.path_bafg)
        # 流域边界采用总出口的累计汇水区。
        self.outlet_catchment = load_outlet_catchment(
            p.path_catchments,
            p.catchment_variable,
            p.down_factor,
        )
        # terrain包含标准化高程、坡度和方向梯度；面积初值为完整粗网格面积。
        self.terrain, self.cell_area_km2 = load_terrain(p)
        # 训练和测试读取准备阶段发布的固定拓扑文件。
        if not getattr(p, "path_prepared_topology", None):
            raise ValueError("Prepared basin manifest does not contain topology")
        self.routing_topology = read_routing_topology(
            Path(p.path_prepared_topology)
        )
        # 边界粗网格保存实际流域覆盖面积，用于守恒的mm水深到水体积转换。
        self.cell_area_km2 = (
            self.cell_area_km2 * self.routing_topology.basin_fraction
        )
        if not getattr(p, "path_grid_gauges", None):
            raise ValueError("Prepared basin manifest does not contain grid_gauges")
        self.grid_gauges = load_grid_gauge_mapping(Path(p.path_grid_gauges))
        self.gauge_ids = list(self.grid_gauges.gauge_ids)
        self.gauge_names = [
            f"ID_{station_id} | {name}"
            for station_id, name in zip(
                self.grid_gauges.gauge_ids,
                self.grid_gauges.gauge_names,
            )
        ]
        # 观测顺序严格服从固定 gauge-to-grid 缓存。
        self.discharge = DailyDischargeReader(
            p.path_bafg,
            p.discharge_variable,
            station_ids=self.gauge_ids,
        )
        if self.discharge.station_ids != self.gauge_ids:
            raise ValueError("Discharge observations and grid gauges are not ID-aligned")
        if self.discharge.outlet_index != self.grid_gauges.outlet_gauge_index:
            raise ValueError("Discharge and grid-gauge outlet indices differ")
        if self.river.station_ids != self.gauge_ids:
            raise ValueError("Prepared station network and grid-gauge mapping differ")
        self.outlet_station_index = self.grid_gauges.outlet_gauge_index
        self.outlet_station_name = self.gauge_names[self.outlet_station_index]
        self.gauge_area_km2 = self.grid_gauges.gauge_grid_area_km2.float()
        if (
            not torch.isfinite(self.gauge_area_km2).all()
            or torch.any(self.gauge_area_km2 <= 0)
        ):
            raise ValueError("D8 gauge upstream areas must be finite and positive")
        self._static_ready = True

    def forcing_readers(self, start, end):
        """建立覆盖指定阶段的P、T和PET延迟读取器。"""
        return build_forcing_readers(self.params, start, end)

    def prepare_training_background(self, readers=None):
        """为训练、独立测试和诊断准备同一背景序列及其身份。

        训练可复用已有日读取器；独立测试临时打开训练期读取器并在完成
        后关闭。标准化特征只用于参数编码器，不改写HBV的日物理输入。
        """
        p = self.params
        own_readers = readers is None
        if own_readers:
            readers = self.forcing_readers(p.train_start, p.train_end)
        try:
            days = int(
                (pd.Timestamp(p.train_end) - pd.Timestamp(p.train_start))
                / pd.Timedelta(days=1)
            )
            for reader in readers:
                if not reader.covers(p.train_start, days):
                    raise ValueError(
                        f"Cannot normalize incomplete {reader.variable!r} training data"
                    )
                reader.load_or_create_normalization(p.normalization_dir)
            self.parameter_climate_sequence = build_parameter_climate_sequence(
                readers,
                p.train_start,
                p.train_end,
                p.hbv_parameter_context_block_days,
                self.routing_topology.basin_mask,
                cache_dir=p.normalization_dir,
            )
            self.describe_parameter_context()
        finally:
            if own_readers:
                for reader in readers:
                    reader.close()

    def describe_parameter_context(self):
        """记录固定背景与空间张量的身份，供检查点和独立测试核对。

        校验内容来自实际输入值和网格身份。相同的数据内容、网格与训练背景
        对应同一参数背景。
        """
        digest = hashlib.sha256()
        tensors = (
            self.parameter_climate_sequence, self.terrain,
            self.routing_topology.basin_mask, self.routing_topology.downstream_index,
            self.routing_topology.routing_level, self.routing_topology.channel_mask,
            self.routing_topology.flow_distance_km, self.routing_topology.slope,
            self.routing_topology.flow_accumulation,
            self.cell_area_km2,
            self.grid_gauges.gauge_cell_index,
            self.gauge_area_km2,
        )
        for tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        digest.update("|".join(self.gauge_ids).encode("utf-8"))
        digest.update(str(self.outlet_station_index).encode())
        p = self.params
        self.parameter_context_metadata = {
            "schema": 17,
            "temporal_resolution": "daily",
            "routing": "d8_kinematic",
            "runoff_model": "single_component_hbv92",
            "routing_input": "q0_plus_q1_plus_q2",
            "encoder": p.parameter_encoder,
            "climate_summary": "full_sequence_output_temporal_mean_and_std",
            "parameter_mode": "static_training_background",
            "train_start": str(p.train_start),
            "train_end_exclusive": str(p.train_end),
            "block_days": p.hbv_parameter_context_block_days,
            "shape": list(self.parameter_climate_sequence.shape),
            "channels": ["prec_log_sqrt_zscore", "t_zscore", "pet_zscore"],
            "context_transforms": [
                "log10_sqrt_plus_0.1", "identity", "identity"
            ],
            "terrain_channels": p.terrain_channels,
            "gauge_ids": list(self.gauge_ids),
            "outlet_gauge_index": self.outlet_station_index,
            "gauge_d8_upstream_area_km2": self.gauge_area_km2.tolist(),
            "input_sha256": digest.hexdigest(),
        }
        if p.parameter_encoder == "bigru":
            self.parameter_context_metadata.update({
                "hidden": p.hbv_parameter_hidden,
                "layers": p.hbv_parameter_layers,
            })
        elif p.parameter_encoder == "patch_transformer":
            self.parameter_context_metadata.update({
                "patch_days": 30,
                "model_width": 64,
                "attention_heads": 4,
                "layers": 2,
            })
        elif p.parameter_encoder == "tcn":
            self.parameter_context_metadata.update({
                "temporal_channels": 32,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                "receptive_field_days": 4095,
            })
        else:
            raise ValueError(f"Unsupported parameter_encoder={p.parameter_encoder!r}")

    @staticmethod
    def require_complete(windows, spinup_days, readers, stage):
        """逐窗口检查预热期和目标期是否包含每一个连续日。"""
        missing = []
        for window in windows:
            start = window.target_start - pd.Timedelta(days=spinup_days)
            days = spinup_days + window.target_days
            if not all(reader.covers(start, days) for reader in readers):
                missing.append((start, window.target_start, window.target_days))
        if missing:
            start, target, days = missing[0]
            raise ValueError(
                f"{stage} forcings are incomplete for {len(missing)} windows; "
                f"first spin-up={start}, target={target}, days={days}"
            )

    def setup(self, stage=None):
        """按Lightning阶段构建训练或连续测试数据集。"""
        self.setup_static()
        p = self.params
        if stage in (None, "fit") and self.train_dataset is None:
            # 训练窗口严格连续；每轮只在第一个窗口前建立一次初态。
            windows = build_training_windows(
                p.train_start, p.train_end, p.spinup_days,
                p.train_window_days,
            )
            readers = self.forcing_readers(p.train_start, p.train_end)
            try:
                self.require_complete(windows, p.spinup_days, readers, "Training")
                self.prepare_training_background(readers)
            except Exception:
                for reader in readers:
                    reader.close()
                raise
            self.train_dataset = self.build_dataset(windows, readers)

        if stage in (None, "test") and self.test_dataset is None:
            if self.train_dataset is None:
                self.prepare_training_background()
            # 测试窗口连续覆盖完整测试期；正式测试初态由
            # training_module从train_start连续建立。
            windows = build_test_windows(
                p.test_start, p.test_end, p.test_window_days
            )
            # 连续评价从固定训练起点推进历史状态；状态来源是历史强迫的
            # 正向积分。
            start = min(pd.Timestamp(p.train_start),
                        pd.Timestamp(p.test_start) - pd.Timedelta(days=p.spinup_days))
            readers = self.forcing_readers(start, p.test_end)
            self.require_complete(windows, p.spinup_days, readers, "Test")
            self.test_dataset = self.build_dataset(windows, readers)

    def training_evaluation_dataset(self):
        """重建覆盖训练目标期的连续、非重叠回放数据集。"""
        self.setup("fit")
        source = self.train_dataset
        windows = sorted(source.windows, key=lambda window: window.target_start)
        end = windows[-1].target_start + pd.Timedelta(days=windows[-1].target_days)
        windows = build_test_windows(
            windows[0].target_start, end, self.params.train_window_days
        )
        readers = (source.precipitation, source.temperature, source.pet)
        return self.build_dataset(windows, readers)

    def build_dataset(self, windows, readers):
        """把时间窗口、动态读取器和全部静态张量封装为Dataset。"""
        return DailyHydroDataset(
            windows,
            self.params.spinup_days,
            *readers,
            self.parameter_climate_sequence,
            self.discharge,
            self.terrain,
            self.routing_topology.basin_mask,
            self.cell_area_km2,
            self.gauge_area_km2,
        )

    def train_dataloader(self):
        """返回严格按时间顺序、batch_size=1的训练DataLoader。"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
        )

    def test_dataloader(self):
        """返回严格按时间顺序、batch_size=1的连续测试DataLoader。"""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
        )

    def close(self):
        """关闭全部xarray/NetCDF文件句柄。"""
        readers = []
        for dataset in (self.train_dataset, self.test_dataset):
            if dataset is not None:
                readers.extend([
                    dataset.precipitation, dataset.temperature, dataset.pet
                ])
        for reader in {id(item): item for item in readers}.values():
            reader.close()
        if self._static_ready:
            self.discharge.close()
