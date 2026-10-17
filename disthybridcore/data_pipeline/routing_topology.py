"""D8地形算法以及已准备路由拓扑的严格读取和验证。

``disthybridcore.prepare_data.terrain_hydrology`` 调用本模块中的纯地形算法
构建拓扑。训练和测试读取manifest指定的NetCDF固定拓扑。

D8 拓扑本身不包含随时间变化的水量；它定义每个栅格唯一的下游。实际
水量传播由运行时选择的动态栅格路由器完成。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path

import numpy as np
import torch
import xarray as xr

@dataclass
class GridRoutingTopology:
    """模型网格上的静态地形、D8连接和河网分区。

    主要字段含义：

    - ``original_dem``：降尺度后的原始高程(m)；
    - ``conditioned_dem``：刻蚀河道并填洼后的高程(m)；
    - ``basin_mask``：模型粗网格是否与流域相交；
    - ``basin_fraction``：粗网格位于流域内的面积比例；
    - ``downstream_index``：每个栅格唯一的下游展平索引；
    - ``flow_direction``：D8八方向编码；
    - ``flow_accumulation``：每个栅格上游累积面积(km²)；
    - ``stream_order``：河道栅格的 Strahler 等级；
    - ``slope``、``flow_distance_km``：到直接下游的坡度和距离；
    - ``channel_mask``：使用河道传播参数的D8栅格；
    - ``routing_level``：可并行处理的上游到下游拓扑层级；
    - ``station_index``：唯一总出口的流量提取索引，长度恒为1；
    - ``station_snap_distance_km``：出口坐标映射到D8出口格的距离；
    - ``outlet_row/outlet_col``：全流域出口所在模型栅格。
    """

    original_dem: torch.Tensor
    conditioned_dem: torch.Tensor
    basin_mask: torch.Tensor
    basin_fraction: torch.Tensor
    downstream_index: torch.Tensor
    flow_direction: torch.Tensor
    flow_accumulation: torch.Tensor
    stream_order: torch.Tensor
    slope: torch.Tensor
    flow_distance_km: torch.Tensor
    channel_mask: torch.Tensor
    routing_level: torch.Tensor
    station_index: torch.Tensor
    station_snap_distance_km: torch.Tensor
    station_area_relative_error: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    outlet_row: int
    outlet_col: int
    channel_area_threshold_km2: float
    cache_path: Path


def priority_flood(
    dem: np.ndarray,
    basin_mask: np.ndarray,
    drainage_seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从流域出口向外执行优先队列填洼并记录排水父节点。

    唯一流域出口作为排水边界加入优先队列。算法每次访问当前最低的已连通
    栅格，并把相邻未访问流域栅格提升到至少略高于当前栅格。这样可消除
    封闭洼地，同时为被填平区域记录一个明确通向出口的 ``parent``。

    ``conditioned`` 用于选择最陡下降方向；``parent`` 在没有严格下降邻格的
    平坦区提供确定的排水方向。
    """
    height, width = dem.shape
    conditioned = dem.astype(np.float64, copy=True)
    parent = np.full((height, width), -1, dtype=np.int64)
    visited = np.zeros((height, width), dtype=bool)
    seed_rows, seed_cols = np.where(drainage_seeds & basin_mask)
    if len(seed_rows) == 0:
        raise ValueError("没有可作为排水边界的流域出口网格")
    queue = []
    for row, col in zip(seed_rows, seed_cols):
        flat = int(np.ravel_multi_index((row, col), dem.shape))
        heapq.heappush(queue, (float(conditioned[row, col]), flat))
        visited[row, col] = True
    offsets = (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    )
    # 从河道向外施加极小正高程增量，使填洼后的平坦区具有可追踪的微坡度。
    # 增量相对于流域总高差极小，不改变地形的宏观高程结构。
    minimum_gradient_m = max(
        1e-3,
        float(np.nanmax(conditioned[basin_mask]) - np.nanmin(conditioned[basin_mask]))
        * 1e-8,
    )
    while queue:
        elevation, flat = heapq.heappop(queue)
        row, col = divmod(flat, width)
        for row_offset, col_offset in offsets:
            next_row, next_col = row + row_offset, col + col_offset
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not basin_mask[next_row, next_col] or visited[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            parent[next_row, next_col] = flat
            next_elevation = max(
                float(conditioned[next_row, next_col]),
                elevation + minimum_gradient_m,
            )
            conditioned[next_row, next_col] = next_elevation
            heapq.heappush(
                queue,
                (next_elevation, int(np.ravel_multi_index((next_row, next_col), dem.shape))),
            )
    if not np.array_equal(visited, basin_mask):
        missing = int(np.logical_and(basin_mask, ~visited).sum())
        raise ValueError(f"流域掩膜包含与出口不连通的网格：{missing} 个")
    return conditioned.astype(np.float32), parent


def d8_from_conditioned(
    conditioned: np.ndarray,
    basin_mask: np.ndarray,
    parent: np.ndarray,
    outlet_mask: np.ndarray,
    cell_size_km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从水文校正 DEM 生成每个流域栅格的唯一 D8 下游连接。

    对八个相邻方向分别计算 ``高程降差 / 水平距离``，选择正坡度最大的
    邻格作为下游；对角方向距离为 ``sqrt(2) * cell_size``。若填洼区没有
    严格下降邻格，则使用优先队列填洼阶段记录的 parent。只有全流域出口
    的下游索引指向自身；坡面格和河道格都沿同一张D8图继续传播。

    返回下游展平索引、D8方向编码、无量纲坡度和相邻传播距离(km)。
    """
    height, width = conditioned.shape
    downstream = np.full((height, width), -1, dtype=np.int64)
    direction = np.full((height, width), -1, dtype=np.int8)
    slope = np.zeros((height, width), dtype=np.float32)
    distance_km = np.zeros((height, width), dtype=np.float32)
    offsets = (
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    )
    diagonal = float(np.sqrt(2.0))
    for row, col in zip(*np.where(basin_mask)):
        if outlet_mask[row, col]:
            downstream[row, col] = np.ravel_multi_index((row, col), conditioned.shape)
            direction[row, col] = 0
            continue
        best = None
        best_slope = 0.0
        best_distance = cell_size_km
        # 枚举八邻域，选择正向下降坡度最大的候选栅格。
        for code, (row_offset, col_offset) in enumerate(offsets):
            next_row, next_col = row + row_offset, col + col_offset
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not basin_mask[next_row, next_col]:
                continue
            distance = cell_size_km * (diagonal if row_offset and col_offset else 1.0)
            candidate = (
                float(conditioned[row, col]) - float(conditioned[next_row, next_col])
            ) / (distance * 1000.0)
            if candidate > best_slope + 1e-12:
                best_slope = candidate
                best = (next_row, next_col, code)
                best_distance = distance
        if best is None:
            parent_flat = int(parent[row, col])
            if parent_flat < 0:
                raise ValueError(f"D8网格 ({row}, {col}) 没有有效下游")
            next_row, next_col = divmod(parent_flat, width)
            row_offset, col_offset = next_row - row, next_col - col
            code = offsets.index((row_offset, col_offset))
            best = (next_row, next_col, code)
            best_distance = cell_size_km * (
                diagonal if row_offset and col_offset else 1.0
            )
            best_slope = max(
                0.0,
                (float(conditioned[row, col]) - float(conditioned[next_row, next_col]))
                / (best_distance * 1000.0),
            )
        next_row, next_col, code = best
        downstream[row, col] = np.ravel_multi_index(
            (next_row, next_col), conditioned.shape
        )
        direction[row, col] = code
        slope[row, col] = best_slope
        distance_km[row, col] = best_distance
    return downstream, direction, slope, distance_km


def compute_flow_accumulation(
    downstream: np.ndarray,
    basin_mask: np.ndarray,
    source_area_km2: np.ndarray,
) -> np.ndarray:
    """沿 D8 有向图累加每个栅格的实际上游汇水面积。

    ``source_area_km2`` 已包含边界粗网格的流域覆盖比例。算法按上游到下游
    的拓扑顺序将每个栅格累计面积传给唯一的下游栅格，因此出口处累积值
    等于整个流域面积。遍历数量和自指向出口数量同时用于检查拓扑闭合性。
    """
    flat_mask = basin_mask.reshape(-1)
    flat_downstream = downstream.reshape(-1)
    flat_area = source_area_km2.reshape(-1)
    indegree = np.zeros(flat_mask.size, dtype=np.int32)
    sources = np.where(flat_mask)[0]
    for source in sources:
        target = int(flat_downstream[source])
        if target >= 0 and target != source:
            indegree[target] += 1
    queue = [int(index) for index in sources[indegree[sources] == 0]]
    accumulation = np.zeros(flat_mask.size, dtype=np.float64)
    accumulation[sources] = flat_area[sources]
    visited = 0
    while queue:
        source = queue.pop()
        visited += 1
        target = int(flat_downstream[source])
        if target >= 0 and target != source:
            accumulation[target] += accumulation[source]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    outlet_count = int(np.sum((flat_downstream[sources] == sources)))
    if visited != len(sources) or outlet_count < 1:
        raise ValueError(
            f"D8 topology must be an acyclic single-outlet graph: "
            f"visited={visited}/{len(sources)}, outlets={outlet_count}"
        )
    return accumulation.reshape(basin_mask.shape).astype(np.float32)


def routing_levels(downstream: np.ndarray, basin_mask: np.ndarray) -> np.ndarray:
    """生成严格递增的D8拓扑层级，同层栅格可在GPU上并行计算。"""
    flat_mask = basin_mask.reshape(-1)
    flat_downstream = downstream.reshape(-1)
    sources = np.where(flat_mask)[0]
    indegree = np.zeros(flat_mask.size, dtype=np.int32)
    for source in sources:
        target = int(flat_downstream[source])
        if target != source:
            indegree[target] += 1
    current = [int(index) for index in sources[indegree[sources] == 0]]
    levels = np.full(flat_mask.size, -1, dtype=np.int32)
    level = 0
    visited = 0
    while current:
        following = []
        for source in current:
            levels[source] = level
            visited += 1
            target = int(flat_downstream[source])
            if target != source:
                indegree[target] -= 1
                if indegree[target] == 0:
                    following.append(target)
        current = following
        level += 1
    if visited != len(sources):
        raise ValueError(f"D8 topology is not acyclic: visited={visited}/{len(sources)}")
    return levels.reshape(basin_mask.shape)


def compute_stream_order(
    downstream: np.ndarray,
    channel_mask: np.ndarray,
) -> np.ndarray:
    """在提取出的 D8 河道图上计算逐栅格 Strahler 等级。

    源头河道为1级；若一个河道栅格接收两个及以上相同最高等级的上游河道，
    其等级在最高值上加1，否则继承最高上游等级。计算按河道入度进行拓扑
    遍历，并检查河道图中不存在循环。
    """
    flat_channel = channel_mask.reshape(-1)
    flat_downstream = downstream.reshape(-1)
    channel_cells = np.where(flat_channel)[0]
    indegree = np.zeros(flat_channel.size, dtype=np.int32)
    for source in channel_cells:
        target = int(flat_downstream[source])
        if target != source and flat_channel[target]:
            indegree[target] += 1
    queue = [int(index) for index in channel_cells[indegree[channel_cells] == 0]]
    order = np.zeros(flat_channel.size, dtype=np.int16)
    order[channel_cells] = 1
    maximum_upstream = np.zeros(flat_channel.size, dtype=np.int16)
    maximum_count = np.zeros(flat_channel.size, dtype=np.int16)
    visited = 0
    while queue:
        source = queue.pop()
        visited += 1
        target = int(flat_downstream[source])
        if target == source or not flat_channel[target]:
            continue
        source_order = int(order[source])
        if source_order > maximum_upstream[target]:
            maximum_upstream[target] = source_order
            maximum_count[target] = 1
        elif source_order == maximum_upstream[target]:
            maximum_count[target] += 1
        indegree[target] -= 1
        if indegree[target] == 0:
            upstream_order = int(maximum_upstream[target])
            order[target] = upstream_order + int(maximum_count[target] >= 2)
            queue.append(target)
    if visited != len(channel_cells):
        raise ValueError("Extracted channel network contains a cycle")
    return order.reshape(channel_mask.shape).astype(np.int8)


def validate_loaded_topology(topology: GridRoutingTopology) -> None:
    """在拓扑进入模型前检查索引、分区、出口和累积面积。

    检查内容包括：数组尺寸一致；所有下游索引位于流域内；存在配置出口；
    每个流域栅格恰好分配给一个节点；每个河道栅格恰好分配给一个河段；
    根据 D8 连接重新计算的汇水面积与缓存完全一致。
    """
    basin = topology.basin_mask.bool()
    channel = topology.channel_mask.bool()
    downstream = topology.downstream_index.long()
    if basin.shape != channel.shape or basin.shape != downstream.shape:
        raise ValueError("D8 cache arrays do not share one grid shape")
    flat_basin = basin.reshape(-1)
    flat_downstream = downstream.reshape(-1)
    sources = torch.where(flat_basin)[0]
    targets = flat_downstream[sources]
    if (targets < 0).any() or (targets >= flat_basin.numel()).any():
        raise ValueError("D8 cache contains an invalid downstream index")
    if not flat_basin[targets].all():
        raise ValueError("A D8 link leaves the basin mask")
    outlet = targets == sources
    if int(outlet.sum()) != 1:
        raise ValueError("Unified D8 cache must contain exactly one basin outlet")
    expected_outlet = topology.outlet_row * basin.shape[1] + topology.outlet_col
    if not bool((sources[outlet] == expected_outlet).any()):
        raise ValueError("Basin outlet is not a D8 channel root")
    if not channel.reshape(-1)[sources[outlet]].all():
        raise ValueError("A D8 root is not classified as a channel cell")
    station_index = topology.station_index.long()
    if station_index.numel() == 0:
        raise ValueError("D8 cache contains no discharge station index")
    if station_index.unique().numel() != station_index.numel():
        raise ValueError("Multiple discharge stations occupy the same D8 grid cell")
    if not flat_basin[station_index].all():
        raise ValueError("A discharge station lies outside the basin D8 graph")
    level = topology.routing_level.reshape(-1)
    non_root = sources[targets != sources]
    if not torch.all(level[flat_downstream[non_root]] > level[non_root]):
        raise ValueError("D8 routing levels do not increase downstream")

    cell_size_km = float(np.median(np.abs(np.diff(topology.x.numpy()))))
    recomputed = compute_flow_accumulation(
        downstream.detach().cpu().numpy(),
        basin.detach().cpu().numpy(),
        topology.basin_fraction.detach().cpu().numpy() * cell_size_km**2,
    )
    cached = topology.flow_accumulation.detach().cpu().numpy()
    if not np.allclose(recomputed, cached, rtol=1e-5, atol=1e-5):
        raise ValueError("Cached flow accumulation does not match D8 links")


def validate_outlet_diagnostic(topology) -> None:
    """检查唯一总出口的流量提取索引和汇水面积闭合。"""
    stations = topology.station_index.long()
    if stations.numel() != 1:
        raise ValueError(
            f"D8 topology must contain one outlet diagnostic cell, got {stations.numel()}"
        )
    outlet_cell = topology.outlet_row * topology.basin_mask.shape[1] + topology.outlet_col
    if int(stations[0]) != outlet_cell:
        raise ValueError("The cached outlet diagnostic is not mapped to the D8 outlet")
    cell_size_km = float(np.median(np.abs(np.diff(topology.x.numpy()))))
    basin_area = topology.basin_fraction.sum() * cell_size_km**2
    represented_area = topology.flow_accumulation.reshape(-1)[stations[0]]
    if not torch.isclose(
        represented_area, basin_area, rtol=1e-5, atol=1e-3
    ):
        raise ValueError("The D8 outlet does not collect the complete basin")
    expected_error = (represented_area - basin_area) / basin_area.clamp_min(1e-6)
    if topology.station_area_relative_error.numel() != 1 or not torch.isclose(
        topology.station_area_relative_error[0], expected_error,
        rtol=1e-5, atol=1e-6,
    ):
        raise ValueError("Cached outlet drainage-area diagnostic is inconsistent")


def read_routing_topology(cache_path: Path) -> GridRoutingTopology:
    """读取并严格核验 manifest 指定的已准备 D8 拓扑。"""
    with xr.open_dataset(cache_path, engine="h5netcdf") as data:
        tensor = lambda name, dtype: torch.from_numpy(
            np.asarray(data[name]).astype(dtype, copy=True)
        )
        topology = GridRoutingTopology(
            original_dem=tensor("dem_original", np.float32),
            conditioned_dem=tensor("dem_conditioned", np.float32),
            basin_mask=tensor("basin_mask", np.bool_),
            basin_fraction=tensor("basin_fraction", np.float32),
            downstream_index=tensor("downstream_index", np.int64),
            flow_direction=tensor("flow_direction", np.int64),
            flow_accumulation=tensor("flow_accumulation", np.float32),
            stream_order=tensor("stream_order", np.int64),
            slope=tensor("slope", np.float32),
            flow_distance_km=tensor("flow_distance_km", np.float32),
            channel_mask=tensor("channel_mask", np.bool_),
            routing_level=tensor("routing_level", np.int64),
            station_index=tensor("station_index", np.int64),
            station_snap_distance_km=tensor("station_snap_distance_km", np.float32),
            station_area_relative_error=tensor(
                "station_area_relative_error", np.float32
            ),
            x=tensor("x", np.float64),
            y=tensor("y", np.float64),
            outlet_row=int(data.attrs["outlet_row"]),
            outlet_col=int(data.attrs["outlet_col"]),
            channel_area_threshold_km2=float(
                data.attrs["channel_area_threshold_km2"]
            ),
            cache_path=cache_path,
        )
    validate_loaded_topology(topology)
    validate_outlet_diagnostic(topology)
    return topology
