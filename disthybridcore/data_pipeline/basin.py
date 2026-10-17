"""站点河网拓扑和站点控制汇水区处理。"""

from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr

from .spatial import average_downsample


@dataclass
class RiverNetworkData:
    """站点尺度有向河网及其静态信息。

    ``adjacency[i, j] = 1`` 表示站点 i 的河水直接流向站点 j；因此矩阵行
    表示上游源节点，列表示下游目标节点。``topological_order`` 保证所有
    上游节点在其下游节点之前计算。
    """

    adjacency: torch.Tensor
    distance_m: torch.Tensor
    altitude_drop_m: torch.Tensor
    topological_order: torch.Tensor
    station_ids: list[str]
    station_names: list[str]
    station_coordinates: torch.Tensor
    artificial: torch.Tensor
    outlet_index: int


def load_river_network(path: str) -> RiverNetworkData:
    """从 BAFG 派生文件读取站点河网，并建立拓扑计算顺序。

    文件中的距离和高程差矩阵与邻接矩阵使用相同的 ``[source, target]``
    索引。函数通过 Kahn 拓扑排序检查河网是否存在有向环；有环河网无法按
    上游到下游顺序进行河道路由，因此会直接报错。
    """
    with xr.open_dataset(path, engine="scipy", decode_timedelta=True) as data:
        adjacency = torch.from_numpy(
            np.asarray(data["adjacency_matrix"]).astype(np.float32, copy=True)
        )
        altitude = torch.from_numpy(
            np.asarray(data["altitude_delta"]).astype(np.float32, copy=True)
        )
        distance = torch.from_numpy(
            np.asarray(data["distance"]).astype(np.float32, copy=True)
        )
        names = [str(value) for value in np.asarray(data["name"]).tolist()]
        if "station_id" not in data or "is_outlet" not in data:
            raise ValueError("Prepared discharge file lacks station_id/is_outlet")
        station_ids = [str(value) for value in np.asarray(data["station_id"]).tolist()]
        outlet_flags = np.asarray(data["is_outlet"]).astype(bool)
        if outlet_flags.sum() != 1:
            raise ValueError("Prepared discharge network must contain one outlet")
        outlet_index = int(np.flatnonzero(outlet_flags)[0])
        coordinates = torch.from_numpy(
            np.asarray(data["coord"]).astype(np.float32, copy=True)
        )
        artificial = (
            torch.from_numpy(np.asarray(data["artificial"]).astype(bool, copy=True))
            if "artificial" in data
            else torch.zeros(adjacency.shape[0], dtype=torch.bool)
        )
    # 邻接矩阵只表达连接关系，任何正值统一转换为1。
    adjacency = (adjacency > 0).float()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("River adjacency matrix must be square")
    # 入度等于每个节点的直接上游数量。入度为0的节点作为拓扑排序起点。
    indegree = adjacency.sum(dim=0).to(torch.int64)
    queue = torch.where(indegree == 0)[0].tolist()
    order = []
    while queue:
        source = queue.pop(0)
        order.append(source)
        for target in torch.where(adjacency[source] > 0)[0].tolist():
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != adjacency.shape[0]:
        raise ValueError("River adjacency contains a directed cycle")
    graph_outlets = torch.where(adjacency.sum(dim=1) == 0)[0]
    if graph_outlets.numel() != 1 or int(graph_outlets[0]) != outlet_index:
        raise ValueError("Station adjacency and is_outlet metadata disagree")
    return RiverNetworkData(
        adjacency=adjacency,
        distance_m=distance,
        altitude_drop_m=altitude,
        topological_order=torch.tensor(order, dtype=torch.int64),
        station_ids=station_ids,
        station_names=names,
        station_coordinates=coordinates,
        artificial=artificial,
        outlet_index=outlet_index,
    )


def load_outlet_catchment(
    path: str,
    variable: str,
    down_factor: int,
) -> torch.Tensor:
    """读取总出口累计汇水区并降尺度为模型网格面积覆盖比例。

    原始变量为 ``[station, H, W]`` 的0/1累计汇水区。这里只选择总出口
    对应的一层，块平均后得到 ``[model_H, model_W]`` 的分数型覆盖比例。
    边界粗格的0--1值用于守恒地计算实际流域面积。
    """
    with xr.open_dataset(path, engine="h5netcdf") as data:
        if variable not in data:
            raise KeyError(f"Catchment variable {variable!r} not found")
        values = np.asarray(data[variable])
        if values.ndim == 2:
            outlet = torch.from_numpy(values.astype(np.float32, copy=True))
            return average_downsample(outlet, down_factor)
        raise ValueError(f"Prepared basin fraction must be [H,W], got {values.shape}")
