"""Prepare the fixed gauge mapping used by grid routing.

The dynamic routers operate on the complete active D8 grid.  This module only
maps observation gauges to cells of that graph; it does not create reaches or
hydrologic response units.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .basin_workspace import normalize_station_id, read_vector


@dataclass(frozen=True)
class GridGaugeMapping:
    gauge_cell_index: torch.Tensor
    gauge_ids: tuple[str, ...]
    gauge_names: tuple[str, ...]
    gauge_snap_distance_m: torch.Tensor
    gauge_official_area_km2: torch.Tensor
    gauge_grid_area_km2: torch.Tensor
    outlet_gauge_index: int
    cache_path: Path

    @property
    def n_gauges(self) -> int:
        return len(self.gauge_ids)


def snap_gauges(
    topology,
    features: list[dict],
    outlet_id: str,
    *,
    max_distance_km: float,
    area_relative_tolerance: float,
) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    channel = topology.channel_mask.numpy().astype(bool)
    downstream = topology.downstream_index.numpy().reshape(-1)
    accumulation = topology.flow_accumulation.numpy().reshape(-1)
    x_km = topology.x.numpy()
    y_km = topology.y.numpy()
    _, width = channel.shape
    rows, cols = np.where(channel)
    cells = rows * width + cols
    channel_flat = channel.reshape(-1)
    all_cells = np.arange(downstream.size)
    incoming_count = np.asarray([
        np.sum((downstream == cell) & channel_flat & (all_cells != cell))
        for cell in cells
    ])

    by_id = {normalize_station_id(f["properties"].get("ID")): f for f in features}
    if outlet_id not in by_id:
        raise ValueError(f"Gauge SHP does not contain outlet ID={outlet_id}")

    selected: dict[str, int] = {}
    distances_m: dict[str, float] = {}
    official_area: dict[str, float] = {}
    used: set[int] = set()
    outlet_cell = topology.outlet_row * width + topology.outlet_col
    for station_id, feature in by_id.items():
        xy_m = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        area = float(feature["properties"].get("area_gov", np.nan))
        official_area[station_id] = area
        if station_id == outlet_id:
            selected[station_id] = int(outlet_cell)
            row, col = divmod(outlet_cell, width)
            distances_m[station_id] = float(np.hypot(
                x_km[col] * 1000.0 - xy_m[0], y_km[row] * 1000.0 - xy_m[1]
            ))
            used.add(int(outlet_cell))
            continue

        distance_km = np.hypot(
            x_km[cols] - xy_m[0] / 1000.0,
            y_km[rows] - xy_m[1] / 1000.0,
        )
        candidate = np.flatnonzero(distance_km <= max_distance_km)
        candidate = np.asarray(
            [i for i in candidate if int(cells[i]) not in used], dtype=np.int64
        )
        if candidate.size == 0:
            raise ValueError(
                f"Gauge {station_id} has no unused D8 channel cell within "
                f"{max_distance_km:g} km"
            )
        if np.isfinite(area) and area > 0:
            error = np.abs(accumulation[cells[candidate]] - area) / area
            compatible = candidate[error <= area_relative_tolerance]
            if compatible.size:
                candidate = compatible
        section = candidate[incoming_count[candidate] == 1]
        if section.size:
            candidate = section
        choice = int(candidate[np.argmin(distance_km[candidate])])
        selected[station_id] = int(cells[choice])
        distances_m[station_id] = float(distance_km[choice] * 1000.0)
        used.add(int(cells[choice]))

    for station_id, feature in by_id.items():
        target_id = normalize_station_id(feature["properties"].get("NEXTDOWNID", ""))
        if target_id not in selected:
            continue
        current, target = selected[station_id], selected[target_id]
        visited: set[int] = set()
        while current != target and current not in visited:
            visited.add(current)
            following = int(downstream[current])
            if following < 0 or following == current:
                break
            current = following
        if current != target:
            raise ValueError(
                f"Gauge topology mismatch: {station_id} does not drain to "
                f"NEXTDOWNID={target_id} on the prepared D8 graph"
            )
    return selected, distances_m, official_area


def prepare_grid_gauge_cache(
    topology_path: Path,
    gauge_path: Path,
    outlet_id: str,
    output_path: Path,
    gauge_snap_max_distance_km: float = 3.0,
    gauge_area_relative_tolerance: float = 0.15,
    observed_station_ids: list[str] | tuple[str, ...] | None = None,
) -> GridGaugeMapping:
    from disthybridcore.data_pipeline.routing_topology import read_routing_topology

    topology_path, output_path = Path(topology_path), Path(output_path)
    topology = read_routing_topology(topology_path)
    with xr.open_dataset(topology_path, engine="h5netcdf") as source:
        topology_signature = source.attrs.get("input_signature")
    if topology_signature is None:
        raise ValueError("D8 topology lacks input_signature metadata")

    features = sorted(
        read_vector(Path(gauge_path)),
        key=lambda f: (0, int(f["properties"]["ID"]))
        if str(f["properties"]["ID"]).isdigit()
        else (1, str(f["properties"]["ID"])),
    )
    outlet_id = str(outlet_id)
    if observed_station_ids is not None:
        requested = [str(value) for value in observed_station_ids]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("observed_station_ids must be a non-empty unique sequence")
        available = {normalize_station_id(f["properties"].get("ID")) for f in features}
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError(f"Observed station IDs are absent from gauge SHP: {missing}")
        if outlet_id not in requested:
            raise ValueError("Observed stations must contain the basin outlet")
        requested_set = set(requested)
        features = [
            f for f in features
            if normalize_station_id(f["properties"].get("ID")) in requested_set
        ]

    selected, distances, official = snap_gauges(
        topology, features, outlet_id,
        max_distance_km=float(gauge_snap_max_distance_km),
        area_relative_tolerance=float(gauge_area_relative_tolerance),
    )
    ids = tuple(normalize_station_id(f["properties"].get("ID")) for f in features)
    names = tuple(
        str(f["properties"].get("name") or station_id)
        for f, station_id in zip(features, ids)
    )
    cells = torch.tensor([selected[value] for value in ids], dtype=torch.long)
    mapping = GridGaugeMapping(
        gauge_cell_index=cells,
        gauge_ids=ids,
        gauge_names=names,
        gauge_snap_distance_m=torch.tensor(
            [distances[value] for value in ids], dtype=torch.float32
        ),
        gauge_official_area_km2=torch.tensor(
            [official[value] for value in ids], dtype=torch.float32
        ),
        gauge_grid_area_km2=topology.flow_accumulation.reshape(-1)[cells].float(),
        outlet_gauge_index=ids.index(outlet_id),
        cache_path=output_path.resolve(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": 1,
        "topology_input_signature": topology_signature,
        "gauge_cell_index": mapping.gauge_cell_index,
        "gauge_ids": mapping.gauge_ids,
        "gauge_names": mapping.gauge_names,
        "gauge_snap_distance_m": mapping.gauge_snap_distance_m,
        "gauge_official_area_km2": mapping.gauge_official_area_km2,
        "gauge_grid_area_km2": mapping.gauge_grid_area_km2,
        "outlet_gauge_index": mapping.outlet_gauge_index,
    }, output_path)
    write_gauge_table(mapping, output_path.parent)
    return mapping


def load_grid_gauge_mapping(path: Path) -> GridGaugeMapping:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1:
        raise ValueError(f"Unsupported grid-gauge cache schema in {path}")
    mapping = GridGaugeMapping(
        gauge_cell_index=payload["gauge_cell_index"].long(),
        gauge_ids=tuple(str(v) for v in payload["gauge_ids"]),
        gauge_names=tuple(str(v) for v in payload["gauge_names"]),
        gauge_snap_distance_m=payload["gauge_snap_distance_m"].float(),
        gauge_official_area_km2=payload["gauge_official_area_km2"].float(),
        gauge_grid_area_km2=payload["gauge_grid_area_km2"].float(),
        outlet_gauge_index=int(payload["outlet_gauge_index"]),
        cache_path=path.resolve(),
    )
    count = mapping.gauge_cell_index.numel()
    if not (
        count == len(mapping.gauge_ids) == len(mapping.gauge_names)
        == mapping.gauge_snap_distance_m.numel()
        == mapping.gauge_official_area_km2.numel()
        == mapping.gauge_grid_area_km2.numel()
    ):
        raise ValueError("Grid-gauge cache arrays have inconsistent lengths")
    if mapping.gauge_cell_index.unique().numel() != count:
        raise ValueError("Grid-gauge cache maps multiple stations to one cell")
    if not 0 <= mapping.outlet_gauge_index < count:
        raise ValueError("Grid-gauge cache has an invalid outlet index")
    return mapping


def write_gauge_table(mapping: GridGaugeMapping, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "gauge_to_grid.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "gauge_id", "gauge_name", "gauge_role", "grid_cell_index",
            "snap_distance_m", "official_area_km2", "d8_area_km2",
        ])
        for index in range(mapping.n_gauges):
            writer.writerow([
                mapping.gauge_ids[index], mapping.gauge_names[index],
                "outlet" if index == mapping.outlet_gauge_index else "internal",
                int(mapping.gauge_cell_index[index]),
                float(mapping.gauge_snap_distance_m[index]),
                float(mapping.gauge_official_area_km2[index]),
                float(mapping.gauge_grid_area_km2[index]),
            ])
    return path
