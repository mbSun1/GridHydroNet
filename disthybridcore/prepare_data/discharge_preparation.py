"""Prepare one or more gauge observations on one union daily calendar."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyproj import Transformer
import xarray as xr

from .basin_workspace import normalize_station_id


def read_station_daily(path):
    data = pd.read_csv(path, sep=";")
    missing = {"YYYY", "MM", "DD", "qobs"}.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} 缺少 LamaH 日流量字段：{sorted(missing)}")
    dates = pd.DatetimeIndex(
        pd.to_datetime(dict(year=data.YYYY, month=data.MM, day=data.DD))
    )
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError(f"{path.name} 日期必须唯一、递增且非空")
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
        raise ValueError(f"{path.name} 日期不连续；不自动重建站内日历")
    qobs = pd.to_numeric(data.qobs, errors="coerce").to_numpy(
        dtype=np.float64, copy=True
    )
    qobs[~np.isfinite(qobs) | (qobs < 0)] = np.nan
    quality = {}
    for field in ("ckhs", "qceq", "qcol"):
        if field in data:
            quality[field] = pd.to_numeric(data[field], errors="coerce").to_numpy(
                dtype=np.float64
            )
    return dates, qobs, quality


def prepare_discharge_observations(
    workspace,
    basin_name,
    vectors,
    source_paths: dict[str, object],
):
    """Merge observed SHP gauges without filling values outside their records."""
    all_features = list(vectors["gauges"])
    features = [
        feature for feature in all_features
        if normalize_station_id(feature["properties"]["ID"]) in source_paths
    ]
    station_ids = [normalize_station_id(feature["properties"]["ID"]) for feature in features]
    if not station_ids or set(station_ids) != set(source_paths):
        raise ValueError("日流量源必须是测站 SHP 的非空子集")

    series = {}
    qualities = {}
    starts, ends = [], []
    for station_id in station_ids:
        dates, qobs, quality = read_station_daily(source_paths[station_id])
        series[station_id] = pd.Series(qobs, index=dates)
        qualities[station_id] = {
            name: pd.Series(values, index=dates) for name, values in quality.items()
        }
        starts.append(dates[0])
        ends.append(dates[-1])
    calendar = pd.date_range(min(starts), max(ends), freq="D")
    discharge = np.column_stack(
        [series[station_id].reindex(calendar).to_numpy() for station_id in station_ids]
    )

    transformer = Transformer.from_crs(3035, 4326, always_xy=True)
    projected_coordinates = np.asarray(
        [feature["geometry"]["coordinates"] for feature in features], dtype=np.float64
    )
    lon, lat = transformer.transform(
        projected_coordinates[:, 0], projected_coordinates[:, 1]
    )
    station_names = [
        str(feature["properties"].get("name") or f"ID_{station_id}")
        for feature, station_id in zip(features, station_ids)
    ]
    outlet_id = str(vectors["outlet_id"])
    is_outlet = np.asarray(
        [station_id == outlet_id for station_id in station_ids], dtype=np.int8
    )
    if int(is_outlet.sum()) != 1:
        raise ValueError("流量观测必须包含唯一总出口")

    index_by_id = {station_id: index for index, station_id in enumerate(station_ids)}
    all_by_id = {
        normalize_station_id(feature["properties"]["ID"]): feature
        for feature in all_features
    }
    adjacency = np.zeros((len(features), len(features)), dtype=np.int8)
    distance = np.zeros_like(adjacency, dtype=np.float64)
    altitude_delta = np.zeros_like(adjacency, dtype=np.float64)
    nextdown_ids = []
    for source, feature in enumerate(features):
        source_id = station_ids[source]
        if source_id == outlet_id:
            nextdown_ids.append("")
            continue
        current = feature
        target_id = normalize_station_id(current["properties"].get("NEXTDOWNID", ""))
        visited = {source_id}
        reported_distance_m = 0.0
        reported_distance_complete = True
        while True:
            reported_km = pd.to_numeric(
                current["properties"].get("dist_hdn"), errors="coerce"
            )
            if np.isfinite(reported_km) and float(reported_km) > 0:
                reported_distance_m += float(reported_km) * 1000.0
            else:
                reported_distance_complete = False
            if target_id in visited:
                raise ValueError(
                    f"Observed station {source_id} has no downstream path to another "
                    "observed station or the outlet in the gauge SHP"
                )
            if target_id in index_by_id:
                break
            if target_id not in all_by_id:
                raise ValueError(
                    f"Observed station {source_id} has no downstream path to another "
                    "observed station or the outlet in the gauge SHP"
                )
            visited.add(target_id)
            current = all_by_id[target_id]
            target_id = normalize_station_id(
                current["properties"].get("NEXTDOWNID", "")
            )
        nextdown_ids.append(target_id)
        target = index_by_id[target_id]
        adjacency[source, target] = 1
        if reported_distance_complete:
            distance[source, target] = reported_distance_m
        else:
            distance[source, target] = float(
                np.linalg.norm(projected_coordinates[source] - projected_coordinates[target])
            )
        source_elevation = pd.to_numeric(
            feature["properties"].get("elev"), errors="coerce"
        )
        target_elevation = pd.to_numeric(
            features[target]["properties"].get("elev"), errors="coerce"
        )
        if np.isfinite(source_elevation) and np.isfinite(target_elevation):
            altitude_delta[source, target] = float(source_elevation - target_elevation)

    output = xr.Dataset(
        {
            "discharge": (("t", "s"), discharge.astype(np.float32)),
            "name": ("s", np.asarray(station_names, dtype=str)),
            "station_id": ("s", np.asarray(station_ids, dtype=str)),
            "station_role": (
                "s", np.asarray(
                    ["outlet" if flag else "internal" for flag in is_outlet],
                    dtype=str,
                )
            ),
            "is_outlet": ("s", is_outlet),
            "nextdown_id": ("s", np.asarray(nextdown_ids, dtype=str)),
            "official_area_km2": (
                "s",
                np.asarray(
                    [feature["properties"].get("area_gov", np.nan) for feature in features],
                    dtype=np.float64,
                ),
            ),
            "coord": (("s", "coordinate"), np.column_stack((lon, lat))),
            "projected_coord": (
                ("s", "projected_coordinate"), projected_coordinates
            ),
            "adjacency_matrix": (("s", "target_station"), adjacency),
            "distance": (("s", "target_station"), distance),
            "altitude_delta": (("s", "target_station"), altitude_delta),
        },
        coords={
            "t": calendar,
            "s": np.arange(len(features)),
            "target_station": np.arange(len(features)),
            "coordinate": ["longitude", "latitude"],
            "projected_coordinate": ["x", "y"],
        },
    )
    output.coord.attrs.update(units="degrees", crs="EPSG:4326")
    output.projected_coord.attrs.update(units="m", crs="EPSG:3035")
    output.discharge.attrs.update(units="m3/s")
    output.distance.attrs.update(units="m")
    output.altitude_delta.attrs.update(units="m")
    output.official_area_km2.attrs.update(units="km2")
    for field in ("ckhs", "qceq", "qcol"):
        values = np.column_stack([
            qualities[station_id].get(
                field, pd.Series(np.nan, index=series[station_id].index)
            ).reindex(calendar).to_numpy()
            for station_id in station_ids
        ])
        output[field] = (("t", "s"), values)
        output[field].attrs["note"] = (
            "Original LamaH quality field; retained without invented thresholds"
        )
    output.attrs.update(
        basin=str(basin_name),
        outlet_station_id=outlet_id,
        station_count=len(station_ids),
        source_files=json.dumps(
            {station_id: source_paths[station_id].name for station_id in station_ids},
            sort_keys=True,
        ),
        observation_policy=(
            "Union daily calendar; original finite nonnegative qobs; "
            "outside-record and invalid observations remain NaN; no imputation"
        ),
    )
    path = (
        workspace.discharge_processed
        / f"basin_{basin_name}_discharge.nc"
    )
    temporary = path.with_suffix(".nc.part")
    output.to_netcdf(temporary, engine="scipy")
    temporary.replace(path)
    output.close()
    valid_counts = np.isfinite(discharge).sum(axis=0)
    print(
        f"[流量观测] stations={len(station_ids)} | "
        f"calendar={calendar[0].date()}..{calendar[-1].date()} | "
        f"valid_days={int(valid_counts.min())}..{int(valid_counts.max())}",
        flush=True,
    )
    return path
