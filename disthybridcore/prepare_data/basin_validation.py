"""检查空间、日历、物理单位、静态属性和观测覆盖；输出可追溯报告。"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from disthybridcore.data_pipeline.routing_topology import read_routing_topology
from .basin_workspace import write_json


def validate_basin_data(manifest, report_dir, *, inspect_values=True):
    """只核验，不偷偷修补；逐块读取强迫，避免多年区域数据同时驻留内存。"""
    with xr.open_dataset(manifest["dem"], engine="h5netcdf") as data:
        x, y = data.x.values, data.y.values
        fraction = data.basin_fraction.values
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
            raise ValueError("DEM 的 x/y 必须严格递增")
        spacing = float(np.median(np.diff(x)))
        if not np.allclose(np.diff(x), spacing) or not np.allclose(np.diff(y), spacing):
            raise ValueError("分析像元不是规则米制正方形")
        if not np.isfinite(data.dem.values).all() or not ((fraction >= 0) & (fraction <= 1)).all():
            raise ValueError("DEM 高程或面积比例无效")
        area_km2 = float(fraction.sum(dtype=np.float64)) * spacing**2 / 1e6
        crs_wkt = data.attrs["crs_wkt"]
    topology = read_routing_topology(Path(manifest["topology"]))
    with xr.open_dataset(manifest["topology"], engine="h5netcdf") as data:
        terrain_report = dict(data.attrs)
    with xr.open_dataset(manifest["static_attributes"], engine="h5netcdf") as data:
        if not (np.allclose(data.x.values / 1000.0, topology.x.numpy(), rtol=0, atol=1e-9)
                and np.allclose(data.y.values / 1000.0, topology.y.numpy(), rtol=0, atol=1e-9)):
            raise ValueError("静态属性与D8网格不一致")
        if data.attrs.get("crs_wkt") != crs_wkt or not np.isfinite(data.terrain_features.values).all():
            raise ValueError("静态属性投影或数值无效")
    summaries, records = {}, {}
    for variable, path in manifest["forcing"].items():
        with xr.open_dataset(path, engine="h5netcdf") as data:
            if variable not in data or not np.array_equal(data.x.values, x) or not np.array_equal(data.y.values, y):
                raise ValueError(f"{variable} 没有逐像元对齐 DEM")
            if data.attrs.get("crs_wkt") != crs_wkt:
                raise ValueError(f"{variable} 投影与 DEM 不一致")
            dates = pd.DatetimeIndex(data.time.values)
            expected = pd.date_range(manifest["forcing_start"], manifest["forcing_end_inclusive"], freq="D")
            if not dates.equals(expected):
                raise ValueError(f"{variable} 不是完整的共同逐日时间轴")
            expected_units = "degree_Celsius" if variable == "t" else "mm/day"
            if data[variable].attrs.get("units") != expected_units:
                raise ValueError(f"{variable} 单位不是 {expected_units}")
            if inspect_values:
                means = []
                minimum, maximum = np.inf, -np.inf
                for start in range(0, len(dates), 365):
                    values = data[variable].isel(time=slice(start, start+365)).values
                    if not np.isfinite(values).all() or (variable != "t" and (values < 0).any()):
                        raise ValueError(f"{variable} 文件仍有缺测或非法负通量，日块起点 {dates[start]}")
                    means.extend((values * fraction).sum(axis=(1, 2)) / fraction.sum())
                    minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
                records[variable] = pd.Series(means, index=dates)
                summaries[variable] = {"minimum": minimum, "maximum": maximum, "units": expected_units,
                                       "missing_pixels": 0, "imputation": "none"}
    with xr.open_dataset(manifest["discharge"], engine="scipy") as discharge:
        dates_q = pd.DatetimeIndex(discharge.t.values)
        if not dates_q.equals(pd.date_range(dates_q[0], dates_q[-1], freq="D")):
            raise ValueError("站点流量不是连续联合日历")
        if discharge.discharge.attrs.get("units") != "m3/s":
            raise ValueError("流量单位应为 m3/s")
        original = discharge.get("discharge_observed", discharge.discharge)
        if "station_id" not in discharge or "is_outlet" not in discharge:
            raise ValueError("站点流量缺少 station_id/is_outlet")
        station_ids = [str(value) for value in discharge.station_id.values.tolist()]
        outlet_flags = discharge.is_outlet.values.astype(bool)
        if outlet_flags.sum() != 1 or len(set(station_ids)) != len(station_ids):
            raise ValueError("站点流量必须包含唯一站点ID和唯一总出口")
        outlet_index = int(np.flatnonzero(outlet_flags)[0])
        values = original.values
        q = pd.Series(values[:, outlet_index], index=dates_q)
        stations = {
            station_id: {
                "valid_observed_days": int(np.isfinite(values[:, index]).sum()),
                "missing_days_on_union_calendar": int((~np.isfinite(values[:, index])).sum()),
                "role": "outlet" if outlet_flags[index] else "internal",
            }
            for index, station_id in enumerate(station_ids)
        }
        discharge_report = {"start": str(dates_q[0].date()), "end_inclusive": str(dates_q[-1].date()),
                            "station_count": len(station_ids), "stations": stations,
                            "outlet_station_id": station_ids[outlet_index], "imputation": "none",
                            "quality_note": "ckhs/qceq/qcol retained; no unverified quality-threshold filtering"}
    report = {"basin": manifest["basin"], "grid_crs": "EPSG:3035", "grid_shape": [len(y), len(x)],
              "cell_size_m": spacing, "area_km2": area_km2, "forcing": summaries,
              "discharge": discharge_report, "terrain": terrain_report,
              "figure_contract": {"claim": "Forcings, terrain and gauges are co-located on one daily hydro-grid",
                                  "backend": "Python", "export": "JPG only", "archetype": "quantitative grid",
                                  "review_risks": ["Copernicus DEM is a DSM", "Boundary-constrained D8 can condition flats",
                                                   "Reprojection does not create subgrid meteorological information",
                                                   "P-PET-Q is not a mass-balance residual (PET is not AET)"]}}
    report_dir = Path(report_dir)
    write_json(report_dir / "basin_data_checks.json", report)
    if records:
        table = pd.DataFrame(records)
        table["qobs_m3_s"] = q.reindex(table.index)
        table["qobs_mm_day"] = table.qobs_m3_s * 86400 / (area_km2 * 1000)
        table.index.name = "date"
        table.to_csv(report_dir / "basin_daily_climate_discharge.csv", encoding="utf-8-sig")
    print(f"[数据核验通过] {manifest['basin']} | 强迫单位/网格/日历一致 | {area_km2:.2f} km2", flush=True)
    return report


def validate_simulation_period(manifest, params):
    """强迫保存全共同区间，但本次训练/测试必须落在强迫和日流量覆盖交集。"""
    required_start = min(pd.Timestamp(params.train_start), pd.Timestamp(params.test_start) - pd.Timedelta(days=params.spinup_days))
    required_end = max(pd.Timestamp(params.train_end), pd.Timestamp(params.test_end))
    available_start = pd.Timestamp(manifest["forcing_start"])
    available_end = pd.Timestamp(manifest["forcing_end_inclusive"]) + pd.Timedelta(days=1)
    if required_start < available_start or required_end > available_end:
        raise ValueError(f"运行期超出强迫覆盖 [{available_start.date()}, {available_end.date()})；预热也需真实强迫")
    with xr.open_dataset(manifest["discharge"], engine="scipy") as data:
        dates = pd.DatetimeIndex(data.t.values)
        for label, start, end in (("训练", params.train_start, params.train_end), ("测试", params.test_start, params.test_end)):
            expected = pd.date_range(start, end, freq="D", inclusive="left")
            if not expected.isin(dates).all():
                raise ValueError(f"{label}期超出出口日流量覆盖 [{dates[0].date()}, {(dates[-1]+pd.Timedelta(days=1)).date()})")
            observation = data.get("discharge_observed", data.discharge).sel(t=expected).values
            valid = np.isfinite(observation) & (observation >= 0)
            valid_by_station = valid.sum(axis=0)
            if np.any(valid_by_station < 2):
                station_ids = [str(value) for value in data.station_id.values.tolist()]
                invalid = [
                    station_ids[index] for index in np.flatnonzero(valid_by_station < 2)
                ]
                raise ValueError(
                    f"{label}期以下站点有效原始流量不足2日：{invalid}"
                )
            if not valid.all():
                print(
                    f"[观测缺测] {label}期 {int((~valid).sum())} station-days；"
                    "保留NaN并从对应站点损失/指标中排除",
                    flush=True,
                )
