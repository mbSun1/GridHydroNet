"""Continuous daily P/T/PET series aligned with outlet discharge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from dlcore.data_pipeline.data_catalog import DLDataCatalog
from dlcore.data_pipeline.observations import read_outlet_discharge


@dataclass(frozen=True)
class DLBasinSeries:
    dates: pd.DatetimeIndex
    forcing: np.ndarray
    observed_discharge: np.ndarray
    basin_area_km2: float
    outlet_id: str
    outlet_name: str

    def period_mask(self, start, end_exclusive) -> np.ndarray:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end_exclusive)
        return np.asarray((self.dates >= start) & (self.dates < end))


def load_basin_series(
    catalog: DLDataCatalog,
    start,
    end_exclusive,
) -> DLBasinSeries:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end_exclusive)
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("Dataset bounds must be increasing midnight day boundaries")
    expected = pd.date_range(start, end, freq="D", inclusive="left")
    with xr.open_dataset(catalog.forcing, engine="h5netcdf") as data:
        expected_units = {
            "prec": "mm/day",
            "t": "degree_Celsius",
            "pet": "mm/day",
        }
        missing = sorted(set(expected_units).difference(data.variables))
        if missing:
            raise ValueError(f"Basin-mean forcing lacks: {', '.join(missing)}")
        for variable, units in expected_units.items():
            if data[variable].attrs.get("units") != units:
                raise ValueError(
                    f"Unexpected units for {variable}: {data[variable].attrs.get('units')}"
                )
        forcing_dates = pd.DatetimeIndex(np.asarray(data.time))
        if not expected.isin(forcing_dates).all():
            raise ValueError(
                f"Requested period [{start.date()}, {end.date()}) exceeds basin-mean forcing"
            )
        selected = data.sel(time=expected)
        arrays = [
            np.asarray(selected[variable], dtype=np.float64).copy()
            for variable in ("prec", "t", "pet")
        ]
    forcing = np.column_stack(arrays)
    if not np.isfinite(forcing).all():
        raise ValueError("P/T/PET forcing contains NaN or Inf")
    if (forcing[:, (0, 2)] < 0).any():
        raise ValueError("Precipitation and PET must be nonnegative")
    observations = (
        read_outlet_discharge(catalog.discharge).reindex(expected).to_numpy().copy()
    )
    return DLBasinSeries(
        dates=expected,
        forcing=forcing,
        observed_discharge=np.asarray(observations, dtype=np.float64),
        basin_area_km2=catalog.basin_area_km2,
        outlet_id=catalog.outlet_id,
        outlet_name=catalog.outlet_name,
    )
