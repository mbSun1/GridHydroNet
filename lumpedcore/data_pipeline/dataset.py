"""Continuous daily basin-mean forcings aligned with outlet discharge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from .data_catalog import LumpedDataCatalog
from .observations import read_outlet_discharge


@dataclass(frozen=True)
class LumpedBasinDataset:
    dates: pd.DatetimeIndex
    precipitation: np.ndarray
    temperature: np.ndarray
    pet: np.ndarray
    observed_discharge: np.ndarray
    basin_area_km2: float
    outlet_id: str
    outlet_name: str

    def period_mask(self, start, end_exclusive) -> np.ndarray:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end_exclusive)
        return np.asarray((self.dates >= start) & (self.dates < end))

    def require_observations(self, start, end_exclusive, label: str) -> None:
        values = self.observed_discharge[self.period_mask(start, end_exclusive)]
        valid = np.isfinite(values) & (values >= 0)
        if valid.sum() < 2:
            raise ValueError(f"{label} has fewer than two valid outlet observations")


def load_lumped_dataset(
    catalog: LumpedDataCatalog,
    start,
    end_exclusive,
) -> LumpedBasinDataset:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end_exclusive)
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("Dataset bounds must be increasing midnight day boundaries")
    expected = pd.date_range(start, end, freq="D", inclusive="left")
    with xr.open_dataset(catalog.forcing, engine="h5netcdf") as data:
        forcing_dates = pd.DatetimeIndex(np.asarray(data.time))
        if not expected.isin(forcing_dates).all():
            raise ValueError(
                f"Requested period [{start.date()}, {end.date()}) exceeds basin-mean forcing"
            )
        selected = data.sel(time=expected)
        arrays = {
            variable: np.asarray(selected[variable], dtype=np.float64).copy()
            for variable in ("prec", "t", "pet")
        }
    for variable, values in arrays.items():
        if not np.isfinite(values).all() or (variable != "t" and (values < 0).any()):
            raise ValueError(f"Invalid basin-mean forcing values in {variable}")
    observations = read_outlet_discharge(catalog.discharge).reindex(expected).to_numpy()
    return LumpedBasinDataset(
        dates=expected,
        precipitation=arrays["prec"],
        temperature=arrays["t"],
        pet=arrays["pet"],
        observed_discharge=np.asarray(observations, dtype=np.float64),
        basin_area_km2=catalog.basin_area_km2,
        outlet_id=catalog.outlet_id,
        outlet_name=catalog.outlet_name,
    )
