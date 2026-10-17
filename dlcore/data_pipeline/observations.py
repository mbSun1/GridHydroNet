"""Read the unique prepared outlet discharge without interpolation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def outlet_metadata(path: str | Path) -> dict:
    path = Path(path)
    with xr.open_dataset(path, engine="scipy", decode_timedelta=True) as data:
        required = {"station_id", "name", "is_outlet", "discharge"}
        missing = sorted(required.difference(data.variables))
        if missing:
            raise ValueError(f"Prepared discharge lacks: {', '.join(missing)}")
        outlet_flags = np.asarray(data.is_outlet).astype(bool)
        if outlet_flags.sum() != 1:
            raise ValueError("Prepared discharge must contain exactly one outlet")
        index = int(np.flatnonzero(outlet_flags)[0])
        station_ids = [str(value) for value in np.asarray(data.station_id).tolist()]
        station_names = [str(value) for value in np.asarray(data.name).tolist()]
        official_area = (
            float(np.asarray(data.official_area_km2)[index])
            if "official_area_km2" in data
            else np.nan
        )
    return {
        "outlet_index": index,
        "outlet_id": station_ids[index],
        "outlet_name": station_names[index],
        "official_area_km2": official_area,
    }


def read_outlet_discharge(path: str | Path) -> pd.Series:
    path = Path(path)
    metadata = outlet_metadata(path)
    with xr.open_dataset(path, engine="scipy", decode_timedelta=True) as data:
        source = data.get("discharge_observed", data["discharge"])
        if source.attrs.get("units") != "m3/s":
            raise ValueError("Prepared outlet discharge units must be m3/s")
        values = np.asarray(source[:, metadata["outlet_index"]], dtype=np.float64)
        values[~np.isfinite(values) | (values < 0)] = np.nan
        dates = pd.DatetimeIndex(np.asarray(data.t))
    expected = pd.date_range(dates[0], dates[-1], freq="D")
    if not dates.equals(expected):
        raise ValueError("Prepared discharge does not use a continuous daily calendar")
    return pd.Series(values, index=dates, name="observed_discharge_m3_s")
