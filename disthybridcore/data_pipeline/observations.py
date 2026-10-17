"""Read aligned daily multi-gauge discharge observations."""

import numpy as np
import pandas as pd
import torch
import xarray as xr


class DailyDischargeReader:
    """Read prepared daily discharge in the requested gauge-ID order."""

    def __init__(
        self,
        path: str,
        variable: str = "discharge",
        station_ids: list[str] | tuple[str, ...] = (),
    ):
        self.dataset = xr.open_dataset(path, engine="scipy", decode_timedelta=True)
        if variable not in self.dataset:
            raise KeyError(f"Discharge variable {variable!r} not found")
        original = self.dataset.get(f"{variable}_observed", self.dataset[variable])
        self.data = original.where(np.isfinite(original) & (original >= 0))
        if "station_id" not in self.dataset or "name" not in self.dataset:
            raise ValueError("Prepared discharge file lacks station_id/name")
        source_ids = [str(value) for value in np.asarray(self.dataset.station_id).tolist()]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Prepared discharge station ids must be unique")
        requested = [str(value) for value in station_ids]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("station_ids must be a non-empty unique sequence")
        missing = [value for value in requested if value not in source_ids]
        if missing:
            raise ValueError(f"Discharge stations are missing: {missing}")
        indices = np.asarray([source_ids.index(value) for value in requested])
        self.data = self.data.isel(s=indices)
        self.station_ids = [source_ids[index] for index in indices]
        source_names = [str(value) for value in np.asarray(self.dataset.name).tolist()]
        self.station_names = [source_names[index] for index in indices]
        if "is_outlet" not in self.dataset:
            raise ValueError("Prepared discharge file lacks is_outlet")
        outlet_flags = np.asarray(self.dataset.is_outlet).astype(bool)[indices]
        if outlet_flags.sum() != 1:
            raise ValueError("Selected discharge stations must contain one outlet")
        self.outlet_index = int(np.flatnonzero(outlet_flags)[0])

    @property
    def stations(self) -> int:
        return int(self.data.sizes["s"])

    def read(self, start_date, days: int) -> torch.Tensor:
        """Return observed discharge as [day, 1, station] in m3/s."""
        dates = pd.date_range(pd.Timestamp(start_date), periods=days, freq="D")
        values = np.asarray(self.data.sel(t=dates).transpose("t", "s"))
        return torch.from_numpy(values.astype(np.float32, copy=True)).unsqueeze(1)

    def close(self) -> None:
        self.dataset.close()
