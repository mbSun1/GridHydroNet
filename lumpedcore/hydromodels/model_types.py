"""Parameter, state, and result types for the lumped HBV92 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class HBV92Parameters:
    beta: float
    fc: float
    k0: float
    k1: float
    k2: float
    lp: float
    perc: float
    uzl: float
    tt: float
    cfmax: float
    cfr: float
    cwh: float

    @staticmethod
    def bounds() -> tuple[tuple[str, float, float], ...]:
        """Return fresh, immutable parameter-bound definitions."""
        return (
            ("beta", 1.0, 6.0),
            ("fc", 50.0, 1000.0),
            ("k0", 0.05, 0.9),
            ("k1", 0.01, 0.5),
            ("k2", 0.001, 0.2),
            ("lp", 0.2, 1.0),
            ("perc", 0.0, 10.0),
            ("uzl", 0.0, 100.0),
            ("tt", -2.5, 2.5),
            ("cfmax", 0.5, 10.0),
            ("cfr", 0.0, 0.1),
            ("cwh", 0.0, 0.2),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "HBV92Parameters":
        bounds = cls.bounds()
        missing = [name for name, _, _ in bounds if name not in values]
        if missing:
            raise KeyError("Missing HBV92 parameters: " + ", ".join(missing))
        parameters = cls(**{
            name: float(values[name]) for name, _, _ in bounds
        })
        parameters.validate()
        return parameters

    def validate(self) -> None:
        values = asdict(self)
        for name, lower, upper in self.bounds():
            value = values[name]
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name}={value} is outside [{lower}, {upper}]")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class HBV92State:
    snowpack: float
    meltwater: float
    soil_moisture: float
    upper_zone: float
    lower_zone: float


@dataclass(frozen=True)
class HBV92Result:
    runoff_mm_day: np.ndarray
    fluxes: dict[str, np.ndarray]
    states: dict[str, np.ndarray]
    final_state: HBV92State
