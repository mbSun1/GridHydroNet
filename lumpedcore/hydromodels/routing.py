"""Optional Gamma unit-hydrograph routing for basin-generated runoff."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RoutingResult:
    routed_runoff_mm_day: np.ndarray
    unit_hydrograph: np.ndarray
    tail_runoff_mm: np.ndarray


def gamma_routing_parameter_bounds() -> tuple[tuple[str, float, float], ...]:
    """Return Gamma-routing calibration bounds without module-level state."""
    return (
        ("routing_shape", 0.0, 2.9),
        ("routing_scale", 0.0, 6.5),
    )


def gamma_unit_hydrograph(shape: float, scale: float, length: int = 15) -> np.ndarray:
    """Match the physDL-HBV Gamma kernel and normalize it to unit mass."""
    if length < 1:
        raise ValueError("Gamma unit-hydrograph length must be positive")
    if not 0.0 <= shape <= 2.9 or not 0.0 <= scale <= 6.5:
        raise ValueError("Gamma routing parameters are outside their physical bounds")
    alpha = max(float(shape), 0.0) + 0.1
    theta = max(float(scale), 0.0) + 0.5
    time = np.arange(0.5, float(length), 1.0, dtype=np.float64)
    log_weights = (
        (alpha - 1.0) * np.log(time)
        - time / theta
        - math.lgamma(alpha)
        - alpha * math.log(theta)
    )
    weights = np.exp(log_weights)
    total = weights.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("Gamma routing produced invalid unit-hydrograph weights")
    return weights / total


def route_runoff_gamma(
    runoff_mm_day,
    shape: float,
    scale: float,
    length: int = 15,
) -> RoutingResult:
    runoff = np.asarray(runoff_mm_day, dtype=np.float64)
    if runoff.ndim != 1 or not np.isfinite(runoff).all() or (runoff < 0).any():
        raise ValueError("Runoff for Gamma routing must be finite, non-negative, and 1D")
    kernel = gamma_unit_hydrograph(shape, scale, length)
    full = np.convolve(runoff, kernel, mode="full")
    return RoutingResult(
        routed_runoff_mm_day=full[: len(runoff)],
        unit_hydrograph=kernel,
        tail_runoff_mm=full[len(runoff):],
    )


def runoff_mm_day_to_discharge(runoff_mm_day, basin_area_km2: float) -> np.ndarray:
    """Convert catchment runoff depth to mean daily discharge in m3/s."""
    runoff = np.asarray(runoff_mm_day, dtype=np.float64)
    if not np.isfinite(basin_area_km2) or basin_area_km2 <= 0:
        raise ValueError("basin_area_km2 must be finite and positive")
    return runoff * float(basin_area_km2) / 86.4
