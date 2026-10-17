"""Shared state, flux, and output types for the distributed HBV92 model."""

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class HBV92State:
    """Five end-of-day HBV92 grid storages, all in mm."""

    snowpack: torch.Tensor
    meltwater: torch.Tensor
    soil_moisture: torch.Tensor
    upper_zone: torch.Tensor
    lower_zone: torch.Tensor


@dataclass
class DistributedHydroState:
    """Grid HBV92 state and D8 kinematic-routing state."""

    runoff: HBV92State
    routing: Any


@dataclass
class HBV92Fluxes:
    """One-day HBV92 grid fluxes; water quantities are in mm/day."""

    rain: torch.Tensor
    snow: torch.Tensor
    melt: torch.Tensor
    refreezing: torch.Tensor
    to_soil: torch.Tensor
    soil_wetness: torch.Tensor
    recharge: torch.Tensor
    excess: torch.Tensor
    evaporation_factor: torch.Tensor
    actual_et: torch.Tensor
    percolation: torch.Tensor
    q0: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    water_balance: torch.Tensor


@dataclass
class HydroModelOutput:
    """Gauge discharge, final coupled state, and window diagnostics."""
    discharge_daily: torch.Tensor
    final_state: DistributedHydroState
    diagnostics: Dict[str, torch.Tensor]
