"""Single-component, daily, grid-distributed HBV92 runoff generation.

The equations and 12-parameter meanings follow
``referenceCode/physDL_HBV/.../rnn.py::HBVMul``.  Each active grid cell has
one parameter set and one set of five storages.  The reference model's
multi-component averaging and Gamma routing are not part of this module.
"""

import torch
from torch import nn

from .model_types import HBV92Fluxes, HBV92State


class GridHBV92Daily(nn.Module):
    """Advance the single-component HBV92 water balance by one day."""

    def initial_state(self, params: dict[str, torch.Tensor]) -> HBV92State:
        """Initialize the five storages as in the reference implementation."""
        initial = torch.full_like(params["fc"], 1.0e-3)
        return HBV92State(
            snowpack=initial.clone(),
            meltwater=initial.clone(),
            soil_moisture=initial.clone(),
            upper_zone=initial.clone(),
            lower_zone=initial.clone(),
        )

    def forward_step(
        self,
        precipitation: torch.Tensor,
        temperature: torch.Tensor,
        pet: torch.Tensor,
        state: HBV92State,
        params: dict[str, torch.Tensor],
    ) -> tuple[HBV92Fluxes, HBV92State]:
        """Run the reference HBV92 precipitation, soil, ET, and response steps.

        Inputs and fluxes are daily water depths in mm except temperature in
        degrees Celsius.  State and parameter fields have shape ``[B,1,H,W]``.
        The hard ``temperature >= TT`` phase split intentionally matches the
        reference implementation; the comparison has no derivative with
        respect to ``TT``.
        """
        beta = params["beta"]
        fc = params["fc"]
        k0 = params["k0"]
        k1 = params["k1"]
        k2 = params["k2"]
        lp = params["lp"]
        perc_parameter = params["perc"]
        uzl = params["uzl"]
        tt = params["tt"]
        cfmax = params["cfmax"]
        cfr = params["cfr"]
        cwh = params["cwh"]

        # Precipitation phase partition.
        rain = precipitation * (temperature >= tt).to(precipitation.dtype)
        snow = precipitation * (temperature < tt).to(precipitation.dtype)

        # Snow accumulation, melt, refreezing, and liquid-water release.
        snowpack = state.snowpack + snow
        melt = torch.minimum(
            torch.clamp(cfmax * (temperature - tt), min=0.0),
            snowpack,
        )
        meltwater = state.meltwater + melt
        snowpack = snowpack - melt

        refreezing = torch.minimum(
            torch.clamp(cfr * cfmax * (tt - temperature), min=0.0),
            meltwater,
        )
        snowpack = snowpack + refreezing
        meltwater = meltwater - refreezing
        to_soil = torch.clamp(meltwater - cwh * snowpack, min=0.0)
        meltwater = meltwater - to_soil

        # Soil moisture, recharge, capacity excess, and actual ET.
        soil_wetness = torch.clamp(
            torch.pow(state.soil_moisture / fc, beta), min=0.0, max=1.0
        )
        recharge = (rain + to_soil) * soil_wetness
        soil_moisture = state.soil_moisture + rain + to_soil - recharge
        excess = torch.clamp(soil_moisture - fc, min=0.0)
        soil_moisture = soil_moisture - excess

        evaporation_factor = torch.clamp(
            soil_moisture / (lp * fc), min=0.0, max=1.0
        )
        actual_et = torch.minimum(soil_moisture, pet * evaporation_factor)
        soil_moisture = torch.clamp(
            soil_moisture - actual_et, min=1.0e-5
        )

        # Upper- and lower-zone response.  The update order is part of the
        # reference formulation: PERC, then Q0, then Q1, then Q2.
        upper_zone = state.upper_zone + recharge + excess
        percolation = torch.minimum(upper_zone, perc_parameter)
        upper_zone = upper_zone - percolation

        q0 = k0 * torch.clamp(upper_zone - uzl, min=0.0)
        upper_zone = upper_zone - q0
        q1 = k1 * upper_zone
        upper_zone = upper_zone - q1

        lower_zone = state.lower_zone + percolation
        q2 = k2 * lower_zone
        lower_zone = lower_zone - q2

        new_state = HBV92State(
            snowpack=snowpack,
            meltwater=meltwater,
            soil_moisture=soil_moisture,
            upper_zone=upper_zone,
            lower_zone=lower_zone,
        )
        storage_old = (
            state.snowpack + state.meltwater + state.soil_moisture
            + state.upper_zone + state.lower_zone
        )
        storage_new = (
            snowpack + meltwater + soil_moisture + upper_zone + lower_zone
        )
        water_balance = (
            storage_old + precipitation - actual_et - q0 - q1 - q2 - storage_new
        )
        return HBV92Fluxes(
            rain=rain,
            snow=snow,
            melt=melt,
            refreezing=refreezing,
            to_soil=to_soil,
            soil_wetness=soil_wetness,
            recharge=recharge,
            excess=excess,
            evaporation_factor=evaporation_factor,
            actual_et=actual_et,
            percolation=percolation,
            q0=q0,
            q1=q1,
            q2=q2,
            water_balance=water_balance,
        ), new_state
