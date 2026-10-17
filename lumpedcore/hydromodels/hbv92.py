"""Single-component, daily, lumped HBV92 water balance."""

from __future__ import annotations

import numpy as np

from .model_types import HBV92Parameters, HBV92Result, HBV92State


class LumpedHBV92:
    """Advance the same five-store HBV92 equations used by GridHBV92Daily."""

    @staticmethod
    def initial_state() -> HBV92State:
        return HBV92State(
            snowpack=1.0e-3,
            meltwater=1.0e-3,
            soil_moisture=1.0e-3,
            upper_zone=1.0e-3,
            lower_zone=1.0e-3,
        )

    @staticmethod
    def forward_step(
        precipitation: float,
        temperature: float,
        pet: float,
        state: HBV92State,
        parameters: HBV92Parameters,
    ) -> tuple[dict[str, float], HBV92State]:
        if not np.isfinite([precipitation, temperature, pet]).all():
            raise ValueError("HBV92 forcing must be finite")
        if precipitation < 0 or pet < 0:
            raise ValueError("HBV92 precipitation and PET must be non-negative")

        rain = precipitation if temperature >= parameters.tt else 0.0
        snow = precipitation if temperature < parameters.tt else 0.0

        snowpack = state.snowpack + snow
        melt = min(max(parameters.cfmax * (temperature - parameters.tt), 0.0), snowpack)
        meltwater = state.meltwater + melt
        snowpack -= melt

        refreezing = min(
            max(parameters.cfr * parameters.cfmax * (parameters.tt - temperature), 0.0),
            meltwater,
        )
        snowpack += refreezing
        meltwater -= refreezing
        to_soil = max(meltwater - parameters.cwh * snowpack, 0.0)
        meltwater -= to_soil

        soil_wetness = min(
            max((state.soil_moisture / parameters.fc) ** parameters.beta, 0.0),
            1.0,
        )
        recharge = (rain + to_soil) * soil_wetness
        soil_moisture = state.soil_moisture + rain + to_soil - recharge
        excess = max(soil_moisture - parameters.fc, 0.0)
        soil_moisture -= excess

        evaporation_factor = min(
            max(soil_moisture / (parameters.lp * parameters.fc), 0.0),
            1.0,
        )
        actual_et = min(soil_moisture, pet * evaporation_factor)
        soil_moisture = max(soil_moisture - actual_et, 1.0e-5)

        upper_zone = state.upper_zone + recharge + excess
        percolation = min(upper_zone, parameters.perc)
        upper_zone -= percolation
        q0 = parameters.k0 * max(upper_zone - parameters.uzl, 0.0)
        upper_zone -= q0
        q1 = parameters.k1 * upper_zone
        upper_zone -= q1

        lower_zone = state.lower_zone + percolation
        q2 = parameters.k2 * lower_zone
        lower_zone -= q2

        new_state = HBV92State(
            snowpack=snowpack,
            meltwater=meltwater,
            soil_moisture=soil_moisture,
            upper_zone=upper_zone,
            lower_zone=lower_zone,
        )
        state_names = (
            "snowpack",
            "meltwater",
            "soil_moisture",
            "upper_zone",
            "lower_zone",
        )
        storage_old = sum(getattr(state, name) for name in state_names)
        storage_new = sum(getattr(new_state, name) for name in state_names)
        water_balance = (
            storage_old + precipitation - actual_et - q0 - q1 - q2 - storage_new
        )
        fluxes = {
            "rain": rain,
            "snow": snow,
            "melt": melt,
            "refreezing": refreezing,
            "to_soil": to_soil,
            "soil_wetness": soil_wetness,
            "recharge": recharge,
            "excess": excess,
            "evaporation_factor": evaporation_factor,
            "actual_et": actual_et,
            "percolation": percolation,
            "q0": q0,
            "q1": q1,
            "q2": q2,
            "water_balance": water_balance,
        }
        return fluxes, new_state

    def simulate(
        self,
        precipitation,
        temperature,
        pet,
        parameters: HBV92Parameters,
        initial_state: HBV92State | None = None,
    ) -> HBV92Result:
        parameters.validate()
        precipitation = np.asarray(precipitation, dtype=np.float64)
        temperature = np.asarray(temperature, dtype=np.float64)
        pet = np.asarray(pet, dtype=np.float64)
        if precipitation.ndim != 1 or not (
            precipitation.shape == temperature.shape == pet.shape
        ):
            raise ValueError("P, T, and PET must be matching one-dimensional arrays")
        flux_names = (
            "rain",
            "snow",
            "melt",
            "refreezing",
            "to_soil",
            "soil_wetness",
            "recharge",
            "excess",
            "evaporation_factor",
            "actual_et",
            "percolation",
            "q0",
            "q1",
            "q2",
            "water_balance",
        )
        state_names = (
            "snowpack",
            "meltwater",
            "soil_moisture",
            "upper_zone",
            "lower_zone",
        )
        flux_arrays = {
            name: np.empty(len(precipitation), dtype=np.float64)
            for name in flux_names
        }
        state_arrays = {
            name: np.empty(len(precipitation), dtype=np.float64)
            for name in state_names
        }
        state = initial_state or self.initial_state()
        for index, (p_value, t_value, pet_value) in enumerate(
            zip(precipitation, temperature, pet)
        ):
            fluxes, state = self.forward_step(
                float(p_value), float(t_value), float(pet_value), state, parameters
            )
            for name in flux_names:
                flux_arrays[name][index] = fluxes[name]
            for name in state_names:
                state_arrays[name][index] = getattr(state, name)
        runoff = flux_arrays["q0"] + flux_arrays["q1"] + flux_arrays["q2"]
        return HBV92Result(
            runoff_mm_day=runoff,
            fluxes=flux_arrays,
            states=state_arrays,
            final_state=state,
        )
