"""Grid-distributed HBV coupled to D8 kinematic-wave routing."""

from __future__ import annotations

import torch
from torch import nn

from .d8_kinematic_wave import D8KinematicWave
from .model_types import DistributedHydroState, HydroModelOutput
from .parameterized_grid_hbv import ParameterizedGridHBV92


class DistributedHBVGridRouting(nn.Module):
    """Neural-parameterized grid HBV followed by dynamic grid routing."""

    def __init__(self, runoff_generator: ParameterizedGridHBV92, grid_router: nn.Module):
        super().__init__()
        self.runoff_generator = runoff_generator
        self.grid_router = grid_router

    def generate_runoff_parameters(self, terrain, parameter_climate_sequence, basin_mask):
        return self.runoff_generator.generate_parameters(
            terrain, parameter_climate_sequence, basin_mask
        )

    def initial_state(self, terrain, runoff_parameters):
        if terrain.ndim != 4:
            raise ValueError("terrain must have shape [B,C,H,W]")
        return DistributedHydroState(
            runoff=self.runoff_generator.initial_state(runoff_parameters),
            routing=self.grid_router.initialize_state(
                terrain.shape[0], dtype=terrain.dtype, device=terrain.device
            ),
        )

    def generate_runoff_sequence(
        self, precipitation, temperature, pet,
        cell_area_km2, runoff_parameters, runoff_state, *, retain_fluxes,
    ):
        if precipitation.ndim != 5 or precipitation.shape[2] != 1:
            raise ValueError("Forcing tensors must have shape [B,T,1,H,W]")
        if any(value.shape != precipitation.shape for value in (temperature, pet)):
            raise ValueError("P, T and PET shapes must match")
        area = cell_area_km2
        if area.ndim == 2:
            area = area[None, None]
        elif area.ndim == 3:
            area = area[:, None]
        area = area.to(dtype=precipitation.dtype, device=precipitation.device)
        basin_area = area.sum(dim=(1, 2, 3)).clamp_min(1.0e-8)
        q0_steps, q1_steps, q2_steps = [], [], []
        flux_steps = [] if retain_fluxes else None
        state_steps = ({
            name: [] for name in (
                "snowpack_storage", "meltwater_storage",
                "soil_moisture_storage", "upper_zone_storage",
                "lower_zone_storage",
            )
        } if retain_fluxes else None)
        balance_steps = []
        for time_index in range(precipitation.shape[1]):
            fluxes, runoff_state = self.runoff_generator.forward_step(
                precipitation[:, time_index], temperature[:, time_index],
                pet[:, time_index], runoff_state, runoff_parameters,
            )
            q0_steps.append(fluxes.q0)
            q1_steps.append(fluxes.q1)
            q2_steps.append(fluxes.q2)
            if flux_steps is not None:
                flux_steps.append(fluxes)
                for name, field in (
                    ("snowpack_storage", runoff_state.snowpack),
                    ("meltwater_storage", runoff_state.meltwater),
                    ("soil_moisture_storage", runoff_state.soil_moisture),
                    ("upper_zone_storage", runoff_state.upper_zone),
                    ("lower_zone_storage", runoff_state.lower_zone),
                ):
                    state_steps[name].append(
                        ((field.detach() * area).sum(dim=(1, 2, 3)) / basin_area)
                    )
            balance_steps.append(
                (fluxes.water_balance * area).abs().sum(dim=(1, 2, 3)) / basin_area
            )
        return (
            torch.stack(q0_steps, dim=1), torch.stack(q1_steps, dim=1),
            torch.stack(q2_steps, dim=1),
            flux_steps,
            ({
                name: torch.stack(sequence, dim=1)
                for name, sequence in state_steps.items()
            } if state_steps is not None else None),
            torch.stack(balance_steps, dim=1), runoff_state,
        )

    def advance_state(self, arguments, state, runoff_parameters=None):
        if runoff_parameters is None:
            runoff_parameters = self.generate_runoff_parameters(
                arguments["terrain"], arguments["parameter_climate_sequence"],
                arguments["basin_mask"],
            )
        q0, q1, q2, _, _, _, runoff_state = self.generate_runoff_sequence(
            arguments["precipitation"], arguments["temperature"], arguments["pet"],
            arguments["cell_area_km2"], runoff_parameters, state.runoff,
            retain_fluxes=False,
        )
        _, routing_state, _ = self.grid_router(
            q0 + q1 + q2, arguments["cell_area_km2"], state.routing
        )
        return DistributedHydroState(runoff=runoff_state, routing=routing_state)

    def warmup(self, **arguments):
        parameters = self.generate_runoff_parameters(
            arguments["terrain"], arguments["parameter_climate_sequence"],
            arguments["basin_mask"],
        )
        state = self.initial_state(arguments["terrain"], parameters)
        return self.advance_state(arguments, state, parameters)

    def forward(
        self, precipitation, temperature, pet, terrain,
        parameter_climate_sequence, basin_mask, cell_area_km2, state=None,
    ):
        parameters = self.generate_runoff_parameters(
            terrain, parameter_climate_sequence, basin_mask
        )
        if state is None:
            state = self.initial_state(terrain, parameters)
        area = cell_area_km2
        if area.ndim == 2:
            area = area[None, None]
        elif area.ndim == 3:
            area = area[:, None]
        if area.ndim != 4:
            raise ValueError(
                "cell_area_km2 must have shape [H,W], [B,H,W], or [B,1,H,W]"
            )
        area = area.to(
            dtype=precipitation.dtype, device=precipitation.device
        )
        basin_area = area.sum(dim=(1, 2, 3)).clamp_min(1.0e-8)

        def area_mean(field):
            return (field * area).sum(dim=(1, 2, 3)) / basin_area

        initial_storage = area_mean(
            state.runoff.snowpack + state.runoff.meltwater
            + state.runoff.soil_moisture + state.runoff.upper_zone
            + state.runoff.lower_zone
        )
        (
            q0, q1, q2, flux_steps, state_diagnostics,
            runoff_balance, runoff_state,
        ) = self.generate_runoff_sequence(
            precipitation, temperature, pet,
            cell_area_km2, parameters, state.runoff, retain_fluxes=True,
        )
        combined_runoff = q0 + q1 + q2
        discharge, routing_state, routing_diagnostics = self.grid_router(
            combined_runoff, cell_area_km2, state.routing
        )
        discharge_factor = 1000.0 / 86400.0
        generated_q0 = (q0 * area[:, None]).sum(dim=(2, 3, 4)) * discharge_factor
        generated_q1 = (q1 * area[:, None]).sum(dim=(2, 3, 4)) * discharge_factor
        generated_q2 = (q2 * area[:, None]).sum(dim=(2, 3, 4)) * discharge_factor
        final_storage = area_mean(
            runoff_state.snowpack + runoff_state.meltwater
            + runoff_state.soil_moisture + runoff_state.upper_zone
            + runoff_state.lower_zone
        )

        names = (
            "prec", "temp", "ep", "precipitation", "rainfall", "snowfall",
            "snowmelt", "refreezing", "snow_liquid_release", "soil_wetness",
            "soil_excess", "recharge", "actual_et", "evaporation_factor",
            "percolation", "q0", "q1", "q2", "total_runoff",
        )
        values = {name: [] for name in names}
        for index, fluxes in enumerate(flux_steps):
            values["prec"].append(area_mean(precipitation[:, index]))
            values["temp"].append(area_mean(temperature[:, index]))
            values["ep"].append(area_mean(pet[:, index]))
            values["precipitation"].append(area_mean(fluxes.rain + fluxes.snow))
            for name, field in (
                ("rainfall", fluxes.rain), ("snowfall", fluxes.snow),
                ("snowmelt", fluxes.melt), ("refreezing", fluxes.refreezing),
                ("snow_liquid_release", fluxes.to_soil),
                ("soil_wetness", fluxes.soil_wetness),
                ("soil_excess", fluxes.excess), ("recharge", fluxes.recharge),
                ("actual_et", fluxes.actual_et),
                ("evaporation_factor", fluxes.evaporation_factor),
                ("percolation", fluxes.percolation),
                ("q0", fluxes.q0), ("q1", fluxes.q1), ("q2", fluxes.q2),
            ):
                values[name].append(area_mean(field))
            values["total_runoff"].append(area_mean(combined_runoff[:, index]))
        diagnostics = {name: torch.stack(sequence, dim=1) for name, sequence in values.items()}
        diagnostics.update(state_diagnostics)

        active = self.grid_router.active_global_index
        gauge_mask = self.grid_router.gauge_upstream_mask.to(
            device=area.device, dtype=area.dtype
        )
        area_active = area.reshape(area.shape[0], -1)[:, active]
        gauge_area = torch.einsum("bn,gn->bg", area_active, gauge_mask).clamp_min(1.0e-8)

        def gauge_mean(field):
            flat = field[:, :, 0].reshape(field.shape[0], field.shape[1], -1)[
                :, :, active
            ]
            return torch.einsum(
                "btn,bn,gn->btg", flat, area_active, gauge_mask
            ) / gauge_area[:, None, :]

        with torch.no_grad():
            actual_et_grid = torch.stack(
                [fluxes.actual_et for fluxes in flux_steps], dim=1
            )
            recharge_grid = torch.stack(
                [fluxes.recharge for fluxes in flux_steps], dim=1
            )
            gauge_precipitation = gauge_mean(precipitation.detach())
            gauge_actual_et = gauge_mean(actual_et_grid.detach())
            gauge_recharge = gauge_mean(recharge_grid.detach())
            gauge_q0 = gauge_mean(q0.detach())
            gauge_q1 = gauge_mean(q1.detach())
            gauge_q2 = gauge_mean(q2.detach())
            gauge_total_runoff = gauge_q0 + gauge_q1 + gauge_q2
            diagnostics.update({
                "gauge_upstream_area_km2": gauge_area.detach(),
                "gauge_precipitation": gauge_precipitation,
                "gauge_actual_et": gauge_actual_et,
                "gauge_recharge": gauge_recharge,
                "gauge_q0": gauge_q0,
                "gauge_q1": gauge_q1,
                "gauge_q2": gauge_q2,
                "gauge_total_runoff": gauge_total_runoff,
                "gauge_generated_total_m3_s": (
                    gauge_total_runoff * gauge_area[:, None, :] * discharge_factor
                ),
            })
            parameter_mask = basin_mask
            if parameter_mask.ndim == 4 and parameter_mask.shape[1] == 1:
                parameter_mask = parameter_mask[:, 0]
            parameter_mask = parameter_mask.bool()
            for name, lower, upper in (
                self.runoff_generator.parameterization.parameter_bounds
            ):
                field = parameters[name][:, 0][parameter_mask].detach()
                fraction = (field - lower) / (upper - lower)
                diagnostics.update({
                    f"hbv_parameter_{name}_mean": field.mean(),
                    f"hbv_parameter_{name}_q05": torch.quantile(field, 0.05),
                    f"hbv_parameter_{name}_median": torch.quantile(field, 0.50),
                    f"hbv_parameter_{name}_q95": torch.quantile(field, 0.95),
                    f"hbv_parameter_{name}_near_lower_fraction": (
                        fraction <= 0.01
                    ).float().mean(),
                    f"hbv_parameter_{name}_near_upper_fraction": (
                        fraction >= 0.99
                    ).float().mean(),
                })
        diagnostics.update({
            "quickflow": diagnostics["q0"],
            "interflow": diagnostics["q1"],
            "baseflow": diagnostics["q2"],
            "runoff_storage_initial": initial_storage,
            "runoff_storage_final": final_storage,
            "runoff_water_balance": runoff_balance,
            "generated_q0_m3_s": generated_q0,
            "generated_q1_m3_s": generated_q1,
            "generated_q2_m3_s": generated_q2,
            "generated_total_m3_s": generated_q0 + generated_q1 + generated_q2,
        })
        diagnostics.update(routing_diagnostics)
        return HydroModelOutput(
            discharge_daily=discharge,
            final_state=DistributedHydroState(runoff=runoff_state, routing=routing_state),
            diagnostics=diagnostics,
        )


def build_distributed_hbv_model(params, routing_topology, grid_gauges):
    if routing_topology is None or grid_gauges is None:
        raise ValueError("routing_topology and grid_gauges are required")
    runoff_generator = ParameterizedGridHBV92(
        terrain_channels=params.terrain_channels,
        parameter_hidden=params.hbv_parameter_hidden,
        parameter_layers=params.hbv_parameter_layers,
        parameter_cell_batch_size=params.hbv_parameter_cell_batch_size,
        parameter_encoder=params.parameter_encoder,
    )
    router = D8KinematicWave(
        routing_topology,
        grid_gauges,
        routing_substeps_per_day=params.d8_substeps_per_day,
    )
    return DistributedHBVGridRouting(runoff_generator, router)
