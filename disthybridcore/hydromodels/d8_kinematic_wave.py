"""Differentiable D8 kinematic-wave routing with hillslope/channel fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class D8KinematicWaveState:
    discharge_m3_s: torch.Tensor


class D8KinematicWave(nn.Module):
    """Route combined HBV runoff through hillslope and channel D8 cells."""

    def __init__(
        self,
        topology,
        gauge_mapping,
        *,
        routing_substeps_per_day: int,
        a_min: float = 1.0e-3,
        a_max: float = 50.0,
        initial_a: float = 5.0,
        b_min: float = 1.0e-3,
        b_max: float = 1.0,
        initial_b: float = 0.60,
        reference_discharge_m3_s: float = 1.0e-4,
    ):
        super().__init__()
        if routing_substeps_per_day <= 0:
            raise ValueError("routing_substeps_per_day must be positive")
        if not (0 < a_min < initial_a < a_max):
            raise ValueError("kinematic-wave a bounds/initial value are invalid")
        if not (0 < b_min < initial_b < b_max <= 1.0):
            raise ValueError("kinematic-wave b bounds/initial value are invalid")
        if reference_discharge_m3_s <= 0:
            raise ValueError("reference_discharge_m3_s must be positive")

        mask = topology.basin_mask.bool().reshape(-1)
        active_global = torch.where(mask)[0]
        global_to_local = torch.full((mask.numel(),), -1, dtype=torch.long)
        global_to_local[active_global] = torch.arange(active_global.numel())
        downstream_global = topology.downstream_index.reshape(-1)[active_global].long()
        downstream_local = global_to_local[downstream_global]
        if torch.any(downstream_local < 0):
            raise ValueError("A D8 link leaves the active basin")
        level = topology.routing_level.reshape(-1)[active_global].long()
        gauge_local = global_to_local[gauge_mapping.gauge_cell_index.long()]
        if torch.any(gauge_local < 0):
            raise ValueError("A gauge is outside the active D8 graph")

        accumulation = topology.flow_accumulation.reshape(-1)[active_global].float()
        slope = topology.slope.reshape(-1)[active_global].float()
        elevation = topology.conditioned_dem.reshape(-1)[active_global].float()
        channel = topology.channel_mask.reshape(-1)[active_global].float()
        features = torch.stack(
            (
                torch.log1p(accumulation),
                torch.log10(slope.clamp_min(1.0e-6)),
                elevation,
                channel,
            ),
            dim=1,
        )
        feature_mean = features.mean(dim=0, keepdim=True)
        feature_std = features.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1.0e-6)
        features = (features - feature_mean) / feature_std

        self.register_buffer("active_global_index", active_global)
        self.register_buffer("downstream_local_index", downstream_local)
        self.register_buffer("channel_mask", channel.bool())
        self.register_buffer(
            "edge_length_m",
            topology.flow_distance_km.reshape(-1)[active_global].float() * 1000.0,
        )
        self.register_buffer("gauge_local_index", gauge_local)
        reverse_links = [[] for _ in range(int(active_global.numel()))]
        for source, destination in enumerate(downstream_local.tolist()):
            if source != destination:
                reverse_links[destination].append(source)
        gauge_upstream = torch.zeros(
            (gauge_local.numel(), active_global.numel()), dtype=torch.bool
        )
        for gauge_index, gauge_cell in enumerate(gauge_local.tolist()):
            stack = [gauge_cell]
            gauge_upstream[gauge_index, gauge_cell] = True
            while stack:
                destination = stack.pop()
                for source in reverse_links[destination]:
                    if not gauge_upstream[gauge_index, source]:
                        gauge_upstream[gauge_index, source] = True
                        stack.append(source)
        self.register_buffer("gauge_upstream_mask", gauge_upstream)
        self.register_buffer("regionalization_features", features)
        self.register_buffer(
            "outlet_local_index",
            global_to_local[torch.tensor([
                topology.outlet_row * topology.basin_mask.shape[1] + topology.outlet_col
            ])],
        )
        self.outlet_gauge_index = int(gauge_mapping.outlet_gauge_index)
        self.routing_substeps_per_day = int(routing_substeps_per_day)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.b_min = float(b_min)
        self.b_max = float(b_max)
        self.reference_discharge_m3_s = float(reference_discharge_m3_s)
        self.newton_iterations = 2
        self.n_active = int(active_global.numel())

        a_fraction = (initial_a - a_min) / (a_max - a_min)
        b_fraction = (initial_b - b_min) / (b_max - b_min)
        self.a_raw = nn.Parameter(torch.tensor(
            math.log(a_fraction / (1.0 - a_fraction)), dtype=torch.float32
        ))
        self.b_raw = nn.Parameter(torch.tensor(
            math.log(b_fraction / (1.0 - b_fraction)), dtype=torch.float32
        ))
        self.a_feature_weights = nn.Parameter(torch.zeros(features.shape[1]))
        self.b_feature_weights = nn.Parameter(torch.zeros(features.shape[1]))
        self.channel_a_ratio_feature_weights = nn.Parameter(
            torch.zeros(features.shape[1])
        )
        channel_a_ratio = 0.30
        channel_a_ratio_fraction = (channel_a_ratio - 0.05) / (1.0 - 0.05)
        self.channel_a_ratio_raw = nn.Parameter(torch.tensor(
            math.log(channel_a_ratio_fraction / (1.0 - channel_a_ratio_fraction)),
            dtype=torch.float32,
        ))
        b_fraction = (initial_b - b_min) / (b_max - b_min)
        self.channel_b_raw = nn.Parameter(torch.tensor(
            math.log(b_fraction / (1.0 - b_fraction)), dtype=torch.float32
        ))
        self.channel_b_feature_weights = nn.Parameter(
            torch.zeros(features.shape[1])
        )

        sorted_index = torch.argsort(level, stable=True)
        counts = torch.bincount(level, minlength=int(level.max()) + 1)
        starts = torch.cat((
            torch.zeros(1, dtype=torch.long),
            counts.cumsum(0)[:-1],
        ))
        self.register_buffer("level_sorted_index", sorted_index)
        self.level_slices = tuple(
            (int(start), int(start + count))
            for start, count in zip(starts, counts)
        )

    def parameter_fields(self, *, include_classes: bool = False):
        raw_hillslope_a = (
            self.a_raw
            + self.regionalization_features @ self.a_feature_weights
        )
        raw_hillslope_b = (
            self.b_raw
            + self.regionalization_features @ self.b_feature_weights
        )
        hillslope_a = self.a_min + (self.a_max - self.a_min) * torch.sigmoid(
            raw_hillslope_a
        )
        hillslope_b = self.b_min + (self.b_max - self.b_min) * torch.sigmoid(
            raw_hillslope_b
        )

        raw_channel_ratio = (
            self.channel_a_ratio_raw
            + self.regionalization_features
            @ self.channel_a_ratio_feature_weights
        )
        channel_ratio = 0.05 + 0.95 * torch.sigmoid(raw_channel_ratio)
        channel_a = self.a_min + (hillslope_a - self.a_min) * channel_ratio

        raw_channel_b = (
            self.channel_b_raw
            + self.regionalization_features
            @ self.channel_b_feature_weights
        )
        channel_b = self.b_min + (self.b_max - self.b_min) * torch.sigmoid(
            raw_channel_b
        )
        a = torch.where(self.channel_mask, channel_a, hillslope_a)
        b = torch.where(self.channel_mask, channel_b, hillslope_b)
        if include_classes:
            return a, b, hillslope_a, channel_a, hillslope_b, channel_b
        return a, b

    def storage_response(
        self,
        discharge: torch.Tensor,
        exponent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the regularized storage curve and its discharge derivative."""
        squared = discharge.square() + self.reference_discharge_m3_s**2
        curve = (
            squared.pow(0.5 * exponent)
            - self.reference_discharge_m3_s**exponent
        )
        derivative = (
            exponent
            * discharge
            * squared.pow(0.5 * exponent - 1.0)
        )
        return curve, derivative

    def route_substep(
        self,
        lateral: torch.Tensor,
        previous_discharge: torch.Tensor,
        *,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
        dt_dx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one D8 routing substep with piecewise-constant lateral flow."""
        if a is None or b is None:
            a, b = self.parameter_fields()
        if dt_dx is None:
            dt_dx = (
                (86400.0 / self.routing_substeps_per_day)
                / self.edge_length_m.clamp_min(1.0)
            )
        incoming = torch.zeros_like(previous_discharge)
        current = torch.zeros_like(previous_discharge)
        for start, end in self.level_slices:
            index = self.level_sorted_index[start:end]
            upstream = incoming[:, index]
            previous = previous_discharge[:, index]
            coefficient = a[index]
            exponent = b[index]
            reference = 0.5 * (previous + upstream)
            _, storage_derivative = self.storage_response(reference, exponent)
            weight = dt_dx[index]
            value = (
                weight * upstream
                + coefficient * storage_derivative * previous
                + weight * lateral[:, index]
            ) / (weight + coefficient * storage_derivative)

            previous_storage, _ = self.storage_response(previous, exponent)
            right_hand_side = (
                weight * upstream
                + coefficient * previous_storage
                + weight * lateral[:, index]
            )
            for iteration in range(self.newton_iterations):
                value_storage, value_derivative = self.storage_response(
                    value, exponent
                )
                residual = (
                    weight * value
                    + coefficient * value_storage
                    - right_hand_side
                )
                derivative = weight + coefficient * value_derivative
                candidate = value - residual / derivative
                value = torch.where(candidate >= 0, candidate, 0.5 * value)
            value = torch.where(
                right_hand_side > 0, value, torch.zeros_like(value)
            )
            current = current.index_copy(1, index, value)
            incoming = incoming.index_add(
                1, self.downstream_local_index[index], value
            )
        return current

    def initialize_state(
        self,
        batch_size: int,
        *,
        dtype,
        device,
    ) -> D8KinematicWaveState:
        zero = torch.zeros(
            (batch_size, self.n_active), dtype=dtype, device=device
        )
        return D8KinematicWaveState(zero)

    def diagnostics(self) -> dict:
        with torch.no_grad():
            a, b, hillslope_a, channel_a, hillslope_b, channel_b = (
                self.parameter_fields(include_classes=True)
            )
            return {
                "mode": "d8_kinematic_hillslope_channel",
                "active_cells": self.n_active,
                "hillslope_cells": int((~self.channel_mask).sum()),
                "channel_cells": int(self.channel_mask.sum()),
                "routing_substeps_per_day": self.routing_substeps_per_day,
                "lateral_input_time_discretization": (
                    "current_daily_runoff_held_constant_within_day"
                ),
                "a_min_mean_max": [
                    float(a.min()), float(a.mean()), float(a.max())
                ],
                "b_min_mean_max": [
                    float(b.min()), float(b.mean()), float(b.max())
                ],
                "hillslope_a_mean": float(hillslope_a.mean()),
                "channel_a_mean": float(channel_a[self.channel_mask].mean()),
                "hillslope_b_mean": float(hillslope_b.mean()),
                "channel_b_mean": float(channel_b[self.channel_mask].mean()),
                "channel_a_to_hillslope_a_mean_ratio": float(
                    channel_a[self.channel_mask].mean()
                    / hillslope_a[self.channel_mask].mean().clamp_min(1.0e-6)
                ),
            }

    def checkpoint_configuration(self) -> dict:
        return {
            "mode": "d8_kinematic_hillslope_channel",
            "routing_substeps_per_day": self.routing_substeps_per_day,
            "lateral_input_time_discretization": (
                "current_daily_runoff_held_constant_within_day"
            ),
            "state_fields": ["discharge_m3_s"],
            "parameterization": "hillslope_channel_fields",
            "gauge_upstream_diagnostics": True,
            "channel_a_ratio_bounds": [0.05, 1.0],
            "channel_b_bounds": [self.b_min, self.b_max],
            "channel_parameter_features": [
                "log1p_flow_accumulation", "log10_slope", "elevation", "channel_mask"
            ],
            "a_bounds": [self.a_min, self.a_max],
            "b_bounds": [self.b_min, self.b_max],
            "newton_iterations": self.newton_iterations,
            "active_cells": self.n_active,
        }

    def forward(
        self,
        runoff_mm_day: torch.Tensor,
        cell_area_km2: torch.Tensor,
        state: D8KinematicWaveState,
    ):
        if runoff_mm_day.ndim != 5 or runoff_mm_day.shape[2] != 1:
            raise ValueError("runoff_mm_day must have shape [B,T,1,H,W]")
        batch, days = runoff_mm_day.shape[:2]
        if state.discharge_m3_s.shape != (batch, self.n_active):
            raise ValueError(
                "D8 kinematic-wave state shape does not match the input batch"
            )
        area = cell_area_km2
        if area.ndim == 2:
            area = area[None, None]
        elif area.ndim == 3:
            area = area[:, None]
        area = area.to(dtype=runoff_mm_day.dtype, device=runoff_mm_day.device)
        if area.ndim != 4:
            raise ValueError(
                "cell_area_km2 must have shape [H,W], [B,H,W], or [B,1,H,W]"
            )
        area_flat = area.reshape(area.shape[0], -1)[:, self.active_global_index]
        if area_flat.shape[0] == 1 and batch != 1:
            area_flat = area_flat.expand(batch, -1)
        local_daily = (
            runoff_mm_day[:, :, 0].reshape(batch, days, -1)[
                :, :, self.active_global_index
            ]
            * area_flat[:, None]
            * (1000.0 / 86400.0)
        )

        a, b = self.parameter_fields()
        dt_dx = (
            (86400.0 / self.routing_substeps_per_day)
            / self.edge_length_m.clamp_min(1.0)
        )
        initial_curve, _ = self.storage_response(state.discharge_m3_s, b)
        initial_storage = (
            a * initial_curve * self.edge_length_m.clamp_min(1.0)
        )
        gauge_mask = self.gauge_upstream_mask.to(dtype=runoff_mm_day.dtype)
        initial_gauge_storage = torch.einsum(
            "bn,gn->bg", initial_storage, gauge_mask
        )
        initial_storage = initial_storage.sum(dim=1)
        discharge = state.discharge_m3_s
        gauge_steps = []
        outlet_steps = []
        effective_input_steps = []
        routing_storage_steps = []
        gauge_storage_steps = []
        for day in range(days):
            lateral = local_daily[:, day]
            day_gauges = []
            for routing_step in range(self.routing_substeps_per_day):
                effective_input_steps.append(lateral.sum(dim=1))
                discharge = self.route_substep(
                    lateral, discharge, a=a, b=b, dt_dx=dt_dx
                )
                day_gauges.append(discharge[:, self.gauge_local_index])
                outlet_steps.append(discharge[:, self.outlet_local_index[0]])
            gauge_steps.append(torch.stack(day_gauges, dim=1).mean(dim=1))
            day_curve, _ = self.storage_response(discharge, b)
            day_storage = a * day_curve * self.edge_length_m.clamp_min(1.0)
            routing_storage_steps.append(day_storage.sum(dim=1))
            gauge_storage_steps.append(torch.einsum(
                "bn,gn->bg", day_storage, gauge_mask
            ))

        discharge_daily = torch.stack(gauge_steps, dim=1).unsqueeze(-1)
        final_state = D8KinematicWaveState(discharge)
        dt_seconds = 86400.0 / self.routing_substeps_per_day
        runoff_volume = local_daily.sum(dim=(1, 2)) * 86400.0
        input_volume = (
            torch.stack(effective_input_steps, dim=1).sum(dim=1) * dt_seconds
        )
        outlet_volume = (
            torch.stack(outlet_steps, dim=1).sum(dim=1) * dt_seconds
        )
        final_curve, _ = self.storage_response(discharge, b)
        final_storage = (
            a * final_curve * self.edge_length_m.clamp_min(1.0)
        )
        final_gauge_storage = torch.einsum(
            "bn,gn->bg", final_storage, gauge_mask
        )
        final_storage = final_storage.sum(dim=1)
        gauge_input_volume = torch.einsum(
            "btn,gn->bg", local_daily, gauge_mask
        ) * 86400.0
        gauge_outflow_volume = discharge_daily[..., 0].sum(dim=1) * 86400.0
        diagnostics = {
            "routing_storage_initial_m3": initial_storage,
            "routing_storage_final_m3": final_storage,
            "routing_storage_daily_m3": torch.stack(
                routing_storage_steps, dim=1
            ),
            "routing_local_inflow_volume_m3": input_volume,
            "routing_generated_runoff_volume_m3": runoff_volume,
            "routing_outlet_volume_m3": outlet_volume,
            "routing_budget_residual_m3": (
                input_volume + initial_storage - outlet_volume - final_storage
            ),
            "routing_outlet_daily_m3_s": discharge_daily[
                :, :, self.outlet_gauge_index, 0
            ],
            "routing_gauge_daily_m3_s": discharge_daily[..., 0],
            "routing_gauge_storage_initial_m3": initial_gauge_storage,
            "routing_gauge_storage_final_m3": final_gauge_storage,
            "routing_gauge_storage_daily_m3": torch.stack(
                gauge_storage_steps, dim=1
            ),
            "routing_gauge_local_inflow_volume_m3": gauge_input_volume,
            "routing_gauge_outflow_volume_m3": gauge_outflow_volume,
            "routing_gauge_budget_residual_m3": (
                gauge_input_volume + initial_gauge_storage
                - gauge_outflow_volume - final_gauge_storage
            ),
            "kinematic_a_mean": a.mean(),
            "kinematic_b_mean": b.mean(),
        }
        with torch.no_grad():
            for name, values, lower, upper in (
                ("a", a.detach(), self.a_min, self.a_max),
                ("b", b.detach(), self.b_min, self.b_max),
            ):
                fraction = (values - lower) / (upper - lower)
                diagnostics.update({
                    f"routing_parameter_{name}_mean": values.mean(),
                    f"routing_parameter_{name}_q05": torch.quantile(values, 0.05),
                    f"routing_parameter_{name}_median": torch.quantile(values, 0.50),
                    f"routing_parameter_{name}_q95": torch.quantile(values, 0.95),
                    f"routing_parameter_{name}_near_lower_fraction": (
                        fraction <= 0.01
                    ).float().mean(),
                    f"routing_parameter_{name}_near_upper_fraction": (
                        fraction >= 0.99
                    ).float().mean(),
                })
        return discharge_daily, final_state, diagnostics
