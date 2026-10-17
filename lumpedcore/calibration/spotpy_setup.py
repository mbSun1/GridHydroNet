"""SPOTPY setup for continuous warm-up and outlet-only HBV92 calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import spotpy

from lumpedcore.data_pipeline.dataset import LumpedBasinDataset
from lumpedcore.hydromodels.hbv92 import LumpedHBV92
from lumpedcore.hydromodels.model_types import HBV92Parameters
from lumpedcore.hydromodels.routing import (
    gamma_routing_parameter_bounds,
    route_runoff_gamma,
    runoff_mm_day_to_discharge,
)


class HBV92SpotpySetup:
    """Expose named physical parameters and a non-rounded ``1-NSE`` loss."""

    def __init__(
        self,
        dataset: LumpedBasinDataset,
        calibration_start,
        calibration_end_exclusive,
        *,
        routing: str = "gamma",
        routing_length: int = 15,
    ):
        self.dataset = dataset
        self.calibration_start = pd.Timestamp(calibration_start)
        self.calibration_end_exclusive = pd.Timestamp(calibration_end_exclusive)
        self.routing = routing
        self.routing_length = int(routing_length)
        if routing not in {"none", "gamma"}:
            raise ValueError("routing must be 'none' or 'gamma'")
        self.calibration_mask = dataset.period_mask(
            self.calibration_start, self.calibration_end_exclusive
        )
        if not self.calibration_mask.any():
            raise ValueError("Calibration period is empty")
        last_calibration_index = int(np.flatnonzero(self.calibration_mask)[-1]) + 1
        self.precipitation = dataset.precipitation[:last_calibration_index]
        self.temperature = dataset.temperature[:last_calibration_index]
        self.pet = dataset.pet[:last_calibration_index]
        self.target_mask = self.calibration_mask[:last_calibration_index]
        self.observations = dataset.observed_discharge[:last_calibration_index][
            self.target_mask
        ]
        valid = np.isfinite(self.observations) & (self.observations >= 0)
        if valid.sum() < 2:
            raise ValueError("Calibration requires at least two valid outlet observations")
        self.parameter_bounds = list(HBV92Parameters.bounds())
        if routing == "gamma":
            self.parameter_bounds.extend(gamma_routing_parameter_bounds())
        self.parameter_names = [name for name, _, _ in self.parameter_bounds]
        self.params = [
            spotpy.parameter.Uniform(name, low=lower, high=upper)
            for name, lower, upper in self.parameter_bounds
        ]
        self.evaluation_history: list[dict[str, float | int]] = []

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def parameter_mapping(self, values) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(array) != len(self.parameter_names):
            raise ValueError(
                f"Expected {len(self.parameter_names)} parameters, got {len(array)}"
            )
        return dict(zip(self.parameter_names, array.tolist()))

    def simulation(self, values):
        mapping = self.parameter_mapping(values)
        parameters = HBV92Parameters.from_mapping(mapping)
        result = LumpedHBV92().simulate(
            self.precipitation, self.temperature, self.pet, parameters
        )
        runoff = result.runoff_mm_day
        if self.routing == "gamma":
            runoff = route_runoff_gamma(
                runoff,
                mapping["routing_shape"],
                mapping["routing_scale"],
                self.routing_length,
            ).routed_runoff_mm_day
        discharge = runoff_mm_day_to_discharge(
            runoff, self.dataset.basin_area_km2
        )
        return discharge[self.target_mask]

    def evaluation(self):
        return self.observations.copy()

    @staticmethod
    def nse_loss(simulation, evaluation) -> float:
        simulation = np.asarray(simulation, dtype=np.float64)
        evaluation = np.asarray(evaluation, dtype=np.float64)
        valid = (
            np.isfinite(simulation)
            & np.isfinite(evaluation)
            & (evaluation >= 0)
        )
        if valid.sum() < 2:
            return np.inf
        simulation = simulation[valid]
        evaluation = evaluation[valid]
        denominator = np.sum((evaluation - evaluation.mean()) ** 2)
        if denominator <= 0:
            return np.inf
        return float(np.sum((simulation - evaluation) ** 2) / denominator)

    def objectivefunction(self, simulation, evaluation, params=None):
        loss = self.nse_loss(simulation, evaluation)
        record: dict[str, float | int] = {
            "evaluation": len(self.evaluation_history) + 1,
            "like1": loss,
        }
        if params is not None:
            values, names = params
            for name, value in zip(names, values):
                record[f"par{name}"] = float(value)
        self.evaluation_history.append(record)
        return loss
