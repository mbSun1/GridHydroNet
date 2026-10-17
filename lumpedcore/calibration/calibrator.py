"""Run reproducible SCE-UA calibration and extract named best parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import spotpy

from .spotpy_setup import HBV92SpotpySetup


@dataclass(frozen=True)
class CalibrationResult:
    parameters: dict[str, float]
    objective: float
    evaluations: int
    history: pd.DataFrame


def calibrate_sceua(
    setup: HBV92SpotpySetup,
    repetitions: int,
    *,
    seed: int = 42,
    ngs: int = 5,
) -> CalibrationResult:
    if repetitions < 1 or ngs < 1:
        raise ValueError("repetitions and ngs must be positive")
    sampler = spotpy.algorithms.sceua(
        setup,
        dbname="lumped_hbv92_sceua",
        dbformat="ram",
        save_sim=False,
        random_state=int(seed),
    )
    sampler.sample(
        int(repetitions),
        ngs=int(ngs),
        kstop=3,
        peps=0.1,
        pcento=0.1,
    )
    history = pd.DataFrame(setup.evaluation_history)
    if history.empty or "like1" not in history:
        raise RuntimeError("SCE-UA returned no recorded calibration evaluations")
    status = sampler.status
    objective = float(status.objectivefunction_min)
    status_mapping = {
        name: float(value)
        for name, value in zip(sampler.parnames, status.params_min)
    }
    parameters = {}
    for name in setup.parameter_names:
        if name not in status_mapping:
            raise RuntimeError(f"SCE-UA status lacks parameter {name}")
        parameters[name] = status_mapping[name]
    recomputed = setup.nse_loss(
        setup.simulation([parameters[name] for name in setup.parameter_names]),
        setup.evaluation(),
    )
    if not np.isclose(recomputed, objective, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(
            f"SCE-UA best objective mismatch: status={objective}, recomputed={recomputed}"
        )
    return CalibrationResult(
        parameters=parameters,
        objective=objective,
        evaluations=int(status.rep),
        history=history,
    )
