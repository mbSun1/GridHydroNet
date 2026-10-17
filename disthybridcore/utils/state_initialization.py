"""Initialize and detach continuous hydrological model states."""

from dataclasses import fields

import pandas as pd
import torch
from torch.utils.data import default_collate


def detach_state(state):
    """Detach every HBV and routing-state tensor at a window boundary."""
    components = {}
    for component_name in ("runoff", "routing"):
        component = getattr(state, component_name)
        components[component_name] = type(component)(**{
            field.name: getattr(component, field.name).detach().clone()
            for field in fields(component)
        })
    return type(state)(**components)


@torch.no_grad()
def warmup_to_date(
    model,
    source_dataset,
    initialization_start,
    target_start,
    device,
    chunk_days=30,
):
    """Advance the model-defined initial state to ``target_start``."""
    start = pd.Timestamp(initialization_start)
    target = pd.Timestamp(target_start)
    if pd.isna(start) or pd.isna(target):
        raise ValueError("State initialization dates must be valid")
    if start.tz is not None or target.tz is not None:
        raise ValueError("Daily state initialization dates must not have time zones")
    if start != start.normalize() or target != target.normalize():
        raise ValueError("Daily state initialization dates must lie on day boundaries")
    if target < start:
        raise ValueError("State initialization target must not precede its start")
    if isinstance(chunk_days, bool) or not isinstance(chunk_days, int) or chunk_days < 1:
        raise ValueError("chunk_days must be a positive integer")

    static_names = (
        "terrain", "parameter_climate_sequence", "basin_mask", "cell_area_km2",
    )
    static = default_collate([
        {name: getattr(source_dataset, name) for name in static_names}
    ])
    static = {name: value.to(device) for name, value in static.items()}
    days = int((target - start).days)
    readers = (
        source_dataset.precipitation,
        source_dataset.temperature,
        source_dataset.pet,
    )
    for name, reader in zip(("precipitation", "temperature", "pet"), readers):
        if not reader.covers(start, days):
            raise ValueError(
                f"Historical {name} is incomplete for state initialization: "
                f"{start.date()} to {(target - pd.Timedelta(days=1)).date()}"
            )

    current = start
    chunks = 0
    was_training = model.training
    model.eval()
    try:
        runoff_parameters = model.generate_runoff_parameters(
            static["terrain"],
            static["parameter_climate_sequence"],
            static["basin_mask"],
        )
        state = model.initial_state(static["terrain"], runoff_parameters)
        while current < target:
            count = min(chunk_days, int((target - current).days))
            forcing = default_collate([source_dataset.read_forcing(current, count)])
            forcing = {name: value.to(device) for name, value in forcing.items()}
            for name, value in forcing.items():
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"Historical state initialization contains non-finite {name}: "
                        f"{current.date()}, {count} days"
                    )
            state = model.advance_state(
                {**static, **forcing}, state, runoff_parameters
            )
            current += pd.Timedelta(days=count)
            chunks += 1
    finally:
        model.train(was_training)

    last_forcing_date = (
        (target - pd.Timedelta(days=1)).isoformat() if days else None
    )
    return detach_state(state), {
        "method": "continuous_historical_forcing",
        "initialization_start": start.isoformat(),
        "initialization_end_exclusive": target.isoformat(),
        "last_forcing_date": last_forcing_date,
        "initialization_days": days,
        "chunk_days": chunk_days,
        "chunks": chunks,
        "initial_state": "HBV92 five storages at 0.001 mm; routing state at zero",
        "uses_discharge_observations": False,
        "uses_target_period_forcing": False,
        "weights_updated": False,
    }
