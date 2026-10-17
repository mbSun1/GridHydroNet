"""Explicit reproducibility setup called only by experiment entry points."""

from __future__ import annotations

import random

import lightning as L
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
