"""Deterministic pilot-profile sampling with split-safe identifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PilotProfile:
    profile_id: str
    baseline_workload: float
    persistence_half_life_s: float
    maneuver_sensitivity: float
    control_change_sensitivity: float
    overload_sensitivity: float
    fatigue_slope_per_s: float
    physiology_response_lag_s: float
    physiology_sensitivity: float
    recovery_rate: float
    physiology_baseline_shift: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_pilot_profile(*, split_id: str, profile_index: int, seed: int) -> PilotProfile:
    """Sample one profile without depending on a model or evaluation result."""

    rng = np.random.default_rng(seed)
    return PilotProfile(
        profile_id=f"{split_id}_profile_{profile_index:03d}",
        baseline_workload=float(rng.uniform(0.10, 0.30)),
        persistence_half_life_s=float(rng.uniform(8.0, 30.0)),
        maneuver_sensitivity=float(rng.uniform(0.6, 1.4)),
        control_change_sensitivity=float(rng.uniform(0.5, 1.5)),
        overload_sensitivity=float(rng.uniform(0.5, 1.5)),
        fatigue_slope_per_s=float(rng.uniform(0.0, 0.0015)),
        physiology_response_lag_s=float(rng.uniform(2.0, 30.0)),
        physiology_sensitivity=float(rng.uniform(0.6, 1.4)),
        recovery_rate=float(rng.uniform(0.5, 1.5)),
        physiology_baseline_shift=float(rng.uniform(-0.08, 0.08)),
    )
