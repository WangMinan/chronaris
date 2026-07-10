"""Family-specific latent workload processes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .profiles import PilotProfile


def generate_workload(
    *,
    family: str,
    true_time_s: NDArray[np.float64],
    maneuver_state: NDArray[np.int8],
    controls: NDArray[np.float64],
    vehicle_state: NDArray[np.float64],
    profile: PilotProfile,
    dt_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    event_energy = _event_energy(maneuver_state, vehicle_state)
    control_change = np.linalg.norm(np.gradient(controls, dt_s, axis=0), axis=1)
    control_change = np.tanh(control_change / 0.8)
    overload = np.tanh(np.abs(vehicle_state[:, 11] - 1.0) / 0.35)
    if family == "g1_state_space":
        drive = (
            0.95 * profile.maneuver_sensitivity * event_energy
            + 0.20 * profile.control_change_sensitivity * control_change
            + 0.20 * profile.overload_sensitivity * overload
            + profile.fatigue_slope_per_s * true_time_s
        )
        workload = np.empty_like(true_time_s)
        workload[0] = profile.baseline_workload
        decay = np.exp(-np.log(2.0) * dt_s / profile.persistence_half_life_s)
        for index in range(1, len(workload)):
            target = np.clip(profile.baseline_workload + drive[index], 0.0, 1.0)
            if maneuver_state[index] in (1, 2, 3):
                local_decay = decay ** 15.0
            elif maneuver_state[index] == 4:
                local_decay = decay ** profile.recovery_rate
            else:
                local_decay = decay ** (1.0 + 0.5 * profile.recovery_rate)
            workload[index] = (
                local_decay * workload[index - 1]
                + (1.0 - local_decay) * target
                + rng.normal(0.0, 0.0015)
            )
        return np.clip(workload, 0.0, 1.0)

    interaction = event_energy * control_change
    logits = (
        -1.0
        + 5.5 * profile.maneuver_sensitivity * event_energy
        + 1.4 * profile.control_change_sensitivity * control_change
        + 1.2 * profile.overload_sensitivity * interaction
    )
    resting_pulse = 1.0 / (1.0 + np.exp(0.75))
    pulses = np.maximum(1.0 / (1.0 + np.exp(-logits)) - resting_pulse, 0.0)
    kernel = _gamma_kernel(dt_s, shape=3.2, scale_s=profile.persistence_half_life_s / 15.0)
    convolved = np.convolve(pulses, kernel, mode="full")[: len(pulses)]
    workload = (
        profile.baseline_workload
        + 1.05 * convolved
        + profile.fatigue_slope_per_s * true_time_s
        + rng.normal(0.0, 0.002, size=len(true_time_s))
    )
    return np.clip(workload, 0.0, 1.0)


def _event_energy(
    maneuver_state: NDArray[np.int8],
    vehicle_state: NDArray[np.float64],
) -> NDArray[np.float64]:
    state_weight = np.choose(
        maneuver_state,
        (0.02, 0.65, 1.0, 0.55, 0.18),
    )
    dynamic = np.tanh(
        np.linalg.norm(vehicle_state[:, 6:9], axis=1) / 0.25
        + np.abs(vehicle_state[:, 9]) / 3.0
    )
    active = (maneuver_state != 0).astype(np.float64)
    return np.clip(0.7 * state_weight + 0.3 * dynamic * active, 0.0, 1.0)


def _gamma_kernel(dt_s: float, *, shape: float, scale_s: float) -> NDArray[np.float64]:
    support = np.arange(0.0, max(8.0 * scale_s, dt_s), dt_s)
    safe = np.maximum(support, 1e-9)
    kernel = safe ** (shape - 1.0) * np.exp(-safe / max(scale_s, 1e-6))
    kernel[0] = 0.0
    total = float(kernel.sum())
    return kernel / total if total > 0 else np.array([1.0])
