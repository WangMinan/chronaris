"""G1 semi-Markov controlled state-space vehicle generator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import ManeuverEvent
from .maneuvers import event_envelope


def generate_g1_vehicle(
    true_time_s: NDArray[np.float64],
    events: tuple[ManeuverEvent, ...],
    *,
    dt_s: float,
    process_noise_scale: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return controls, 12-state vehicle truth and normalized physics residuals."""

    count = len(true_time_s)
    controls = _ou_controls(count, dt_s, rng)
    for event in events:
        envelope = event_envelope(true_time_s, event, spline=False)
        _apply_control_template(controls, envelope, event.maneuver_type)
    controls = np.clip(controls, -1.0, 1.0)

    state = np.zeros((count, 12), dtype=np.float64)
    state[0, 0] = 250.0 + rng.normal(0.0, 2.0)
    state[0, 1] = 6_000.0 + rng.normal(0.0, 25.0)
    state[0, 11] = 1.0
    for index in range(1, count):
        previous = state[index - 1]
        control = controls[index]
        roll_rate = previous[6] + dt_s * (
            -1.3 * previous[6] - 0.32 * previous[3] + 0.75 * control[1]
        )
        pitch_rate = previous[7] + dt_s * (
            -1.1 * previous[7] - 0.38 * previous[4] + 0.62 * control[0]
        )
        yaw_rate = previous[8] + dt_s * (
            -1.0 * previous[8]
            - 0.20 * previous[5]
            + 0.48 * control[2]
            + 0.12 * control[1]
        )
        longitudinal_acc = 2.8 * control[3] - 0.012 * (previous[0] - 250.0)
        lateral_acc = 2.2 * control[1] + 0.8 * control[2]
        speed = previous[0] + dt_s * longitudinal_acc
        roll = previous[3] + dt_s * roll_rate
        pitch = previous[4] + dt_s * pitch_rate
        yaw = previous[5] + dt_s * yaw_rate
        vertical_target = 0.18 * speed * np.sin(np.clip(pitch, -0.35, 0.35))
        vertical_speed = previous[2] + dt_s * 0.8 * (vertical_target - previous[2])
        altitude = previous[1] + dt_s * vertical_speed
        vertical_acc = (vertical_speed - previous[2]) / dt_s
        normal_load = 1.0 + vertical_acc / 9.80665
        noise = rng.normal(0.0, process_noise_scale, size=12)
        noise[[0, 1, 2]] *= np.array([0.20, 0.50, 0.15])
        noise[[3, 4, 5, 6, 7, 8]] *= 0.04
        noise[[9, 10, 11]] *= 0.08
        next_state = np.array(
            [
                speed,
                altitude,
                vertical_speed,
                roll,
                pitch,
                yaw,
                roll_rate,
                pitch_rate,
                yaw_rate,
                longitudinal_acc,
                lateral_acc,
                normal_load,
            ]
        )
        state[index] = next_state + noise
    state[:, 0] = np.clip(state[:, 0], 160.0, 360.0)
    state[:, 1] = np.clip(state[:, 1], 3_000.0, 12_000.0)
    state[:, 3:6] = np.clip(state[:, 3:6], -1.2, 1.2)
    state[:, 11] = np.clip(state[:, 11], 0.0, 3.5)
    return controls, state, physics_residuals(state, dt_s)


def physics_residuals(
    state: NDArray[np.float64],
    dt_s: float,
) -> NDArray[np.float64]:
    derivative = np.gradient(state, dt_s, axis=0)
    residual = np.column_stack(
        (
            (derivative[:, 1] - state[:, 2]) / 8.0,
            (derivative[:, 0] - state[:, 9]) / 3.0,
            (derivative[:, 3] - state[:, 6]) / 0.35,
            (derivative[:, 4] - state[:, 7]) / 0.35,
            (derivative[:, 5] - state[:, 8]) / 0.35,
            ((state[:, 11] - 1.0) - derivative[:, 2] / 9.80665) / 0.5,
        )
    )
    return np.clip(residual, -2.0, 2.0)


def _ou_controls(
    count: int,
    dt_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    controls = np.zeros((count, 4), dtype=np.float64)
    controls[0] = rng.normal(0.0, 0.02, size=4)
    for index in range(1, count):
        controls[index] = (
            controls[index - 1]
            + 0.8 * (0.0 - controls[index - 1]) * dt_s
            + 0.05 * np.sqrt(dt_s) * rng.normal(size=4)
        )
    return controls


def _apply_control_template(
    controls: NDArray[np.float64],
    envelope: NDArray[np.float64],
    maneuver_type: int,
) -> None:
    templates = np.array(
        [
            [0.75, 0.10, 0.00, 0.10],
            [0.05, 0.80, 0.15, 0.05],
            [0.05, 0.35, 0.75, 0.05],
            [0.10, 0.05, 0.00, 0.85],
            [0.55, 0.60, 0.40, 0.45],
        ],
        dtype=np.float64,
    )
    controls += envelope[:, None] * templates[maneuver_type][None, :]
