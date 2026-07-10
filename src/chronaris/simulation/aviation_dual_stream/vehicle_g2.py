"""G2 event-spline vehicle generator without G1 transition matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import ManeuverEvent
from .maneuvers import event_envelope
from .vehicle_g1 import physics_residuals


def generate_g2_vehicle(
    true_time_s: NDArray[np.float64],
    events: tuple[ManeuverEvent, ...],
    *,
    dt_s: float,
    process_noise_scale: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Construct state trajectories from smooth event nodes and numerical derivatives."""

    count = len(true_time_s)
    roll = np.zeros(count)
    pitch = np.zeros(count)
    yaw = np.zeros(count)
    speed_delta = np.zeros(count)
    controls = np.zeros((count, 4), dtype=np.float64)
    for event in events:
        envelope = event_envelope(true_time_s, event, spline=True)
        oscillation = 0.08 * np.sin(0.7 * true_time_s + event.event_id) * np.abs(envelope)
        if event.maneuver_type == 0:
            pitch += 0.22 * envelope + oscillation
            controls[:, 0] += 0.75 * envelope
        elif event.maneuver_type == 1:
            roll += 0.45 * envelope + oscillation
            controls[:, 1] += 0.80 * envelope
        elif event.maneuver_type == 2:
            yaw += 0.38 * envelope + oscillation
            roll += 0.18 * np.tanh(1.4 * envelope)
            controls[:, 2] += 0.78 * envelope
        elif event.maneuver_type == 3:
            speed_delta += 24.0 * np.tanh(envelope)
            controls[:, 3] += 0.85 * envelope
        else:
            pitch += 0.16 * envelope + oscillation
            roll += 0.30 * np.tanh(1.2 * envelope)
            yaw += 0.15 * envelope
            speed_delta += 12.0 * np.tanh(envelope)
            controls += envelope[:, None] * np.array([0.55, 0.60, 0.42, 0.45])
    controls += rng.normal(0.0, 0.015, size=controls.shape)
    controls = np.clip(controls, -1.0, 1.0)
    roll += 0.012 * np.sin(0.08 * true_time_s)
    pitch += 0.008 * np.sin(0.11 * true_time_s + 0.3)
    yaw += 0.01 * np.sin(0.05 * true_time_s)
    roll_rate = np.gradient(roll, dt_s)
    pitch_rate = np.gradient(pitch, dt_s)
    yaw_rate = np.gradient(yaw, dt_s)
    speed = 245.0 + speed_delta + 1.2 * np.sin(0.025 * true_time_s)
    longitudinal_acc = np.gradient(speed, dt_s)
    vertical_speed = 0.42 * speed * np.sin(np.clip(pitch, -0.4, 0.4))
    altitude = 6_100.0 + np.cumsum(vertical_speed) * dt_s
    vertical_acc = np.gradient(vertical_speed, dt_s)
    lateral_acc = 2.0 * np.tanh(roll) + 0.7 * np.tanh(yaw_rate)
    normal_load = 1.0 + vertical_acc / 9.80665 + 0.025 * np.tanh(roll * pitch * 10.0)
    state = np.column_stack(
        (
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
        )
    )
    noise = rng.normal(0.0, process_noise_scale * 0.35, size=state.shape)
    noise[:, 1] *= 0.5
    noise[:, 3:9] *= 0.03
    noise[:, 11] *= 0.05
    state += noise
    state[:, 0] = np.clip(state[:, 0], 160.0, 360.0)
    state[:, 1] = np.clip(state[:, 1], 3_000.0, 12_000.0)
    state[:, 3:6] = np.clip(state[:, 3:6], -1.2, 1.2)
    state[:, 11] = np.clip(state[:, 11], 0.0, 3.5)
    residual = physics_residuals(state, dt_s)
    residual += 0.025 * np.column_stack(
        [np.sin((column + 1) * 0.03 * true_time_s) for column in range(6)]
    )
    return controls, state, np.clip(residual, -2.0, 2.0)
