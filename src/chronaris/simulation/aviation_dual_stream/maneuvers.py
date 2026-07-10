"""Maneuver event plans and state labels for both generator families."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import AviationScenarioConfig
from .contracts import ManeuverEvent


def generate_maneuver_plan(
    config: AviationScenarioConfig,
    rng: np.random.Generator,
) -> tuple[ManeuverEvent, ...]:
    """Generate at least the configured number of complete maneuver events."""

    short_run = config.duration_s < 120.0
    for _attempt in range(256):
        cursor = float(rng.uniform(3.0, 6.0) if short_run else rng.uniform(30.0, 45.0))
        events: list[ManeuverEvent] = []
        feasible = True
        for event_index in range(config.minimum_complete_events):
            entry, sustained, exit_duration, recovery = _sample_durations(
                config,
                short_run=short_run,
                rng=rng,
            )
            event_end = cursor + entry + sustained + exit_duration + recovery
            remaining_events = config.minimum_complete_events - event_index - 1
            minimum_remaining = remaining_events * (12.0 if short_run else 20.0)
            if event_end + minimum_remaining + 2.0 >= config.duration_s:
                feasible = False
                break
            maneuver_type = int((config.trajectory_index * 2 + event_index) % 5)
            events.append(
                ManeuverEvent(
                    event_id=event_index,
                    maneuver_type=maneuver_type,
                    entry_start_s=cursor,
                    sustained_start_s=cursor + entry,
                    exit_start_s=cursor + entry + sustained,
                    recovery_start_s=cursor + entry + sustained + exit_duration,
                    event_end_s=event_end,
                    amplitude=float(rng.uniform(0.65, 1.15)),
                    direction=float(rng.choice((-1.0, 1.0))),
                )
            )
            if event_index < config.minimum_complete_events - 1:
                cursor = event_end + float(
                    rng.uniform(2.0, 5.0) if short_run else rng.uniform(20.0, 35.0)
                )
        if feasible and len(events) == config.minimum_complete_events:
            return tuple(events)
    raise RuntimeError("could not place the required complete maneuver events")


def _sample_durations(
    config: AviationScenarioConfig,
    *,
    short_run: bool,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    if config.generator_family == "g1_state_space":
        return (
            float(rng.uniform(1.5 if short_run else 2.0, 3.0 if short_run else 5.0)),
            float(rng.uniform(4.0, 7.0 if short_run else 22.0)),
            float(rng.uniform(1.5 if short_run else 2.0, 3.0 if short_run else 5.0)),
            float(rng.uniform(2.5 if short_run else 5.0, 5.0 if short_run else 18.0)),
        )
    return (
        float(rng.uniform(1.5, 3.0 if short_run else 6.0)),
        float(rng.uniform(4.0, 7.5 if short_run else 25.0)),
        float(rng.uniform(1.5, 3.0 if short_run else 6.0)),
        float(rng.uniform(2.5 if short_run else 4.0, 5.5 if short_run else 18.0)),
    )


def render_maneuver_states(
    true_time_s: NDArray[np.float64],
    events: tuple[ManeuverEvent, ...],
) -> tuple[NDArray[np.int8], NDArray[np.int8]]:
    state = np.zeros(true_time_s.shape, dtype=np.int8)
    maneuver_type = np.full(true_time_s.shape, -1, dtype=np.int8)
    for event in events:
        masks = (
            (true_time_s >= event.entry_start_s) & (true_time_s < event.sustained_start_s),
            (true_time_s >= event.sustained_start_s) & (true_time_s < event.exit_start_s),
            (true_time_s >= event.exit_start_s) & (true_time_s < event.recovery_start_s),
            (true_time_s >= event.recovery_start_s) & (true_time_s < event.event_end_s),
        )
        for state_id, mask in enumerate(masks, start=1):
            state[mask] = state_id
            maneuver_type[mask] = event.maneuver_type
    return state, maneuver_type


def event_envelope(
    true_time_s: NDArray[np.float64],
    event: ManeuverEvent,
    *,
    spline: bool,
) -> NDArray[np.float64]:
    envelope = np.zeros_like(true_time_s)
    entry_mask = (true_time_s >= event.entry_start_s) & (true_time_s < event.sustained_start_s)
    sustained_mask = (true_time_s >= event.sustained_start_s) & (true_time_s < event.exit_start_s)
    exit_mask = (true_time_s >= event.exit_start_s) & (true_time_s < event.recovery_start_s)
    recovery_mask = (true_time_s >= event.recovery_start_s) & (true_time_s < event.event_end_s)
    entry_phase = _phase(true_time_s, entry_mask, event.entry_start_s, event.sustained_start_s)
    exit_phase = _phase(true_time_s, exit_mask, event.exit_start_s, event.recovery_start_s)
    recovery_phase = _phase(true_time_s, recovery_mask, event.recovery_start_s, event.event_end_s)
    if spline:
        envelope[entry_mask] = _smoothstep(entry_phase)
        sustained_phase = _phase(
            true_time_s,
            sustained_mask,
            event.sustained_start_s,
            event.exit_start_s,
        )
        envelope[sustained_mask] = 1.0 + 0.12 * np.sin(3.0 * np.pi * sustained_phase)
        envelope[exit_mask] = 1.0 - _smoothstep(exit_phase)
        envelope[recovery_mask] = -0.12 * np.sin(np.pi * recovery_phase) * np.exp(-2 * recovery_phase)
    else:
        envelope[entry_mask] = entry_phase
        envelope[sustained_mask] = 1.0
        envelope[exit_mask] = 1.0 - exit_phase
        envelope[recovery_mask] = 0.15 * np.exp(-4.0 * recovery_phase)
    return envelope * event.amplitude * event.direction


def _phase(
    time_s: NDArray[np.float64],
    mask: NDArray[np.bool_],
    start_s: float,
    stop_s: float,
) -> NDArray[np.float64]:
    return np.clip((time_s[mask] - start_s) / max(stop_s - start_s, 1e-9), 0.0, 1.0)


def _smoothstep(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return value * value * (3.0 - 2.0 * value)
