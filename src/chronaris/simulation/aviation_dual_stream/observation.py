"""Independent asynchronous sampling, clock, missingness and noise process."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import ObservationScenarioConfig
from .contracts import (
    PHYSIOLOGY_FEATURE_NAMES,
    VEHICLE_FEATURE_NAMES,
    LatentTrajectory,
    ObservedStream,
    ObservationTrace,
)


def render_observations(
    latent: LatentTrajectory,
    config: ObservationScenarioConfig,
    *,
    observation_seed: int,
) -> tuple[ObservedStream, ObservedStream, ObservationTrace, ObservationTrace, NDArray[np.float64]]:
    seed_sequence = np.random.SeedSequence(observation_seed)
    vehicle_rng, physiology_rng = (
        np.random.default_rng(child) for child in seed_sequence.spawn(2)
    )
    vehicle, vehicle_trace = _render_stream(
        stream_kind="vehicle",
        true_time_s=latent.true_time_s,
        source_values=latent.vehicle_state,
        feature_names=VEHICLE_FEATURE_NAMES,
        duration_s=latent.config.duration_s,
        rate_hz=config.vehicle_rate_hz,
        jitter_std_ms=config.vehicle_jitter_std_ms,
        offset_s=config.vehicle_clock_offset_s,
        drift_ppm=config.vehicle_clock_drift_ppm,
        random_missing_rate=config.vehicle_random_missing_rate,
        block_gap_s=config.vehicle_block_gap_s,
        snr_db=config.observation_snr_db,
        value_lag_s=0.0,
        rng=vehicle_rng,
    )
    physiology, physiology_trace = _render_stream(
        stream_kind="physiology",
        true_time_s=latent.true_time_s,
        source_values=latent.physiology_state,
        feature_names=PHYSIOLOGY_FEATURE_NAMES,
        duration_s=latent.config.duration_s,
        rate_hz=config.physiology_rate_hz,
        jitter_std_ms=config.physiology_jitter_std_ms,
        offset_s=config.physiology_clock_offset_s,
        drift_ppm=config.physiology_clock_drift_ppm,
        random_missing_rate=config.physiology_random_missing_rate,
        block_gap_s=config.physiology_block_gap_s,
        snr_db=config.observation_snr_db,
        value_lag_s=config.additional_physiology_lag_s,
        rng=physiology_rng,
    )
    realized_lag = latent.physiology_field_lag_s + config.additional_physiology_lag_s
    return vehicle, physiology, vehicle_trace, physiology_trace, realized_lag


def _render_stream(
    *,
    stream_kind: str,
    true_time_s: NDArray[np.float64],
    source_values: NDArray[np.float64],
    feature_names: tuple[str, ...],
    duration_s: float,
    rate_hz: float,
    jitter_std_ms: float,
    offset_s: float,
    drift_ppm: float,
    random_missing_rate: float,
    block_gap_s: float,
    snr_db: float,
    value_lag_s: float,
    rng: np.random.Generator,
) -> tuple[ObservedStream, ObservationTrace]:
    candidate_time = np.arange(0.0, duration_s, 1.0 / rate_hz, dtype=np.float64)
    retained = _retained_mask(
        len(candidate_time),
        missing_rate=random_missing_rate,
        rng=rng,
    )
    gap_start: float | None = None
    if block_gap_s > 0:
        upper = max(duration_s - block_gap_s - 0.1 * duration_s, 0.1 * duration_s)
        gap_start = float(rng.uniform(0.1 * duration_s, upper))
        retained &= ~(
            (candidate_time >= gap_start)
            & (candidate_time < gap_start + block_gap_s)
        )
    retained_time = candidate_time[retained]
    source_time = np.clip(retained_time - value_lag_s, true_time_s[0], true_time_s[-1])
    values = np.column_stack(
        [
            np.interp(source_time, true_time_s, source_values[:, index])
            for index in range(source_values.shape[1])
        ]
    )
    signal_std = np.std(values, axis=0)
    fallback = np.maximum(np.abs(np.mean(values, axis=0)) * 0.01, 1e-6)
    noise_std = np.where(signal_std > 1e-9, signal_std, fallback) / (10.0 ** (snr_db / 20.0))
    values = values + rng.normal(0.0, noise_std, size=values.shape)
    jitter_s = rng.normal(0.0, jitter_std_ms / 1000.0, size=len(retained_time))
    observed_time = retained_time + offset_s + drift_ppm * 1e-6 * retained_time + jitter_s
    order = np.argsort(observed_time, kind="mergesort")
    stream = ObservedStream(
        stream_kind=stream_kind,
        feature_names=feature_names,
        true_sample_time_s=retained_time[order],
        observed_time_s=observed_time[order],
        values=values[order],
    )
    trace = ObservationTrace(
        stream_kind=stream_kind,
        candidate_true_time_s=candidate_time,
        retained_mask=retained,
        retained_jitter_s=jitter_s[order],
        clock_offset_s=offset_s,
        clock_drift_ppm=drift_ppm,
        random_missing_rate_config=random_missing_rate,
        block_gap_start_s=gap_start,
        block_gap_duration_s=block_gap_s,
    )
    return stream, trace


def _retained_mask(
    count: int,
    *,
    missing_rate: float,
    rng: np.random.Generator,
) -> NDArray[np.bool_]:
    if missing_rate <= 0:
        return np.ones(count, dtype=np.bool_)
    best_mask = np.ones(count, dtype=np.bool_)
    best_error = float("inf")
    for _ in range(128):
        mask = rng.random(count) >= missing_rate
        error = abs((1.0 - float(np.mean(mask))) - missing_rate)
        if error < best_error:
            best_mask, best_error = mask, error
        if error <= 0.02:
            return mask
    return best_mask
