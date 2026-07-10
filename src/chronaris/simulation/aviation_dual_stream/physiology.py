"""Family-specific feature-level physiology generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .profiles import PilotProfile


_BASELINES = np.array([0.58, 0.42, 0.32, 97.2, 72.0, 0.22, 0.50])
_DIRECTIONS = np.array([0.28, -0.22, 0.18, -1.4, 24.0, 0.35, 0.04])


def generate_physiology(
    *,
    family: str,
    true_time_s: NDArray[np.float64],
    workload: NDArray[np.float64],
    profile: PilotProfile,
    dt_s: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    base_field_lags = profile.physiology_response_lag_s * np.array(
        [0.86, 0.95, 1.04, 1.18, 0.92, 1.08, 1.00]
    )
    base_field_lags = np.clip(base_field_lags, 2.0, 30.0)
    if family == "g1_state_space":
        values = np.zeros((len(true_time_s), len(_BASELINES)), dtype=np.float64)
        values[0] = _BASELINES + profile.physiology_baseline_shift
        retention = np.exp(-dt_s / 1.4)
        for index in range(1, len(true_time_s)):
            delayed = np.array(
                [
                    workload[max(0, index - int(round(lag / dt_s)))]
                    for lag in base_field_lags
                ]
            )
            response = np.tanh(1.7 * (delayed - profile.baseline_workload))
            target = (
                _BASELINES
                + profile.physiology_baseline_shift
                + profile.physiology_sensitivity * _DIRECTIONS * response
            )
            values[index] = (
                retention * values[index - 1]
                + (1.0 - retention) * target
                + rng.normal(0.0, _noise_scale() * 0.20)
            )
        field_lags = np.clip(base_field_lags + 0.8, 2.0, 30.0)
    else:
        values = np.empty((len(true_time_s), len(_BASELINES)), dtype=np.float64)
        kernel_peak_lags = np.empty(len(_BASELINES), dtype=np.float64)
        for field_index, lag_s in enumerate(base_field_lags):
            delayed = np.interp(
                true_time_s - lag_s,
                true_time_s,
                workload,
                left=workload[0],
                right=workload[-1],
            )
            if field_index == 0:
                response = delayed - profile.baseline_workload
                hysteresis = response
                kernel_peak_lags[field_index] = 0.0
            else:
                thresholded = np.maximum(delayed - (profile.baseline_workload + 0.04), 0.0)
                shape = 2.4 + 0.25 * field_index
                scale_s = 1.2 + 0.2 * field_index
                kernel = _gamma_kernel(dt_s, shape=shape, scale_s=scale_s)
                kernel_peak_lags[field_index] = (shape - 1.0) * scale_s
                response = _causal_convolve_with_initial(thresholded, kernel)
                response = np.tanh(2.5 * response)
                hysteresis = _hysteresis(
                    response,
                    recovery=0.995 - 0.01 * min(field_index, 4),
                )
            values[:, field_index] = (
                _BASELINES[field_index]
                + profile.physiology_baseline_shift
                + profile.physiology_sensitivity * _DIRECTIONS[field_index] * hysteresis
                + rng.normal(0.0, _noise_scale()[field_index] * 0.25, size=len(true_time_s))
            )
        field_lags = np.clip(base_field_lags + kernel_peak_lags, 2.0, 40.0)
        field_lags[0] = base_field_lags[0]
    values[:, 0:3] = np.clip(values[:, 0:3], 0.0, 1.0)
    values[:, 3] = np.clip(values[:, 3], 88.0, 100.0)
    values[:, 4] = np.clip(values[:, 4], 45.0, 180.0)
    values[:, 5:7] = np.clip(values[:, 5:7], 0.0, 1.0)
    return values, field_lags


def _noise_scale() -> NDArray[np.float64]:
    return np.array([0.012, 0.012, 0.015, 0.05, 0.35, 0.015, 0.004])


def _gamma_kernel(dt_s: float, *, shape: float, scale_s: float) -> NDArray[np.float64]:
    support = np.arange(0.0, 8.0 * scale_s, dt_s)
    safe = np.maximum(support, 1e-9)
    kernel = safe ** (shape - 1.0) * np.exp(-safe / scale_s)
    kernel[0] = 0.0
    return kernel / max(float(kernel.sum()), 1e-12)


def _hysteresis(signal: NDArray[np.float64], *, recovery: float) -> NDArray[np.float64]:
    output = np.empty_like(signal)
    output[0] = signal[0]
    for index in range(1, len(signal)):
        if signal[index] >= output[index - 1]:
            output[index] = signal[index]
        else:
            output[index] = recovery * output[index - 1] + (1.0 - recovery) * signal[index]
    return output


def _causal_convolve_with_initial(
    signal: NDArray[np.float64],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    padding = np.full(max(len(kernel) - 1, 0), signal[0], dtype=np.float64)
    padded = np.concatenate((padding, signal))
    return np.convolve(padded, kernel, mode="valid")[: len(signal)]
