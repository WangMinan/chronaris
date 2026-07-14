"""Leakage-safe input stabilization for Dingxin development candidates."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from chronaris.evaluation.application_tasks.core_feasibility_features import (
    DingxinFeatureCache,
    _history_slice,
    _summary,
)


STABILIZATION_MODES = (
    "none",
    "train_global_robust",
    "window_causal_robust",
    "pilot_session_baseline_relative",
    "train_rank_quantile",
)


def stabilized_summary_pair(
    cache: DingxinFeatureCache,
    *,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    modality: str,
    history_s: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in STABILIZATION_MODES:
        raise ValueError(f"unsupported stabilization mode: {mode}")
    train_positions = cache.positions(train_sample_ids)
    validation_positions = cache.positions(validation_sample_ids)
    train_parts = []
    validation_parts = []
    for values, mask, age in _streams(cache, modality):
        train_values = _history_slice(
            values[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_values = _history_slice(
            values[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        train_mask = _history_slice(
            mask[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_mask = _history_slice(
            mask[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        train_age = _history_slice(
            age[train_positions], cache.timestamps_s[train_positions], history_s
        )
        validation_age = _history_slice(
            age[validation_positions],
            cache.timestamps_s[validation_positions],
            history_s,
        )
        stabilized_train, stabilized_validation = _stabilize(
            train_values,
            validation_values,
            train_mask,
            validation_mask,
            mode=mode,
        )
        train_parts.append(
            _summary(
                stabilized_train,
                train_mask,
                train_age,
                _history_slice(
                    cache.timestamps_s[train_positions],
                    cache.timestamps_s[train_positions],
                    history_s,
                ),
                history_s=30.0,
            )
        )
        validation_parts.append(
            _summary(
                stabilized_validation,
                validation_mask,
                validation_age,
                _history_slice(
                    cache.timestamps_s[validation_positions],
                    cache.timestamps_s[validation_positions],
                    history_s,
                ),
                history_s=30.0,
            )
        )
    return (
        np.ascontiguousarray(np.concatenate(train_parts, axis=1), dtype=np.float32),
        np.ascontiguousarray(
            np.concatenate(validation_parts, axis=1), dtype=np.float32
        ),
    )


def distribution_shift_rows(
    cache: DingxinFeatureCache,
    *,
    split_id: str,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    channel_budget: int = 64,
) -> list[dict[str, object]]:
    rows = []
    train_positions = cache.positions(train_sample_ids)
    validation_positions = cache.positions(validation_sample_ids)
    for modality, values, mask, _age in (
        ("physiology", cache.physiology_values, cache.physiology_mask, cache.physiology_age_s),
        ("vehicle", cache.vehicle_values, cache.vehicle_mask, cache.vehicle_age_s),
    ):
        train = values[train_positions]
        validation = values[validation_positions]
        train_mask = mask[train_positions]
        validation_mask = mask[validation_positions]
        observed = np.flatnonzero(train_mask.any(axis=(0, 1)))
        variances = np.asarray(
            [
                np.var(train[:, :, index][train_mask[:, :, index]])
                if train_mask[:, :, index].any()
                else 0.0
                for index in observed
            ]
        )
        selected = observed[np.argsort(variances, kind="mergesort")[::-1][:channel_budget]]
        smd_values = []
        ks_values = []
        wasserstein_values = []
        for index in selected:
            left = train[:, :, index][train_mask[:, :, index]].astype(np.float64)
            right = validation[:, :, index][validation_mask[:, :, index]].astype(
                np.float64
            )
            if not len(left) or not len(right):
                continue
            pooled = np.sqrt((np.var(left) + np.var(right)) / 2.0)
            smd_values.append(
                abs(float(np.mean(left) - np.mean(right))) / max(float(pooled), 1e-8)
            )
            ks_values.append(float(ks_2samp(left, right).statistic))
            scale = np.quantile(left, 0.75) - np.quantile(left, 0.25)
            wasserstein_values.append(
                float(wasserstein_distance(left, right)) / max(float(scale), 1e-8)
            )
        rows.append(
            {
                "split_id": split_id,
                "modality": modality,
                "selected_channel_count": len(selected),
                "mean_absolute_standardized_mean_difference": float(
                    np.mean(smd_values)
                ),
                "median_ks_statistic": float(np.median(ks_values)),
                "median_iqr_normalized_wasserstein": float(
                    np.median(wasserstein_values)
                ),
                "selection_fit_role": "inner_train",
                "outer_test_opened": False,
            }
        )
    return rows


def _streams(cache: DingxinFeatureCache, modality: str):
    streams = []
    if modality in {"physiology", "dual"}:
        streams.append(
            (cache.physiology_values, cache.physiology_mask, cache.physiology_age_s)
        )
    if modality in {"vehicle", "dual"}:
        streams.append((cache.vehicle_values, cache.vehicle_mask, cache.vehicle_age_s))
    if not streams:
        raise ValueError(f"unsupported modality: {modality}")
    return streams


def _stabilize(train, validation, train_mask, validation_mask, *, mode):
    if mode == "none":
        return (
            np.where(train_mask, train, 0.0).astype(np.float32),
            np.where(validation_mask, validation, 0.0).astype(np.float32),
        )
    if mode == "window_causal_robust":
        return _window_robust(train, train_mask), _window_robust(
            validation, validation_mask
        )
    if mode == "pilot_session_baseline_relative":
        train_residual = _baseline_residual(train, train_mask)
        validation_residual = _baseline_residual(validation, validation_mask)
        return _train_robust(
            train_residual,
            validation_residual,
            train_mask,
            validation_mask,
        )
    if mode == "train_rank_quantile":
        return _train_rank(train, validation, train_mask, validation_mask)
    return _train_robust(train, validation, train_mask, validation_mask)


def _train_robust(train, validation, train_mask, validation_mask):
    masked = np.where(train_mask, train, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=(0, 1))
        q25 = np.nanquantile(masked, 0.25, axis=(0, 1))
        q75 = np.nanquantile(masked, 0.75, axis=(0, 1))
    median = np.nan_to_num(median, nan=0.0)
    q25 = np.where(np.isfinite(q25), q25, median)
    q75 = np.where(np.isfinite(q75), q75, median)
    scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    left = np.where(train_mask, (train - median) / scale, 0.0)
    right = np.where(validation_mask, (validation - median) / scale, 0.0)
    return np.clip(left, -20, 20).astype(np.float32), np.clip(
        right, -20, 20
    ).astype(np.float32)


def _window_robust(values, mask):
    masked = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=1, keepdims=True)
        q25 = np.nanquantile(masked, 0.25, axis=1, keepdims=True)
        q75 = np.nanquantile(masked, 0.75, axis=1, keepdims=True)
    median = np.nan_to_num(median, nan=0.0)
    q25 = np.where(np.isfinite(q25), q25, median)
    q75 = np.where(np.isfinite(q75), q75, median)
    scale = np.where(q75 - q25 > 1e-6, q75 - q25, 1.0)
    output = np.where(mask, (values - median) / scale, 0.0)
    return np.clip(np.nan_to_num(output), -20, 20).astype(np.float32)


def _baseline_residual(values, mask):
    count = max(1, int(round(values.shape[1] / 6)))
    early = np.where(mask[:, :count], values[:, :count], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        baseline = np.nanmedian(early, axis=1, keepdims=True)
    baseline = np.nan_to_num(baseline, nan=0.0)
    return np.where(mask, values - baseline, 0.0)


def _train_rank(train, validation, train_mask, validation_mask):
    left = np.zeros_like(train, dtype=np.float32)
    right = np.zeros_like(validation, dtype=np.float32)
    for channel in range(train.shape[-1]):
        reference = np.sort(train[:, :, channel][train_mask[:, :, channel]])
        if not len(reference):
            continue
        for values, mask, output in (
            (train, train_mask, left),
            (validation, validation_mask, right),
        ):
            observed = values[:, :, channel][mask[:, :, channel]]
            ranks = np.searchsorted(reference, observed, side="right") / len(reference)
            output[:, :, channel][mask[:, :, channel]] = (2.0 * ranks - 1.0).astype(
                np.float32
            )
    return left, right
