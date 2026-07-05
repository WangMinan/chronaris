"""Shared helpers for task evaluation public-opt runners."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def safe_public_opt_regression_fallback(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0
    return float(np.mean(finite_values, dtype=np.float64))


def should_use_public_opt_regression_fallback(train_y: np.ndarray) -> bool:
    finite_values = np.asarray(train_y, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return True
    return bool(np.allclose(finite_values, finite_values[0]))


def sanitize_public_opt_regression_outputs(
    values: np.ndarray,
    *,
    fallback_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    sanitized = np.asarray(values, dtype=np.float32).copy()
    nonfinite_mask = ~np.isfinite(sanitized)
    if np.any(nonfinite_mask):
        sanitized[nonfinite_mask] = float(fallback_value)
    return sanitized, nonfinite_mask


def safe_public_opt_classification_fallback(
    values: np.ndarray,
    *,
    label_order: Sequence[int | float],
) -> int:
    if values.size == 0:
        return int(label_order[0])
    labels, counts = np.unique(values.astype(int), return_counts=True)
    return int(labels[np.argmax(counts)])


def should_use_public_opt_classification_fallback(train_y: np.ndarray) -> bool:
    if train_y.size == 0:
        return True
    return int(np.unique(train_y).size) <= 1


def sanitize_public_opt_classification_outputs(
    values: np.ndarray,
    *,
    fallback_label: int,
) -> np.ndarray:
    sanitized = np.asarray(values).copy()
    nonfinite_mask = ~np.isfinite(sanitized.astype(float))
    if np.any(nonfinite_mask):
        sanitized = sanitized.astype(np.int32, copy=False)
        sanitized[nonfinite_mask] = int(fallback_label)
    return sanitized.astype(np.int32, copy=False)


def sanitize_public_opt_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            sanitized[key] = 0.0
        else:
            sanitized[key] = value
    return sanitized
