"""Representation-geometry diagnostics for the safe-lag-aware fusion main line.

The audit (`docs/artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md`, Q8)
found that the original pipeline computed no effective-rank / covariance / dimension
utilization diagnostics, leaving dimension collapse invisible. This module provides
the compact, deterministic diagnostics used as gate-2 metrics and as training-time
monitoring. All functions are numpy/torch agnostic at the boundary (accept numpy or
torch tensors) and return plain floats so they can be logged alongside the existing
scalar losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class RepresentationGeometry:
    """Compact geometry summary of a batch of representation vectors.

    All fields are computed on the flattened ``[N, D]`` design matrix (samples x
    feature dims). ``effective_rank`` is the participation ratio ``sum(s)^2/sum(s^2)``
    in ``[1, D]``; ``dimension_utilization`` is the fraction of dims whose variance
    exceeds ``relative_threshold * max_variance``.
    """

    effective_rank: float
    dimension_utilization: float
    covariance_trace: float
    top_singular_values: tuple[float, ...]
    seed: int | None
    sample_count: int
    feature_dim: int


def _to_numpy(features) -> np.ndarray:
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("representation features must have shape [N, D]")
    return array


def compute_representation_geometry(
    features,
    *,
    relative_threshold: float = 0.01,
    top_k: int = 16,
    seed: int | None = None,
) -> RepresentationGeometry:
    """Compute effective rank, dimension utilization and covariance spectrum.

    Singular values are computed on the centered features; the participation ratio
    is a standard, stable effective-rank estimator that equals 1 when a single
    component dominates and equals ``D`` when all components are equal.
    """

    array = _to_numpy(features)
    n_samples, feature_dim = array.shape
    centered = array - array.mean(axis=0, keepdims=True)
    # ponytail: economy SVD is enough for the participation ratio and the spectrum.
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    singular_values = np.clip(singular_values, a_min=0.0, a_max=None)
    participation_ratio = (
        float((singular_values.sum() ** 2) / max((singular_values**2).sum(), 1e-12))
        if singular_values.size
        else 0.0
    )
    variances = (singular_values**2) / max(n_samples - 1, 1)
    max_variance = float(variances.max()) if variances.size else 0.0
    utilization = (
        float((variances > relative_threshold * max_variance).mean())
        if max_variance > 0
        else 0.0
    )
    top = tuple(float(value) for value in singular_values[:top_k])
    return RepresentationGeometry(
        effective_rank=participation_ratio,
        dimension_utilization=utilization,
        covariance_trace=float(variances.sum()),
        top_singular_values=top,
        seed=seed,
        sample_count=int(n_samples),
        feature_dim=int(feature_dim),
    )


def geometry_rows(
    geometries: Sequence[tuple[str, RepresentationGeometry]],
) -> list[dict[str, object]]:
    """Flatten a sequence of (label, geometry) pairs into serializable rows."""

    rows: list[dict[str, object]] = []
    for label, geometry in geometries:
        rows.append(
            {
                "label": label,
                "effective_rank": geometry.effective_rank,
                "dimension_utilization": geometry.dimension_utilization,
                "covariance_trace": geometry.covariance_trace,
                "sample_count": geometry.sample_count,
                "feature_dim": geometry.feature_dim,
                "top_singular_values": list(geometry.top_singular_values),
            }
        )
    return rows
