"""Preprocessing for E3 fusion stream structure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from chronaris.evaluation.fusion_stream_structure.contracts import (
    FusionStreamRecord,
    FusionStreamRunConfig,
)


@dataclass(frozen=True, slots=True)
class PreprocessedFusionStream:
    record: FusionStreamRecord
    matrix: np.ndarray
    feature_columns: tuple[str, ...]
    window_ids: tuple[object, ...]
    times: tuple[float, ...]
    row_mapping: tuple[Mapping[str, object], ...]
    status: str
    manifest: Mapping[str, object]

    @property
    def key(self) -> tuple[str, str, str]:
        return self.record.key

    @property
    def T(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def d(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 else 0


def preprocess_stream(
    record: FusionStreamRecord,
    config: FusionStreamRunConfig | None = None,
) -> PreprocessedFusionStream:
    config = config or FusionStreamRunConfig()
    matrix = record.feature_matrix()
    source_columns = tuple(record.feature_columns)

    missing_result = _fill_missing(matrix, source_columns)
    matrix = missing_result["matrix"]
    retained_columns = missing_result["columns"]

    variance_result = _filter_low_variance(
        matrix,
        retained_columns,
        threshold=config.low_variance_threshold,
    )
    matrix = variance_result["matrix"]
    retained_columns = variance_result["columns"]

    if matrix.shape[1] == 0:
        status = "no_feature_columns"
        pca_manifest = {"status": "not_applied", "reason": "no_feature_columns"}
    else:
        zscore_result = _zscore(matrix)
        matrix = zscore_result["matrix"]
        pca_result = _apply_pca(
            matrix,
            retained_columns,
            explained_variance_threshold=config.pca_explained_variance,
        )
        matrix = pca_result["matrix"]
        retained_columns = pca_result["columns"]
        pca_manifest = pca_result["manifest"]
        status = "too_short" if record.T < config.min_T else "completed"

    row_mapping = tuple(
        {
            "row_index": index,
            "window_id": window_id,
            "time": time_value,
        }
        for index, (window_id, time_value) in enumerate(zip(record.window_ids, record.times))
    )
    manifest = {
        "method_name": record.method_name,
        "sortie_id": record.sortie_id,
        "view_id": record.view_id,
        "status": status,
        "T": record.T,
        "source_feature_count": len(source_columns),
        "retained_feature_count": int(matrix.shape[1]),
        "min_T": int(config.min_T),
        "missing_values": missing_result["manifest"],
        "low_variance_filter": variance_result["manifest"],
        "zscore": {
            "scope": "method_sortie_view",
            "applied": matrix.shape[1] > 0,
        },
        "pca": pca_manifest,
        "row_mapping": list(row_mapping),
    }
    return PreprocessedFusionStream(
        record=record,
        matrix=matrix,
        feature_columns=tuple(retained_columns),
        window_ids=record.window_ids,
        times=record.times,
        row_mapping=row_mapping,
        status=status,
        manifest=manifest,
    )


def preprocess_streams(
    records: Mapping[tuple[str, str, str], FusionStreamRecord],
    config: FusionStreamRunConfig | None = None,
) -> dict[tuple[str, str, str], PreprocessedFusionStream]:
    return {
        key: preprocess_stream(record, config=config)
        for key, record in records.items()
    }


def _fill_missing(matrix: np.ndarray, columns: tuple[str, ...]) -> dict[str, object]:
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    retained_indices = []
    dropped_all_missing = []
    filled_columns = []
    for index, column in enumerate(columns):
        values = matrix[:, index]
        finite = np.isfinite(values)
        if not finite.any():
            dropped_all_missing.append(column)
            continue
        if not finite.all():
            mean_value = float(values[finite].mean())
            values[~finite] = mean_value if np.isfinite(mean_value) else 0.0
            matrix[:, index] = values
            filled_columns.append(column)
        retained_indices.append(index)
    retained = matrix[:, retained_indices] if retained_indices else np.empty((matrix.shape[0], 0), dtype=np.float64)
    return {
        "matrix": retained,
        "columns": tuple(columns[index] for index in retained_indices),
        "manifest": {
            "dropped_all_missing_columns": dropped_all_missing,
            "mean_imputed_columns": filled_columns,
        },
    }


def _filter_low_variance(
    matrix: np.ndarray,
    columns: tuple[str, ...],
    *,
    threshold: float,
) -> dict[str, object]:
    if matrix.shape[1] == 0:
        return {
            "matrix": matrix,
            "columns": columns,
            "manifest": {
                "threshold": float(threshold),
                "dropped_columns": [],
            },
        }
    variances = np.nanvar(matrix, axis=0)
    keep_mask = variances > threshold
    dropped = [column for column, keep in zip(columns, keep_mask) if not keep]
    retained_columns = tuple(column for column, keep in zip(columns, keep_mask) if keep)
    return {
        "matrix": matrix[:, keep_mask] if keep_mask.any() else np.empty((matrix.shape[0], 0), dtype=np.float64),
        "columns": retained_columns,
        "manifest": {
            "threshold": float(threshold),
            "dropped_columns": dropped,
            "source_variances": {column: float(value) for column, value in zip(columns, variances)},
        },
    }


def _zscore(matrix: np.ndarray) -> dict[str, object]:
    if matrix.shape[1] == 0:
        return {"matrix": matrix, "means": [], "stds": []}
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds < 1e-12, 1.0, stds)
    return {
        "matrix": (matrix - means) / safe_stds,
        "means": means.tolist(),
        "stds": safe_stds.tolist(),
    }


def _apply_pca(
    matrix: np.ndarray,
    columns: tuple[str, ...],
    *,
    explained_variance_threshold: float,
) -> dict[str, object]:
    if matrix.shape[1] <= 1 or matrix.shape[0] <= 1:
        return {
            "matrix": matrix,
            "columns": columns,
            "manifest": {
                "status": "not_applied",
                "reason": "single_dimension_or_single_row",
                "retained_components": int(matrix.shape[1]),
            },
        }
    _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    variances = singular_values ** 2
    total = float(variances.sum())
    if total <= 0.0:
        return {
            "matrix": matrix,
            "columns": columns,
            "manifest": {
                "status": "not_applied",
                "reason": "zero_total_variance",
                "retained_components": int(matrix.shape[1]),
            },
        }
    explained = variances / total
    cumulative = np.cumsum(explained)
    component_count = int(np.searchsorted(cumulative, explained_variance_threshold, side="left") + 1)
    component_count = max(1, min(component_count, matrix.shape[1]))
    components = vt[:component_count, :]
    transformed = matrix @ components.T
    retained_columns = tuple(f"pca_component_{index + 1}" for index in range(component_count))
    return {
        "matrix": transformed,
        "columns": retained_columns,
        "manifest": {
            "status": "applied",
            "threshold": float(explained_variance_threshold),
            "source_feature_count": int(matrix.shape[1]),
            "retained_components": component_count,
            "explained_variance_ratio": explained[:component_count].astype(float).tolist(),
            "cumulative_explained_variance": float(cumulative[component_count - 1]),
            "source_columns": list(columns),
        },
    }
