"""Fold-fitted weak-label construction for the two Dingxin application tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    FieldRoleRecord,
    FoldTaskLabelResult,
    OuterFoldDefinition,
    stable_sample_hash,
)


MANEUVER_TASK_ID = "dingxin_maneuver_intensity_weak_classification_v2"
RESPONSE_TASK_ID = "dingxin_future_physiology_response_v2"


class LabelConstructionError(ValueError):
    """Raised when a fold cannot fit a required label contract."""


def build_fold_task_labels(
    records: pd.DataFrame,
    contexts: Sequence[ApplicationContextRecord],
    fold: OuterFoldDefinition,
    field_roles: Sequence[FieldRoleRecord],
    *,
    minimum_maneuver_semantic_count: int = 4,
    minimum_response_field_count: int = 2,
    response_train_valid_ratio: float = 0.80,
    eps: float = 1e-6,
) -> FoldTaskLabelResult:
    """Fit fold-local thresholds and apply them to train/test contexts."""

    record_by_sample = {str(row.sample_id): row for row in records.itertuples(index=False)}
    context_by_id = {context.context_id: context for context in contexts}
    label_rows: list[Mapping[str, object]] = []
    threshold_rows: list[Mapping[str, object]] = []
    warnings: list[str] = []
    fit_hashes = {
        MANEUVER_TASK_ID: stable_sample_hash(fold.classification_train_context_ids),
        RESPONSE_TASK_ID: stable_sample_hash(fold.response_train_context_ids),
    }

    try:
        maneuver_rows, maneuver_thresholds = _build_maneuver_labels(
            record_by_sample=record_by_sample,
            context_by_id=context_by_id,
            train_context_ids=fold.classification_train_context_ids,
            test_context_ids=fold.classification_test_context_ids,
            roles=field_roles,
            fit_sample_hash=fit_hashes[MANEUVER_TASK_ID],
            minimum_semantic_count=minimum_maneuver_semantic_count,
            eps=eps,
        )
        label_rows.extend(_attach_fold(maneuver_rows, fold))
        threshold_rows.extend(_attach_fold(maneuver_thresholds, fold))
    except LabelConstructionError as exc:
        warnings.append(f"{MANEUVER_TASK_ID}: {exc}")

    selected_response_fields: tuple[str, ...] = ()
    try:
        response_rows, response_thresholds, selected_response_fields = _build_response_labels(
            record_by_sample=record_by_sample,
            context_by_id=context_by_id,
            train_context_ids=fold.response_train_context_ids,
            test_context_ids=fold.response_test_context_ids,
            roles=field_roles,
            fit_sample_hash=fit_hashes[RESPONSE_TASK_ID],
            minimum_field_count=minimum_response_field_count,
            train_valid_ratio=response_train_valid_ratio,
            eps=eps,
        )
        label_rows.extend(_attach_fold(response_rows, fold))
        threshold_rows.extend(_attach_fold(response_thresholds, fold))
    except LabelConstructionError as exc:
        warnings.append(f"{RESPONSE_TASK_ID}: {exc}")

    completed_tasks = {str(row["task_id"]) for row in label_rows}
    status = "completed" if completed_tasks == {MANEUVER_TASK_ID, RESPONSE_TASK_ID} else "partial"
    if not completed_tasks:
        status = "unavailable"
    return FoldTaskLabelResult(
        fold_id=fold.fold_id,
        split_strategy=fold.split_strategy,
        status=status,
        label_rows=tuple(label_rows),
        threshold_rows=tuple(threshold_rows),
        selected_response_fields=selected_response_fields,
        fit_sample_hashes=fit_hashes,
        warnings=tuple(warnings),
    )


def _build_maneuver_labels(
    *,
    record_by_sample: Mapping[str, object],
    context_by_id: Mapping[str, ApplicationContextRecord],
    train_context_ids: Sequence[str],
    test_context_ids: Sequence[str],
    roles: Sequence[FieldRoleRecord],
    fit_sample_hash: str,
    minimum_semantic_count: int,
    eps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    roles_by_sortie: dict[str, list[FieldRoleRecord]] = defaultdict(list)
    for role in roles:
        if role.selected_for_maneuver_label and role.semantic_key:
            roles_by_sortie[role.sortie_id].append(role)
    raw_by_context: dict[str, dict[str, tuple[float, float]]] = {}
    for context_id in tuple(train_context_ids) + tuple(test_context_ids):
        context = context_by_id[context_id]
        record = record_by_sample[context.end_sample_id]
        raw_by_context[context_id] = _maneuver_raw_values(
            getattr(record, "raw_vehicle_stats"),
            roles_by_sortie.get(context.sortie_id, ()),
        )

    semantic_keys = sorted(
        {
            key
            for context_id in train_context_ids
            for key in raw_by_context.get(context_id, {})
        }
    )
    if len(semantic_keys) < minimum_semantic_count:
        raise LabelConstructionError(
            f"only {len(semantic_keys)} train semantic groups; require {minimum_semantic_count}"
        )
    scalers: dict[str, tuple[float, float, float, float]] = {}
    thresholds: list[dict[str, object]] = []
    for semantic_key in semantic_keys:
        std_values = [
            raw_by_context[context_id][semantic_key][0]
            for context_id in train_context_ids
            if semantic_key in raw_by_context.get(context_id, {})
        ]
        delta_values = [
            raw_by_context[context_id][semantic_key][1]
            for context_id in train_context_ids
            if semantic_key in raw_by_context.get(context_id, {})
        ]
        if not std_values or not delta_values:
            continue
        median_std, iqr_std = _median_iqr(std_values)
        median_delta, iqr_delta = _median_iqr(delta_values)
        if iqr_std <= eps and iqr_delta <= eps:
            thresholds.append(
                {
                    "task_id": MANEUVER_TASK_ID,
                    "parameter_type": "semantic_scaler_excluded",
                    "parameter_name": semantic_key,
                    "median_std": median_std,
                    "iqr_std": iqr_std,
                    "median_abs_delta": median_delta,
                    "iqr_abs_delta": iqr_delta,
                    "exclusion_reason": "both_train_iqrs_are_zero",
                    "fit_sample_hash": fit_sample_hash,
                }
            )
            continue
        scalers[semantic_key] = (median_std, iqr_std, median_delta, iqr_delta)
        thresholds.append(
            {
                "task_id": MANEUVER_TASK_ID,
                "parameter_type": "semantic_scaler",
                "parameter_name": semantic_key,
                "median_std": median_std,
                "iqr_std": iqr_std,
                "median_abs_delta": median_delta,
                "iqr_abs_delta": iqr_delta,
                "fit_sample_hash": fit_sample_hash,
            }
        )
    if len(scalers) < minimum_semantic_count:
        raise LabelConstructionError(
            f"only {len(scalers)} fitted semantic scalers; require {minimum_semantic_count}"
        )

    scores: dict[str, tuple[float | None, int]] = {
        context_id: _maneuver_score(
            raw_by_context.get(context_id, {}),
            scalers,
            minimum_semantic_count=minimum_semantic_count,
            eps=eps,
        )
        for context_id in tuple(train_context_ids) + tuple(test_context_ids)
    }
    train_scores = [
        float(scores[context_id][0])
        for context_id in train_context_ids
        if scores[context_id][0] is not None
    ]
    if len(train_scores) < 3:
        raise LabelConstructionError("fewer than three valid train maneuver scores")
    lower = float(np.quantile(np.asarray(train_scores), 1.0 / 3.0))
    upper = float(np.quantile(np.asarray(train_scores), 2.0 / 3.0))
    thresholds.append(
        {
            "task_id": MANEUVER_TASK_ID,
            "parameter_type": "class_bounds",
            "parameter_name": "low_medium_high",
            "lower_bound": lower,
            "upper_bound": upper,
            "fit_sample_hash": fit_sample_hash,
        }
    )
    rows = []
    for split_role, context_ids in (("train", train_context_ids), ("test", test_context_ids)):
        for context_id in context_ids:
            score, valid_count = scores[context_id]
            rows.append(
                {
                    "task_id": MANEUVER_TASK_ID,
                    "task_name": "机动强度弱监督分类",
                    "task_type": "classification",
                    "split_role": split_role,
                    "context_id": context_id,
                    "score": score,
                    "class_label": None if score is None else _bucketize(score, lower, upper),
                    "continuous_target": None,
                    "high_response_label": None,
                    "valid_field_count": valid_count,
                    "status": "completed" if score is not None else "insufficient_fields",
                    "fit_sample_hash": fit_sample_hash,
                }
            )
    return rows, thresholds


def _build_response_labels(
    *,
    record_by_sample: Mapping[str, object],
    context_by_id: Mapping[str, ApplicationContextRecord],
    train_context_ids: Sequence[str],
    test_context_ids: Sequence[str],
    roles: Sequence[FieldRoleRecord],
    fit_sample_hash: str,
    minimum_field_count: int,
    train_valid_ratio: float,
    eps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], tuple[str, ...]]:
    candidate_fields = tuple(
        sorted({role.feature_name for role in roles if role.selected_for_response_target})
    )
    raw_deltas: dict[str, dict[str, float]] = {}
    for context_id in tuple(train_context_ids) + tuple(test_context_ids):
        context = context_by_id[context_id]
        if context.target_sample_id is None:
            raw_deltas[context_id] = {}
            continue
        past = getattr(record_by_sample[context.end_sample_id], "raw_physiology_stats")
        future = getattr(record_by_sample[context.target_sample_id], "raw_physiology_stats")
        raw_deltas[context_id] = _response_raw_deltas(past, future, candidate_fields)

    selected_fields: list[str] = []
    field_iqrs: dict[str, float] = {}
    thresholds: list[dict[str, object]] = []
    train_count = max(len(train_context_ids), 1)
    for field_name in candidate_fields:
        values = [
            raw_deltas[context_id][field_name]
            for context_id in train_context_ids
            if field_name in raw_deltas.get(context_id, {})
        ]
        valid_ratio = len(values) / train_count
        if valid_ratio < train_valid_ratio or not values:
            continue
        _median, iqr = _median_iqr(values)
        if iqr <= eps:
            continue
        selected_fields.append(field_name)
        field_iqrs[field_name] = iqr
        thresholds.append(
            {
                "task_id": RESPONSE_TASK_ID,
                "parameter_type": "response_field_scale",
                "parameter_name": field_name,
                "iqr_delta": iqr,
                "train_valid_ratio": valid_ratio,
                "representative_statistic": "window_mean_fallback",
                "fit_sample_hash": fit_sample_hash,
            }
        )
    if len(selected_fields) < minimum_field_count:
        raise LabelConstructionError(
            f"only {len(selected_fields)} response fields pass train coverage/IQR; require {minimum_field_count}"
        )

    scores: dict[str, tuple[float | None, int]] = {
        context_id: _response_score(
            raw_deltas.get(context_id, {}),
            selected_fields,
            field_iqrs,
            minimum_field_count=minimum_field_count,
        )
        for context_id in tuple(train_context_ids) + tuple(test_context_ids)
    }
    train_scores = [
        float(scores[context_id][0])
        for context_id in train_context_ids
        if scores[context_id][0] is not None
    ]
    if len(train_scores) < 4:
        raise LabelConstructionError("fewer than four valid train physiology-response scores")
    high_threshold = float(np.quantile(np.asarray(train_scores), 0.75))
    thresholds.append(
        {
            "task_id": RESPONSE_TASK_ID,
            "parameter_type": "high_response_bound",
            "parameter_name": "train_q75",
            "high_response_threshold": high_threshold,
            "representative_statistic": "window_mean_fallback",
            "fit_sample_hash": fit_sample_hash,
        }
    )

    rows = []
    for split_role, context_ids in (("train", train_context_ids), ("test", test_context_ids)):
        for context_id in context_ids:
            score, valid_count = scores[context_id]
            rows.append(
                {
                    "task_id": RESPONSE_TASK_ID,
                    "task_name": "机动诱发生理响应预测",
                    "task_type": "regression_and_binary",
                    "split_role": split_role,
                    "context_id": context_id,
                    "score": score,
                    "class_label": None,
                    "continuous_target": score,
                    "high_response_label": None if score is None else int(score >= high_threshold),
                    "valid_field_count": valid_count,
                    "status": "completed" if score is not None else "insufficient_fields",
                    "fit_sample_hash": fit_sample_hash,
                    "representative_statistic": "window_mean_fallback",
                }
            )
    return rows, thresholds, tuple(selected_fields)


def _maneuver_raw_values(
    stats: Mapping[str, object],
    roles: Sequence[FieldRoleRecord],
) -> dict[str, tuple[float, float]]:
    feature_map = stats.get("features", {}) if isinstance(stats, Mapping) else {}
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for role in roles:
        payload = feature_map.get(role.feature_name)
        if not isinstance(payload, Mapping) or int(payload.get("count") or 0) <= 0:
            continue
        std = _finite_float(payload.get("std"))
        delta = _finite_float(payload.get("delta"))
        if std is None or delta is None or role.semantic_key is None:
            continue
        grouped[role.semantic_key].append((abs(std), abs(delta)))
    return {
        key: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for key, values in grouped.items()
    }


def _maneuver_score(
    raw: Mapping[str, tuple[float, float]],
    scalers: Mapping[str, tuple[float, float, float, float]],
    *,
    minimum_semantic_count: int,
    eps: float,
) -> tuple[float | None, int]:
    values = []
    for semantic_key, (std, delta) in raw.items():
        scaler = scalers.get(semantic_key)
        if scaler is None:
            continue
        median_std, iqr_std, median_delta, iqr_delta = scaler
        z_std = float(np.clip((std - median_std) / (iqr_std + eps), -5.0, 5.0))
        z_delta = float(np.clip((delta - median_delta) / (iqr_delta + eps), -5.0, 5.0))
        values.append(max(0.0, z_std) + max(0.0, z_delta))
    if len(values) < minimum_semantic_count:
        return None, len(values)
    return float(np.mean(values)), len(values)


def _response_raw_deltas(
    past_stats: Mapping[str, object],
    future_stats: Mapping[str, object],
    fields: Sequence[str],
) -> dict[str, float]:
    past_map = past_stats.get("features", {}) if isinstance(past_stats, Mapping) else {}
    future_map = future_stats.get("features", {}) if isinstance(future_stats, Mapping) else {}
    result = {}
    for field_name in fields:
        past = past_map.get(field_name)
        future = future_map.get(field_name)
        if not isinstance(past, Mapping) or not isinstance(future, Mapping):
            continue
        if int(past.get("count") or 0) <= 0 or int(future.get("count") or 0) <= 0:
            continue
        past_mean = _finite_float(past.get("mean"))
        future_mean = _finite_float(future.get("mean"))
        if past_mean is None or future_mean is None:
            continue
        result[field_name] = abs(future_mean - past_mean)
    return result


def _response_score(
    raw: Mapping[str, float],
    selected_fields: Sequence[str],
    field_iqrs: Mapping[str, float],
    *,
    minimum_field_count: int,
) -> tuple[float | None, int]:
    values = [
        float(np.clip(raw[field_name] / field_iqrs[field_name], 0.0, 10.0))
        for field_name in selected_fields
        if field_name in raw and field_name in field_iqrs
    ]
    if len(values) < minimum_field_count:
        return None, len(values)
    return float(np.mean(values)), len(values)


def _attach_fold(
    rows: Sequence[Mapping[str, object]],
    fold: OuterFoldDefinition,
) -> list[dict[str, object]]:
    return [
        {
            "fold_id": fold.fold_id,
            "split_strategy": fold.split_strategy,
            "held_out_group": fold.held_out_group,
            **dict(row),
        }
        for row in rows
    ]


def _median_iqr(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.median(array)), float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def _bucketize(score: float, lower: float, upper: float) -> str:
    if score <= lower:
        return "low"
    if score <= upper:
        return "medium"
    return "high"


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None
