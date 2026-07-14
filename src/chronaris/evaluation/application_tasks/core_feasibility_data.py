"""Fold-local targets and historical references for feasibility screening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.dataset.application_evaluation.labels import _build_maneuver_labels
from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.core_feasibility_protocol import PRIMARY_FOLDS
from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    _load_contexts,
    _load_field_roles,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    build_raw_median_response_targets,
    load_dingxin_target_source_data,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    load_aligned_private_records,
)


def fold_task_data(
    *, fold_id, train_ids, validation_ids, target_frame, maneuver_scores
):
    maneuver = _target_lookup(
        target_frame, fold_id, "maneuver_intensity_classification"
    )
    response = _target_lookup(
        target_frame, fold_id, "physiology_response_prediction"
    )
    maneuver_train = tuple(value for value in train_ids if value in maneuver)
    maneuver_validation = tuple(value for value in validation_ids if value in maneuver)
    response_train = tuple(
        value for value in train_ids if value in response and response[value][1] is not None
    )
    response_validation = tuple(
        value
        for value in validation_ids
        if value in response and response[value][1] is not None
    )
    return {
        "maneuver": {
            "train_ids": maneuver_train,
            "validation_ids": maneuver_validation,
            "targets": FoldTargets(
                train_maneuver=np.asarray([maneuver[value][0] for value in maneuver_train]),
                validation_maneuver=np.asarray(
                    [maneuver[value][0] for value in maneuver_validation]
                ),
                train_maneuver_score=np.asarray(
                    [maneuver_scores[value] for value in maneuver_train]
                ),
                validation_maneuver_score=np.asarray(
                    [maneuver_scores[value] for value in maneuver_validation]
                ),
                train_response=np.zeros(len(maneuver_train)),
                validation_response=np.zeros(len(maneuver_validation)),
                train_high_response=np.zeros(len(maneuver_train), dtype=np.int64),
                validation_high_response=np.zeros(
                    len(maneuver_validation), dtype=np.int64
                ),
            ),
        },
        "response": _response_task_payload(response_train, response_validation, response),
        "high_response": _response_task_payload(
            response_train, response_validation, response
        ),
    }


def maneuver_scores_by_fold(
    *, plans, fixed_root, e_run_manifest_path, f_run_manifest_path
):
    contexts = _load_contexts(Path(fixed_root) / "context_sample_manifest.jsonl")
    context_by_id = {context.context_id: context for context in contexts}
    roles = _load_field_roles(Path(fixed_root) / "field_role_manifest.csv")
    records = load_aligned_private_records(
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
    )
    record_by_sample = {
        str(row.sample_id): row for row in records.itertuples(index=False)
    }
    outputs = {}
    for fold_id, plan in plans.items():
        rows, _thresholds = _build_maneuver_labels(
            record_by_sample=record_by_sample,
            context_by_id=context_by_id,
            train_context_ids=plan["train_sample_ids"],
            test_context_ids=plan["validation_sample_ids"],
            roles=roles,
            fit_sample_hash=stable_sample_hash(plan["train_sample_ids"]),
            minimum_semantic_count=4,
            eps=1e-6,
        )
        outputs[fold_id] = {
            str(row["context_id"]): float(row["score"])
            for row in rows
            if row["status"] == "completed"
        }
    return outputs


def response_delta_index(*, fixed_root, snapshot_root):
    source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
    )
    result = build_raw_median_response_targets(source, snapshot_root=snapshot_root)
    return {
        (str(row.context_id), str(row.feature_name)): float(row.absolute_delta)
        for row in result.field_delta_rows.itertuples(index=False)
        if np.isfinite(row.absolute_delta)
    }


def field_target_matrix(
    *, fold_id, sample_ids, threshold_frame, field_delta_index
):
    selected = threshold_frame[
        (threshold_frame["fold_id"] == fold_id)
        & (threshold_frame["task_slug"] == "physiology_response_prediction")
        & (threshold_frame["parameter_type"] == "response_field_scale")
    ]
    fields = tuple(str(value) for value in selected["parameter_name"])
    scales = {
        str(row.parameter_name): float(row.iqr_delta)
        for row in selected.itertuples(index=False)
    }
    if not fields:
        raise ValueError("no fold-local response fields available")
    return np.asarray(
        [
            [
                np.clip(
                    field_delta_index.get((sample_id, field), np.nan) / scales[field],
                    0,
                    10,
                )
                for field in fields
            ]
            for sample_id in sample_ids
        ],
        dtype=np.float64,
    )


def historical_frozen_rows(path):
    frame = pd.read_csv(path)
    specifications = {
        "maneuver": ("maneuver_intensity_classification", "macro_f1", "higher"),
        "response": ("physiology_response_regression", "rmse", "lower"),
        "high_response": (
            "high_physiology_response_classification",
            "macro_auprc",
            "higher",
        ),
    }
    comparisons = []
    selected = []
    for fold_id in PRIMARY_FOLDS:
        for task, (task_name, metric, direction) in specifications.items():
            subset = frame[
                (frame["fold"] == fold_id)
                & (frame["role"] == "validation")
                & (frame["task"] == task_name)
                & (frame["metric"] == metric)
                & (frame["status"] == "available")
            ]
            task_rows = []
            for row in subset.itertuples(index=False):
                item = {
                    "fold_id": fold_id,
                    "candidate_id": "historical_frozen_panel",
                    "task": task,
                    "metric": "auprc" if task == "high_response" else metric,
                    "direction": direction,
                    "head_variant": f"{row.method}:{row.consumer}",
                    "value": float(row.value),
                    "fit_role": "inner_train",
                    "evaluation_role": "inner_validation",
                    "outer_test_accessed": False,
                    "status": "completed",
                }
                comparisons.append(item)
                task_rows.append(item)
            best = (
                max(task_rows, key=lambda row: row["value"])
                if direction == "higher"
                else min(task_rows, key=lambda row: row["value"])
            )
            selected.append({**best, "selected_head_variant": best["head_variant"]})
    return comparisons, selected


def _response_task_payload(train_ids, validation_ids, response):
    train_response = np.asarray([response[value][1] for value in train_ids], dtype=float)
    validation_response = np.asarray(
        [response[value][1] for value in validation_ids], dtype=float
    )
    train_high = np.asarray([response[value][2] for value in train_ids], dtype=np.int64)
    validation_high = np.asarray(
        [response[value][2] for value in validation_ids], dtype=np.int64
    )
    return {
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "targets": FoldTargets(
            train_maneuver=np.zeros(len(train_ids), dtype=np.int64),
            validation_maneuver=np.zeros(len(validation_ids), dtype=np.int64),
            train_maneuver_score=np.zeros(len(train_ids)),
            validation_maneuver_score=np.zeros(len(validation_ids)),
            train_response=train_response,
            validation_response=validation_response,
            train_high_response=train_high,
            validation_high_response=validation_high,
        ),
    }


def _target_lookup(frame, fold_id, task_slug):
    subset = frame[
        (frame["fold_id"] == fold_id)
        & (frame["task_slug"] == task_slug)
        & (frame["status"] == "completed")
    ]
    result = {}
    for row in subset.itertuples(index=False):
        continuous = (
            None if pd.isna(row.continuous_target) else float(row.continuous_target)
        )
        result[str(row.context_id)] = (
            int(row.class_target),
            continuous,
            int(row.binary_target),
        )
    return result
