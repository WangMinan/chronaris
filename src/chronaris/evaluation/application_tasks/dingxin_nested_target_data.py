"""Recompute Dingxin task targets using only each inner-train role."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    FieldRoleRecord,
    stable_sample_hash,
)
from chronaris.dataset.application_evaluation.labels import _build_maneuver_labels
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    build_raw_median_response_targets,
    load_dingxin_target_source_data,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    load_aligned_private_records,
)


CLASS_MAPPING = {"low": 0, "medium": 1, "high": 2}


def build_dingxin_nested_targets(
    *,
    fixed_audit_root: str | Path,
    snapshot_root: str | Path,
    inner_split_root: str | Path,
    e_run_manifest_path: str,
    f_run_manifest_path: str,
    minimum_response_field_count: int = 2,
    response_train_valid_ratio: float = 0.80,
    eps: float = 1e-6,
):
    fixed_root = Path(fixed_audit_root)
    contexts = _load_contexts(fixed_root / "context_sample_manifest.jsonl")
    context_by_id = {context.context_id: context for context in contexts}
    roles = _load_field_roles(fixed_root / "field_role_manifest.csv")
    records = load_aligned_private_records(
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
    )
    record_by_sample = {
        str(row.sample_id): row for row in records.itertuples(index=False)
    }
    response_source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
    )
    response_raw = build_raw_median_response_targets(
        response_source,
        snapshot_root=snapshot_root,
    )
    delta_frame = response_raw.field_delta_rows
    delta_index = {
        (str(row.context_id), str(row.feature_name)): float(row.absolute_delta)
        for row in delta_frame.itertuples(index=False)
        if np.isfinite(row.absolute_delta)
    }
    candidate_fields = tuple(sorted(delta_frame["feature_name"].astype(str).unique()))
    split_payload = json.loads(
        (Path(inner_split_root) / "split_manifest.json").read_text(encoding="utf-8")
    )
    label_rows = []
    threshold_rows = []
    fold_rows = []
    for plan in split_payload["folds"]:
        fold_id = str(plan["fold_id"])
        role_ids = {
            "train": tuple(str(value) for value in plan["train_sample_ids"]),
            "validation": tuple(
                str(value) for value in plan["validation_sample_ids"]
            ),
            "held_out": tuple(str(value) for value in plan["held_out_sample_ids"]),
        }
        evaluation_ids = role_ids["validation"] + role_ids["held_out"]
        classification_fit_hash = stable_sample_hash(role_ids["train"])
        maneuver_rows, maneuver_thresholds = _build_maneuver_labels(
            record_by_sample=record_by_sample,
            context_by_id=context_by_id,
            train_context_ids=role_ids["train"],
            test_context_ids=evaluation_ids,
            roles=roles,
            fit_sample_hash=classification_fit_hash,
            minimum_semantic_count=4,
            eps=eps,
        )
        role_by_id = {
            context_id: role
            for role, sample_ids in role_ids.items()
            for context_id in sample_ids
        }
        for row in maneuver_rows:
            label_rows.append(
                {
                    "fold_id": fold_id,
                    "task_slug": "maneuver_intensity_classification",
                    "role": role_by_id[row["context_id"]],
                    "context_id": row["context_id"],
                    "class_target": CLASS_MAPPING.get(row["class_label"], -1),
                    "continuous_target": None,
                    "binary_target": -1,
                    "status": row["status"],
                    "fit_sample_hash": classification_fit_hash,
                    "threshold_scope": "inner_train_nested",
                }
            )
        threshold_rows.extend(
            {
                "fold_id": fold_id,
                "task_slug": "maneuver_intensity_classification",
                "threshold_scope": "inner_train_nested",
                **row,
            }
            for row in maneuver_thresholds
        )
        response_rows, response_thresholds, response_fit_hash = (
            _build_nested_response_rows(
                fold_id=fold_id,
                role_ids=role_ids,
                delta_index=delta_index,
                candidate_fields=candidate_fields,
                minimum_field_count=minimum_response_field_count,
                train_valid_ratio=response_train_valid_ratio,
                eps=eps,
            )
        )
        label_rows.extend(response_rows)
        threshold_rows.extend(response_thresholds)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "classification_fit_sample_hash": classification_fit_hash,
                "response_fit_sample_hash": response_fit_hash,
                "train_count": len(role_ids["train"]),
                "validation_count": len(role_ids["validation"]),
                "held_out_count": len(role_ids["held_out"]),
            }
        )
    return pd.DataFrame(label_rows), pd.DataFrame(threshold_rows), pd.DataFrame(fold_rows)


def _build_nested_response_rows(
    *, fold_id, role_ids, delta_index, candidate_fields,
    minimum_field_count, train_valid_ratio, eps
):
    available_train_ids = tuple(
        context_id
        for context_id in role_ids["train"]
        if any((context_id, field) in delta_index for field in candidate_fields)
    )
    fit_hash = stable_sample_hash(available_train_ids)
    selected_fields = []
    field_iqrs = {}
    threshold_rows = []
    for field in candidate_fields:
        values = [
            delta_index[(context_id, field)]
            for context_id in available_train_ids
            if (context_id, field) in delta_index
        ]
        valid_ratio = len(values) / max(len(available_train_ids), 1)
        iqr = (
            float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
            if values
            else 0.0
        )
        selected = valid_ratio >= train_valid_ratio and iqr > eps
        if selected:
            selected_fields.append(field)
            field_iqrs[field] = iqr
        threshold_rows.append(
            {
                "fold_id": fold_id,
                "task_slug": "physiology_response_prediction",
                "threshold_scope": "inner_train_nested",
                "parameter_type": (
                    "response_field_scale" if selected else "response_field_excluded"
                ),
                "parameter_name": field,
                "iqr_delta": iqr,
                "train_valid_ratio": valid_ratio,
                "fit_sample_hash": fit_hash,
                "exclusion_reason": None if selected else "coverage_or_iqr_failed",
            }
        )
    if len(selected_fields) < minimum_field_count:
        raise ValueError(f"{fold_id} has too few nested response fields")
    scores = {
        context_id: _response_score(
            context_id,
            selected_fields,
            field_iqrs,
            delta_index,
            minimum_field_count,
        )
        for sample_ids in role_ids.values()
        for context_id in sample_ids
    }
    train_scores = [
        scores[context_id]
        for context_id in available_train_ids
        if scores[context_id] is not None
    ]
    if len(train_scores) < 4:
        raise ValueError(f"{fold_id} has too few nested response scores")
    high_threshold = float(np.quantile(train_scores, 0.75))
    threshold_rows.append(
        {
            "fold_id": fold_id,
            "task_slug": "physiology_response_prediction",
            "threshold_scope": "inner_train_nested",
            "parameter_type": "high_response_bound",
            "parameter_name": "train_q75",
            "iqr_delta": None,
            "train_valid_ratio": None,
            "fit_sample_hash": fit_hash,
            "high_response_threshold": high_threshold,
            "exclusion_reason": None,
        }
    )
    rows = []
    for role, sample_ids in role_ids.items():
        for context_id in sample_ids:
            score = scores[context_id]
            rows.append(
                {
                    "fold_id": fold_id,
                    "task_slug": "physiology_response_prediction",
                    "role": role,
                    "context_id": context_id,
                    "class_target": -1,
                    "continuous_target": score,
                    "binary_target": (
                        -1 if score is None else int(score >= high_threshold)
                    ),
                    "status": "completed" if score is not None else "unavailable",
                    "fit_sample_hash": fit_hash,
                    "threshold_scope": "inner_train_nested",
                }
            )
    return rows, threshold_rows, fit_hash


def _response_score(
    context_id, selected_fields, field_iqrs, delta_index, minimum_field_count
):
    values = [
        float(np.clip(delta_index[(context_id, field)] / field_iqrs[field], 0, 10))
        for field in selected_fields
        if (context_id, field) in delta_index
    ]
    return float(np.mean(values)) if len(values) >= minimum_field_count else None


def _load_contexts(path: Path):
    result = []
    for payload in pd.read_json(path, lines=True).to_dict("records"):
        payload = dict(payload)
        payload["source_sample_ids"] = tuple(payload["source_sample_ids"])
        result.append(ApplicationContextRecord(**payload))
    return tuple(result)


def _load_field_roles(path: Path):
    frame = pd.read_csv(path)
    result = []
    for row in frame.to_dict("records"):
        payload = {key: value for key, value in row.items() if key != "valid_window_ratio"}
        for key in ("display_label", "unit_hint", "semantic_key", "exclusion_reason"):
            if pd.isna(payload[key]):
                payload[key] = None
        for key in (
            "selected_for_maneuver_label",
            "selected_for_response_target",
            "allowed_in_maneuver_input",
        ):
            payload[key] = bool(payload[key])
        result.append(FieldRoleRecord(**payload))
    return tuple(result)
