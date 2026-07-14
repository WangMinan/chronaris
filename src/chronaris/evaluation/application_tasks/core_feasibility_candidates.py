"""Resumable candidate runners for Dingxin feasibility screening."""

from __future__ import annotations

import json
from pathlib import Path

from chronaris.evaluation.application_tasks.core_feasibility_data import (
    field_target_matrix,
)
from chronaris.evaluation.application_tasks.core_feasibility_features import (
    sequence_features,
    summary_features,
)
from chronaris.evaluation.application_tasks.core_feasibility_models import (
    evaluate_feature_family,
    evaluate_fieldwise_response,
    transform_sequence_family,
)


CANDIDATE_STATE_VERSION = 2


def run_summary_candidate(
    *, state_root, resume, fold_id, candidate_id, modality, history_s, family,
    cache, task_data, random_state
):
    state_path = Path(state_root) / fold_id / f"{candidate_id}.json"
    resumed = _load_candidate_state(state_path) if resume else None
    if resumed is not None:
        return resumed["comparison_rows"], resumed["fold_rows"]
    comparison_rows = []
    fold_rows = []
    try:
        for task in ("maneuver", "response", "high_response"):
            data = task_data[task]
            train = summary_features(
                cache,
                sample_ids=data["train_ids"],
                modality=modality,
                history_s=history_s,
            )
            validation = summary_features(
                cache,
                sample_ids=data["validation_ids"],
                modality=modality,
                history_s=history_s,
            )
            rows, selected = evaluate_feature_family(
                fold_id=fold_id,
                candidate_id=candidate_id,
                train_features=train,
                validation_features=validation,
                targets=data["targets"],
                family=family,
                random_state=random_state,
                tasks=(task,),
            )
            comparison_rows.extend(rows)
            fold_rows.extend(selected)
        _save_candidate_state(state_path, comparison_rows, fold_rows)
    except Exception as exc:
        comparison_rows, fold_rows = unavailable_candidate_rows(
            fold_id, candidate_id, str(exc)
        )
        _save_candidate_state(
            state_path, comparison_rows, fold_rows, status="unavailable"
        )
    return comparison_rows, fold_rows


def run_sequence_candidate(
    *, state_root, resume, fold_id, candidate_id, family, cache, task_data,
    random_state
):
    state_path = Path(state_root) / fold_id / f"{candidate_id}.json"
    resumed = _load_candidate_state(state_path) if resume else None
    if resumed is not None:
        return resumed["comparison_rows"], resumed["fold_rows"]
    comparison_rows = []
    fold_rows = []
    try:
        for task, modality, target_kind in (
            ("maneuver", "vehicle", "classification"),
            ("response", "dual", "regression"),
            ("high_response", "dual", "classification"),
        ):
            data = task_data[task]
            if task == "maneuver":
                target = data["targets"].train_maneuver
            elif task == "response":
                target = data["targets"].train_response
            else:
                target = data["targets"].train_high_response
            train_sequence, validation_sequence, _selected_count = sequence_features(
                cache,
                train_sample_ids=data["train_ids"],
                validation_sample_ids=data["validation_ids"],
                modality=modality,
                target=target,
                target_kind=target_kind,
                channel_budget=32,
            )
            train, validation = transform_sequence_family(
                train_sequence,
                validation_sequence,
                family=family,
                random_state=random_state,
            )
            rows, selected = evaluate_feature_family(
                fold_id=fold_id,
                candidate_id=candidate_id,
                train_features=train,
                validation_features=validation,
                targets=data["targets"],
                family="linear",
                random_state=random_state,
                tasks=(task,),
            )
            comparison_rows.extend(rows)
            fold_rows.extend(selected)
        _save_candidate_state(state_path, comparison_rows, fold_rows)
    except Exception as exc:
        comparison_rows, fold_rows = unavailable_candidate_rows(
            fold_id, candidate_id, str(exc)
        )
        _save_candidate_state(
            state_path, comparison_rows, fold_rows, status="unavailable"
        )
    return comparison_rows, fold_rows


def run_fieldwise_candidate(
    *, state_root, resume, fold_id, cache, task_data, threshold_frame,
    field_delta_index
):
    candidate_id = "fieldwise_physiology_30s"
    state_path = Path(state_root) / fold_id / f"{candidate_id}.json"
    resumed = _load_candidate_state(state_path) if resume else None
    if resumed is not None:
        return resumed["comparison_rows"], resumed["fold_rows"]
    data = task_data["response"]
    try:
        train = summary_features(
            cache,
            sample_ids=data["train_ids"],
            modality="physiology",
            history_s=30.0,
        )
        validation = summary_features(
            cache,
            sample_ids=data["validation_ids"],
            modality="physiology",
            history_s=30.0,
        )
        train_fields = field_target_matrix(
            fold_id=fold_id,
            sample_ids=data["train_ids"],
            threshold_frame=threshold_frame,
            field_delta_index=field_delta_index,
        )
        comparisons, selected = evaluate_fieldwise_response(
            fold_id=fold_id,
            candidate_id=candidate_id,
            train_features=train,
            validation_features=validation,
            train_field_targets=train_fields,
            validation_response=data["targets"].validation_response,
        )
        fold_rows = [selected]
        _save_candidate_state(state_path, comparisons, fold_rows)
    except Exception as exc:
        comparisons, fold_rows = unavailable_candidate_rows(
            fold_id, candidate_id, str(exc), tasks=("response",)
        )
        _save_candidate_state(
            state_path, comparisons, fold_rows, status="unavailable"
        )
    return comparisons, fold_rows


def unavailable_candidate_rows(
    fold_id, candidate_id, reason, *, tasks=("maneuver", "response", "high_response")
):
    metrics = {
        "maneuver": ("macro_f1", "higher"),
        "response": ("rmse", "lower"),
        "high_response": ("auprc", "higher"),
    }
    comparisons = []
    fold_rows = []
    for task in tasks:
        metric, direction = metrics[task]
        row = {
            "fold_id": fold_id,
            "candidate_id": candidate_id,
            "task": task,
            "metric": metric,
            "direction": direction,
            "head_variant": "unavailable",
            "value": None,
            "fit_role": "inner_train",
            "evaluation_role": "inner_validation",
            "outer_test_accessed": False,
            "status": "unavailable",
            "reason": reason,
        }
        comparisons.append(row)
        fold_rows.append({**row, "selected_head_variant": "unavailable"})
    return comparisons, fold_rows


def _load_candidate_state(path):
    if not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("candidate_state_version") != CANDIDATE_STATE_VERSION:
        return None
    return payload


def _save_candidate_state(path, comparison_rows, fold_rows, *, status="completed"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_state_version": CANDIDATE_STATE_VERSION,
                "status": status,
                "comparison_rows": comparison_rows,
                "fold_rows": fold_rows,
                "outer_test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
