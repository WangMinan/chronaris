"""Bounded, resumable candidate panel for Dingxin task stability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.core_feasibility_data import (
    field_target_matrix,
)
from chronaris.evaluation.application_tasks.core_feasibility_features import (
    sequence_features,
)
from chronaris.evaluation.application_tasks.core_feasibility_models import (
    transform_sequence_family,
)
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    high_response_metrics,
    maneuver_metrics,
    response_metrics,
    stable_sha256,
)
from chronaris.evaluation.application_tasks.task_stability_features import (
    stabilized_summary_pair,
)
from chronaris.evaluation.application_tasks.task_stability_models import (
    predict_fieldwise_response,
    predict_high_response,
    predict_maneuver,
    predict_response,
)


def candidate_manifest() -> list[dict[str, object]]:
    rows = [
        _summary("vehicle_5s_global", "vehicle", 5.0, "train_global_robust"),
        _summary("vehicle_30s_global", "vehicle", 30.0, "train_global_robust"),
        _summary("dual_30s_global", "dual", 30.0, "train_global_robust"),
        _summary("physiology_30s_global", "physiology", 30.0, "train_global_robust"),
        _summary("vehicle_30s_no_adaptation", "vehicle", 30.0, "none"),
        _summary("vehicle_30s_window", "vehicle", 30.0, "window_causal_robust"),
        _summary(
            "vehicle_30s_baseline_relative",
            "vehicle",
            30.0,
            "pilot_session_baseline_relative",
        ),
        _summary("vehicle_30s_rank", "vehicle", 30.0, "train_rank_quantile"),
        _summary("dual_30s_window", "dual", 30.0, "window_causal_robust"),
        _summary(
            "dual_30s_baseline_relative",
            "dual",
            30.0,
            "pilot_session_baseline_relative",
        ),
        _summary("dual_30s_rank", "dual", 30.0, "train_rank_quantile"),
        _summary("vehicle_30s_ordinal", "vehicle", 30.0, "train_global_robust", maneuver_head="ordinal"),
        _summary("vehicle_30s_score_bucket", "vehicle", 30.0, "train_global_robust", maneuver_head="score_bucket"),
        _summary(
            "vehicle_30s_histgb",
            "vehicle",
            30.0,
            "train_global_robust",
            maneuver_head="histgb",
            response_head="histgb",
        ),
        _summary(
            "dual_30s_histgb",
            "dual",
            30.0,
            "train_global_robust",
            maneuver_head="histgb",
            response_head="histgb",
        ),
        _sequence("dual_minirocket_1000", "minirocket_1000"),
        _sequence("dual_minirocket_5000", "minirocket_5000"),
        _sequence("dual_multirocket", "multirocket"),
        _sequence("dual_hydra", "hydra"),
        _summary(
            "dual_30s_joint_risk",
            "dual",
            30.0,
            "train_global_robust",
            high_head="joint",
        ),
        {
            **_summary(
                "fieldwise_physiology_30s",
                "physiology",
                30.0,
                "train_global_robust",
            ),
            "fieldwise_response": True,
        },
        {
            "candidate_id": "historical_frozen_panel",
            "representation": "historical_frozen_64d",
            "ranking_eligible": False,
            "diagnostic_only": True,
            "reason": "historical checkpoints are not fit to the repaired development splits",
        },
    ]
    if len(rows) > 24:
        raise ValueError("task-stability candidate budget exceeds 24")
    for index, row in enumerate(rows):
        row["candidate_order"] = index
        row["config_sha256"] = stable_sha256(row)
    return rows


def run_candidate_split(
    *,
    candidate: dict[str, object],
    plan: dict[str, object],
    cache,
    targets: pd.DataFrame,
    thresholds: pd.DataFrame,
    maneuver_scores,
    field_delta_index,
    state_root: str | Path,
    random_state: int = 17,
) -> list[dict[str, object]]:
    state_path = Path(state_root) / str(candidate["candidate_id"]) / f"{plan['fold_id']}.json"
    lineage = {
        "candidate_config_sha256": candidate["config_sha256"],
        "split_sha256": stable_sha256(plan),
        "train_sample_sha256": stable_sha256(sorted(plan["train_sample_ids"])),
        "validation_sample_sha256": stable_sha256(sorted(plan["validation_sample_ids"])),
    }
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("lineage") == lineage and payload.get("status") == "completed":
            return payload["metric_rows"]
        raise ValueError(f"candidate state lineage mismatch: {state_path}")
    if candidate.get("diagnostic_only"):
        rows = [
            _row(
                candidate,
                plan,
                task=task,
                metric=metric,
                value=None,
                status="unavailable_new_split_contract",
            )
            for task, metric in (
                ("maneuver", "macro_f1"),
                ("response", "rmse_ratio"),
                ("high_response", "normalized_ap"),
            )
        ]
        _save_state(state_path, lineage, rows)
        return rows
    try:
        rows = _evaluate(
            candidate=candidate,
            plan=plan,
            cache=cache,
            targets=targets,
            thresholds=thresholds,
            maneuver_scores=maneuver_scores,
            field_delta_index=field_delta_index,
            random_state=random_state,
        )
    except Exception as exc:  # a single optional candidate must not erase progress
        rows = [
            _row(
                candidate,
                plan,
                task=task,
                metric=metric,
                value=None,
                status=f"unavailable:{type(exc).__name__}:{exc}",
            )
            for task, metric in (
                ("maneuver", "macro_f1"),
                ("response", "rmse_ratio"),
                ("high_response", "normalized_ap"),
            )
        ]
    _save_state(state_path, lineage, rows)
    return rows


def _evaluate(
    *, candidate, plan, cache, targets, thresholds,
    maneuver_scores, field_delta_index, random_state,
):
    split_id = str(plan["fold_id"])
    maneuver_train = _subset(targets, split_id, "maneuver_intensity_classification", "train")
    maneuver_validation = _subset(
        targets, split_id, "maneuver_intensity_classification", "validation"
    )
    response_train = _subset(targets, split_id, "physiology_response_prediction", "train")
    response_validation = _subset(
        targets, split_id, "physiology_response_prediction", "validation"
    )
    train_m_ids = tuple(maneuver_train["context_id"].astype(str))
    val_m_ids = tuple(maneuver_validation["context_id"].astype(str))
    train_r_ids = tuple(response_train["context_id"].astype(str))
    val_r_ids = tuple(response_validation["context_id"].astype(str))
    train_m_score = np.asarray(
        [maneuver_scores[split_id][value] for value in train_m_ids], dtype=np.float64
    )
    val_m_score = np.asarray(
        [maneuver_scores[split_id][value] for value in val_m_ids], dtype=np.float64
    )
    m_train, m_validation = _features(
        candidate,
        cache,
        train_ids=train_m_ids,
        validation_ids=val_m_ids,
        selection_target=maneuver_train["class_target"].to_numpy(dtype=np.int64),
        target_kind="classification",
        random_state=random_state,
    )
    m_prediction, score_prediction = predict_maneuver(
        m_train,
        m_validation,
        train_labels=maneuver_train["class_target"].to_numpy(dtype=np.int64),
        train_score=train_m_score,
        head=str(candidate["maneuver_head"]),
        random_state=random_state,
    )
    maneuver_values = maneuver_metrics(
        train_labels=maneuver_train["class_target"].to_numpy(dtype=np.int64),
        validation_labels=maneuver_validation["class_target"].to_numpy(dtype=np.int64),
        prediction=m_prediction,
        validation_score=val_m_score,
        score_prediction=score_prediction,
    )

    r_train, r_validation = _features(
        candidate,
        cache,
        train_ids=train_r_ids,
        validation_ids=val_r_ids,
        selection_target=response_train["continuous_target"].to_numpy(dtype=np.float64),
        target_kind="regression",
        random_state=random_state,
    )
    if candidate.get("fieldwise_response"):
        fields = field_target_matrix(
            fold_id=split_id,
            sample_ids=train_r_ids,
            threshold_frame=thresholds,
            field_delta_index=field_delta_index,
        )
        response_prediction = predict_fieldwise_response(
            r_train,
            r_validation,
            train_field_targets=fields,
        )
    else:
        response_prediction = predict_response(
            r_train,
            r_validation,
            train_target=response_train["continuous_target"].to_numpy(dtype=np.float64),
            head=str(candidate["response_head"]),
            random_state=random_state,
        )
    response_values = response_metrics(
        train_target=response_train["continuous_target"].to_numpy(dtype=np.float64),
        validation_target=response_validation["continuous_target"].to_numpy(
            dtype=np.float64
        ),
        prediction=response_prediction,
    )
    high_probability = predict_high_response(
        r_train,
        r_validation,
        train_binary=response_train["binary_target"].to_numpy(dtype=np.int64),
        train_response=response_train["continuous_target"].to_numpy(dtype=np.float64),
        head=str(candidate["high_head"]),
        random_state=random_state,
    )
    high_values = high_response_metrics(
        validation_target=response_validation["binary_target"].to_numpy(dtype=np.int64),
        probability=high_probability,
    )
    rows = []
    for task, values in (
        ("maneuver", maneuver_values),
        ("response", response_values),
        ("high_response", high_values),
    ):
        rows.extend(
            _row(candidate, plan, task=task, metric=metric, value=value)
            for metric, value in values.items()
        )
    return rows


def _features(
    candidate, cache, *, train_ids, validation_ids,
    selection_target, target_kind, random_state,
):
    if candidate["representation"] == "causal_summary":
        return stabilized_summary_pair(
            cache,
            train_sample_ids=train_ids,
            validation_sample_ids=validation_ids,
            modality=str(candidate["modality"]),
            history_s=float(candidate["history_s"]),
            mode=str(candidate["stabilization"]),
        )
    train_sequence, validation_sequence, _ = sequence_features(
        cache,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        modality=str(candidate["modality"]),
        target=np.asarray(selection_target),
        target_kind=target_kind,
        history_s=30.0,
        channel_budget=32,
    )
    return transform_sequence_family(
        train_sequence,
        validation_sequence,
        family=str(candidate["sequence_family"]),
        random_state=random_state,
    )


def _summary(
    candidate_id, modality, history_s, stabilization,
    *, maneuver_head="balanced_logistic", response_head="ridge_log1p",
    high_head="balanced_logistic",
):
    return {
        "candidate_id": candidate_id,
        "representation": "causal_summary",
        "modality": modality,
        "history_s": history_s,
        "stabilization": stabilization,
        "maneuver_head": maneuver_head,
        "response_head": response_head,
        "high_head": high_head,
        "ranking_eligible": True,
        "diagnostic_only": False,
    }


def _sequence(candidate_id, family):
    return {
        "candidate_id": candidate_id,
        "representation": "sequence_transform",
        "modality": "dual",
        "history_s": 30.0,
        "stabilization": "train_global_robust",
        "sequence_family": family,
        "maneuver_head": "balanced_logistic",
        "response_head": "ridge_log1p",
        "high_head": "balanced_logistic",
        "ranking_eligible": True,
        "diagnostic_only": False,
    }


def _row(candidate, plan, *, task, metric, value, status="completed"):
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_order": candidate["candidate_order"],
        "split_id": plan["fold_id"],
        "outer_pool_id": plan["outer_pool_id"],
        "validation_support_hash": plan["validation_support_hash"],
        "task": task,
        "metric": metric,
        "value": None if value is None else float(value),
        "status": status,
        "ranking_eligible": bool(candidate["ranking_eligible"]),
        "fit_role": "inner_train",
        "evaluation_role": "inner_validation",
        "outer_test_opened": False,
    }


def _subset(frame, split_id, task_slug, role):
    return frame[
        (frame["fold_id"] == split_id)
        & (frame["task_slug"] == task_slug)
        & (frame["role"] == role)
        & (frame["status"] == "completed")
    ]


def _save_state(path: Path, lineage, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"status": "completed", "lineage": lineage, "metric_rows": rows},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
