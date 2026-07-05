"""Package export helpers for the optimized private Chronaris candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    merge_task_features,
    select_feature_names,
)


@dataclass(frozen=True, slots=True)
class StageIPrivateOptimizedPackageResult:
    """Package artifacts written for one optimized Chronaris candidate."""

    package_path: str
    report_path: str
    package_summary: Mapping[str, object]


def write_optimized_candidate_package(
    *,
    run_id: str,
    target_variant_name: str,
    e_run_manifest_path: str,
    f_run_manifest_path: str,
    lag_window_points: int,
    residual_mode: str,
    records: pd.DataFrame,
    task_payload: Mapping[str, object],
    variant_frames: Mapping[str, pd.DataFrame],
    task_results: Mapping[str, object],
    diagnostics: Mapping[str, object],
    artifact_root: str | Path,
    report_root: str | Path,
) -> StageIPrivateOptimizedPackageResult:
    """Write a reusable JSON package for the current optimized candidate."""

    artifact_path = Path(artifact_root)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_root)
    report_path.mkdir(parents=True, exist_ok=True)

    selected_vehicle_fields = select_feature_names(
        records["raw_vehicle_stats"],
        preferred_keywords=(
            "speed",
            "acc",
            "pitch",
            "roll",
            "yaw",
            "rate",
            "overload",
            "rudder",
            "stick",
            "angle",
            "heading",
        ),
    )
    selected_physiology_fields = select_feature_names(
        records["raw_physiology_stats"],
        preferred_keywords=("eeg", "spo2"),
    )
    target_frame = variant_frames[target_variant_name]
    package = {
        "package_version": "chronaris_private_opt_v1",
        "run_id": run_id,
        "target_variant_name": target_variant_name,
        "source_manifests": {
            "e_run_manifest_path": e_run_manifest_path,
            "f_run_manifest_path": f_run_manifest_path,
        },
        "dependency_contracts": {
            "requires_feature_export_all_window_contract": True,
            "requires_f_full_reference_hidden": True,
            "requires_stage_g_causal_fusion": True,
            "use_causal_mask": True,
            "fusion_output_mode": "pooled_with_residual",
            "lag_window_points": lag_window_points,
            "residual_mode": residual_mode,
        },
        "records_summary": {
            "sample_count": int(len(records)),
            "view_count": int(records["view_id"].nunique()),
            "sortie_count": int(records["sortie_id"].nunique()),
        },
        "selected_vehicle_fields": list(selected_vehicle_fields),
        "selected_physiology_fields": list(selected_physiology_fields),
        "tasks": {
            TASK_MANEUVER: _build_threshold_head_package(
                task_entries=task_payload["by_task"][TASK_MANEUVER],
                variant_frame=target_frame,
                task_variant_payload=task_results[TASK_MANEUVER]["variants"][target_variant_name],
            ),
            TASK_RESPONSE: _build_regression_head_package(
                task_entries=task_payload["by_task"][TASK_RESPONSE],
                variant_frame=target_frame,
                task_variant_payload=task_results[TASK_RESPONSE]["variants"][target_variant_name],
            ),
            TASK_RETRIEVAL: _build_retrieval_head_package(
                task_entries=task_payload["by_task"][TASK_RETRIEVAL],
                variant_frame=target_frame,
                task_variant_payload=task_results[TASK_RETRIEVAL]["variants"][target_variant_name],
            ),
        },
        "diagnostics": diagnostics.get(target_variant_name, {}),
    }
    package_path = artifact_path / "optimized_candidate_package.json"
    package_path.write_text(
        json.dumps(package, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report_file = report_path / f"private-optimized-package-{run_id}.md"
    report_file.write_text(_render_package_report(package, package_path), encoding="utf-8")
    return StageIPrivateOptimizedPackageResult(
        package_path=str(package_path),
        report_path=str(report_file),
        package_summary=package,
    )


def _build_threshold_head_package(
    *,
    task_entries: Sequence[object],
    variant_frame: pd.DataFrame,
    task_variant_payload: Mapping[str, object],
) -> dict[str, object]:
    frame = merge_task_features(task_entries, variant_frame, task_type="classification")
    score_column = "feat__residual__vehicle_proxy_score"
    if frame.empty or score_column not in frame:
        return {"status": "not_exported"}
    scores = frame[score_column].to_numpy(dtype=float)
    labels = frame["y_label"].to_numpy(dtype=int)
    low_threshold, high_threshold = _fit_threshold_classifier_params(scores, labels)
    return {
        "status": "exported",
        "task_type": "classification",
        "head_family": "class_balanced_threshold",
        "score_column": score_column,
        "label_order": [0, 1, 2],
        "thresholds": {
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
        },
        "cross_validated_best_metrics": dict(task_variant_payload.get("best_metrics", {})),
    }


def _build_regression_head_package(
    *,
    task_entries: Sequence[object],
    variant_frame: pd.DataFrame,
    task_variant_payload: Mapping[str, object],
) -> dict[str, object]:
    frame = merge_task_features(task_entries, variant_frame, task_type="regression")
    if frame.empty:
        return {"status": "not_exported"}
    score_column = "feat__residual__physiology_proxy_score"
    preferred_columns = [
        "feat__residual__physiology_proxy_score",
        "feat__residual__vehicle_proxy_score",
        "feat__residual__ctx__window_fraction",
    ]
    available_heads: dict[str, object] = {}
    if score_column in frame:
        available_heads["physiology_persistence"] = {
            "head_family": "physiology_persistence",
            "score_column": score_column,
        }
    feature_columns = [column for column in preferred_columns if column in frame]
    if feature_columns:
        feature_matrix = frame[feature_columns].to_numpy(dtype=float)
        labels = frame["y_label"].to_numpy(dtype=float)
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(feature_matrix)
        model = Ridge(alpha=1.0)
        model.fit(scaled_matrix, labels)
        available_heads["ridge_residual"] = {
            "head_family": "ridge_residual",
            "feature_columns": feature_columns,
            "scaler_mean": scaler.mean_.astype(float),
            "scaler_scale": scaler.scale_.astype(float),
            "coef": np.asarray(model.coef_, dtype=float),
            "intercept": float(model.intercept_),
        }
    return {
        "status": "exported",
        "task_type": "regression",
        "recommended_head": str(task_variant_payload.get("best_model", "ridge_residual")),
        "available_heads": available_heads,
        "cross_validated_model_metrics": dict(task_variant_payload.get("model_metrics", {})),
        "cross_validated_best_metrics": dict(task_variant_payload.get("best_metrics", {})),
    }


def _build_retrieval_head_package(
    *,
    task_entries: Sequence[object],
    variant_frame: pd.DataFrame,
    task_variant_payload: Mapping[str, object],
) -> dict[str, object]:
    frame = merge_task_features(task_entries, variant_frame, task_type="retrieval")
    if frame.empty:
        return {"status": "not_exported"}
    feature_columns = _optimized_retrieval_feature_columns(frame)
    if not feature_columns:
        return {"status": "not_exported"}
    matrix = frame[list(feature_columns)].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return {
        "status": "exported",
        "task_type": "retrieval",
        "head_family": "chronaris_time_residual_retrieval",
        "feature_columns": list(feature_columns),
        "scale": scale.astype(float),
        "cross_validated_metrics": {
            "sample_count": int(task_variant_payload.get("sample_count", 0)),
            "top1_accuracy": float(task_variant_payload.get("top1_accuracy", 0.0)),
            "mrr": float(task_variant_payload.get("mrr", 0.0)),
        },
    }


def _fit_threshold_classifier_params(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    finite_train = np.isfinite(scores)
    if not finite_train.any():
        return 0.0, 0.0
    values = sorted(set(float(value) for value in scores[finite_train]))
    if len(values) < 2:
        return values[0], values[0]
    mids = [(left + right) / 2.0 for left, right in zip(values[:-1], values[1:])]
    cuts = [values[0] - 1e-6, *mids, values[-1] + 1e-6]
    best_score: tuple[float, float] | None = None
    best_thresholds = (cuts[0], cuts[-1])
    for low_threshold in cuts:
        for high_threshold in cuts:
            if low_threshold > high_threshold:
                continue
            predicted = _apply_thresholds(scores, low_threshold, high_threshold)
            macro_f1 = _macro_f1(labels, predicted)
            balanced_accuracy = _balanced_accuracy(labels, predicted)
            score = (macro_f1, balanced_accuracy)
            if best_score is None or score > best_score:
                best_score = score
                best_thresholds = (low_threshold, high_threshold)
    return best_thresholds


def _apply_thresholds(values: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    finite = np.nan_to_num(values, nan=low_threshold, posinf=high_threshold, neginf=low_threshold)
    return np.where(finite <= low_threshold, 0, np.where(finite <= high_threshold, 1, 2)).astype(int)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for value in [0, 1, 2]:
        tp = int(np.logical_and(y_true == value, y_pred == value).sum())
        fp = int(np.logical_and(y_true != value, y_pred == value).sum())
        fn = int(np.logical_and(y_true == value, y_pred != value).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
            continue
        scores.append((2.0 * precision * recall) / (precision + recall))
    return float(np.mean(scores))


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls: list[float] = []
    for value in [0, 1, 2]:
        positives = y_true == value
        if not positives.any():
            continue
        recalls.append(float((y_pred[positives] == value).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _optimized_retrieval_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    preferred = (
        "feat__residual__ctx__start_offset_s",
        "feat__residual__ctx__window_index",
        "feat__residual__ctx__window_fraction",
    )
    columns = tuple(column for column in preferred if column in frame.columns)
    if columns:
        return columns
    return tuple(column for column in frame.columns if column.startswith("feat__"))


def _render_package_report(package: Mapping[str, object], package_path: Path) -> str:
    tasks = package["tasks"]
    lines = [
        f"# Private Optimized Package - {package['run_id']}",
        "",
        f"- package path: `{package_path}`",
        f"- target variant: `{package['target_variant_name']}`",
        f"- source E manifest: `{package['source_manifests']['e_run_manifest_path']}`",
        f"- source F manifest: `{package['source_manifests']['f_run_manifest_path']}`",
        "",
        "## Dependency Contracts",
        "",
        f"- feature export all-window contract: `{package['dependency_contracts']['requires_feature_export_all_window_contract']}`",
        f"- F full reference hidden: `{package['dependency_contracts']['requires_f_full_reference_hidden']}`",
        f"- Stage G causal fusion: `{package['dependency_contracts']['requires_stage_g_causal_fusion']}`",
        f"- lag window points: `{package['dependency_contracts']['lag_window_points']}`",
        f"- residual mode: `{package['dependency_contracts']['residual_mode']}`",
        "",
        "## Exported Heads",
        "",
        f"- `{TASK_MANEUVER}`: `{tasks[TASK_MANEUVER].get('head_family', tasks[TASK_MANEUVER].get('status'))}`",
        f"- `{TASK_RESPONSE}`: `{tasks[TASK_RESPONSE].get('recommended_head', tasks[TASK_RESPONSE].get('status'))}`",
        f"- `{TASK_RETRIEVAL}`: `{tasks[TASK_RETRIEVAL].get('head_family', tasks[TASK_RETRIEVAL].get('status'))}`",
    ]
    return "\n".join(lines) + "\n"


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
