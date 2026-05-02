"""Optimized Chronaris candidate helpers for the private Stage I benchmark."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.evaluation import (
    evaluate_classification_predictions,
    evaluate_regression_predictions,
)
from chronaris.pipelines.causal_fusion import (
    StageGCausalFusionConfig,
    export_stage_g_causal_fusion_tensors,
    run_stage_g_causal_fusion,
)
from chronaris.pipelines.stage_i_baseline_models import build_loso_splits
from chronaris.pipelines.stage_i_private_benchmark_data import (
    TASK_MANEUVER,
    TASK_RESPONSE,
    TASK_RETRIEVAL,
    aggregate_field_score,
    base_feature_row,
    build_intermediate_export_from_view,
    compute_event_mask_interference,
    flatten_stream_stats,
    pool_sequence_features,
)
from chronaris.pipelines.stage_i_private_feature_utils import safe_mean

DEFAULT_OPTIMIZED_VARIANT_NAME = "chronaris_opt"


def optimized_no_mask_variant_name(target_variant_name: str) -> str:
    """Return the paired no-causal-mask variant name for an optimized target."""

    return f"{target_variant_name}_no_causal_mask"


def build_private_variant_order(
    base_variant_order: Sequence[str],
    *,
    enable_optimized_chronaris: bool,
    target_variant_name: str,
) -> tuple[str, ...]:
    """Build report/evaluation order without mutating the frozen base variants."""

    order = list(base_variant_order)
    if enable_optimized_chronaris:
        for variant_name in (
            target_variant_name,
            optimized_no_mask_variant_name(target_variant_name),
        ):
            if variant_name not in order:
                order.append(variant_name)
    return tuple(order)


def is_optimized_variant_name(variant_name: str, *, target_variant_name: str) -> bool:
    return variant_name in {
        target_variant_name,
        optimized_no_mask_variant_name(target_variant_name),
    }


def build_optimized_chronaris_feature_frames(
    records: pd.DataFrame,
    *,
    target_variant_name: str,
    lag_window_points: int,
    residual_mode: str,
    selected_vehicle_fields: Sequence[str],
    selected_physiology_fields: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build masked and no-mask optimized Chronaris feature frames."""

    frames: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, object] = {}
    for variant_name, use_causal_mask in (
        (target_variant_name, True),
        (optimized_no_mask_variant_name(target_variant_name), False),
    ):
        frame, summary = _build_one_optimized_frame(
            records,
            use_causal_mask=use_causal_mask,
            lag_window_points=lag_window_points,
            residual_mode=residual_mode,
            selected_vehicle_fields=selected_vehicle_fields,
            selected_physiology_fields=selected_physiology_fields,
        )
        frames[variant_name] = frame
        diagnostics[variant_name] = summary
    return frames, diagnostics


def _build_one_optimized_frame(
    records: pd.DataFrame,
    *,
    use_causal_mask: bool,
    lag_window_points: int,
    residual_mode: str,
    selected_vehicle_fields: Sequence[str],
    selected_physiology_fields: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    config = StageGCausalFusionConfig(
        use_causal_mask=use_causal_mask,
        lag_window_points=lag_window_points,
        fusion_output_mode="pooled_with_residual",
        residual_mode=residual_mode,
    )
    rows: list[dict[str, object]] = []
    attention_entropies: list[float] = []
    top_concentrations: list[float] = []
    event_interference: list[float] = []
    residual_gates: list[float] = []
    for _view_id, frame in records.groupby("view_id", sort=False):
        f_view = frame["f_view"].iloc[0]
        intermediate = build_intermediate_export_from_view(f_view)
        fusion_result = run_stage_g_causal_fusion(intermediate, config=config)
        tensor_export = export_stage_g_causal_fusion_tensors(intermediate, config=config)
        interference_by_sample = compute_event_mask_interference(intermediate, config=config)
        max_window_index = max(int(value) for value in frame["window_index"].to_list()) or 1
        for sample_index, sample in enumerate(fusion_result.samples):
            record = frame.loc[frame["raw_sample_id"] == sample.sample_id]
            if record.empty:
                continue
            base_row = record.iloc[0]
            fused = np.asarray(tensor_export.fused_states[sample_index], dtype=np.float32)
            features = pool_sequence_features(fused, prefix="opt_fused")
            top_concentration = float(
                np.asarray(sample.attention_weights, dtype=np.float32).max(axis=-1).mean()
            )
            interference = float(interference_by_sample.get(sample.sample_id, 0.0))
            residual_gate = 1.0 if use_causal_mask and residual_mode == "raw_window_stats" else 0.0
            features.update(
                {
                    "diag__attention_entropy": float(sample.mean_attention_entropy),
                    "diag__top_event_concentration": top_concentration,
                    "diag__event_mask_interference": interference,
                    "diag__causal_residual_gate": residual_gate,
                    "diag__lag_window_points": float(lag_window_points),
                }
            )
            if residual_mode == "raw_window_stats":
                residual_features = _build_raw_residual_features(
                    base_row,
                    selected_vehicle_fields=selected_vehicle_fields,
                    selected_physiology_fields=selected_physiology_fields,
                    max_window_index=max_window_index,
                )
                for key, value in residual_features.items():
                    features[f"residual__{key}"] = float(value) * residual_gate
            rows.append(base_feature_row(base_row, features))
            attention_entropies.append(float(sample.mean_attention_entropy))
            top_concentrations.append(top_concentration)
            event_interference.append(interference)
            residual_gates.append(residual_gate)
    return pd.DataFrame(rows), {
        "mean_attention_entropy": safe_mean(attention_entropies),
        "mean_top_event_concentration": safe_mean(top_concentrations),
        "mean_event_mask_interference": safe_mean(event_interference),
        "mean_causal_residual_gate": safe_mean(residual_gates),
        "lag_window_points": int(lag_window_points),
        "residual_mode": residual_mode,
        "use_causal_mask": use_causal_mask,
    }


def _build_raw_residual_features(
    row: pd.Series,
    *,
    selected_vehicle_fields: Sequence[str],
    selected_physiology_fields: Sequence[str],
    max_window_index: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features.update(flatten_stream_stats(row["raw_vehicle_stats"], prefix="vehicle_raw"))
    features.update(flatten_stream_stats(row["raw_physiology_stats"], prefix="physiology_raw"))
    features["vehicle_proxy_score"] = _safe_score(
        aggregate_field_score(row["raw_vehicle_stats"], selected_vehicle_fields)
    )
    features["physiology_proxy_score"] = _safe_score(
        aggregate_field_score(row["raw_physiology_stats"], selected_physiology_fields)
    )
    features["ctx__start_offset_s"] = float(row["start_offset_ms"]) / 1000.0
    features["ctx__end_offset_s"] = float(row["end_offset_ms"]) / 1000.0
    features["ctx__duration_s"] = (
        float(row["end_offset_ms"]) - float(row["start_offset_ms"])
    ) / 1000.0
    features["ctx__window_index"] = float(row["window_index"])
    features["ctx__window_fraction"] = float(row["window_index"]) / float(max(max_window_index, 1))
    return features


def run_optimized_supervised_variant(
    frame: pd.DataFrame,
    *,
    task_type: str,
    variant_name: str,
) -> dict[str, object]:
    """Run task-aware heads for the optimized Chronaris candidate."""

    if frame.empty:
        return {"status": "not_run"}
    if task_type == "classification":
        model_predictions = {
            "class_balanced_threshold": _run_threshold_classifier(frame),
        }
        model_metrics = {
            model_name: evaluate_classification_predictions(predictions, label_order=(0, 1, 2))
            for model_name, predictions in model_predictions.items()
        }
    elif task_type == "regression":
        model_predictions = {
            "physiology_persistence": _run_persistence_regressor(frame),
            "ridge_residual": _run_ridge_residual_regressor(frame),
        }
        model_metrics = {
            model_name: evaluate_regression_predictions(predictions)
            for model_name, predictions in model_predictions.items()
        }
    else:
        raise ValueError(f"unsupported optimized supervised task type: {task_type}")
    best_model_name, best_metrics = _select_best_model(model_metrics, task_type=task_type)
    return {
        "status": "completed",
        "sample_count": int(len(frame)),
        "head_family": "chronaris_task_aware",
        "variant_name": variant_name,
        "model_metrics": model_metrics,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "predictions": {
            model_name: predictions.to_dict(orient="records")
            for model_name, predictions in model_predictions.items()
        },
    }


def run_optimized_retrieval_variant(
    frame: pd.DataFrame,
    *,
    variant_name: str,
) -> dict[str, object]:
    """Run the private T3 time-residual retrieval projection."""

    if frame.empty:
        return {"status": "not_run"}
    feature_columns = _optimized_retrieval_feature_columns(frame)
    if not feature_columns:
        return {"status": "not_run"}
    working = frame.copy()
    matrix = working[list(feature_columns)].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    working["retrieval_vector"] = [row.astype(np.float32) for row in matrix / scale.reshape(1, -1)]

    rows: list[dict[str, object]] = []
    for sortie_id, sortie_frame in working.groupby("sortie_id", sort=False):
        for row in sortie_frame.itertuples(index=False):
            candidates = sortie_frame.loc[sortie_frame["pilot_id"] != row.pilot_id].copy()
            if candidates.empty or row.paired_sample_id not in set(candidates["sample_id"]):
                continue
            query = row.retrieval_vector
            candidate_matrix = np.stack(candidates["retrieval_vector"].to_list(), axis=0)
            distances = np.linalg.norm(candidate_matrix - query.reshape(1, -1), axis=1)
            ranked = candidates.iloc[np.argsort(distances)].reset_index(drop=True)
            rank = int(ranked.index[ranked["sample_id"] == row.paired_sample_id][0]) + 1
            rows.append(
                {
                    "sortie_id": sortie_id,
                    "sample_id": row.sample_id,
                    "paired_sample_id": row.paired_sample_id,
                    "top1_hit": int(rank == 1),
                    "reciprocal_rank": 1.0 / rank,
                }
            )
    metric_frame = pd.DataFrame(rows)
    if metric_frame.empty:
        return {"status": "not_run"}
    return {
        "status": "completed",
        "sample_count": int(metric_frame.shape[0]),
        "head_family": "chronaris_time_residual_retrieval",
        "variant_name": variant_name,
        "top1_accuracy": float(metric_frame["top1_hit"].mean()),
        "mrr": float(metric_frame["reciprocal_rank"].mean()),
    }


def write_optimized_candidate_artifacts(
    summary: Mapping[str, object],
    *,
    artifact_root: str | Path,
    target_variant_name: str,
) -> tuple[str, str]:
    """Write compact JSON/CSV views of optimized candidate evidence."""

    artifact_path = Path(artifact_root)
    metric_rows = _optimized_metric_rows(summary, target_variant_name=target_variant_name)
    summary_path = artifact_path / "optimized_candidate_summary.json"
    metrics_path = artifact_path / "optimized_candidate_metrics.csv"
    payload = {
        "run_id": summary["run_id"],
        "target_variant_name": target_variant_name,
        "no_mask_variant_name": optimized_no_mask_variant_name(target_variant_name),
        "private_optimality_supported": summary["conclusion"]["private_optimality_supported"],
        "criterion_details": summary["conclusion"]["criterion_details"],
        "task_metrics": _compact_task_metrics(summary, target_variant_name=target_variant_name),
        "diagnostics": {
            target_variant_name: summary["diagnostics"].get(target_variant_name),
            optimized_no_mask_variant_name(target_variant_name): summary["diagnostics"].get(
                optimized_no_mask_variant_name(target_variant_name)
            ),
        },
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "variant", "metric", "value"))
        writer.writeheader()
        writer.writerows(metric_rows)
    return str(summary_path), str(metrics_path)


def _run_threshold_classifier(frame: pd.DataFrame) -> pd.DataFrame:
    score_column = "feat__residual__vehicle_proxy_score"
    if score_column not in frame:
        return _constant_prediction_frame(frame, task_type="classification", model_name="class_balanced_threshold")
    frames: list[pd.DataFrame] = []
    loso_splits = build_loso_splits(frame["split_group"].to_numpy(dtype=object))
    for split in loso_splits:
        train_scores = frame.iloc[split.train_indices][score_column].to_numpy(dtype=float)
        train_y = frame.iloc[split.train_indices]["y_label"].to_numpy(dtype=int)
        test_scores = frame.iloc[split.test_indices][score_column].to_numpy(dtype=float)
        predicted = _predict_threshold_classes(train_scores, train_y, test_scores)
        frames.append(
            _prediction_frame(
                frame,
                split.test_indices,
                predicted,
                task_type="classification",
                model_name="class_balanced_threshold",
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _predict_threshold_classes(
    train_scores: np.ndarray,
    train_y: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    finite_train = np.isfinite(train_scores)
    if not finite_train.any():
        return np.full(test_scores.shape, _majority_label(train_y), dtype=int)
    values = sorted(set(float(value) for value in train_scores[finite_train]))
    if len(values) < 2:
        return np.full(test_scores.shape, _majority_label(train_y), dtype=int)
    mids = [(left + right) / 2.0 for left, right in zip(values[:-1], values[1:])]
    cuts = [values[0] - 1e-6, *mids, values[-1] + 1e-6]
    best_score: tuple[float, float] | None = None
    best_thresholds = (cuts[0], cuts[-1])
    for low_threshold in cuts:
        for high_threshold in cuts:
            if low_threshold > high_threshold:
                continue
            train_predicted = _apply_thresholds(train_scores, low_threshold, high_threshold)
            score = (
                float(f1_score(train_y, train_predicted, average="macro", zero_division=0)),
                float(balanced_accuracy_score(train_y, train_predicted)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_thresholds = (low_threshold, high_threshold)
    return _apply_thresholds(test_scores, *best_thresholds)


def _apply_thresholds(values: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    finite = np.nan_to_num(values, nan=low_threshold, posinf=high_threshold, neginf=low_threshold)
    return np.where(finite <= low_threshold, 0, np.where(finite <= high_threshold, 1, 2)).astype(int)


def _run_persistence_regressor(frame: pd.DataFrame) -> pd.DataFrame:
    score_column = "feat__residual__physiology_proxy_score"
    if score_column not in frame:
        return _constant_prediction_frame(frame, task_type="regression", model_name="physiology_persistence")
    frames: list[pd.DataFrame] = []
    loso_splits = build_loso_splits(frame["split_group"].to_numpy(dtype=object))
    for split in loso_splits:
        predicted = frame.iloc[split.test_indices][score_column].to_numpy(dtype=float)
        predicted = np.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
        frames.append(
            _prediction_frame(
                frame,
                split.test_indices,
                predicted,
                task_type="regression",
                model_name="physiology_persistence",
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _run_ridge_residual_regressor(frame: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "feat__residual__physiology_proxy_score",
        "feat__residual__vehicle_proxy_score",
        "feat__residual__ctx__window_fraction",
    ]
    feature_columns = [column for column in preferred_columns if column in frame]
    if not feature_columns:
        return _constant_prediction_frame(frame, task_type="regression", model_name="ridge_residual")
    frames: list[pd.DataFrame] = []
    loso_splits = build_loso_splits(frame["split_group"].to_numpy(dtype=object))
    for split in loso_splits:
        train_x = frame.iloc[split.train_indices][feature_columns].to_numpy(dtype=float)
        test_x = frame.iloc[split.test_indices][feature_columns].to_numpy(dtype=float)
        train_y = frame.iloc[split.train_indices]["y_label"].to_numpy(dtype=float)
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_x, train_y)
        predicted = model.predict(test_x)
        frames.append(
            _prediction_frame(
                frame,
                split.test_indices,
                predicted,
                task_type="regression",
                model_name="ridge_residual",
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _prediction_frame(
    frame: pd.DataFrame,
    indices: Sequence[int],
    predicted: np.ndarray,
    *,
    task_type: str,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, frame_index in enumerate(indices):
        row = frame.iloc[int(frame_index)]
        entry = row["task_entry"]
        rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset_id": "private_stage_h_benchmark",
                "subset_id": entry.task_name,
                "subject_id": f"pilot_{entry.pilot_id}",
                "split_group": row["split_group"],
                "evaluation_group": "leave_one_view_out",
                "track": "objective" if task_type == "classification" else "subjective",
                "model_name": model_name,
                "y_true": row["y_label"],
                "y_pred": predicted[position],
            }
        )
    return pd.DataFrame(rows)


def _constant_prediction_frame(
    frame: pd.DataFrame,
    *,
    task_type: str,
    model_name: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in build_loso_splits(frame["split_group"].to_numpy(dtype=object)):
        train_y = frame.iloc[split.train_indices]["y_label"].to_numpy()
        if task_type == "classification":
            value = _majority_label(train_y.astype(int))
            predicted = np.full(len(split.test_indices), value, dtype=int)
        else:
            value = float(np.mean(train_y.astype(float))) if len(train_y) else 0.0
            predicted = np.full(len(split.test_indices), value, dtype=float)
        frames.append(
            _prediction_frame(
                frame,
                split.test_indices,
                predicted,
                task_type=task_type,
                model_name=model_name,
            )
        )
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


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


def _select_best_model(
    metrics_by_name: Mapping[str, Mapping[str, object]],
    *,
    task_type: str,
) -> tuple[str, Mapping[str, object]]:
    if task_type == "classification":
        best_name = max(
            metrics_by_name,
            key=lambda name: (
                metrics_by_name[name]["macro_f1"],
                metrics_by_name[name]["balanced_accuracy"],
            ),
        )
    else:
        best_name = min(
            metrics_by_name,
            key=lambda name: (
                metrics_by_name[name]["rmse"],
                metrics_by_name[name]["mae"],
            ),
        )
    return best_name, metrics_by_name[best_name]


def _majority_label(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def _safe_score(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _compact_task_metrics(
    summary: Mapping[str, object],
    *,
    target_variant_name: str,
) -> dict[str, object]:
    no_mask_name = optimized_no_mask_variant_name(target_variant_name)
    compact: dict[str, object] = {}
    for task_name in (TASK_MANEUVER, TASK_RESPONSE, TASK_RETRIEVAL):
        task = summary["tasks"][task_name]
        compact[task_name] = {
            target_variant_name: _variant_metric_payload(task["variants"].get(target_variant_name)),
            no_mask_name: _variant_metric_payload(task["variants"].get(no_mask_name)),
            "best_variant": task.get("best_variant"),
        }
    return compact


def _optimized_metric_rows(
    summary: Mapping[str, object],
    *,
    target_variant_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    no_mask_name = optimized_no_mask_variant_name(target_variant_name)
    for task_name in (TASK_MANEUVER, TASK_RESPONSE, TASK_RETRIEVAL):
        task = summary["tasks"][task_name]
        for variant_name in (target_variant_name, no_mask_name):
            metrics = _variant_metric_payload(task["variants"].get(variant_name))
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    rows.append(
                        {
                            "task": task_name,
                            "variant": variant_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )
    return rows


def _variant_metric_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping) or payload.get("status") != "completed":
        return {}
    if "best_metrics" in payload:
        return dict(payload["best_metrics"])
    return {
        key: payload[key]
        for key in ("sample_count", "top1_accuracy", "mrr")
        if key in payload
    }


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
