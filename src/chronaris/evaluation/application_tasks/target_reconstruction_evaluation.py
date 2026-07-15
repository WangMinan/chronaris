"""Matched-clean task heads, time probes, aggregation, and 1C gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from chronaris.evaluation.application_tasks.clean_input_contract import (
    normalized_phase,
    phase_residualize,
)
from chronaris.evaluation.application_tasks.target_reconstruction_contracts import (
    MATCHED_METHODS,
)
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    response_metrics,
    safe_spearman,
)
from chronaris.representation import load_fusion_stream_batch


TASKS = (
    "future_maneuver_score",
    "future_maneuver_trend",
    "physiology_residual",
    "high_residual_response",
)


def evaluate_matched_clean_panel(
    *,
    plans: Sequence[Mapping[str, object]],
    representation_root: str | Path,
    seed: int,
    contexts: pd.DataFrame,
    maneuver_targets: pd.DataFrame,
    physiology_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit the same fixed-capacity heads for all methods and all inner splits."""

    metric_rows = []
    prediction_rows = []
    target_map = {
        "future_maneuver_score": (maneuver_targets, "future_maneuver_score"),
        "future_maneuver_trend": (maneuver_targets, "maneuver_trend_class"),
        "physiology_residual": (physiology_targets, "residual_response"),
        "high_residual_response": (physiology_targets, "high_residual_response"),
    }
    for plan in plans:
        split_id = str(plan["fold_id"])
        time_predictions = {}
        for task, (target_frame, column) in target_map.items():
            train_ids, train_target = _aligned_target(
                target_frame, split_id, "train", column, plan["train_sample_ids"]
            )
            validation_ids, validation_target = _aligned_target(
                target_frame, split_id, "validation", column, plan["validation_sample_ids"]
            )
            train_phase = normalized_phase(contexts, train_ids)
            validation_phase = normalized_phase(contexts, validation_ids)
            time_predictions[task] = _time_only_prediction(
                train_phase,
                validation_phase,
                train_target,
                classification=task in {"future_maneuver_trend", "high_residual_response"},
                probability=task == "high_residual_response",
            )
            time_value = _primary_metric(task, train_target, validation_target, time_predictions[task])
            metric_rows.append(
                _metric_row(
                    plan,
                    method="time_only",
                    task=task,
                    value=time_value,
                    metric=_metric_name(task),
                    representation_variant="explicit_phase_diagnostic",
                    time_baseline_value=time_value,
                    standardized_gain=0.0,
                    status="completed",
                )
            )
        for method in MATCHED_METHODS:
            batches = {
                role: load_fusion_stream_batch(
                    Path(representation_root)
                    / f"seed_{seed}"
                    / split_id
                    / method
                    / split_id
                    / role
                )
                for role in ("train", "validation")
            }
            for task, (target_frame, column) in target_map.items():
                train_ids, train_target = _aligned_target(
                    target_frame, split_id, "train", column, plan["train_sample_ids"]
                )
                validation_ids, validation_target = _aligned_target(
                    target_frame, split_id, "validation", column, plan["validation_sample_ids"]
                )
                raw_train = _select_embedding(batches["train"], train_ids)
                raw_validation = _select_embedding(batches["validation"], validation_ids)
                train_phase = normalized_phase(contexts, train_ids)
                validation_phase = normalized_phase(contexts, validation_ids)
                clean_train, clean_validation = phase_residualize(
                    raw_train,
                    raw_validation,
                    train_phase=train_phase,
                    validation_phase=validation_phase,
                )
                if task == "high_residual_response":
                    prediction = _classification_probability(
                        clean_train, clean_validation, train_target
                    )
                    raw_prediction = _classification_probability(
                        raw_train, raw_validation, train_target
                    )
                elif task == "future_maneuver_trend":
                    prediction = _classification_prediction(
                        clean_train, clean_validation, train_target
                    )
                    raw_prediction = _classification_prediction(
                        raw_train, raw_validation, train_target
                    )
                else:
                    prediction = _regression_prediction(
                        clean_train, clean_validation, train_target
                    )
                    raw_prediction = _regression_prediction(
                        raw_train, raw_validation, train_target
                    )
                primary = _primary_metric(task, train_target, validation_target, prediction)
                raw_primary = _primary_metric(task, train_target, validation_target, raw_prediction)
                time_value = _primary_metric(
                    task, train_target, validation_target, time_predictions[task]
                )
                gain = _standardized_gain(task, primary, time_value)
                metric_rows.append(
                    _metric_row(
                        plan,
                        method=method,
                        task=task,
                        value=primary,
                        metric=_metric_name(task),
                        representation_variant="train_phase_residualized_pooled_64d",
                        time_baseline_value=time_value,
                        standardized_gain=gain,
                        status="completed",
                    )
                )
                metric_rows.append(
                    _metric_row(
                        plan,
                        method=method,
                        task=task,
                        value=raw_primary,
                        metric=_metric_name(task),
                        representation_variant="raw_pooled_64d_diagnostic",
                        time_baseline_value=time_value,
                        standardized_gain=_standardized_gain(task, raw_primary, time_value),
                        status="completed",
                    )
                )
                extras = _extra_metrics(task, train_target, validation_target, prediction)
                for name, value in extras.items():
                    metric_rows.append(
                        _metric_row(
                            plan,
                            method=method,
                            task=task,
                            value=value,
                            metric=name,
                            representation_variant="train_phase_residualized_pooled_64d",
                            time_baseline_value=time_value,
                            standardized_gain=gain,
                            status="completed",
                        )
                    )
                for sample_id, actual, predicted in zip(
                    validation_ids, validation_target, prediction, strict=True
                ):
                    prediction_rows.append(
                        {
                            "split_id": split_id,
                            "outer_pool_id": str(plan["outer_pool_id"]),
                            "main_selection": bool(plan["main_selection"]),
                            "method": method,
                            "task": task,
                            "context_id": sample_id,
                            "actual": float(actual),
                            "prediction": float(predicted),
                            "outer_test_opened": False,
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    summary, best = aggregate_matched_clean_metrics(metrics)
    return metrics, summary, best, pd.DataFrame(prediction_rows)


def aggregate_matched_clean_metrics(metrics: pd.DataFrame):
    formal = metrics[
        metrics["main_selection"].astype(bool)
        & (metrics["representation_variant"] == "train_phase_residualized_pooled_64d")
        & metrics["method"].isin(MATCHED_METHODS)
    ]
    rows = []
    best = {}
    for task in TASKS:
        metric = _metric_name(task)
        task_rows = formal[(formal["task"] == task) & (formal["metric"] == metric)]
        candidates = []
        for method in MATCHED_METHODS:
            subset = task_rows[task_rows["method"] == method]
            values = subset["value"].to_numpy(dtype=np.float64)
            if len(values) != 6:
                continue
            higher = task != "physiology_residual"
            item = {
                "task": task,
                "method": method,
                "metric": metric,
                "mean": _pool_balanced_mean(subset, "value"),
                "median": float(np.median(values)),
                "worst": float(np.min(values) if higher else np.max(values)),
                "positive_split_count": int(
                    np.sum(
                        subset["standardized_gain"].to_numpy(dtype=np.float64) > 0
                    )
                ),
                "median_standardized_gain_over_time": float(
                    np.median(subset["standardized_gain"].to_numpy(dtype=np.float64))
                ),
                "completed_split_count": len(values),
            }
            if task == "physiology_residual":
                skill = formal[
                    (formal["task"] == task)
                    & (formal["method"] == method)
                    & (formal["metric"] == "response_skill")
                ]
                ratio = formal[
                    (formal["task"] == task)
                    & (formal["method"] == method)
                    & (formal["metric"] == "rmse_ratio")
                ]
                item.update(
                    {
                        "mean_skill": _pool_balanced_mean(skill, "value"),
                        "median_rmse_ratio": float(np.median(ratio["value"])),
                        "positive_skill_split_count": int(np.sum(skill["value"] > 0)),
                    }
                )
            if task == "high_residual_response":
                normalized = formal[
                    (formal["task"] == task)
                    & (formal["method"] == method)
                    & (formal["metric"] == "normalized_ap")
                ]
                item.update(
                    {
                        "mean_normalized_ap": _pool_balanced_mean(normalized, "value"),
                        "median_normalized_ap": float(np.median(normalized["value"])),
                        "positive_normalized_ap_split_count": int(
                            np.sum(normalized["value"] > 0)
                        ),
                    }
                )
            rows.append(item)
            candidates.append(item)
        if not candidates:
            continue
        best[task] = (
            min(candidates, key=lambda row: (row["mean"], row["median"]))
            if task == "physiology_residual"
            else max(candidates, key=lambda row: (row["mean"], row["median"]))
        )
    return pd.DataFrame(rows), best


def _aligned_target(frame, split_id, role, column, ordered_ids):
    subset = frame[
        (frame["split_id"] == split_id)
        & (frame["role"] == role)
        & (frame["status"] == "completed")
    ].set_index("context_id")
    ids = tuple(str(value) for value in ordered_ids if str(value) in subset.index)
    return ids, subset.loc[list(ids), column].to_numpy()


def _select_embedding(batch, sample_ids):
    lookup = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
    missing = sorted(set(sample_ids) - set(lookup))
    if missing:
        raise ValueError(f"matched representation lacks target samples: {missing[:3]}")
    return batch.pooled_embedding.detach().cpu().numpy()[[lookup[value] for value in sample_ids]]


def _regression_prediction(train, validation, target):
    left, right = _usable_features(train, validation)
    if left.shape[1] == 0:
        return np.full(len(validation), float(np.mean(target)))
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(left, target).predict(right)


def _classification_prediction(train, validation, target):
    labels = np.asarray(target, dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.full(len(validation), int(classes[0]), dtype=np.int64)
    left, right = _usable_features(train, validation)
    if left.shape[1] == 0:
        return np.full(len(validation), int(np.bincount(labels).argmax()), dtype=np.int64)
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=17,
        ),
    ).fit(left, labels).predict(right)


def _classification_probability(train, validation, target):
    labels = np.asarray(target, dtype=np.int64)
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.full(len(validation), float(classes[0]))
    left, right = _usable_features(train, validation)
    if left.shape[1] == 0:
        return np.full(len(validation), float(np.mean(labels)))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=17,
        ),
    ).fit(left, labels)
    return model.predict_proba(right)[:, list(model.classes_).index(1)]


def _time_only_prediction(
    train_phase, validation_phase, target, *, classification, probability=False
):
    polynomial = PolynomialFeatures(degree=3, include_bias=True)
    train = polynomial.fit_transform(np.asarray(train_phase).reshape(-1, 1))
    validation = polynomial.transform(np.asarray(validation_phase).reshape(-1, 1))
    if classification:
        if probability:
            return _classification_probability(train, validation, target)
        return _classification_prediction(train, validation, target)
    return _regression_prediction(train, validation, target)


def _usable_features(train, validation):
    left = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    right = np.asarray(validation, dtype=np.float64).reshape(len(validation), -1)
    keep = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0) & (np.ptp(left, axis=0) > 1e-10)
    return left[:, keep], right[:, keep]


def _primary_metric(task, train_target, validation_target, prediction):
    if task == "future_maneuver_score":
        return safe_spearman(validation_target, prediction)
    if task == "future_maneuver_trend":
        return float(f1_score(validation_target, prediction, labels=(0, 1, 2), average="macro", zero_division=0))
    if task == "physiology_residual":
        return response_metrics(train_target=train_target, validation_target=validation_target, prediction=prediction)["rmse"]
    probability = prediction
    if np.asarray(prediction).dtype.kind in "iu":
        probability = np.asarray(prediction, dtype=np.float64)
    return _safe_high_response_metrics(validation_target, probability)["normalized_ap"]


def _extra_metrics(task, train_target, validation_target, prediction):
    if task == "physiology_residual":
        values = response_metrics(train_target=train_target, validation_target=validation_target, prediction=prediction)
        return {key: values[key] for key in ("rmse_ratio", "response_skill", "spearman")}
    if task == "high_residual_response":
        values = _safe_high_response_metrics(validation_target, prediction)
        return {key: values[key] for key in ("normalized_ap", "auprc", "prevalence")}
    return {}


def _safe_high_response_metrics(validation_target, probability):
    """Compute AP-derived metrics even when one validation class is absent.

    Residual labels are fit on inner-train only, so a naturally shifted validation
    support can legitimately contain only one class.  AP remains defined for an
    all-negative support (with a warning in sklearn), while normalized AP is not
    informative for an all-positive support.  Both cases are retained as pressure
    evidence instead of aborting the matched-clean panel.
    """

    target = np.asarray(validation_target, dtype=np.int64)
    score = np.asarray(probability, dtype=np.float64)
    prevalence = float(np.mean(target)) if len(target) else 0.0
    if len(target) == 0:
        auprc = 0.0
    elif prevalence <= 0.0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(target, score))
    normalized_ap = (
        float((auprc - prevalence) / (1.0 - prevalence))
        if prevalence < 1.0
        else 0.0
    )
    return {
        "normalized_ap": normalized_ap,
        "auprc": auprc,
        "prevalence": prevalence,
    }


def _metric_name(task):
    return {
        "future_maneuver_score": "spearman",
        "future_maneuver_trend": "macro_f1",
        "physiology_residual": "rmse",
        "high_residual_response": "normalized_ap",
    }[task]


def _standardized_gain(task, value, baseline):
    if task == "physiology_residual":
        return float((baseline - value) / max(abs(baseline), 1e-12))
    return float((value - baseline) / max(1.0 - baseline, 1e-12))


def _metric_row(plan, *, method, task, value, metric, representation_variant, time_baseline_value, standardized_gain, status):
    return {
        "split_id": str(plan["fold_id"]),
        "outer_pool_id": str(plan["outer_pool_id"]),
        "split_kind": str(plan["split_kind"]),
        "main_selection": bool(plan["main_selection"]),
        "method": method,
        "task": task,
        "metric": metric,
        "value": float(value),
        "representation_variant": representation_variant,
        "time_baseline_value": float(time_baseline_value),
        "standardized_gain": float(standardized_gain),
        "status": status,
        "fit_role": "inner_train",
        "evaluation_role": "inner_validation",
        "outer_test_opened": False,
    }


def _pool_balanced_mean(frame, column):
    return float(frame.groupby("outer_pool_id", sort=True)[column].mean().mean())
