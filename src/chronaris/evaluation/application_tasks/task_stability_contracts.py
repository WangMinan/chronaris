"""Metric and decision contracts for Dingxin inner-development stability audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)


MANEUVER_LABELS = (0, 1, 2)
FINAL_OUTER_THRESHOLDS = {
    "maneuver_macro_f1": {"operator": ">", "value": 0.9409316262},
    "response_rmse": {"operator": "<", "value": 0.2911709799},
    "high_response_auprc": {"operator": ">", "value": 0.8838786894},
}
RELATIVE_GATES = {
    "maneuver": {
        "mean_macro_f1": 0.85,
        "median_macro_f1": 0.85,
        "worst_macro_f1": 0.65,
        "minimum_class_recall": 0.55,
        "median_score_spearman": 0.50,
    },
    "response": {
        "median_rmse_ratio": 0.95,
        "mean_response_skill": 0.0,
        "positive_skill_fraction": 2 / 3,
        "median_spearman": 0.20,
    },
    "response_near_miss": {
        "median_rmse_ratio": 1.0,
        "positive_skill_fraction": 0.5,
    },
    "high_response": {
        "mean_normalized_ap": 0.20,
        "median_normalized_ap": 0.20,
        "positive_normalized_ap_fraction": 2 / 3,
        "mean_auprc_lift": 0.15,
    },
}


def stable_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def maneuver_metric_ceiling(
    validation_labels: Sequence[int], labels: Sequence[int] = MANEUVER_LABELS
) -> dict[str, object]:
    values = np.asarray(validation_labels, dtype=np.int64)
    support = tuple(int(value) for value in sorted(np.unique(values)))
    counts = {int(label): int(np.sum(values == label)) for label in labels}
    theoretical = len(set(support).intersection(labels)) / max(len(labels), 1)
    return {
        "validation_classes": support,
        "class_counts": counts,
        "fixed_macro_f1_ceiling": float(theoretical),
        "support_aware_macro_f1_ceiling": 1.0 if len(values) else 0.0,
        "minimum_class_count": min(counts.values()) if counts else 0,
        "complete_class_support": set(support) == set(labels),
    }


def maneuver_metrics(
    *,
    train_labels: Sequence[int],
    validation_labels: Sequence[int],
    prediction: Sequence[int],
    validation_score: Sequence[float],
    score_prediction: Sequence[float] | None = None,
) -> dict[str, float]:
    train = np.asarray(train_labels, dtype=np.int64)
    actual = np.asarray(validation_labels, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    support = tuple(int(value) for value in sorted(np.unique(actual)))
    majority = int(np.bincount(train, minlength=3).argmax())
    baseline = np.full(len(actual), majority, dtype=np.int64)
    recalls = recall_score(
        actual,
        predicted,
        labels=MANEUVER_LABELS,
        average=None,
        zero_division=0,
    )
    score_actual = np.asarray(validation_score, dtype=np.float64)
    score_predicted = (
        predicted.astype(np.float64)
        if score_prediction is None
        else np.asarray(score_prediction, dtype=np.float64)
    )
    return {
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=MANEUVER_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "support_aware_macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=support,
                average="macro",
                zero_division=0,
            )
        ),
        "majority_macro_f1": float(
            f1_score(
                actual,
                baseline,
                labels=MANEUVER_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_lift": float(
            f1_score(
                actual,
                predicted,
                labels=MANEUVER_LABELS,
                average="macro",
                zero_division=0,
            )
            - f1_score(
                actual,
                baseline,
                labels=MANEUVER_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_low": float(recalls[0]),
        "recall_medium": float(recalls[1]),
        "recall_high": float(recalls[2]),
        "minimum_class_recall": float(np.min(recalls)),
        "ordinal_mae": float(mean_absolute_error(actual, predicted)),
        "score_mae": float(mean_absolute_error(score_actual, score_predicted)),
        "score_spearman": safe_spearman(score_actual, score_predicted),
    }


def response_metrics(
    *,
    train_target: Sequence[float],
    validation_target: Sequence[float],
    prediction: Sequence[float],
) -> dict[str, float]:
    train = np.asarray(train_target, dtype=np.float64)
    actual = np.asarray(validation_target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    baseline = np.full(len(actual), float(np.mean(train)), dtype=np.float64)
    mse = float(mean_squared_error(actual, predicted))
    baseline_mse = float(mean_squared_error(actual, baseline))
    rmse = float(np.sqrt(mse))
    baseline_rmse = float(np.sqrt(baseline_mse))
    std = float(np.std(actual, ddof=0))
    iqr = float(np.quantile(actual, 0.75) - np.quantile(actual, 0.25))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(actual, predicted)),
        "validation_std": std,
        "validation_iqr": iqr,
        "nrmse_std": _safe_divide(rmse, std),
        "nrmse_iqr": _safe_divide(rmse, iqr),
        "train_mean_baseline_rmse": baseline_rmse,
        "rmse_ratio": _safe_divide(rmse, baseline_rmse),
        "response_skill": (
            float(1.0 - mse / baseline_mse) if baseline_mse > 1e-12 else 0.0
        ),
        "spearman": safe_spearman(actual, predicted),
    }


def high_response_metrics(
    *, validation_target: Sequence[int], probability: Sequence[float]
) -> dict[str, float]:
    actual = np.asarray(validation_target, dtype=np.int64)
    score = np.asarray(probability, dtype=np.float64)
    prevalence = float(np.mean(actual))
    auprc = float(average_precision_score(actual, score))
    normalized = _safe_divide(auprc - prevalence, 1.0 - prevalence)
    predicted = (score >= 0.5).astype(np.int64)
    return {
        "auprc": auprc,
        "prevalence": prevalence,
        "random_baseline_auprc": prevalence,
        "normalized_ap": normalized,
        "auprc_lift": auprc - prevalence,
        "auroc": float(roc_auc_score(actual, score)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
    }


def aggregate_values(values: Sequence[float], *, higher_is_better: bool) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot aggregate an empty metric panel")
    rng = np.random.default_rng(17)
    samples = np.asarray(
        [np.mean(rng.choice(array, size=len(array), replace=True)) for _ in range(2_000)]
    )
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "worst": float(np.min(array) if higher_is_better else np.max(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
        "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
        "split_count": int(len(array)),
    }


def decide_allowance(
    *, protocol_valid: bool, maneuver: Mapping[str, float],
    response: Mapping[str, float], high_response: Mapping[str, float],
    field_or_dual_gain: bool,
) -> dict[str, object]:
    mg = RELATIVE_GATES["maneuver"]
    rg = RELATIVE_GATES["response"]
    rn = RELATIVE_GATES["response_near_miss"]
    hg = RELATIVE_GATES["high_response"]
    maneuver_pass = bool(
        maneuver["mean_macro_f1"] >= mg["mean_macro_f1"]
        and maneuver["median_macro_f1"] >= mg["median_macro_f1"]
        and maneuver["worst_macro_f1"] >= mg["worst_macro_f1"]
        and maneuver["minimum_class_recall"] >= mg["minimum_class_recall"]
        and maneuver["mean_macro_f1_lift"] > 0
        and maneuver["median_score_spearman"] >= mg["median_score_spearman"]
    )
    response_pass = bool(
        response["median_rmse_ratio"] <= rg["median_rmse_ratio"]
        and response["mean_response_skill"] > rg["mean_response_skill"]
        and response["positive_skill_fraction"] >= rg["positive_skill_fraction"]
        and response["median_spearman"] >= rg["median_spearman"]
        and field_or_dual_gain
    )
    response_near = bool(
        response["median_rmse_ratio"] <= rn["median_rmse_ratio"]
        and response["positive_skill_fraction"] >= rn["positive_skill_fraction"]
        and field_or_dual_gain
    )
    high_pass = bool(
        high_response["mean_normalized_ap"] >= hg["mean_normalized_ap"]
        and high_response["median_normalized_ap"] >= hg["median_normalized_ap"]
        and high_response["positive_normalized_ap_fraction"]
        >= hg["positive_normalized_ap_fraction"]
        and high_response["mean_auprc_lift"] >= hg["mean_auprc_lift"]
    )
    full = protocol_valid and maneuver_pass and response_pass and high_pass
    risk_path = protocol_valid and maneuver_pass and response_near and high_pass
    gate_failures = []
    if not protocol_valid:
        gate_failures.append("protocol")
    if not maneuver_pass:
        gate_failures.append("maneuver")
    if not response_pass:
        gate_failures.append("response")
    if not response_near:
        gate_failures.append("response_near_miss")
    if not high_pass:
        gate_failures.append("high_response")
    if not protocol_valid:
        primary_blocker = "protocol_invalid"
    elif not maneuver_pass and not high_pass:
        primary_blocker = "cross_support_generalization_instability"
    elif not maneuver_pass:
        primary_blocker = "maneuver_relative_gate"
    elif not high_pass:
        primary_blocker = "high_response_relative_gate"
    elif not response_near:
        primary_blocker = "response_relative_and_near_miss_gates"
    else:
        primary_blocker = None
    return {
        "protocol_valid": bool(protocol_valid),
        "maneuver_gate_passed": maneuver_pass,
        "response_gate_passed": response_pass,
        "response_near_miss_passed": response_near,
        "high_response_gate_passed": high_pass,
        "allow_safe_fusion": bool(full or risk_path),
        "allow_task_aware_research": bool(full or risk_path),
        "response_task_risk": "none" if full else ("high" if risk_path else "blocked"),
        "gate_failures": gate_failures,
        "primary_blocker": primary_blocker,
        "gate_checks": {
            "maneuver": {
                "mean_macro_f1": maneuver["mean_macro_f1"] >= mg["mean_macro_f1"],
                "median_macro_f1": maneuver["median_macro_f1"]
                >= mg["median_macro_f1"],
                "worst_macro_f1": maneuver["worst_macro_f1"]
                >= mg["worst_macro_f1"],
                "minimum_class_recall": maneuver["minimum_class_recall"]
                >= mg["minimum_class_recall"],
                "positive_baseline_lift": maneuver["mean_macro_f1_lift"] > 0,
                "median_score_spearman": maneuver["median_score_spearman"]
                >= mg["median_score_spearman"],
            },
            "response": {
                "median_rmse_ratio": response["median_rmse_ratio"]
                <= rg["median_rmse_ratio"],
                "mean_response_skill": response["mean_response_skill"]
                > rg["mean_response_skill"],
                "positive_skill_fraction": response["positive_skill_fraction"]
                >= rg["positive_skill_fraction"],
                "median_spearman": response["median_spearman"]
                >= rg["median_spearman"],
                "field_or_dual_stable_gain": field_or_dual_gain,
            },
            "high_response": {
                "mean_normalized_ap": high_response["mean_normalized_ap"]
                >= hg["mean_normalized_ap"],
                "median_normalized_ap": high_response["median_normalized_ap"]
                >= hg["median_normalized_ap"],
                "positive_normalized_ap_fraction": high_response[
                    "positive_normalized_ap_fraction"
                ]
                >= hg["positive_normalized_ap_fraction"],
                "mean_auprc_lift": high_response["mean_auprc_lift"]
                >= hg["mean_auprc_lift"],
            },
        },
        "decision": (
            "passed"
            if full
            else "passed_with_response_risk"
            if risk_path
            else "protocol_unrecoverable"
            if not protocol_valid
            else "gap"
        ),
    }


def safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(predicted, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) <= 1e-12 or np.ptp(right) <= 1e-12:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0
