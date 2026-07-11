"""Direction-aware application metrics, segmentation scores, and paired statistics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    y_true,
    y_pred,
    probabilities,
    classes,
    *,
    calibration_bins: int = 10,
):
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    probability = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes, dtype=int)
    binary = label_binarize(truth, classes=classes)
    if len(classes) == 2:
        binary = np.column_stack((1 - binary[:, 0], binary[:, 0]))
    per_class = [
        average_precision_score(binary[:, index], probability[:, index])
        for index in range(len(classes))
        if binary[:, index].sum() > 0
    ]
    one_hot = binary.astype(float)
    return {
        "macro_f1": (f1_score(truth, prediction, average="macro", zero_division=0), "higher"),
        "balanced_accuracy": (balanced_accuracy_score(truth, prediction), "higher"),
        "macro_auprc": (float(np.mean(per_class)) if per_class else None, "higher"),
        "brier_score": (float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))), "lower"),
        "ece_10_bins": (_expected_calibration_error(truth, probability, classes, calibration_bins), "lower"),
    }


def regression_metrics(y_true, y_pred):
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    correlation = spearmanr(truth, prediction).statistic
    return {
        "mae": (mean_absolute_error(truth, prediction), "lower"),
        "rmse": (mean_squared_error(truth, prediction) ** 0.5, "lower"),
        "spearman": (float(correlation) if np.isfinite(correlation) else None, "higher"),
    }


def segmentation_metrics(
    y_true,
    y_pred,
    *,
    query_step_s: float = 30.0 / 96.0,
):
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("segmentation labels must share shape [N,T]")
    metrics = {
        "frame_macro_f1": (
            f1_score(truth.ravel(), prediction.ravel(), average="macro", zero_division=0),
            "higher",
        ),
    }
    for threshold in (0.10, 0.25, 0.50):
        metrics[f"segmental_f1_iou_{threshold:.2f}"] = (
            _mean_segmental_f1(truth, prediction, threshold),
            "higher",
        )
    for tolerance_s in (1.0, 2.0):
        metrics[f"boundary_f1_{tolerance_s:.0f}s"] = (
            _mean_boundary_f1(
                truth,
                prediction,
                tolerance_s=tolerance_s,
                query_step_s=query_step_s,
            ),
            "higher",
        )
    metrics["edit_score"] = (_mean_edit_score(truth, prediction), "higher")
    delay = _mean_detection_delay(truth, prediction, query_step_s=query_step_s)
    metrics["boundary_detection_delay_s"] = (delay, "lower")
    return metrics


def compute_fusion_gain_rows(metric_rows, *, fusion_methods):
    index = {}
    for row in metric_rows:
        key = (
            row["dataset"],
            row["task"],
            row["consumer"],
            row["seed"],
            row["fold"],
            row["role"],
            row["metric"],
        )
        index[(key, row["method"])] = row
    result = []
    keys = {key for key, method in index if method in fusion_methods}
    for key in sorted(keys):
        physiology = index.get((key, "physiology_only"))
        vehicle = index.get((key, "vehicle_only"))
        if physiology is None or vehicle is None:
            continue
        for method in fusion_methods:
            fusion = index.get((key, method))
            if fusion is None:
                continue
            values = (physiology["value"], vehicle["value"], fusion["value"])
            if any(value is None for value in values):
                gain = None
                status = "unavailable"
            elif fusion["direction"] == "higher":
                gain = fusion["value"] - max(physiology["value"], vehicle["value"])
                status = "available"
            else:
                gain = min(physiology["value"], vehicle["value"]) - fusion["value"]
                status = "available"
            result.append(
                {
                    "dataset": key[0],
                    "task": key[1],
                    "consumer": key[2],
                    "seed": key[3],
                    "fold": key[4],
                    "role": key[5],
                    "metric": key[6],
                    "fusion_method": method,
                    "best_single_value": (
                        max(physiology["value"], vehicle["value"])
                        if fusion["direction"] == "higher" and status == "available"
                        else min(physiology["value"], vehicle["value"])
                        if status == "available"
                        else None
                    ),
                    "fusion_value": fusion["value"],
                    "fusion_gain": gain,
                    "direction_normalized": "higher_is_better",
                    "status": status,
                }
            )
    return result


@dataclass(frozen=True, slots=True)
class PairedStatistic:
    mean_difference: float
    bootstrap_lower: float
    bootstrap_upper: float
    permutation_p_value: float
    independent_unit_count: int


def paired_trajectory_statistic(
    first,
    second,
    *,
    seed: int = 17,
    bootstrap_repetitions: int = 2000,
) -> PairedStatistic:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape or first_values.ndim != 1:
        raise ValueError("paired statistics require equal one-dimensional arrays")
    if len(first_values) < 2:
        raise ValueError("paired statistics require at least two independent units")
    differences = first_values - second_values
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences,
        size=(bootstrap_repetitions, len(differences)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    observed = abs(differences.mean())
    if len(differences) <= 16:
        sign_means = np.asarray(
            [
                np.mean(differences * np.asarray(signs))
                for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
            ]
        )
    else:
        signs = rng.choice((-1.0, 1.0), size=(10_000, len(differences)))
        sign_means = (signs * differences).mean(axis=1)
    p_value = (np.sum(np.abs(sign_means) >= observed) + 1) / (
        len(sign_means) + 1
    )
    return PairedStatistic(
        mean_difference=float(differences.mean()),
        bootstrap_lower=float(lower),
        bootstrap_upper=float(upper),
        permutation_p_value=float(p_value),
        independent_unit_count=len(differences),
    )


def _segments(values):
    result = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            result.append((int(values[start]), start, index))
            start = index
    return result


def _mean_segmental_f1(truth, prediction, threshold):
    values = []
    for true_row, pred_row in zip(truth, prediction, strict=True):
        true_segments = _segments(true_row)
        pred_segments = _segments(pred_row)
        matched = set()
        true_positive = 0
        for pred_class, pred_start, pred_end in pred_segments:
            candidates = []
            for index, (true_class, true_start, true_end) in enumerate(true_segments):
                if index in matched or true_class != pred_class:
                    continue
                intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
                union = max(pred_end, true_end) - min(pred_start, true_start)
                candidates.append((intersection / max(union, 1), index))
            if candidates:
                best_iou, best_index = max(candidates)
                if best_iou >= threshold:
                    matched.add(best_index)
                    true_positive += 1
        precision = true_positive / max(len(pred_segments), 1)
        recall = true_positive / max(len(true_segments), 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def _mean_boundary_f1(truth, prediction, *, tolerance_s, query_step_s):
    values = []
    tolerance_points = int(np.floor(tolerance_s / query_step_s + 1e-9))
    for true_row, pred_row in zip(truth, prediction, strict=True):
        true_boundaries = np.flatnonzero(true_row[1:] != true_row[:-1]) + 1
        pred_boundaries = np.flatnonzero(pred_row[1:] != pred_row[:-1]) + 1
        matched = set()
        true_positive = 0
        for boundary in pred_boundaries:
            candidates = [
                (abs(int(boundary) - int(value)), index)
                for index, value in enumerate(true_boundaries)
                if index not in matched
                and abs(int(boundary) - int(value)) <= tolerance_points
            ]
            if candidates:
                _, index = min(candidates)
                matched.add(index)
                true_positive += 1
        precision = true_positive / max(len(pred_boundaries), 1)
        recall = true_positive / max(len(true_boundaries), 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def _mean_edit_score(truth, prediction):
    values = []
    for true_row, pred_row in zip(truth, prediction, strict=True):
        true_tokens = [item[0] for item in _segments(true_row)]
        pred_tokens = [item[0] for item in _segments(pred_row)]
        distance = _levenshtein(true_tokens, pred_tokens)
        values.append(1.0 - distance / max(len(true_tokens), len(pred_tokens), 1))
    return float(np.mean(values))


def _mean_detection_delay(truth, prediction, *, query_step_s):
    delays = []
    for true_row, pred_row in zip(truth, prediction, strict=True):
        true_boundaries = np.flatnonzero(true_row[1:] != true_row[:-1]) + 1
        pred_boundaries = np.flatnonzero(pred_row[1:] != pred_row[:-1]) + 1
        for boundary in true_boundaries:
            later = pred_boundaries[pred_boundaries >= boundary]
            if len(later):
                delays.append((later[0] - boundary) * query_step_s)
    return float(np.mean(delays)) if delays else None


def _levenshtein(first, second):
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, start=1):
        current = [row]
        for column, right in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def _expected_calibration_error(truth, probability, classes, bins):
    confidence = probability.max(axis=1)
    prediction = classes[probability.argmax(axis=1)]
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if not selected.any():
            continue
        accuracy = np.mean(prediction[selected] == truth[selected])
        error += selected.mean() * abs(accuracy - confidence[selected].mean())
    return float(error)
