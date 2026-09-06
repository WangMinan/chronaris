"""Validation-only selection for the fixed linear downstream heads."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def grouped_score(metric, target, prediction, groups=None):
    """Average subject metrics, rather than giving long recordings more votes."""
    if groups is None:
        return float(metric(target, prediction))
    groups = np.asarray(groups)
    if groups.shape != (len(target),) or not len(groups):
        raise ValueError("validation groups must cover every target")
    return float(np.mean([metric(target[groups == group], prediction[groups == group])
                          for group in np.unique(groups)]))


def _sample_weights(values, sample_weight):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("consumer sample weights must be finite, positive and aligned")
    return weights


def fit_classifier(
    train_values,
    train_target,
    validation_values,
    validation_target,
    *,
    c_values,
    random_state,
    scaler_with_mean,
    classification_labels=None,
    solver="lbfgs",
    train_sample_weight=None,
    validation_groups=None,
):
    weights = _sample_weights(train_values, train_sample_weight)
    target = np.asarray(train_target, dtype=np.int64)
    class_weight = "balanced"
    if weights is not None:
        if solver == "liblinear_ovr":
            raise ValueError("weighted consumers require a direct logistic solver")
        classes = np.unique(target)
        class_weight = {int(label): weights.sum() / (len(classes) * weights[target == label].sum())
                        for label in classes}
    best = None
    best_score = float("-inf")
    selected = None
    for c_value in c_values:
        estimator = LogisticRegression(
            C=float(c_value),
            class_weight=class_weight,
            max_iter=5_000,
            random_state=random_state,
            solver="liblinear" if solver == "liblinear_ovr" else solver,
        )
        if solver == "liblinear_ovr":
            estimator = OneVsRestClassifier(estimator, n_jobs=1)
        candidate = make_pipeline(
            StandardScaler(with_mean=scaler_with_mean),
            estimator,
        )
        fit_weights = {} if weights is None else {
            "standardscaler__sample_weight": weights, "logisticregression__sample_weight": weights}
        candidate.fit(train_values, target, **fit_weights)
        score = (
            grouped_score(lambda truth, prediction: f1_score(truth, prediction, labels=classification_labels,
                average="macro", zero_division=0), np.asarray(validation_target, dtype=np.int64),
                candidate.predict(validation_values), validation_groups)
            if validation_values is not None
            else 0.0
        )
        if score > best_score:
            best = candidate
            best_score = float(score)
            selected = float(c_value)
    return best, selected


def classifier_classes(classifier) -> np.ndarray:
    """Return fitted class order for either direct or explicit OvR logistic heads."""
    return np.asarray(classifier[-1].classes_, dtype=np.int64)


def fit_regressor(
    train_values,
    train_target,
    validation_values,
    validation_target,
    *,
    alpha_values,
    scaler_with_mean,
    train_sample_weight=None,
    validation_groups=None,
):
    weights = _sample_weights(train_values, train_sample_weight)
    best = None
    best_score = float("inf")
    selected = None
    for alpha in alpha_values:
        candidate = make_pipeline(
            StandardScaler(with_mean=scaler_with_mean),
            Ridge(alpha=float(alpha)),
        )
        fit_weights = {} if weights is None else {
            "standardscaler__sample_weight": weights, "ridge__sample_weight": weights}
        candidate.fit(train_values, np.asarray(train_target, dtype=np.float64), **fit_weights)
        score = (
            grouped_score(lambda truth, prediction: mean_squared_error(truth, prediction) ** .5,
                np.asarray(validation_target, dtype=np.float64), candidate.predict(validation_values), validation_groups)
            if validation_values is not None
            else 0.0
        )
        if score < best_score:
            best = candidate
            best_score = score
            selected = float(alpha)
    return best, selected
