"""Validation-only selection for the fixed linear downstream heads."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_classifier(
    train_values,
    train_target,
    validation_values,
    validation_target,
    *,
    c_values,
    random_state,
    scaler_with_mean,
):
    best = None
    best_score = float("-inf")
    selected = None
    for c_value in c_values:
        candidate = make_pipeline(
            StandardScaler(with_mean=scaler_with_mean),
            LogisticRegression(
                C=float(c_value),
                class_weight="balanced",
                max_iter=5_000,
                random_state=random_state,
            ),
        ).fit(train_values, np.asarray(train_target, dtype=np.int64))
        score = (
            f1_score(
                np.asarray(validation_target, dtype=np.int64),
                candidate.predict(validation_values),
                average="macro",
                zero_division=0,
            )
            if validation_values is not None
            else 0.0
        )
        if score > best_score:
            best = candidate
            best_score = float(score)
            selected = float(c_value)
    return best, selected


def fit_regressor(
    train_values,
    train_target,
    validation_values,
    validation_target,
    *,
    alpha_values,
    scaler_with_mean,
):
    best = None
    best_score = float("inf")
    selected = None
    for alpha in alpha_values:
        candidate = make_pipeline(
            StandardScaler(with_mean=scaler_with_mean),
            Ridge(alpha=float(alpha)),
        ).fit(train_values, np.asarray(train_target, dtype=np.float64))
        score = (
            float(
                mean_squared_error(
                    np.asarray(validation_target, dtype=np.float64),
                    candidate.predict(validation_values),
                )
                ** 0.5
            )
            if validation_values is not None
            else 0.0
        )
        if score < best_score:
            best = candidate
            best_score = score
            selected = float(alpha)
    return best, selected
