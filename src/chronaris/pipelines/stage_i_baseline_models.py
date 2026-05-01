"""Model-spec and fold-cache helpers for Stage I baselines."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVR


@dataclass(frozen=True, slots=True)
class StageIBaselineModelSpec:
    """One estimator plus the shared preprocessing mode it expects."""

    estimator: object
    preprocessing: str


@dataclass(frozen=True, slots=True)
class StageIBaselineLosoSplit:
    """One cached leave-one-subject-out fold."""

    split_group: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def objective_model_specs(profile: str) -> dict[str, StageIBaselineModelSpec]:
    if profile == "window_v2":
        return {
            "logistic_regression": StageIBaselineModelSpec(
                estimator=_build_window_logistic_regression(),
                preprocessing="scaled",
            ),
            "linear_svc": StageIBaselineModelSpec(
                estimator=LinearSVC(
                    class_weight="balanced",
                    dual=False,
                    max_iter=500,
                    tol=5e-2,
                ),
                preprocessing="scaled",
            ),
        }
    return {
        "logistic_regression": StageIBaselineModelSpec(
            estimator=LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=42),
            preprocessing="scaled",
        ),
        "linear_svc": StageIBaselineModelSpec(
            estimator=LinearSVC(class_weight="balanced", dual=False, max_iter=5_000),
            preprocessing="scaled",
        ),
        "random_forest_classifier": StageIBaselineModelSpec(
            estimator=RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
            preprocessing="imputed",
        ),
    }


def subjective_model_specs(profile: str) -> dict[str, StageIBaselineModelSpec]:
    if profile == "window_v2":
        return {
            "ridge_regression": StageIBaselineModelSpec(
                estimator=Ridge(alpha=1.0),
                preprocessing="scaled",
            ),
            "linear_svr": StageIBaselineModelSpec(
                estimator=LinearSVR(
                    C=1.0,
                    epsilon=0.1,
                    random_state=42,
                    max_iter=500,
                    tol=5e-2,
                ),
                preprocessing="scaled",
            ),
        }
    return {
        "ridge_regression": StageIBaselineModelSpec(
            estimator=Ridge(alpha=1.0),
            preprocessing="scaled",
        ),
        "svr": StageIBaselineModelSpec(
            estimator=SVR(C=1.0, epsilon=0.1),
            preprocessing="scaled",
        ),
        "random_forest_regressor": StageIBaselineModelSpec(
            estimator=RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),
            preprocessing="imputed",
        ),
    }


def build_loso_splits(split_groups: np.ndarray) -> tuple[StageIBaselineLosoSplit, ...]:
    unique_groups = tuple(sorted(set(split_groups.tolist())))
    return tuple(
        StageIBaselineLosoSplit(
            split_group=split_group,
            train_indices=np.flatnonzero(split_groups != split_group),
            test_indices=np.flatnonzero(split_groups == split_group),
        )
        for split_group in unique_groups
    )


def build_fold_cache(
    *,
    feature_matrix: np.ndarray,
    loso_splits: Sequence[StageIBaselineLosoSplit],
    preprocess_modes: set[str],
) -> tuple[tuple[StageIBaselineLosoSplit, Mapping[str, tuple[np.ndarray, np.ndarray]]], ...]:
    cached_folds: list[tuple[StageIBaselineLosoSplit, Mapping[str, tuple[np.ndarray, np.ndarray]]]] = []
    for split in loso_splits:
        imputer = _build_median_imputer()
        train_imputed = imputer.fit_transform(feature_matrix[split.train_indices])
        test_imputed = imputer.transform(feature_matrix[split.test_indices])
        prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "imputed": (train_imputed, test_imputed),
        }
        if "scaled" in preprocess_modes:
            scaler = StandardScaler()
            prepared["scaled"] = (
                scaler.fit_transform(train_imputed),
                scaler.transform(test_imputed),
            )
        cached_folds.append((split, prepared))
    return tuple(cached_folds)


def clone_estimator(model_spec: StageIBaselineModelSpec):
    return clone(model_spec.estimator)


def _build_window_logistic_regression() -> LogisticRegression:
    kwargs: dict[str, object] = {
        "max_iter": 100,
        "tol": 5e-2,
        "class_weight": "balanced",
        "random_state": 42,
        "solver": "sag",
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "ovr"
    return LogisticRegression(**kwargs)


def _build_median_imputer() -> SimpleImputer:
    kwargs: dict[str, object] = {"strategy": "median"}
    if "keep_empty_features" in signature(SimpleImputer).parameters:
        kwargs["keep_empty_features"] = True
    return SimpleImputer(**kwargs)
