"""Bounded upper-bound model panel for Dingxin inner validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from aeon.transformations.collection.convolution_based import (
    HydraTransformer,
    MiniRocket,
    MultiRocket,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True, slots=True)
class FoldTargets:
    train_maneuver: np.ndarray
    validation_maneuver: np.ndarray
    train_maneuver_score: np.ndarray
    validation_maneuver_score: np.ndarray
    train_response: np.ndarray
    validation_response: np.ndarray
    train_high_response: np.ndarray
    validation_high_response: np.ndarray


def evaluate_feature_family(
    *,
    fold_id: str,
    candidate_id: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    targets: FoldTargets,
    family: str,
    random_state: int = 17,
    tasks: Sequence[str] = ("maneuver", "response", "high_response"),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate a fixed representation with bounded task-specific heads."""

    train_features, validation_features = _drop_constant_train_columns(
        train_features,
        validation_features,
    )
    comparisons = []
    selected = []
    task_rows = []
    if "maneuver" in tasks:
        task_rows.append(
            (
                "maneuver",
                "macro_f1",
                "higher",
                _maneuver_heads(
                    train_features,
                    validation_features,
                    targets,
                    family=family,
                    random_state=random_state,
                ),
            )
        )
    if "response" in tasks:
        task_rows.append(
            (
                "response",
                "rmse",
                "lower",
                _response_heads(
                    train_features,
                    validation_features,
                    targets,
                    family=family,
                    random_state=random_state,
                ),
            )
        )
    if "high_response" in tasks:
        task_rows.append(
            (
                "high_response",
                "auprc",
                "higher",
                _high_response_heads(
                    train_features,
                    validation_features,
                    targets,
                    family=family,
                    random_state=random_state,
                ),
            )
        )
    for task, metric, direction, rows in task_rows:
        for row in rows:
            comparisons.append(
                {
                    "fold_id": fold_id,
                    "candidate_id": candidate_id,
                    "task": task,
                    "metric": metric,
                    "direction": direction,
                    "head_variant": row["head_variant"],
                    "value": row["value"],
                    "fit_role": "inner_train",
                    "evaluation_role": "inner_validation",
                    "outer_test_accessed": False,
                    "status": row.get("status", "completed"),
                }
            )
        available = [row for row in rows if row.get("value") is not None]
        best = (
            max(available, key=lambda row: row["value"])
            if direction == "higher"
            else min(available, key=lambda row: row["value"])
        )
        selected.append(
            {
                "fold_id": fold_id,
                "candidate_id": candidate_id,
                "task": task,
                "metric": metric,
                "direction": direction,
                "selected_head_variant": best["head_variant"],
                "value": float(best["value"]),
                "fit_role": "inner_train",
                "evaluation_role": "inner_validation",
                "outer_test_accessed": False,
                "status": "completed",
            }
        )
    return comparisons, selected


def transform_sequence_family(
    train_sequence: np.ndarray,
    validation_sequence: np.ndarray,
    *,
    family: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "minirocket_1000":
        transformer = MiniRocket(
            n_kernels=1_000,
            n_jobs=1,
            random_state=random_state,
        )
    elif family == "minirocket_5000":
        transformer = MiniRocket(
            n_kernels=5_000,
            n_jobs=1,
            random_state=random_state,
        )
    elif family == "multirocket":
        transformer = MultiRocket(
            n_kernels=1_250,
            n_features_per_kernel=4,
            n_jobs=1,
            random_state=random_state,
        )
    elif family == "hydra":
        transformer = HydraTransformer(
            n_kernels=8,
            n_groups=32,
            max_num_channels=8,
            n_jobs=1,
            random_state=random_state,
            output_type="numpy",
        )
    else:
        raise ValueError(f"unknown sequence family: {family}")
    transformed_train = np.asarray(transformer.fit_transform(train_sequence))
    transformed_validation = np.asarray(transformer.transform(validation_sequence))
    return (
        transformed_train.reshape(len(train_sequence), -1).astype(np.float32),
        transformed_validation.reshape(len(validation_sequence), -1).astype(np.float32),
    )


def evaluate_fieldwise_response(
    *,
    fold_id: str,
    candidate_id: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_field_targets: np.ndarray,
    validation_response: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    comparisons = []
    train_features, validation_features = _drop_constant_train_columns(
        train_features,
        validation_features,
    )
    train_target = np.nan_to_num(
        train_field_targets,
        nan=np.nanmedian(train_field_targets, axis=0),
    )
    selector = SelectKBest(
        f_regression,
        k=_k(train_features, 128),
    ).fit(train_features, train_target.mean(axis=1))
    selected_train = selector.transform(train_features)
    selected_validation = selector.transform(validation_features)
    for alpha in (0.1, 1.0, 10.0, 100.0):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha),
        )
        model.fit(selected_train, train_target)
        prediction = np.clip(
            model.predict(selected_validation),
            0.0,
            10.0,
        ).mean(axis=1)
        value = float(mean_squared_error(validation_response, prediction) ** 0.5)
        comparisons.append(
            {
                "fold_id": fold_id,
                "candidate_id": candidate_id,
                "task": "response",
                "metric": "rmse",
                "direction": "lower",
                "head_variant": f"fieldwise_ridge_alpha={alpha:g}",
                "value": value,
                "fit_role": "inner_train",
                "evaluation_role": "inner_validation",
                "outer_test_accessed": False,
                "status": "completed",
            }
        )
    best = min(comparisons, key=lambda row: row["value"])
    selected = {
        **best,
        "selected_head_variant": best["head_variant"],
    }
    return comparisons, selected


def _maneuver_heads(train, validation, targets, *, family, random_state):
    outputs = []
    if family == "histgb":
        model = make_pipeline(
            SelectKBest(f_classif, k=_k(train, 128)),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ).fit(train, targets.train_maneuver)
        outputs.append(
            _metric("hist_gradient_boosting", f1_score(
                targets.validation_maneuver,
                model.predict(validation),
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            ))
        )
        return outputs
    base = _classification_pipeline(train, random_state=random_state)
    base.fit(train, targets.train_maneuver)
    outputs.append(
        _metric("balanced_logistic", f1_score(
            targets.validation_maneuver,
            base.predict(validation),
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        ))
    )
    svm = make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=_k(train, 256)),
        LinearSVC(C=1.0, class_weight="balanced", random_state=random_state),
    ).fit(train, targets.train_maneuver)
    outputs.append(
        _metric("linear_svm", f1_score(
            targets.validation_maneuver,
            svm.predict(validation),
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        ))
    )
    ordinal_probabilities = []
    for boundary in (0, 1):
        classifier = _classification_pipeline(
            train,
            random_state=random_state,
            binary=True,
        ).fit(train, (targets.train_maneuver > boundary).astype(np.int64))
        ordinal_probabilities.append(classifier.predict_proba(validation)[:, 1])
    ordinal_prediction = sum(probability >= 0.5 for probability in ordinal_probabilities)
    outputs.append(
        _metric("ordinal_cumulative", f1_score(
            targets.validation_maneuver,
            ordinal_prediction,
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        ))
    )
    score_model = _regression_pipeline(train, alpha=10.0).fit(
        train, targets.train_maneuver_score
    )
    score_prediction = score_model.predict(validation)
    lower, upper = np.quantile(targets.train_maneuver_score, (1 / 3, 2 / 3))
    bucket = np.digitize(score_prediction, (lower, upper)).astype(np.int64)
    outputs.append(
        _metric("continuous_score_then_bucket", f1_score(
            targets.validation_maneuver,
            bucket,
            labels=(0, 1, 2),
            average="macro",
            zero_division=0,
        ))
    )
    return outputs


def _response_heads(train, validation, targets, *, family, random_state):
    outputs = []
    if family == "histgb":
        model = make_pipeline(
            SelectKBest(f_regression, k=_k(train, 128)),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ).fit(train, targets.train_response)
        prediction = np.clip(model.predict(validation), 0.0, 10.0)
        return [_response_metric("hist_gradient_boosting", prediction, targets)]
    for alpha in (0.1, 1.0, 10.0, 100.0):
        for transform in ("raw", "log1p"):
            fit_target = (
                np.log1p(targets.train_response)
                if transform == "log1p"
                else targets.train_response
            )
            model = _regression_pipeline(train, alpha=alpha).fit(train, fit_target)
            prediction = model.predict(validation)
            if transform == "log1p":
                prediction = np.expm1(prediction)
            outputs.append(
                _response_metric(
                    f"ridge_alpha={alpha:g};target={transform}",
                    np.clip(prediction, 0.0, 10.0),
                    targets,
                )
            )
    for components in (2, 4):
        if components >= len(train):
            continue
        model = make_pipeline(
            StandardScaler(),
            SelectKBest(f_regression, k=_k(train, 128)),
            PLSRegression(n_components=components, scale=True, max_iter=1_000),
        ).fit(train, targets.train_response)
        outputs.append(
            _response_metric(
                f"pls_components={components}",
                np.asarray(model.predict(validation)).reshape(-1),
                targets,
            )
        )
    huber = make_pipeline(
        StandardScaler(),
        SelectKBest(f_regression, k=_k(train, 64)),
        HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1_000),
    ).fit(train, targets.train_response)
    outputs.append(_response_metric("huber", huber.predict(validation), targets))
    kernel = make_pipeline(
        StandardScaler(),
        SelectKBest(f_regression, k=_k(train, 64)),
        KernelRidge(alpha=1.0, kernel="rbf", gamma=1 / max(_k(train, 64), 1)),
    ).fit(train, targets.train_response)
    outputs.append(
        _response_metric("kernel_ridge_rbf", kernel.predict(validation), targets)
    )
    return outputs


def _high_response_heads(
    train,
    validation,
    targets,
    *,
    family,
    random_state,
):
    outputs = []
    if len(np.unique(targets.train_high_response)) < 2:
        return [_metric("unavailable_single_train_class", None)]
    probabilities = {}
    for class_weight, name in ((None, "bce"), ("balanced", "balanced_bce")):
        model = _classification_pipeline(
            train,
            random_state=random_state,
            binary=True,
            class_weight=class_weight,
        ).fit(train, targets.train_high_response)
        probabilities[name] = model.predict_proba(validation)[:, 1]
        outputs.append(
            _metric(
                name,
                average_precision_score(
                    targets.validation_high_response, probabilities[name]
                ),
            )
        )
    response_model = _regression_pipeline(train, alpha=10.0).fit(
        train, targets.train_response
    )
    train_prediction = response_model.predict(train).reshape(-1, 1)
    validation_prediction = response_model.predict(validation).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    ).fit(train_prediction, targets.train_high_response)
    continuous_probability = calibrator.predict_proba(validation_prediction)[:, 1]
    outputs.append(
        _metric(
            "continuous_regression_threshold",
            average_precision_score(
                targets.validation_high_response, continuous_probability
            ),
        )
    )
    joint_probability = 0.5 * probabilities["balanced_bce"] + 0.5 * continuous_probability
    outputs.append(
        _metric(
            "joint_regression_risk",
            average_precision_score(
                targets.validation_high_response, joint_probability
            ),
        )
    )
    return outputs


def _classification_pipeline(
    train,
    *,
    random_state,
    binary=False,
    class_weight="balanced",
):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=_k(train, 256)),
        LogisticRegression(
            C=1.0,
            class_weight=class_weight,
            max_iter=2_000,
            solver="liblinear" if binary else "lbfgs",
            random_state=random_state,
        ),
    )


def _regression_pipeline(train, *, alpha):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_regression, k=_k(train, 256)),
        Ridge(alpha=alpha),
    )


def _response_metric(name, prediction, targets):
    return _metric(
        name,
        mean_squared_error(targets.validation_response, prediction) ** 0.5,
        prediction=np.asarray(prediction),
    )


def _metric(name, value, **extra):
    return {
        "head_variant": name,
        "value": None if value is None else float(value),
        **extra,
    }


def _k(matrix: np.ndarray, maximum: int) -> int:
    return max(1, min(int(maximum), matrix.shape[1], max(len(matrix) - 1, 1)))


def _drop_constant_train_columns(train, validation):
    train_array = np.nan_to_num(np.asarray(train, dtype=np.float64))
    validation_array = np.nan_to_num(np.asarray(validation, dtype=np.float64))
    selected = np.flatnonzero(np.ptp(train_array, axis=0) > 1e-10)
    if not len(selected):
        return (
            np.zeros((len(train_array), 1), dtype=np.float64),
            np.zeros((len(validation_array), 1), dtype=np.float64),
        )
    return train_array[:, selected], validation_array[:, selected]
