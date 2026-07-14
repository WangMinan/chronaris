"""Fixed, predeclared lightweight heads for Dingxin task-stability screening."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_maneuver(
    train_features,
    validation_features,
    *,
    train_labels,
    train_score,
    head: str,
    random_state: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    train, validation = drop_constant_train_columns(train_features, validation_features)
    labels = np.asarray(train_labels, dtype=np.int64)
    score = np.asarray(train_score, dtype=np.float64)
    if head == "histgb":
        model = make_pipeline(
            SelectKBest(f_classif, k=_k(train, 128)),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ).fit(train, labels)
        prediction = model.predict(validation).astype(np.int64)
        return prediction, prediction.astype(np.float64)
    if head == "ordinal":
        probabilities = []
        for boundary in (0, 1):
            classifier = _classifier(train, random_state=random_state).fit(
                train, (labels > boundary).astype(np.int64)
            )
            probabilities.append(classifier.predict_proba(validation)[:, 1])
        prediction = sum(value >= 0.5 for value in probabilities).astype(np.int64)
        return prediction, probabilities[0] + probabilities[1]
    if head == "score_bucket":
        model = _regressor(train, alpha=10.0).fit(train, score)
        score_prediction = np.asarray(model.predict(validation)).reshape(-1)
        lower, upper = np.quantile(score, (1 / 3, 2 / 3))
        return np.digitize(score_prediction, (lower, upper)).astype(np.int64), score_prediction
    model = _classifier(train, random_state=random_state).fit(train, labels)
    probabilities = model.predict_proba(validation)
    prediction = model.classes_[np.argmax(probabilities, axis=1)].astype(np.int64)
    expected = probabilities @ model.classes_.astype(np.float64)
    return prediction, expected


def predict_response(
    train_features,
    validation_features,
    *,
    train_target,
    head: str,
    random_state: int = 17,
) -> np.ndarray:
    train, validation = drop_constant_train_columns(train_features, validation_features)
    target = np.asarray(train_target, dtype=np.float64)
    if head == "histgb":
        model = make_pipeline(
            SelectKBest(f_regression, k=_k(train, 128)),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ).fit(train, target)
        return np.clip(model.predict(validation), 0.0, 10.0)
    transformed = np.log1p(target) if head == "ridge_log1p" else target
    model = _regressor(train, alpha=10.0).fit(train, transformed)
    prediction = np.asarray(model.predict(validation)).reshape(-1)
    if head == "ridge_log1p":
        prediction = np.expm1(prediction)
    return np.clip(prediction, 0.0, 10.0)


def predict_high_response(
    train_features,
    validation_features,
    *,
    train_binary,
    train_response,
    head: str,
    random_state: int = 17,
) -> np.ndarray:
    train, validation = drop_constant_train_columns(train_features, validation_features)
    binary = np.asarray(train_binary, dtype=np.int64)
    classifier = _classifier(train, random_state=random_state).fit(train, binary)
    direct = classifier.predict_proba(validation)[:, 1]
    if head != "joint":
        return direct
    response_model = _regressor(train, alpha=10.0).fit(train, train_response)
    train_prediction = response_model.predict(train).reshape(-1, 1)
    validation_prediction = response_model.predict(validation).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    ).fit(train_prediction, binary)
    return 0.5 * direct + 0.5 * calibrator.predict_proba(validation_prediction)[:, 1]


def predict_fieldwise_response(
    train_features,
    validation_features,
    *,
    train_field_targets: np.ndarray,
) -> np.ndarray:
    train, validation = drop_constant_train_columns(train_features, validation_features)
    field_target = np.asarray(train_field_targets, dtype=np.float64)
    fill = np.nanmedian(field_target, axis=0)
    field_target = np.where(np.isfinite(field_target), field_target, fill)
    selector = SelectKBest(f_regression, k=_k(train, 128)).fit(
        train, np.mean(field_target, axis=1)
    )
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        selector.transform(train), field_target
    )
    return np.clip(model.predict(selector.transform(validation)), 0.0, 10.0).mean(
        axis=1
    )


def drop_constant_train_columns(train, validation):
    left = np.asarray(train, dtype=np.float32).reshape(len(train), -1)
    right = np.asarray(validation, dtype=np.float32).reshape(len(validation), -1)
    finite = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0)
    variable = np.ptp(left, axis=0) > 1e-10
    selected = finite & variable
    if not np.any(selected):
        return np.zeros((len(left), 1), dtype=np.float32), np.zeros(
            (len(right), 1), dtype=np.float32
        )
    return left[:, selected], right[:, selected]


def _classifier(train, *, random_state):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_classif, k=_k(train, 256)),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=random_state,
        ),
    )


def _regressor(train, *, alpha):
    return make_pipeline(
        StandardScaler(),
        SelectKBest(f_regression, k=_k(train, 256)),
        Ridge(alpha=alpha),
    )


def _k(matrix: np.ndarray, maximum: int) -> int:
    return max(1, min(int(maximum), int(matrix.shape[1])))
