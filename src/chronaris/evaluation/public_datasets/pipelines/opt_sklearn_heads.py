"""Head fitting and ensemble helpers for sklearn-backed public opt."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from chronaris.evaluation import evaluate_classification_predictions, evaluate_regression_predictions
from chronaris.modeling.common.run_observer import StageIRunProgress
from chronaris.evaluation.public_datasets.pipelines.opt_postprocess import merge_public_opt_prediction_frames
from chronaris.evaluation.public_datasets.pipelines.opt_shared import (
    safe_public_opt_classification_fallback,
    safe_public_opt_regression_fallback,
    sanitize_public_opt_classification_outputs,
    sanitize_public_opt_metrics,
    sanitize_public_opt_regression_outputs,
    should_use_public_opt_classification_fallback,
    should_use_public_opt_regression_fallback,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def _run_one_regression_head(
    *,
    subset_frame: pd.DataFrame,
    evaluation_group: str,
    head_name: str,
    feature_columns: Sequence[str],
    loso_splits,
    progress: StageIRunProgress | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_matrix = _extract_feature_matrix(subset_frame, feature_columns)
    y_all = subset_frame["y_true"].to_numpy(dtype=float, copy=True)
    for fold_index, split in enumerate(loso_splits):
        LOGGER.info(
            "task_eval_public_opt_sklearn regression subset=%s head=%s fold=%d/%d train=%d test=%d",
            evaluation_group,
            head_name,
            fold_index + 1,
            len(loso_splits),
            len(split.train_indices),
            len(split.test_indices),
        )
        if progress is not None:
            progress.update(
                "fold_start",
                dataset_id=str(subset_frame["dataset_id"].iloc[0]),
                subset=evaluation_group,
                candidate=head_name,
                head=head_name,
                fold_index=fold_index + 1,
                fold_count=len(loso_splits),
                train_count=len(split.train_indices),
                test_count=len(split.test_indices),
            )
        train_X = feature_matrix[split.train_indices]
        test_X = feature_matrix[split.test_indices]
        train_y = y_all[split.train_indices]
        fallback_value = safe_public_opt_regression_fallback(train_y)
        if should_use_public_opt_regression_fallback(train_y):
            predicted = np.full(
                shape=(len(split.test_indices),),
                fill_value=fallback_value,
                dtype=np.float32,
            )
        else:
            predicted = _fit_regression_head(
                head_name=head_name,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                split_groups=subset_frame.iloc[split.train_indices]["split_group"]
                .astype(str)
                .to_numpy(),
            )
            predicted, _ = sanitize_public_opt_regression_outputs(
                predicted,
                fallback_value=fallback_value,
            )
        test_frame = subset_frame.iloc[split.test_indices].copy()
        rows.append(
            pd.DataFrame(
                {
                    "track": "subjective",
                    "dataset_id": test_frame["dataset_id"].astype(str).to_numpy(),
                    "profile": test_frame["profile"].astype(str).to_numpy(),
                    "evaluation_group": np.full(
                        len(test_frame),
                        evaluation_group,
                        dtype=object,
                    ),
                    "subset_id": test_frame["subset_id"].astype(str).to_numpy(),
                    "head_name": np.full(len(test_frame), head_name, dtype=object),
                    "model_name": np.full(len(test_frame), head_name, dtype=object),
                    "split_group": test_frame["split_group"].astype(str).to_numpy(),
                    "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
                    "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
                    "session_id": test_frame["session_id"].astype(str).to_numpy(),
                    "y_true": test_frame["y_true"].to_numpy(dtype=float, copy=True),
                    "y_pred": predicted.astype(float, copy=False),
                }
            )
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def _run_one_classification_head(
    *,
    subset_frame: pd.DataFrame,
    evaluation_group: str,
    head_name: str,
    feature_columns: Sequence[str],
    loso_splits,
    label_order: Sequence[int | float],
    train_balance_policy: str,
    progress: StageIRunProgress | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_matrix = _extract_feature_matrix(subset_frame, feature_columns)
    y_all = subset_frame["y_true"].to_numpy(dtype=int, copy=True)
    for fold_index, split in enumerate(loso_splits):
        LOGGER.info(
            "task_eval_public_opt_sklearn classification subset=%s head=%s fold=%d/%d train=%d test=%d",
            evaluation_group,
            head_name,
            fold_index + 1,
            len(loso_splits),
            len(split.train_indices),
            len(split.test_indices),
        )
        if progress is not None:
            progress.update(
                "fold_start",
                dataset_id=str(subset_frame["dataset_id"].iloc[0]),
                subset=evaluation_group,
                candidate=head_name,
                head=head_name,
                fold_index=fold_index + 1,
                fold_count=len(loso_splits),
                train_count=len(split.train_indices),
                test_count=len(split.test_indices),
            )
        train_X = feature_matrix[split.train_indices]
        test_X = feature_matrix[split.test_indices]
        train_y = y_all[split.train_indices]
        fallback_label = safe_public_opt_classification_fallback(
            train_y,
            label_order=label_order,
        )
        if should_use_public_opt_classification_fallback(train_y):
            predicted = np.full(
                shape=(len(split.test_indices),),
                fill_value=fallback_label,
                dtype=np.int32,
            )
            confidence = np.ones((len(split.test_indices),), dtype=np.float32)
        else:
            predicted, confidence = _fit_classification_head(
                head_name=head_name,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                split_groups=subset_frame.iloc[split.train_indices]["split_group"]
                .astype(str)
                .to_numpy(),
                train_balance_policy=train_balance_policy,
            )
            predicted = sanitize_public_opt_classification_outputs(
                predicted,
                fallback_label=fallback_label,
            )
        test_frame = subset_frame.iloc[split.test_indices].copy()
        rows.append(
            pd.DataFrame(
                {
                    "track": "objective",
                    "dataset_id": test_frame["dataset_id"].astype(str).to_numpy(),
                    "profile": test_frame["profile"].astype(str).to_numpy(),
                    "evaluation_group": np.full(
                        len(test_frame),
                        evaluation_group,
                        dtype=object,
                    ),
                    "subset_id": test_frame["subset_id"].astype(str).to_numpy(),
                    "head_name": np.full(len(test_frame), head_name, dtype=object),
                    "model_name": np.full(len(test_frame), head_name, dtype=object),
                    "split_group": test_frame["split_group"].astype(str).to_numpy(),
                    "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
                    "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
                    "session_id": test_frame["session_id"].astype(str).to_numpy(),
                    "y_true": test_frame["y_true"].to_numpy(dtype=int, copy=True),
                    "y_pred": predicted.astype(int, copy=False),
                    "prediction_confidence": confidence.astype(float, copy=False),
                }
            )
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def _extract_feature_matrix(
    subset_frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    if not feature_columns:
        return np.zeros((len(subset_frame), 0), dtype=np.float32)
    feature_matrix = subset_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=float,
        copy=True,
    )
    return np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_regression_head(
    *,
    head_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    split_groups: np.ndarray,
) -> np.ndarray:
    if head_name == "target_prior_median":
        return _target_prior_predictions(
            train_y=train_y,
            test_count=len(test_X),
            policy="median",
        )
    if head_name == "target_prior_trimmed_mean":
        return _target_prior_predictions(
            train_y=train_y,
            test_count=len(test_X),
            policy="trimmed_mean",
        )
    if head_name == "physiology_persistence":
        model = Ridge(alpha=1.0)
        model.fit(train_X, train_y)
        return np.asarray(model.predict(test_X), dtype=np.float32)
    if head_name == "ridge_residual_cv":
        search = _fit_regression_search(
            estimator_name="ridge",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
        )
        return np.asarray(search.predict(test_X), dtype=np.float32)
    if head_name == "elasticnet_residual":
        search = _fit_regression_search(
            estimator_name="elasticnet",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
        )
        return np.asarray(search.predict(test_X), dtype=np.float32)
    if head_name == "huber_residual":
        search = _fit_regression_search(
            estimator_name="huber",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
        )
        return np.asarray(search.predict(test_X), dtype=np.float32)
    if head_name == "ridge_heat_physiology_lowdim":
        search = _fit_regression_search(
            estimator_name="ridge_lowdim",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
        )
        return np.asarray(search.predict(test_X), dtype=np.float32)
    if head_name == "huber_heat_physiology_lowdim":
        search = _fit_regression_search(
            estimator_name="huber_lowdim",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
        )
        return np.asarray(search.predict(test_X), dtype=np.float32)
    if head_name == "heat_prior_residual_guarded":
        return _fit_heat_prior_residual_guarded(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            split_groups=split_groups,
        )
    raise ValueError(f"unsupported public opt regression head: {head_name}")


def _target_prior_predictions(
    *,
    train_y: np.ndarray,
    test_count: int,
    policy: str,
) -> np.ndarray:
    prior_value = _target_prior_value(train_y=train_y, policy=policy)
    return np.full((test_count,), prior_value, dtype=np.float32)


def _target_prior_value(
    *,
    train_y: np.ndarray,
    policy: str,
) -> float:
    finite_values = np.asarray(train_y, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0
    if policy == "median":
        return float(np.median(finite_values))
    if policy == "trimmed_mean":
        lower, upper = np.quantile(finite_values, [0.1, 0.9])
        trimmed = finite_values[(finite_values >= lower) & (finite_values <= upper)]
        if trimmed.size == 0:
            trimmed = finite_values
        return float(np.mean(trimmed, dtype=np.float64))
    raise ValueError(f"unsupported target prior policy: {policy}")


def _fit_heat_prior_residual_guarded(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    split_groups: np.ndarray,
) -> np.ndarray:
    prior_value = _target_prior_value(train_y=train_y, policy="median")
    pure_prior = np.full((len(test_X),), prior_value, dtype=np.float32)
    if train_X.shape[1] == 0 or np.unique(split_groups).size < 2:
        return pure_prior
    if not _guarded_residual_improves_inner_cv(
        train_X=train_X,
        train_y=train_y,
        split_groups=split_groups,
    ):
        return pure_prior
    model = _fit_guarded_residual_model(
        train_X=train_X,
        train_residual=train_y - prior_value,
    )
    return np.asarray(prior_value + model.predict(test_X), dtype=np.float32)


def _guarded_residual_improves_inner_cv(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    split_groups: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    prior_predictions: list[np.ndarray] = []
    residual_predictions: list[np.ndarray] = []
    truth_values: list[np.ndarray] = []
    for held_out_group in sorted(np.unique(split_groups).tolist()):
        train_mask = split_groups != held_out_group
        validation_mask = split_groups == held_out_group
        if not np.any(train_mask) or not np.any(validation_mask):
            continue
        inner_train_y = train_y[train_mask]
        inner_prior = _target_prior_value(train_y=inner_train_y, policy="median")
        model = _fit_guarded_residual_model(
            train_X=train_X[train_mask],
            train_residual=inner_train_y - inner_prior,
        )
        validation_X = train_X[validation_mask]
        prior_predictions.append(
            np.full((len(validation_X),), inner_prior, dtype=np.float32)
        )
        residual_predictions.append(
            np.asarray(inner_prior + model.predict(validation_X), dtype=np.float32)
        )
        truth_values.append(train_y[validation_mask])
    if not truth_values:
        return False
    y_true = np.concatenate(truth_values).astype(float, copy=False)
    prior_pred = np.concatenate(prior_predictions).astype(float, copy=False)
    residual_pred = np.concatenate(residual_predictions).astype(float, copy=False)
    prior_rmse = _rmse(y_true, prior_pred)
    residual_rmse = _rmse(y_true, residual_pred)
    return bool(residual_rmse < prior_rmse - tolerance)


def _fit_guarded_residual_model(*, train_X: np.ndarray, train_residual: np.ndarray) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    model.fit(train_X, train_residual)
    return model


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(residual * residual, dtype=np.float64)))


def _fit_classification_head(
    *,
    head_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    split_groups: np.ndarray,
    train_balance_policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if head_name == "physiology_margin_balanced_logistic":
        search = _fit_classification_search(
            estimator_name="balanced_logistic",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
            train_balance_policy=train_balance_policy,
        )
        probabilities = search.predict_proba(test_X)
        predicted = np.asarray(search.predict(test_X), dtype=np.int32)
        confidence = probabilities.max(axis=1).astype(np.float32)
        return predicted, confidence
    if head_name == "balanced_logistic_context":
        search = _fit_classification_search(
            estimator_name="balanced_logistic",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
            train_balance_policy=train_balance_policy,
        )
        probabilities = search.predict_proba(test_X)
        predicted = np.asarray(search.predict(test_X), dtype=np.int32)
        confidence = probabilities.max(axis=1).astype(np.float32)
        return predicted, confidence
    if head_name == "balanced_linear_svc_context":
        search = _fit_classification_search(
            estimator_name="balanced_linear_svc",
            train_X=train_X,
            train_y=train_y,
            split_groups=split_groups,
            train_balance_policy=train_balance_policy,
        )
        decision = search.decision_function(test_X)
        predicted = np.asarray(search.predict(test_X), dtype=np.int32)
        confidence = _decision_confidence(decision)
        return predicted, confidence
    raise ValueError(f"unsupported public opt classification head: {head_name}")


def _fit_regression_search(
    *,
    estimator_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    split_groups: np.ndarray,
):
    pipeline, search_space = _regression_search_spec(estimator_name)
    unique_groups = np.unique(split_groups)
    if unique_groups.size < 2:
        pipeline.set_params(**_single_candidate_defaults(search_space))
        pipeline.fit(train_X, train_y)
        return pipeline
    cv = GroupKFold(n_splits=min(5, unique_groups.size))
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=search_space,
        scoring="neg_mean_squared_error",
        cv=cv.split(train_X, train_y, groups=split_groups),
        n_jobs=None,
        refit=True,
    )
    search.fit(train_X, train_y)
    return search.best_estimator_


def _fit_classification_search(
    *,
    estimator_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    split_groups: np.ndarray,
    train_balance_policy: str,
):
    pipeline, search_space = _classification_search_spec(
        estimator_name=estimator_name,
        train_balance_policy=train_balance_policy,
    )
    unique_groups = np.unique(split_groups)
    if unique_groups.size < 2:
        pipeline.set_params(**_single_candidate_defaults(search_space))
        pipeline.fit(train_X, train_y)
        return pipeline
    cv = GroupKFold(n_splits=min(5, unique_groups.size))
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=search_space,
        scoring="f1_macro",
        cv=cv.split(train_X, train_y, groups=split_groups),
        n_jobs=None,
        refit=True,
    )
    search.fit(train_X, train_y)
    return search.best_estimator_


def _regression_search_spec(
    estimator_name: str,
) -> tuple[Pipeline, dict[str, list[float]]]:
    if estimator_name == "ridge":
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        return pipeline, {"model__alpha": [0.1, 1.0, 4.0, 16.0, 64.0]}
    if estimator_name == "ridge_lowdim":
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        return pipeline, {"model__alpha": [0.01, 0.1, 1.0, 4.0, 16.0]}
    if estimator_name == "elasticnet":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(max_iter=5000, random_state=42)),
            ]
        )
        return pipeline, {
            "model__alpha": [0.01, 0.1, 1.0, 4.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7],
        }
    if estimator_name == "huber":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(max_iter=500)),
            ]
        )
        return pipeline, {
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__epsilon": [1.1, 1.35, 1.5],
        }
    if estimator_name == "huber_lowdim":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(max_iter=500)),
            ]
        )
        return pipeline, {
            "model__alpha": [0.00001, 0.0001, 0.001],
            "model__epsilon": [1.1, 1.35],
        }
    raise ValueError(f"unsupported regression search estimator: {estimator_name}")


def _classification_search_spec(
    *,
    estimator_name: str,
    train_balance_policy: str,
) -> tuple[Pipeline, dict[str, list[float]]]:
    class_weight = (
        "balanced" if train_balance_policy == "class_weight_balanced" else None
    )
    if estimator_name == "balanced_logistic":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight=class_weight),
                ),
            ]
        )
        return pipeline, {"model__C": [0.25, 1.0, 4.0, 16.0]}
    if estimator_name == "balanced_linear_svc":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearSVC(class_weight=class_weight, max_iter=20000),
                ),
            ]
        )
        return pipeline, {"model__C": [0.25, 1.0, 4.0, 16.0]}
    raise ValueError(f"unsupported classification search estimator: {estimator_name}")


def _single_candidate_defaults(
    search_space: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    return {key: float(values[0]) for key, values in search_space.items()}


def _decision_confidence(decision: np.ndarray) -> np.ndarray:
    raw = np.asarray(decision, dtype=np.float32)
    if raw.ndim == 1:
        return np.abs(raw)
    return np.max(raw, axis=1)


def _build_regression_ensemble(
    *,
    head_prediction_frames: Mapping[str, pd.DataFrame],
    head_metrics: Mapping[str, Mapping[str, object]],
    policy: str,
) -> tuple[str | None, pd.DataFrame | None, dict[str, object] | None]:
    if policy != "mean_top2" or len(head_prediction_frames) < 2:
        return None, None, None
    top_two = sorted(
        head_metrics,
        key=lambda name: (
            float(head_metrics[name]["rmse"]),
            float(head_metrics[name]["mae"]),
        ),
    )[:2]
    merged = merge_public_opt_prediction_frames(
        [head_prediction_frames[name] for name in top_two]
    )
    ensemble = merged.loc[
        :,
        [
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
        ],
    ].copy()
    ensemble["track"] = "subjective"
    ensemble["head_name"] = "mean_top2_ensemble"
    ensemble["model_name"] = "mean_top2_ensemble"
    ensemble["y_pred"] = merged[
        [f"y_pred__{index}" for index in range(len(top_two))]
    ].mean(axis=1)
    ensemble = ensemble[
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "head_name",
            "model_name",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
            "y_pred",
        ]
    ]
    metrics = sanitize_public_opt_metrics(evaluate_regression_predictions(ensemble))
    return "mean_top2_ensemble", ensemble, metrics


def _build_classification_ensemble(
    *,
    head_prediction_frames: Mapping[str, pd.DataFrame],
    head_metrics: Mapping[str, Mapping[str, object]],
    label_order: Sequence[int | float],
    policy: str,
) -> tuple[str | None, pd.DataFrame | None, dict[str, object] | None]:
    if policy != "vote_top2" or len(head_prediction_frames) < 2:
        return None, None, None
    top_two = sorted(
        head_metrics,
        key=lambda name: (
            float(head_metrics[name]["macro_f1"]),
            float(head_metrics[name]["balanced_accuracy"]),
        ),
        reverse=True,
    )[:2]
    merged = merge_public_opt_prediction_frames(
        [head_prediction_frames[name] for name in top_two]
    )
    ensemble = merged.loc[
        :,
        [
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
        ],
    ].copy()
    ensemble["track"] = "objective"
    ensemble["head_name"] = "vote_top2_ensemble"
    ensemble["model_name"] = "vote_top2_ensemble"
    first_pred = merged["y_pred__0"].to_numpy(dtype=int, copy=True)
    second_pred = merged["y_pred__1"].to_numpy(dtype=int, copy=True)
    first_conf = merged.get(
        "prediction_confidence__0",
        pd.Series(np.ones(len(merged))),
    ).to_numpy(dtype=float, copy=True)
    second_conf = merged.get(
        "prediction_confidence__1",
        pd.Series(np.ones(len(merged))),
    ).to_numpy(dtype=float, copy=True)
    predicted = np.where(
        first_pred == second_pred,
        first_pred,
        np.where(first_conf >= second_conf, first_pred, second_pred),
    )
    confidence = np.maximum(first_conf, second_conf)
    ensemble["y_pred"] = predicted.astype(int, copy=False)
    ensemble["prediction_confidence"] = confidence.astype(float, copy=False)
    ensemble = ensemble[
        [
            "track",
            "dataset_id",
            "profile",
            "evaluation_group",
            "subset_id",
            "head_name",
            "model_name",
            "split_group",
            "sample_id",
            "subject_id",
            "session_id",
            "y_true",
            "y_pred",
            "prediction_confidence",
        ]
    ]
    metrics = sanitize_public_opt_metrics(
        evaluate_classification_predictions(ensemble, label_order=label_order)
    )
    return "vote_top2_ensemble", ensemble, metrics
