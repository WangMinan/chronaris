"""Sklearn-backed runners for Stage I public-opt heads."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from chronaris.evaluation import evaluate_classification_predictions, evaluate_regression_predictions
from chronaris.pipelines.stage_i.stage_i_baseline_models import build_loso_splits
from chronaris.pipelines.stage_i.stage_i_public_opt_data import PUBLIC_OPT_FEATURE_PROFILES, StageIPublicOptFeatureFrameResult
from chronaris.pipelines.stage_i.stage_i_public_opt_shared import (
    safe_public_opt_classification_fallback,
    safe_public_opt_regression_fallback,
    sanitize_public_opt_classification_outputs,
    sanitize_public_opt_metrics,
    sanitize_public_opt_regression_outputs,
    should_use_public_opt_classification_fallback,
    should_use_public_opt_regression_fallback,
)


def run_public_opt_backend(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    head_feature_columns: Mapping[str, Sequence[str]],
    train_balance_policy: str,
    ensemble_policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if feature_result.task_type == "regression":
        return _run_public_opt_regression(
            feature_result=feature_result,
            head_feature_columns=head_feature_columns,
            ensemble_policy=ensemble_policy,
        )
    if feature_result.task_type == "classification":
        return _run_public_opt_classification(
            feature_result=feature_result,
            head_feature_columns=head_feature_columns,
            train_balance_policy=train_balance_policy,
            ensemble_policy=ensemble_policy,
        )
    raise ValueError(f"unsupported public opt task type: {feature_result.task_type}")


def validate_public_opt_config(
    *,
    feature_profile: str,
    head_catalog: str,
    train_balance_policy: str,
    ensemble_policy: str,
    winner_margin_policy: str,
) -> None:
    if feature_profile not in set(PUBLIC_OPT_FEATURE_PROFILES):
        raise ValueError(f"unsupported public opt feature_profile: {feature_profile}")
    if head_catalog not in {"minimal", "expanded"}:
        raise ValueError(f"unsupported public opt head_catalog: {head_catalog}")
    if train_balance_policy not in {"none", "class_weight_balanced"}:
        raise ValueError(
            "unsupported public opt train_balance_policy: "
            f"{train_balance_policy}"
        )
    if ensemble_policy not in {"none", "mean_top2", "vote_top2"}:
        raise ValueError(f"unsupported public opt ensemble_policy: {ensemble_policy}")
    if winner_margin_policy not in {"paper_gate", "none"}:
        raise ValueError(
            "unsupported public opt winner_margin_policy: "
            f"{winner_margin_policy}"
        )


def resolve_head_feature_columns(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    feature_profile: str,
    head_catalog: str,
) -> dict[str, tuple[str, ...]]:
    if head_catalog == "minimal":
        catalog = {
            "subjective": ("physiology_persistence", "ridge_residual_cv"),
            "objective": (
                "physiology_margin_balanced_logistic",
                "balanced_logistic_context",
            ),
        }
    else:
        catalog = {
            "subjective": tuple(feature_result.head_feature_columns),
            "objective": tuple(feature_result.head_feature_columns),
        }
    allowed_columns = set(feature_result.feature_groups[feature_profile])
    resolved: dict[str, tuple[str, ...]] = {}
    for head_name in catalog[feature_result.track]:
        default_columns = tuple(feature_result.head_feature_columns[head_name])
        if head_name == "physiology_persistence":
            resolved_columns = default_columns
        else:
            filtered = tuple(
                column for column in default_columns if column in allowed_columns
            )
            resolved_columns = filtered or default_columns
        resolved[head_name] = resolved_columns
    return resolved


def _run_public_opt_regression(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    head_feature_columns: Mapping[str, Sequence[str]],
    ensemble_policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    subset_results: dict[str, object] = {}

    for evaluation_group in feature_result.subset_order:
        subset_frame = _select_evaluation_group_frame(
            feature_result.feature_frame,
            evaluation_group=evaluation_group,
            evaluation_groups=feature_result.evaluation_groups,
        )
        if subset_frame.empty:
            continue
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        head_metrics: dict[str, dict[str, object]] = {}
        head_prediction_frames: dict[str, pd.DataFrame] = {}
        for head_name, feature_columns in head_feature_columns.items():
            predictions = _run_one_regression_head(
                subset_frame=subset_frame,
                evaluation_group=evaluation_group,
                head_name=head_name,
                feature_columns=tuple(feature_columns),
                loso_splits=loso_splits,
            )
            metrics = sanitize_public_opt_metrics(
                evaluate_regression_predictions(predictions)
            )
            head_metrics[head_name] = metrics
            head_prediction_frames[head_name] = predictions
            prediction_frames.append(predictions)
        ensemble_name, ensemble_predictions, ensemble_metrics = _build_regression_ensemble(
            head_prediction_frames=head_prediction_frames,
            head_metrics=head_metrics,
            policy=ensemble_policy,
        )
        if (
            ensemble_name
            and ensemble_predictions is not None
            and ensemble_metrics is not None
        ):
            head_metrics[ensemble_name] = ensemble_metrics
            head_prediction_frames[ensemble_name] = ensemble_predictions
            prediction_frames.append(ensemble_predictions)
        best_head = min(
            head_metrics,
            key=lambda name: (
                float(head_metrics[name]["rmse"]),
                float(head_metrics[name]["mae"]),
            ),
        )
        subset_results[evaluation_group] = {
            "sample_count": int(len(subset_frame)),
            "fold_count": int(subset_frame["split_group"].nunique()),
            "best_head": best_head,
            "heads": head_metrics,
        }

    predictions = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return predictions, subset_results


def _run_public_opt_classification(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    head_feature_columns: Mapping[str, Sequence[str]],
    train_balance_policy: str,
    ensemble_policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    subset_results: dict[str, object] = {}
    if feature_result.label_order is None:
        raise ValueError("classification public opt requires explicit label_order.")

    for evaluation_group in feature_result.subset_order:
        subset_frame = _select_evaluation_group_frame(
            feature_result.feature_frame,
            evaluation_group=evaluation_group,
            evaluation_groups=feature_result.evaluation_groups,
        )
        if subset_frame.empty:
            continue
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        head_metrics: dict[str, dict[str, object]] = {}
        head_prediction_frames: dict[str, pd.DataFrame] = {}
        for head_name, feature_columns in head_feature_columns.items():
            predictions = _run_one_classification_head(
                subset_frame=subset_frame,
                evaluation_group=evaluation_group,
                head_name=head_name,
                feature_columns=tuple(feature_columns),
                loso_splits=loso_splits,
                label_order=feature_result.label_order,
                train_balance_policy=train_balance_policy,
            )
            metrics = sanitize_public_opt_metrics(
                evaluate_classification_predictions(
                    predictions,
                    label_order=feature_result.label_order,
                )
            )
            head_metrics[head_name] = metrics
            head_prediction_frames[head_name] = predictions
            prediction_frames.append(predictions)
        ensemble_name, ensemble_predictions, ensemble_metrics = _build_classification_ensemble(
            head_prediction_frames=head_prediction_frames,
            head_metrics=head_metrics,
            label_order=feature_result.label_order,
            policy=ensemble_policy,
        )
        if (
            ensemble_name
            and ensemble_predictions is not None
            and ensemble_metrics is not None
        ):
            head_metrics[ensemble_name] = ensemble_metrics
            head_prediction_frames[ensemble_name] = ensemble_predictions
            prediction_frames.append(ensemble_predictions)
        best_head = max(
            head_metrics,
            key=lambda name: (
                float(head_metrics[name]["macro_f1"]),
                float(head_metrics[name]["balanced_accuracy"]),
            ),
        )
        subset_results[evaluation_group] = {
            "sample_count": int(len(subset_frame)),
            "fold_count": int(subset_frame["split_group"].nunique()),
            "best_head": best_head,
            "heads": head_metrics,
        }

    predictions = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return predictions, subset_results


def _select_evaluation_group_frame(
    feature_frame: pd.DataFrame,
    *,
    evaluation_group: str,
    evaluation_groups: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    subset_ids = tuple(evaluation_groups[evaluation_group])
    return feature_frame.loc[
        feature_frame["subset_id"].astype(str).isin(subset_ids)
    ].copy()


def _run_one_regression_head(
    *,
    subset_frame: pd.DataFrame,
    evaluation_group: str,
    head_name: str,
    feature_columns: Sequence[str],
    loso_splits,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_matrix = _extract_feature_matrix(subset_frame, feature_columns)
    y_all = subset_frame["y_true"].to_numpy(dtype=float, copy=True)
    for split in loso_splits:
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
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_matrix = _extract_feature_matrix(subset_frame, feature_columns)
    y_all = subset_frame["y_true"].to_numpy(dtype=int, copy=True)
    for split in loso_splits:
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
    raise ValueError(f"unsupported public opt regression head: {head_name}")


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
    merged = _merge_prediction_frames([head_prediction_frames[name] for name in top_two])
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
    merged = _merge_prediction_frames([head_prediction_frames[name] for name in top_two])
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
            "y_true",
            "y_pred",
            "prediction_confidence",
        ]
    ]
    metrics = sanitize_public_opt_metrics(
        evaluate_classification_predictions(ensemble, label_order=label_order)
    )
    return "vote_top2_ensemble", ensemble, metrics


def _merge_prediction_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    base_keys = [
        "dataset_id",
        "profile",
        "evaluation_group",
        "subset_id",
        "split_group",
        "sample_id",
        "subject_id",
        "y_true",
    ]
    merged = None
    for index, frame in enumerate(frames):
        selected_columns = base_keys + ["y_pred"]
        if "prediction_confidence" in frame.columns:
            selected_columns.append("prediction_confidence")
        renamed = frame.loc[:, selected_columns].rename(
            columns={
                "y_pred": f"y_pred__{index}",
                "prediction_confidence": f"prediction_confidence__{index}",
            }
        )
        merged = (
            renamed
            if merged is None
            else merged.merge(renamed, on=base_keys, how="inner")
        )
    if merged is None:
        return pd.DataFrame()
    return merged
