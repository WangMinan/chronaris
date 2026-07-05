"""Sklearn-backed runners for task evaluation public-opt heads."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation import evaluate_classification_predictions, evaluate_regression_predictions
from chronaris.modeling.common.baseline_models import build_loso_splits
from chronaris.evaluation.public_datasets.pipelines.opt_data import PUBLIC_OPT_FEATURE_PROFILES, StageIPublicOptFeatureFrameResult
from chronaris.evaluation.public_datasets.pipelines.opt_postprocess import (
    apply_public_opt_regression_prediction_aggregation,
    validate_public_opt_prediction_aggregation_policy,
)
from chronaris.evaluation.public_datasets.pipelines.opt_sklearn_heads import (
    _build_classification_ensemble,
    _build_regression_ensemble,
    _run_one_classification_head,
    _run_one_regression_head,
)
from chronaris.evaluation.public_datasets.pipelines.opt_shared import (
    sanitize_public_opt_metrics,
)
from chronaris.modeling.common.run_observer import StageIRunProgress

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def run_public_opt_backend(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    head_feature_columns: Mapping[str, Sequence[str]],
    head_catalog: str,
    train_balance_policy: str,
    ensemble_policy: str,
    prediction_aggregation_policy: str,
    selected_evaluation_groups: Sequence[str] | None = None,
    progress: StageIRunProgress | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if feature_result.task_type == "regression":
        return _run_public_opt_regression(
            feature_result=feature_result,
            head_feature_columns=head_feature_columns,
            head_catalog=head_catalog,
            ensemble_policy=ensemble_policy,
            prediction_aggregation_policy=prediction_aggregation_policy,
            selected_evaluation_groups=selected_evaluation_groups,
            progress=progress,
        )
    if feature_result.task_type == "classification":
        return _run_public_opt_classification(
            feature_result=feature_result,
            head_feature_columns=head_feature_columns,
            train_balance_policy=train_balance_policy,
            ensemble_policy=ensemble_policy,
            selected_evaluation_groups=selected_evaluation_groups,
            progress=progress,
        )
    raise ValueError(f"unsupported public opt task type: {feature_result.task_type}")


def validate_public_opt_config(
    *,
    feature_profile: str,
    head_catalog: str,
    train_balance_policy: str,
    ensemble_policy: str,
    prediction_aggregation_policy: str,
    winner_margin_policy: str,
) -> None:
    if feature_profile not in set(PUBLIC_OPT_FEATURE_PROFILES):
        raise ValueError(f"unsupported public opt feature_profile: {feature_profile}")
    if head_catalog not in {"minimal", "expanded", "uab_hybrid"}:
        raise ValueError(f"unsupported public opt head_catalog: {head_catalog}")
    if train_balance_policy not in {"none", "class_weight_balanced"}:
        raise ValueError(
            "unsupported public opt train_balance_policy: "
            f"{train_balance_policy}"
        )
    if ensemble_policy not in {"none", "mean_top2", "vote_top2"}:
        raise ValueError(f"unsupported public opt ensemble_policy: {ensemble_policy}")
    validate_public_opt_prediction_aggregation_policy(prediction_aggregation_policy)
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
    elif head_catalog == "uab_hybrid":
        if feature_result.dataset_id != "uab_workload_dataset":
            raise ValueError("head_catalog=uab_hybrid only supports UAB public opt.")
        catalog = {
            "subjective": (
                "target_prior_median",
                "target_prior_trimmed_mean",
                "physiology_persistence",
                "ridge_residual_cv",
                "elasticnet_residual",
                "huber_residual",
                "ridge_heat_physiology_lowdim",
                "huber_heat_physiology_lowdim",
                "heat_prior_residual_guarded",
            ),
            "objective": (),
        }
    else:
        subjective_heads = tuple(feature_result.head_feature_columns)
        if feature_result.dataset_id == "uab_workload_dataset":
            subjective_heads = (
                "physiology_persistence",
                "ridge_residual_cv",
                "elasticnet_residual",
                "huber_residual",
            )
        catalog = {
            "subjective": subjective_heads,
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
    head_catalog: str,
    ensemble_policy: str,
    prediction_aggregation_policy: str,
    selected_evaluation_groups: Sequence[str] | None,
    progress: StageIRunProgress | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    subset_results: dict[str, object] = {}
    subset_order = _resolve_selected_evaluation_groups(
        feature_result=feature_result,
        selected_evaluation_groups=selected_evaluation_groups,
    )

    for evaluation_group in subset_order:
        subset_frame = _select_evaluation_group_frame(
            feature_result.feature_frame,
            evaluation_group=evaluation_group,
            evaluation_groups=feature_result.evaluation_groups,
        )
        if subset_frame.empty:
            continue
        LOGGER.info(
            "task_eval_public_opt_sklearn regression subset start dataset_id=%s subset=%s rows=%d",
            feature_result.dataset_id,
            evaluation_group,
            len(subset_frame),
        )
        if progress is not None:
            progress.update(
                "subset_start",
                dataset_id=feature_result.dataset_id,
                subset=evaluation_group,
                sample_count=len(subset_frame),
            )
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        head_metrics: dict[str, dict[str, object]] = {}
        head_prediction_frames: dict[str, pd.DataFrame] = {}
        active_head_feature_columns = _select_regression_head_feature_columns(
            evaluation_group=evaluation_group,
            head_catalog=head_catalog,
            head_feature_columns=head_feature_columns,
        )
        for head_name, feature_columns in active_head_feature_columns.items():
            LOGGER.info(
                "task_eval_public_opt_sklearn regression subset=%s head=%s start feature_count=%d fold_count=%d",
                evaluation_group,
                head_name,
                len(feature_columns),
                len(loso_splits),
            )
            if progress is not None:
                progress.update(
                    "head_start",
                    dataset_id=feature_result.dataset_id,
                    subset=evaluation_group,
                    candidate=head_name,
                    head=head_name,
                    feature_count=len(feature_columns),
                    fold_count=len(loso_splits),
                )
            predictions = _run_one_regression_head(
                subset_frame=subset_frame,
                evaluation_group=evaluation_group,
                head_name=head_name,
                feature_columns=tuple(feature_columns),
                loso_splits=loso_splits,
                progress=progress,
            )
            predictions = apply_public_opt_regression_prediction_aggregation(
                predictions,
                policy=prediction_aggregation_policy,
            )
            metrics = sanitize_public_opt_metrics(
                evaluate_regression_predictions(predictions)
            )
            head_metrics[head_name] = metrics
            head_prediction_frames[head_name] = predictions
            prediction_frames.append(predictions)
            LOGGER.info(
                "task_eval_public_opt_sklearn regression subset=%s head=%s done rmse=%.4f mae=%.4f",
                evaluation_group,
                head_name,
                float(metrics["rmse"]),
                float(metrics["mae"]),
            )
            if progress is not None:
                progress.update(
                    "head_done",
                    dataset_id=feature_result.dataset_id,
                    subset=evaluation_group,
                    candidate=head_name,
                    head=head_name,
                    rmse=float(metrics["rmse"]),
                    mae=float(metrics["mae"]),
                )
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
        if progress is not None:
            progress.update(
                "subset_done",
                dataset_id=feature_result.dataset_id,
                subset=evaluation_group,
                best_candidate=best_head,
                best_head=best_head,
            )

    predictions = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return predictions, subset_results


def _select_regression_head_feature_columns(
    *,
    evaluation_group: str,
    head_catalog: str,
    head_feature_columns: Mapping[str, Sequence[str]],
) -> dict[str, Sequence[str]]:
    if head_catalog != "uab_hybrid":
        return dict(head_feature_columns)
    if evaluation_group == "n_back":
        selected_names = (
            "physiology_persistence",
            "ridge_residual_cv",
            "elasticnet_residual",
            "huber_residual",
        )
    elif evaluation_group == "heat_the_chair":
        selected_names = (
            "target_prior_median",
            "target_prior_trimmed_mean",
            "heat_prior_residual_guarded",
            "physiology_persistence",
            "ridge_heat_physiology_lowdim",
            "huber_heat_physiology_lowdim",
        )
    else:
        selected_names = tuple(head_feature_columns)
    return {
        head_name: head_feature_columns[head_name]
        for head_name in selected_names
        if head_name in head_feature_columns
    }


def _run_public_opt_classification(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    head_feature_columns: Mapping[str, Sequence[str]],
    train_balance_policy: str,
    ensemble_policy: str,
    selected_evaluation_groups: Sequence[str] | None,
    progress: StageIRunProgress | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    prediction_frames: list[pd.DataFrame] = []
    subset_results: dict[str, object] = {}
    if feature_result.label_order is None:
        raise ValueError("classification public opt requires explicit label_order.")
    subset_order = _resolve_selected_evaluation_groups(
        feature_result=feature_result,
        selected_evaluation_groups=selected_evaluation_groups,
    )

    for evaluation_group in subset_order:
        subset_frame = _select_evaluation_group_frame(
            feature_result.feature_frame,
            evaluation_group=evaluation_group,
            evaluation_groups=feature_result.evaluation_groups,
        )
        if subset_frame.empty:
            continue
        LOGGER.info(
            "task_eval_public_opt_sklearn classification subset start dataset_id=%s subset=%s rows=%d",
            feature_result.dataset_id,
            evaluation_group,
            len(subset_frame),
        )
        if progress is not None:
            progress.update(
                "subset_start",
                dataset_id=feature_result.dataset_id,
                subset=evaluation_group,
                sample_count=len(subset_frame),
            )
        split_groups = subset_frame["split_group"].astype(str).to_numpy()
        loso_splits = build_loso_splits(split_groups)
        head_metrics: dict[str, dict[str, object]] = {}
        head_prediction_frames: dict[str, pd.DataFrame] = {}
        for head_name, feature_columns in head_feature_columns.items():
            LOGGER.info(
                "task_eval_public_opt_sklearn classification subset=%s head=%s start feature_count=%d fold_count=%d",
                evaluation_group,
                head_name,
                len(feature_columns),
                len(loso_splits),
            )
            if progress is not None:
                progress.update(
                    "head_start",
                    dataset_id=feature_result.dataset_id,
                    subset=evaluation_group,
                    candidate=head_name,
                    head=head_name,
                    feature_count=len(feature_columns),
                    fold_count=len(loso_splits),
                )
            predictions = _run_one_classification_head(
                subset_frame=subset_frame,
                evaluation_group=evaluation_group,
                head_name=head_name,
                feature_columns=tuple(feature_columns),
                loso_splits=loso_splits,
                label_order=feature_result.label_order,
                train_balance_policy=train_balance_policy,
                progress=progress,
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
            LOGGER.info(
                "task_eval_public_opt_sklearn classification subset=%s head=%s done macro_f1=%.4f balanced_accuracy=%.4f",
                evaluation_group,
                head_name,
                float(metrics["macro_f1"]),
                float(metrics["balanced_accuracy"]),
            )
            if progress is not None:
                progress.update(
                    "head_done",
                    dataset_id=feature_result.dataset_id,
                    subset=evaluation_group,
                    candidate=head_name,
                    head=head_name,
                    macro_f1=float(metrics["macro_f1"]),
                    balanced_accuracy=float(metrics["balanced_accuracy"]),
                )
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
        if progress is not None:
            progress.update(
                "subset_done",
                dataset_id=feature_result.dataset_id,
                subset=evaluation_group,
                best_candidate=best_head,
                best_head=best_head,
            )

    predictions = (
        pd.concat(prediction_frames, axis=0, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return predictions, subset_results


def _resolve_selected_evaluation_groups(
    *,
    feature_result: StageIPublicOptFeatureFrameResult,
    selected_evaluation_groups: Sequence[str] | None,
) -> tuple[str, ...]:
    if not selected_evaluation_groups:
        return tuple(feature_result.subset_order)
    requested = tuple(str(value) for value in selected_evaluation_groups)
    supported = set(feature_result.subset_order)
    unsupported = tuple(value for value in requested if value not in supported)
    if unsupported:
        raise ValueError(
            "unsupported public opt selected evaluation groups: "
            f"{unsupported}; supported={tuple(feature_result.subset_order)}"
        )
    return requested


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
