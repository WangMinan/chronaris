"""Candidate and fold execution helpers for public UAB torch opt."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import HuberRegressor, Ridge
from torch import nn
from torch.utils.data import DataLoader

from chronaris.evaluation import evaluate_regression_predictions
from chronaris.pipelines.stage_i.common.baseline_models import build_loso_splits
from chronaris.pipelines.stage_i.common.run_observer import StageIRunProgress
from chronaris.pipelines.stage_i.public.opt_postprocess import apply_public_opt_regression_prediction_aggregation
from chronaris.pipelines.stage_i.public.opt_shared import (
    safe_public_opt_regression_fallback,
    sanitize_public_opt_metrics,
    sanitize_public_opt_regression_outputs,
)
from chronaris.pipelines.stage_i.public.opt_torch_catalog import TorchUABCandidateSpec
from chronaris.pipelines.stage_i.public.opt_torch_models import (
    _TabularRegressionDataset,
    _TorchFeatureBundle,
    _TorchSubsetBundle,
    _apply_standardizer,
    _build_torch_uab_model,
    _fit_standardizer,
    _predict_torch_uab,
    _split_train_validation_groups,
    _validation_rmse,
)
from chronaris.pipelines.stage_i.public.opt_torch_supervision import (
    broadcast_torch_uab_session_predictions,
    build_torch_uab_supervision_view,
)
from chronaris.pipelines.torch_runtime import seed_torch

UAB_TORCH_DATASET_ID = "uab_workload_dataset"
LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def _build_feature_profile_bundle(
    feature_result,
    *,
    feature_profile: str,
) -> _TorchFeatureBundle:
    feature_columns = tuple(feature_result.feature_groups[feature_profile])
    feature_frame = feature_result.feature_frame
    feature_matrix = feature_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    feature_matrix = np.nan_to_num(
        feature_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )
    physiology_allowed = set(feature_result.feature_groups["physiology_only"])
    context_allowed = set(feature_result.feature_groups["context_only"])
    physiology_indices = tuple(
        index for index, name in enumerate(feature_columns) if name in physiology_allowed
    )
    context_indices = tuple(
        index for index, name in enumerate(feature_columns) if name in context_allowed
    )
    if not physiology_indices or not context_indices:
        raise ValueError(
            f"feature profile {feature_profile} has empty branch features after filter."
        )
    subset_labels = feature_frame["subset_id"].astype(str).to_numpy()
    subset_bundles = {
        subset_id: _build_subset_feature_bundle(
            subset_id=subset_id,
            feature_frame=feature_frame,
            feature_matrix=feature_matrix,
            subset_labels=subset_labels,
        )
        for subset_id in ("n_back", "heat_the_chair")
    }
    return _TorchFeatureBundle(
        feature_profile=feature_profile,
        feature_columns=feature_columns,
        physiology_indices=physiology_indices,
        context_indices=context_indices,
        feature_matrix=feature_matrix,
        feature_frame=feature_frame,
        subset_bundles=subset_bundles,
    )


def _build_subset_feature_bundle(
    *,
    subset_id: str,
    feature_frame: pd.DataFrame,
    feature_matrix: np.ndarray,
    subset_labels: np.ndarray,
) -> _TorchSubsetBundle:
    subset_indices = np.flatnonzero(subset_labels == subset_id)
    if subset_indices.size == 0:
        raise ValueError(f"torch UAB feature frame is missing subset_id={subset_id}")
    subset_frame = feature_frame.iloc[subset_indices].reset_index(drop=True)
    subset_matrix = feature_matrix[subset_indices].copy()
    split_groups = subset_frame["split_group"].astype(str).to_numpy()
    targets = subset_frame["y_true"].to_numpy(dtype=np.float32, copy=True)
    return _TorchSubsetBundle(
        subset_id=subset_id,
        subset_frame=subset_frame,
        subset_matrix=subset_matrix,
        split_groups=split_groups,
        targets=targets,
        loso_splits=build_loso_splits(split_groups),
    )


def _run_torch_uab_candidate(
    *,
    feature_bundle: _TorchFeatureBundle,
    candidate: TorchUABCandidateSpec,
    selected_subsets: tuple[str, ...],
    seed: int,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    max_folds: int | None,
    prediction_aggregation_policy: str,
    supervision_granularity: str,
    progress: StageIRunProgress | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    group_metrics: dict[str, object] = {}
    for subset_id in selected_subsets:
        LOGGER.info(
            "stage_i_public_opt_torch candidate=%s subset=%s start",
            candidate.candidate_id,
            subset_id,
        )
        if progress is not None:
            progress.update(
                "subset_start",
                dataset_id=UAB_TORCH_DATASET_ID,
                candidate=candidate.candidate_id,
                subset=subset_id,
            )
        subset_bundle = feature_bundle.subset_bundles[subset_id]
        subset_frame = subset_bundle.subset_frame
        subset_matrix = subset_bundle.subset_matrix
        loso_splits = subset_bundle.loso_splits
        if max_folds is not None:
            loso_splits = loso_splits[:max_folds]
        frames: list[pd.DataFrame] = []
        targets = subset_bundle.targets
        for fold_index, split in enumerate(loso_splits):
            held_out_groups = ",".join(
                sorted(
                    subset_frame.iloc[split.test_indices]["split_group"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            )
            LOGGER.info(
                "stage_i_public_opt_torch candidate=%s subset=%s fold=%d/%d train=%d test=%d held_out=%s",
                candidate.candidate_id,
                subset_id,
                fold_index + 1,
                len(loso_splits),
                len(split.train_indices),
                len(split.test_indices),
                held_out_groups,
            )
            if progress is not None:
                progress.update(
                    "fold_start",
                    dataset_id=UAB_TORCH_DATASET_ID,
                    candidate=candidate.candidate_id,
                    subset=subset_id,
                    fold_index=fold_index + 1,
                    fold_count=len(loso_splits),
                    train_count=len(split.train_indices),
                    test_count=len(split.test_indices),
                    held_out_groups=held_out_groups,
                )
            fold_predictions = _run_one_torch_uab_fold(
                subset_frame=subset_frame,
                subset_matrix=subset_matrix,
                candidate=candidate,
                physiology_indices=feature_bundle.physiology_indices,
                context_indices=feature_bundle.context_indices,
                seed=seed + fold_index,
                device=device,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                train_indices=split.train_indices,
                test_indices=split.test_indices,
                targets=targets,
                supervision_granularity=supervision_granularity,
            )
            frames.append(fold_predictions)
        predictions = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
        predictions = apply_public_opt_regression_prediction_aggregation(
            predictions,
            policy=prediction_aggregation_policy,
        )
        metrics = sanitize_public_opt_metrics(evaluate_regression_predictions(predictions))
        group_metrics[subset_id] = metrics
        prediction_frames.append(predictions)
        LOGGER.info(
            "stage_i_public_opt_torch candidate=%s subset=%s done rmse=%.4f mae=%.4f",
            candidate.candidate_id,
            subset_id,
            float(metrics["rmse"]),
            float(metrics["mae"]),
        )
        if progress is not None:
            progress.update(
                "subset_done",
                dataset_id=UAB_TORCH_DATASET_ID,
                candidate=candidate.candidate_id,
                subset=subset_id,
                rmse=float(metrics["rmse"]),
                mae=float(metrics["mae"]),
            )
    predictions = pd.concat(prediction_frames, axis=0, ignore_index=True)
    mean_rmse = float(
        np.mean([float(group_metrics[group]["rmse"]) for group in selected_subsets], dtype=np.float64)
    )
    mean_mae = float(
        np.mean([float(group_metrics[group]["mae"]) for group in selected_subsets], dtype=np.float64)
    )
    return {
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "feature_profile": candidate.feature_profile,
        "hidden_dims": list(candidate.hidden_dims),
        "dropout": candidate.dropout,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "groups": group_metrics,
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
    }, predictions


def _run_one_torch_uab_fold(
    *,
    subset_frame: pd.DataFrame,
    subset_matrix: np.ndarray,
    candidate: TorchUABCandidateSpec,
    physiology_indices: Sequence[int],
    context_indices: Sequence[int],
    seed: int,
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    targets: np.ndarray,
    supervision_granularity: str,
) -> pd.DataFrame:
    seed_torch(seed, device=device)
    train_view = build_torch_uab_supervision_view(
        subset_frame=subset_frame.iloc[train_indices].copy(),
        subset_matrix=subset_matrix[train_indices],
        targets=targets[train_indices],
        supervision_granularity=supervision_granularity,
    )
    test_frame = subset_frame.iloc[test_indices].copy()
    test_view = build_torch_uab_supervision_view(
        subset_frame=test_frame,
        subset_matrix=subset_matrix[test_indices],
        targets=targets[test_indices],
        supervision_granularity=supervision_granularity,
    )
    train_X = train_view.matrix
    train_y = train_view.targets
    test_X = test_view.matrix
    fallback_value = safe_public_opt_regression_fallback(train_y)
    train_scale = _fit_standardizer(train_X)
    train_X = _apply_standardizer(train_X, train_scale)
    test_X = _apply_standardizer(test_X, train_scale)

    train_groups = train_view.split_groups
    if candidate.model_family in {
        "heat_residual_correction",
        "heat_affine_calibrated_blend",
    }:
        predictions = _predict_heat_specialist_fold(
            candidate=candidate,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            train_groups=train_groups,
            seed=seed,
            fallback_value=fallback_value,
        )
        if supervision_granularity == "session_pooled_broadcast":
            predictions = broadcast_torch_uab_session_predictions(
                target_frame=test_frame,
                pooled_frame=test_view.frame,
                pooled_predictions=predictions,
            )
        return _torch_uab_prediction_frame(
            test_frame=test_frame,
            candidate=candidate,
            predictions=predictions,
        )

    inner_train_idx, inner_val_idx = _split_train_validation_groups(train_groups, seed=seed)
    model = _build_torch_uab_model(
        candidate=candidate,
        input_dim=train_X.shape[1],
        physiology_dim=len(physiology_indices),
        context_dim=len(context_indices),
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=candidate.weight_decay,
    )
    criterion = nn.HuberLoss()

    train_dataset = _TabularRegressionDataset(
        train_X[inner_train_idx],
        train_y[inner_train_idx],
        physiology_indices,
        context_indices,
    )
    val_dataset = (
        _TabularRegressionDataset(
            train_X[inner_val_idx],
            train_y[inner_val_idx],
            physiology_indices,
            context_indices,
        )
        if inner_val_idx is not None and len(inner_val_idx) > 0
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(batch_size, 1),
        shuffle=True,
        drop_last=False,
        pin_memory=device == "cuda",
    )
    non_blocking = device == "cuda"

    best_state = None
    best_metric = float("inf")
    stale_epochs = 0
    for epoch in range(epochs):
        del epoch
        model.train()
        for features, physiology_features, context_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                features.to(device=device, non_blocking=non_blocking),
                physiology_features.to(device=device, non_blocking=non_blocking),
                context_features.to(device=device, non_blocking=non_blocking),
            )
            loss = criterion(
                predictions,
                batch_targets.to(device=device, non_blocking=non_blocking).view(-1, 1),
            )
            loss.backward()
            optimizer.step()
        validation_metric = _validation_rmse(
            model=model,
            dataset=val_dataset,
            device=device,
            fallback_value=fallback_value,
        )
        if validation_metric < best_metric - 1e-6:
            best_metric = validation_metric
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if val_dataset is not None and stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(best_state)

    predictions = _predict_torch_uab(
        model=model,
        features=test_X,
        physiology_indices=physiology_indices,
        context_indices=context_indices,
        device=device,
        fallback_value=fallback_value,
    )
    if supervision_granularity == "session_pooled_broadcast":
        predictions = broadcast_torch_uab_session_predictions(
            target_frame=test_frame,
            pooled_frame=test_view.frame,
            pooled_predictions=predictions,
        )
    return _torch_uab_prediction_frame(
        test_frame=test_frame,
        candidate=candidate,
        predictions=predictions,
    )


def _torch_uab_prediction_frame(
    *,
    test_frame: pd.DataFrame,
    candidate: TorchUABCandidateSpec,
    predictions: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track": "subjective",
            "dataset_id": test_frame["dataset_id"].astype(str).to_numpy(),
            "profile": test_frame["profile"].astype(str).to_numpy(),
            "evaluation_group": test_frame["subset_id"].astype(str).to_numpy(),
            "subset_id": test_frame["subset_id"].astype(str).to_numpy(),
            "candidate_id": np.full(len(test_frame), candidate.candidate_id, dtype=object),
            "model_name": np.full(len(test_frame), candidate.model_family, dtype=object),
            "feature_profile": np.full(
                len(test_frame),
                candidate.feature_profile,
                dtype=object,
            ),
            "split_group": test_frame["split_group"].astype(str).to_numpy(),
            "sample_id": test_frame["sample_id"].astype(str).to_numpy(),
            "subject_id": test_frame["subject_id"].astype(str).to_numpy(),
            "session_id": test_frame["session_id"].astype(str).to_numpy(),
            "y_true": test_frame["y_true"].to_numpy(dtype=float, copy=True),
            "y_pred": predictions.astype(float, copy=False),
        }
    )


def _predict_heat_specialist_fold(
    *,
    candidate: TorchUABCandidateSpec,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    train_groups: np.ndarray,
    seed: int,
    fallback_value: float,
) -> np.ndarray:
    if should_use_public_opt_regression_fallback_local(train_y):
        return np.full((len(test_X),), fallback_value, dtype=np.float32)
    if candidate.model_family == "heat_residual_correction":
        baseline = _fit_ridge_safe(train_X, train_y, alpha=1.0)
        baseline_train = np.asarray(baseline.predict(train_X), dtype=np.float32)
        baseline_test = np.asarray(baseline.predict(test_X), dtype=np.float32)
        residual_y = train_y - baseline_train
        correction = _fit_huber_safe(
            train_X,
            residual_y,
            alpha=max(candidate.weight_decay, 1e-6),
        )
        predicted = baseline_test + np.asarray(correction.predict(test_X), dtype=np.float32)
    elif candidate.model_family == "heat_affine_calibrated_blend":
        predicted = _predict_affine_calibrated_blend(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            train_groups=train_groups,
            seed=seed,
            weight_decay=candidate.weight_decay,
        )
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported heat specialist family: {candidate.model_family}")
    predicted, _ = sanitize_public_opt_regression_outputs(
        np.asarray(predicted, dtype=np.float32),
        fallback_value=fallback_value,
    )
    return predicted


def should_use_public_opt_regression_fallback_local(train_y: np.ndarray) -> bool:
    finite_values = np.asarray(train_y, dtype=np.float32)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size <= 1:
        return True
    return bool(np.allclose(finite_values, finite_values[0]))


def _predict_affine_calibrated_blend(
    *,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    train_groups: np.ndarray,
    seed: int,
    weight_decay: float,
) -> np.ndarray:
    inner_train_idx, inner_val_idx = _split_train_validation_groups(train_groups, seed=seed)
    if inner_val_idx is None or len(inner_val_idx) == 0:
        inner_train_idx = np.arange(len(train_y), dtype=int)
        inner_val_idx = inner_train_idx

    baseline = _fit_ridge_safe(train_X[inner_train_idx], train_y[inner_train_idx], alpha=1.0)
    residual_model = _fit_huber_safe(
        train_X[inner_train_idx],
        train_y[inner_train_idx],
        alpha=max(weight_decay, 1e-6),
    )
    val_baseline = np.asarray(baseline.predict(train_X[inner_val_idx]), dtype=np.float32)
    val_raw = np.asarray(residual_model.predict(train_X[inner_val_idx]), dtype=np.float32)
    test_baseline = np.asarray(baseline.predict(test_X), dtype=np.float32)
    test_raw = np.asarray(residual_model.predict(test_X), dtype=np.float32)
    affine_a, affine_b = _fit_affine_calibration(
        predictions=val_raw,
        targets=train_y[inner_val_idx],
    )
    val_calibrated = affine_a * val_raw + affine_b
    test_calibrated = affine_a * test_raw + affine_b
    blend_weight = _select_blend_weight(
        calibrated=val_calibrated,
        baseline=val_baseline,
        targets=train_y[inner_val_idx],
    )
    return blend_weight * test_calibrated + (1.0 - blend_weight) * test_baseline


def _fit_ridge_safe(train_X: np.ndarray, train_y: np.ndarray, *, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(train_X, train_y)
    return model


def _fit_huber_safe(train_X: np.ndarray, train_y: np.ndarray, *, alpha: float) -> object:
    try:
        model = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=500)
        model.fit(train_X, train_y)
        return model
    except Exception:  # pragma: no cover - rare sklearn convergence fallback
        return _fit_ridge_safe(train_X, train_y, alpha=1.0)


def _fit_affine_calibration(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        [np.asarray(predictions, dtype=np.float64), np.ones(len(predictions), dtype=np.float64)]
    )
    try:
        coefficients, *_ = np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return 1.0, 0.0
    return float(coefficients[0]), float(coefficients[1])


def _select_blend_weight(
    *,
    calibrated: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
) -> float:
    best_weight = 1.0
    best_rmse = float("inf")
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        predictions = weight * calibrated + (1.0 - weight) * baseline
        rmse = float(
            np.sqrt(np.mean(np.square(predictions - targets), dtype=np.float64))
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = float(weight)
    return best_weight
