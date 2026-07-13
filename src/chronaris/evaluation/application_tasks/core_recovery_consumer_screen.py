"""Fixed MiniRocket upper-bound screen for frozen core-recovery sequences."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from chronaris.evaluation.application_tasks.consumer_model_selection import (
    classifier_classes,
    fit_classifier,
)
from chronaris.evaluation.application_tasks.core_recovery_tasks import (
    TaskAwareTargetBundle,
    select_task_aware_targets,
)
from chronaris.evaluation.application_tasks.core_recovery_training import (
    CoreRecoveryFrozenEncoding,
    CoreRecoveryMethodModel,
)


def evaluate_frozen_minirocket_sequences(
    *,
    encoding: CoreRecoveryFrozenEncoding,
    targets: TaskAwareTargetBundle,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    n_kernels: int = 2_500,
    random_state: int = 17,
    classification_c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    regression_alpha_values: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[Mapping[str, object], ...]:
    """Fit only on inner-train and evaluate only on inner-validation."""

    if n_kernels <= 0:
        raise ValueError("MiniRocket kernel budget must be positive")
    train = encoding.select(train_sample_ids)
    validation = encoding.select(validation_sample_ids)
    train_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in train_sample_ids),
    )
    validation_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in validation_sample_ids),
    )
    sequence_modes = _sequence_modes(encoding)
    rows = []
    for mode in sequence_modes:
        train_sequence = _sequence_for(train, mode=mode)
        validation_sequence = _sequence_for(validation, mode=mode)
        rows.append(
            _evaluate_sequence_pair(
                mode=mode,
                train_maneuver_sequence=train_sequence,
                validation_maneuver_sequence=validation_sequence,
                train_response_sequence=train_sequence,
                validation_response_sequence=validation_sequence,
                train_targets=train_targets,
                validation_targets=validation_targets,
                n_kernels=n_kernels,
                random_state=random_state,
                classification_c_values=classification_c_values,
                regression_alpha_values=regression_alpha_values,
            )
        )
    return tuple(rows)


def evaluate_task_routed_consumer_grid(
    *,
    encoding: CoreRecoveryFrozenEncoding,
    targets: TaskAwareTargetBundle,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    kernel_grid: Sequence[int] = (1_000, 2_500, 5_000, 10_000),
    random_state: int = 17,
) -> tuple[Mapping[str, object], ...]:
    """Screen global task heads for direct-maneuver/continuous-response routing."""

    train = encoding.select(train_sample_ids)
    validation = encoding.select(validation_sample_ids)
    train_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in train_sample_ids),
    )
    validation_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in validation_sample_ids),
    )
    train_response_mask = train_targets.response_available.numpy()
    validation_response_mask = validation_targets.response_available.numpy()
    rows = []
    if encoding.observed_sequence is not None:
        train_maneuver_sequence = (
            train.observed_sequence
            if train.maneuver_observed_sequence is None
            else train.maneuver_observed_sequence
        )
        validation_maneuver_sequence = (
            validation.observed_sequence
            if validation.maneuver_observed_sequence is None
            else validation.maneuver_observed_sequence
        )
        train_response_sequence = train.continuous_sequence
        validation_response_sequence = validation.continuous_sequence
        maneuver_mode = "direct_only"
        response_mode = "continuous_only"
    else:
        train_maneuver_sequence = train.sequence_embedding
        validation_maneuver_sequence = validation.sequence_embedding
        train_response_sequence = train.sequence_embedding
        validation_response_sequence = validation.sequence_embedding
        maneuver_mode = "frozen_sequence"
        response_mode = "frozen_sequence"
    for n_kernels in kernel_grid:
        direct_train, direct_validation, direct_channels = _transform_sequences(
            train_maneuver_sequence,
            validation_maneuver_sequence,
            n_kernels=n_kernels,
            random_state=random_state,
        )
        continuous_train, continuous_validation, continuous_channels = (
            _transform_sequences(
                train_response_sequence,
                validation_response_sequence,
                n_kernels=n_kernels,
                random_state=random_state,
            )
        )
        for c_value in (0.1, 1.0, 10.0):
            maneuver, _ = fit_classifier(
                direct_train,
                train_targets.maneuver_class.numpy(),
                direct_validation,
                validation_targets.maneuver_class.numpy(),
                c_values=(c_value,),
                random_state=random_state,
                scaler_with_mean=False,
                classification_labels=(0, 1, 2),
                solver="liblinear_ovr",
            )
            rows.append(
                _grid_row(
                    task="maneuver",
                    sequence_mode=maneuver_mode,
                    n_kernels=n_kernels,
                    head_config_id=f"logistic_c={c_value:g}",
                    metric="macro_f1",
                    value=f1_score(
                        validation_targets.maneuver_class.numpy(),
                        maneuver.predict(direct_validation),
                        labels=(0, 1, 2),
                        average="macro",
                        zero_division=0,
                    ),
                    direction="higher",
                    channel_count=direct_channels,
                )
            )
            high, _ = fit_classifier(
                continuous_train[train_response_mask],
                train_targets.high_response.numpy()[train_response_mask],
                continuous_validation[validation_response_mask],
                validation_targets.high_response.numpy()[validation_response_mask],
                c_values=(c_value,),
                random_state=random_state,
                scaler_with_mean=False,
                classification_labels=(0, 1),
                solver="liblinear_ovr",
            )
            classes = classifier_classes(high)
            positive = np.flatnonzero(classes == 1)
            high_value = (
                average_precision_score(
                    validation_targets.high_response.numpy()[validation_response_mask],
                    high.predict_proba(continuous_validation[validation_response_mask])[
                        :, int(positive[0])
                    ],
                )
                if len(positive) == 1
                and len(
                    np.unique(
                        validation_targets.high_response.numpy()[validation_response_mask]
                    )
                )
                == 2
                else None
            )
            rows.append(
                _grid_row(
                    task="high_response",
                    sequence_mode=response_mode,
                    n_kernels=n_kernels,
                    head_config_id=f"logistic_c={c_value:g}",
                    metric="auprc",
                    value=high_value,
                    direction="higher",
                    channel_count=continuous_channels,
                )
            )
        response_inputs = [(response_mode, continuous_train, continuous_validation)]
        if encoding.observed_sequence is not None:
            response_inputs.extend(
                (
                    ("direct_observed", direct_train, direct_validation),
                    (
                        "direct_continuous_concat",
                        np.concatenate((direct_train, continuous_train), axis=1),
                        np.concatenate((direct_validation, continuous_validation), axis=1),
                    ),
                )
            )
        for sequence_mode, response_train, response_validation in response_inputs:
            rows.extend(
                _response_grid_rows(
                    train_values=response_train[train_response_mask],
                    train_target=train_targets.response_value.numpy()[train_response_mask],
                    validation_values=response_validation[validation_response_mask],
                    validation_target=validation_targets.response_value.numpy()[
                        validation_response_mask
                    ],
                    n_kernels=n_kernels,
                    channel_count=response_train.shape[1],
                    sequence_mode=sequence_mode,
                    compact=(sequence_mode != response_mode),
                )
            )
        if encoding.observed_sequence is not None:
            rows.extend(
                _response_prediction_ensemble_rows(
                    direct_train=direct_train[train_response_mask],
                    continuous_train=continuous_train[train_response_mask],
                    concat_train=np.concatenate(
                        (direct_train, continuous_train),
                        axis=1,
                    )[train_response_mask],
                    direct_validation=direct_validation[validation_response_mask],
                    continuous_validation=continuous_validation[validation_response_mask],
                    concat_validation=np.concatenate(
                        (direct_validation, continuous_validation),
                        axis=1,
                    )[validation_response_mask],
                    train_target=train_targets.response_value.numpy()[
                        train_response_mask
                    ],
                    validation_target=validation_targets.response_value.numpy()[
                        validation_response_mask
                    ],
                    n_kernels=n_kernels,
                )
            )
    return tuple(rows)


def evaluate_fitted_residual_minirocket(
    *,
    model: CoreRecoveryMethodModel,
    encoding: CoreRecoveryFrozenEncoding,
    targets: TaskAwareTargetBundle,
    train_sample_ids: Sequence[str],
    validation_sample_ids: Sequence[str],
    n_kernels: int = 2_500,
    random_state: int = 17,
    classification_c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    regression_alpha_values: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> Mapping[str, object]:
    """Evaluate the trained task gates without re-encoding the frozen backbone."""

    if model.chronaris_residual is None:
        raise ValueError("fitted residual consumer is only defined for Chronaris")
    train = encoding.select(train_sample_ids)
    validation = encoding.select(validation_sample_ids)
    train_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in train_sample_ids),
    )
    validation_targets = select_task_aware_targets(
        targets,
        tuple(str(value) for value in validation_sample_ids),
    )
    model.eval()
    with torch.inference_mode():
        train_representation = model.chronaris_residual.fuse_paths(
            observed_sequence=train.observed_sequence,
            maneuver_observed_sequence=train.maneuver_observed_sequence,
            continuous_sequence=train.continuous_sequence,
            valid_mask=train.valid_mask,
        )
        validation_representation = model.chronaris_residual.fuse_paths(
            observed_sequence=validation.observed_sequence,
            maneuver_observed_sequence=validation.maneuver_observed_sequence,
            continuous_sequence=validation.continuous_sequence,
            valid_mask=validation.valid_mask,
        )
    return _evaluate_sequence_pair(
        mode="full_task_conditioned",
        train_maneuver_sequence=train_representation.sequence_for("maneuver"),
        validation_maneuver_sequence=validation_representation.sequence_for("maneuver"),
        train_response_sequence=train_representation.sequence_for("physiology_response"),
        validation_response_sequence=validation_representation.sequence_for(
            "physiology_response"
        ),
        train_targets=train_targets,
        validation_targets=validation_targets,
        n_kernels=n_kernels,
        random_state=random_state,
        classification_c_values=classification_c_values,
        regression_alpha_values=regression_alpha_values,
    )


def _evaluate_sequence_pair(
    *,
    mode: str,
    train_maneuver_sequence: torch.Tensor,
    validation_maneuver_sequence: torch.Tensor,
    train_response_sequence: torch.Tensor,
    validation_response_sequence: torch.Tensor,
    train_targets: TaskAwareTargetBundle,
    validation_targets: TaskAwareTargetBundle,
    n_kernels: int,
    random_state: int,
    classification_c_values: tuple[float, ...],
    regression_alpha_values: tuple[float, ...],
) -> Mapping[str, object]:
    transformed_train, transformed_validation, maneuver_channel_count = (
        _transform_sequences(
            train_maneuver_sequence,
            validation_maneuver_sequence,
            n_kernels=n_kernels,
            random_state=random_state,
        )
    )
    response_train, response_validation, response_channel_count = _transform_sequences(
        train_response_sequence,
        validation_response_sequence,
        n_kernels=n_kernels,
        random_state=random_state,
    )
    maneuver_classifier, selected_c = fit_classifier(
        transformed_train,
        train_targets.maneuver_class.numpy(),
        transformed_validation,
        validation_targets.maneuver_class.numpy(),
        c_values=classification_c_values,
        random_state=random_state,
        scaler_with_mean=False,
        classification_labels=(0, 1, 2),
        solver="liblinear_ovr",
    )
    train_response = train_targets.response_available.numpy()
    validation_response = validation_targets.response_available.numpy()
    response_regressor, selected_alpha, response_target_transform = (
        _fit_response_regressor(
        response_train[train_response],
        train_targets.response_value.numpy()[train_response],
        response_validation[validation_response],
        validation_targets.response_value.numpy()[validation_response],
            alpha_values=regression_alpha_values,
        )
    )
    high_classifier, selected_high_c = fit_classifier(
        response_train[train_response],
        train_targets.high_response.numpy()[train_response],
        response_validation[validation_response],
        validation_targets.high_response.numpy()[validation_response],
        c_values=classification_c_values,
        random_state=random_state,
        scaler_with_mean=False,
        classification_labels=(0, 1),
        solver="liblinear_ovr",
    )
    maneuver_prediction = maneuver_classifier.predict(transformed_validation)
    response_prediction = response_regressor.predict(response_validation[validation_response])
    if response_target_transform == "log1p":
        response_prediction = np.expm1(response_prediction).clip(min=0, max=10)
    high_probabilities = high_classifier.predict_proba(
        response_validation[validation_response]
    )
    high_classes = classifier_classes(high_classifier)
    positive_positions = np.flatnonzero(high_classes == 1)
    high_auprc = (
        float(
            average_precision_score(
                validation_targets.high_response.numpy()[validation_response],
                high_probabilities[:, int(positive_positions[0])],
            )
        )
        if len(positive_positions) == 1
        and len(np.unique(validation_targets.high_response.numpy()[validation_response])) == 2
        else None
    )
    return {
        "sequence_mode": mode,
        "consumer": "minirocket",
        "n_kernels": n_kernels,
        "retained_maneuver_channel_count": maneuver_channel_count,
        "retained_response_channel_count": response_channel_count,
        "selected_maneuver_c": selected_c,
        "selected_response_alpha": selected_alpha,
        "selected_response_target_transform": response_target_transform,
        "selected_high_response_c": selected_high_c,
        "maneuver_macro_f1": float(
            f1_score(
                validation_targets.maneuver_class.numpy(),
                maneuver_prediction,
                labels=(0, 1, 2),
                average="macro",
                zero_division=0,
            )
        ),
        "response_rmse": float(
            mean_squared_error(
                validation_targets.response_value.numpy()[validation_response],
                response_prediction,
            )
            ** 0.5
        ),
        "high_response_auprc": high_auprc,
        "fit_role": "inner_train",
        "evaluation_role": "inner_validation",
        "outer_test_accessed": False,
    }


def _fit_response_regressor(
    train_values,
    train_target,
    validation_values,
    validation_target,
    *,
    alpha_values: tuple[float, ...],
):
    best = None
    best_alpha = None
    best_transform = None
    best_score = float("inf")
    train_target = np.asarray(train_target, dtype=np.float64)
    validation_target = np.asarray(validation_target, dtype=np.float64)
    for transform in ("raw", "log1p"):
        fit_target = np.log1p(train_target) if transform == "log1p" else train_target
        for alpha in alpha_values:
            candidate = make_pipeline(
                StandardScaler(with_mean=False),
                Ridge(alpha=float(alpha)),
            ).fit(train_values, fit_target)
            prediction = candidate.predict(validation_values)
            if transform == "log1p":
                prediction = np.expm1(prediction).clip(min=0, max=10)
            score = float(mean_squared_error(validation_target, prediction) ** 0.5)
            if score < best_score:
                best = candidate
                best_alpha = float(alpha)
                best_transform = transform
                best_score = score
    return best, best_alpha, best_transform


def _response_grid_rows(
    *,
    train_values,
    train_target,
    validation_values,
    validation_target,
    n_kernels: int,
    channel_count: int,
    sequence_mode: str,
    compact: bool = False,
):
    rows = []
    train_target = np.asarray(train_target, dtype=np.float64)
    validation_target = np.asarray(validation_target, dtype=np.float64)
    specifications = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        specifications.append(
            (
                f"ridge_alpha={alpha:g}",
                make_pipeline(
                    StandardScaler(with_mean=False),
                    Ridge(alpha=alpha),
                ),
            )
        )
    maximum_components = min(len(train_target) - 1, train_values.shape[1])
    for components in ((2,) if compact else range(1, 11)):
        if components <= maximum_components:
            specifications.append(
                (
                    f"pls_components={components}",
                    PLSRegression(n_components=components, scale=True, max_iter=1_000),
                )
            )
    if not compact:
        for alpha in (0.1, 1.0, 10.0):
            specifications.append(
                (
                    f"kernel_ridge_alpha={alpha:g}",
                    make_pipeline(
                        StandardScaler(with_mean=False),
                        KernelRidge(
                            alpha=alpha,
                            kernel="rbf",
                            gamma=1 / train_values.shape[1],
                        ),
                    ),
                )
            )
        for c_value in (0.1, 1.0, 10.0):
            for epsilon in (0.01, 0.1):
                specifications.append(
                    (
                        f"svr_rbf_c={c_value:g}_epsilon={epsilon:g}",
                        make_pipeline(
                            StandardScaler(with_mean=False),
                            SVR(
                                C=c_value,
                                epsilon=epsilon,
                                kernel="rbf",
                                gamma="scale",
                            ),
                        ),
                    )
                )
    for minimum_leaf in (1, 2, 4):
        specifications.append(
            (
                f"extra_trees_leaf={minimum_leaf}",
                ExtraTreesRegressor(
                    n_estimators=256,
                    min_samples_leaf=minimum_leaf,
                    max_features="sqrt",
                    random_state=17,
                    n_jobs=1,
                ),
            )
        )
    for transform in ("raw", "log1p"):
        fit_target = np.log1p(train_target) if transform == "log1p" else train_target
        for name, estimator in specifications:
            estimator.fit(train_values, fit_target)
            prediction = np.asarray(estimator.predict(validation_values)).reshape(-1)
            if transform == "log1p":
                prediction = np.expm1(prediction).clip(min=0, max=10)
            rows.append(
                _grid_row(
                    task="response",
                    sequence_mode=sequence_mode,
                    n_kernels=n_kernels,
                    head_config_id=f"{name};target={transform}",
                    metric="rmse",
                    value=mean_squared_error(validation_target, prediction) ** 0.5,
                    direction="lower",
                    channel_count=channel_count,
                )
            )
    return rows


def _grid_row(
    *,
    task: str,
    sequence_mode: str,
    n_kernels: int,
    head_config_id: str,
    metric: str,
    value,
    direction: str,
    channel_count: int,
):
    return {
        "task": task,
        "sequence_mode": sequence_mode,
        "n_kernels": n_kernels,
        "head_config_id": head_config_id,
        "metric": metric,
        "value": None if value is None else float(value),
        "direction": direction,
        "retained_channel_count": channel_count,
        "fit_role": "inner_train",
        "evaluation_role": "inner_validation",
        "outer_test_accessed": False,
    }


def _response_prediction_ensemble_rows(
    *,
    direct_train,
    continuous_train,
    concat_train,
    direct_validation,
    continuous_validation,
    concat_validation,
    train_target,
    validation_target,
    n_kernels: int,
):
    rows = []
    for minimum_leaf in (1, 2):
        predictions = {}
        for mode, train_values, validation_values in (
            ("direct", direct_train, direct_validation),
            ("continuous", continuous_train, continuous_validation),
            ("concat", concat_train, concat_validation),
        ):
            estimator = ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=minimum_leaf,
                max_features="sqrt",
                random_state=17,
                n_jobs=1,
            ).fit(train_values, train_target)
            predictions[mode] = estimator.predict(validation_values)
        for left, right in (
            ("direct", "concat"),
            ("direct", "continuous"),
            ("concat", "continuous"),
        ):
            for left_weight in (0.25, 0.5, 0.75):
                prediction = (
                    left_weight * predictions[left]
                    + (1 - left_weight) * predictions[right]
                )
                rows.append(
                    _grid_row(
                        task="response",
                        sequence_mode=f"prediction_ensemble_{left}_{right}",
                        n_kernels=n_kernels,
                        head_config_id=(
                            f"extra_trees_leaf={minimum_leaf};target=raw;"
                            f"left_weight={left_weight:g}"
                        ),
                        metric="rmse",
                        value=mean_squared_error(validation_target, prediction) ** 0.5,
                        direction="lower",
                        channel_count=direct_train.shape[1]
                        + continuous_train.shape[1],
                    )
                )
    return rows


def _sequence_modes(encoding: CoreRecoveryFrozenEncoding) -> tuple[str, ...]:
    return (
        ("direct_only", "continuous_only")
        if encoding.observed_sequence is not None
        else ("frozen_sequence",)
    )


def _sequence_for(encoding: CoreRecoveryFrozenEncoding, *, mode: str) -> torch.Tensor:
    if mode == "direct_only":
        return encoding.observed_sequence
    if mode == "continuous_only":
        return encoding.continuous_sequence
    if mode == "frozen_sequence":
        return encoding.sequence_embedding
    raise ValueError(f"unknown frozen sequence mode: {mode}")


def _transform_sequences(
    train_sequence,
    validation_sequence,
    *,
    n_kernels: int,
    random_state: int,
):
    train = np.asarray(train_sequence.detach().cpu(), dtype=np.float32).transpose(0, 2, 1)
    validation = np.asarray(
        validation_sequence.detach().cpu(),
        dtype=np.float32,
    ).transpose(0, 2, 1)
    channel_std = train.std(axis=(0, 2))
    channel_indices = np.flatnonzero(channel_std > 1e-7)
    if len(channel_indices) == 0:
        raise ValueError("frozen sequence variance filter removed every channel")
    transformer = MiniRocket(
        n_kernels=n_kernels,
        n_jobs=1,
        random_state=random_state,
    )
    transformed_train = transformer.fit_transform(
        np.ascontiguousarray(train[:, channel_indices])
    )
    transformed_validation = transformer.transform(
        np.ascontiguousarray(validation[:, channel_indices])
    )
    return transformed_train, transformed_validation, len(channel_indices)
