"""Fixed linear and MiniRocket consumers for Dingxin window tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from chronaris.evaluation.application_tasks.consumer_model_selection import (
    fit_classifier,
    fit_regressor,
)
from chronaris.evaluation.application_tasks.dingxin_consumer_targets import (
    DingxinFoldConsumerTargets,
    MANEUVER_TASK,
    RESPONSE_TASK,
)


@dataclass(frozen=True, slots=True)
class DingxinConsumerConfig:
    n_kernels: int = 10_000
    random_state: int = 17
    n_jobs: int = 1
    classification_c: float = 1.0
    regression_alpha: float = 1.0
    minimum_global_channel_std: float = 1e-7
    classification_c_grid: tuple[float, ...] = (0.1, 1.0, 10.0)
    regression_alpha_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    tune_on_validation: bool = False


@dataclass(slots=True)
class DingxinTaskConsumerBundle:
    consumer_name: str
    config: DingxinConsumerConfig
    maneuver_classifier: object
    response_regressor: object
    high_response_classifier: object
    transformer: object | None
    channel_indices: np.ndarray | None
    fit_maneuver_sample_ids: tuple[str, ...]
    fit_response_sample_ids: tuple[str, ...]
    selected_maneuver_c: float
    selected_response_alpha: float
    selected_high_response_c: float

    def predict(self, values):
        transformed = self._transform(values)
        maneuver_step = self.maneuver_classifier.named_steps["logisticregression"]
        high_step = self.high_response_classifier.named_steps["logisticregression"]
        return {
            "maneuver_prediction": self.maneuver_classifier.predict(transformed),
            "maneuver_probability": self.maneuver_classifier.predict_proba(transformed),
            "maneuver_classes": maneuver_step.classes_,
            "response_prediction": self.response_regressor.predict(transformed),
            "high_response_prediction": self.high_response_classifier.predict(
                transformed
            ),
            "high_response_probability": self.high_response_classifier.predict_proba(
                transformed
            ),
            "high_response_classes": high_step.classes_,
            "transformed_feature_count": int(transformed.shape[1]),
        }

    def to_manifest(self):
        return {
            "consumer_name": self.consumer_name,
            "config": asdict(self.config),
            "fit_maneuver_sample_ids": list(self.fit_maneuver_sample_ids),
            "fit_response_sample_ids": list(self.fit_response_sample_ids),
            "input_channel_count_after_train_filter": (
                None if self.channel_indices is None else len(self.channel_indices)
            ),
            "variance_filter_fit_role": (
                None if self.channel_indices is None else "train"
            ),
            "selected_maneuver_c": self.selected_maneuver_c,
            "selected_response_alpha": self.selected_response_alpha,
            "selected_high_response_c": self.selected_high_response_c,
            "hyperparameter_selection_role": (
                "validation" if self.config.tune_on_validation else "fixed"
            ),
        }

    def _transform(self, values):
        array = np.asarray(values, dtype=np.float32)
        if self.consumer_name == "linear":
            if array.ndim != 2:
                raise ValueError("linear Dingxin consumer expects [N,D]")
            return array
        if array.ndim != 3 or self.transformer is None or self.channel_indices is None:
            raise ValueError("MiniRocket Dingxin consumer expects [N,T,D]")
        collection = np.ascontiguousarray(array.transpose(0, 2, 1))
        if not np.isfinite(collection).all():
            raise ValueError("MiniRocket Dingxin evaluation input is non-finite")
        return self.transformer.transform(collection[:, self.channel_indices])


def fit_dingxin_task_consumer(
    *,
    consumer_name: str,
    pooled_embedding,
    sequence_embedding,
    sample_ids: Sequence[str],
    targets: DingxinFoldConsumerTargets,
    config: DingxinConsumerConfig | None = None,
    validation_pooled_embedding=None,
    validation_sequence_embedding=None,
    validation_sample_ids: Sequence[str] = (),
) -> DingxinTaskConsumerBundle:
    resolved = config or DingxinConsumerConfig()
    ordered_ids = tuple(str(value) for value in sample_ids)
    maneuver_ids = targets.sample_ids(role="train", task=MANEUVER_TASK)
    response_ids = targets.sample_ids(role="train", task=RESPONSE_TASK)
    if tuple(ordered_ids) != maneuver_ids:
        raise ValueError("Dingxin maneuver train targets must cover train representation")
    positions = {sample_id: index for index, sample_id in enumerate(ordered_ids)}
    response_positions = np.asarray(
        [positions[sample_id] for sample_id in response_ids], dtype=np.int64
    )
    validation_ids = tuple(str(value) for value in validation_sample_ids)
    validation_maneuver_ids = targets.sample_ids(
        role="validation", task=MANEUVER_TASK
    )
    validation_response_ids = targets.sample_ids(
        role="validation", task=RESPONSE_TASK
    )
    if resolved.tune_on_validation and validation_ids != validation_maneuver_ids:
        raise ValueError("Dingxin validation targets must cover validation representation")
    validation_positions = {
        sample_id: index for index, sample_id in enumerate(validation_ids)
    }
    validation_response_positions = (
        np.asarray(
            [validation_positions[sample_id] for sample_id in validation_response_ids],
            dtype=np.int64,
        )
        if resolved.tune_on_validation
        else np.asarray([], dtype=np.int64)
    )
    transformer = None
    channel_indices = None
    if consumer_name == "linear":
        transformed = np.asarray(pooled_embedding, dtype=np.float32)
        validation_transformed = (
            np.asarray(validation_pooled_embedding, dtype=np.float32)
            if resolved.tune_on_validation
            else None
        )
    elif consumer_name == "minirocket":
        from aeon.transformations.collection.convolution_based import MiniRocket

        values = np.asarray(sequence_embedding, dtype=np.float32)
        collection = np.ascontiguousarray(values.transpose(0, 2, 1))
        global_std = collection.std(axis=(0, 2))
        channel_indices = np.flatnonzero(
            global_std > resolved.minimum_global_channel_std
        )
        if len(channel_indices) == 0:
            raise ValueError("MiniRocket train-only variance filter removed all channels")
        transformer = MiniRocket(
            n_kernels=resolved.n_kernels,
            n_jobs=resolved.n_jobs,
            random_state=resolved.random_state,
        )
        transformed = transformer.fit_transform(collection[:, channel_indices])
        validation_transformed = (
            transformer.transform(
                np.ascontiguousarray(
                    np.asarray(validation_sequence_embedding, dtype=np.float32).transpose(
                        0, 2, 1
                    )
                )[:, channel_indices]
            )
            if resolved.tune_on_validation
            else None
        )
    else:
        raise ValueError(f"unsupported Dingxin consumer: {consumer_name}")
    maneuver_classifier, selected_maneuver_c = fit_classifier(
        transformed,
        targets.maneuver_classes(maneuver_ids),
        validation_transformed,
        (
            targets.maneuver_classes(validation_maneuver_ids)
            if resolved.tune_on_validation
            else None
        ),
        c_values=(
            resolved.classification_c_grid
            if resolved.tune_on_validation
            else (resolved.classification_c,)
        ),
        random_state=resolved.random_state,
        scaler_with_mean=(consumer_name == "linear"),
        classification_labels=(0, 1, 2),
    )
    response_regressor, selected_response_alpha = fit_regressor(
        transformed[response_positions],
        targets.response_values(response_ids),
        (
            validation_transformed[validation_response_positions]
            if resolved.tune_on_validation
            else None
        ),
        (
            targets.response_values(validation_response_ids)
            if resolved.tune_on_validation
            else None
        ),
        alpha_values=(
            resolved.regression_alpha_grid
            if resolved.tune_on_validation
            else (resolved.regression_alpha,)
        ),
        scaler_with_mean=(consumer_name == "linear"),
    )
    high_response_classifier, selected_high_response_c = fit_classifier(
        transformed[response_positions],
        targets.high_response_classes(response_ids),
        (
            validation_transformed[validation_response_positions]
            if resolved.tune_on_validation
            else None
        ),
        (
            targets.high_response_classes(validation_response_ids)
            if resolved.tune_on_validation
            else None
        ),
        c_values=(
            resolved.classification_c_grid
            if resolved.tune_on_validation
            else (resolved.classification_c,)
        ),
        random_state=resolved.random_state,
        scaler_with_mean=(consumer_name == "linear"),
        classification_labels=(0, 1),
    )
    return DingxinTaskConsumerBundle(
        consumer_name=consumer_name,
        config=resolved,
        maneuver_classifier=maneuver_classifier,
        response_regressor=response_regressor,
        high_response_classifier=high_response_classifier,
        transformer=transformer,
        channel_indices=channel_indices,
        fit_maneuver_sample_ids=maneuver_ids,
        fit_response_sample_ids=response_ids,
        selected_maneuver_c=selected_maneuver_c,
        selected_response_alpha=selected_response_alpha,
        selected_high_response_c=selected_high_response_c,
    )
