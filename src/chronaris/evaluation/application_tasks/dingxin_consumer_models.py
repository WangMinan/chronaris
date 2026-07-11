"""Fixed linear and MiniRocket consumers for Dingxin window tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    transformer = None
    channel_indices = None
    if consumer_name == "linear":
        transformed = np.asarray(pooled_embedding, dtype=np.float32)
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
    else:
        raise ValueError(f"unsupported Dingxin consumer: {consumer_name}")
    maneuver_classifier = _classifier(resolved).fit(
        transformed,
        targets.maneuver_classes(maneuver_ids),
    )
    response_regressor = make_pipeline(
        StandardScaler(),
        Ridge(alpha=resolved.regression_alpha),
    ).fit(
        transformed[response_positions],
        targets.response_values(response_ids),
    )
    high_response_classifier = _classifier(resolved).fit(
        transformed[response_positions],
        targets.high_response_classes(response_ids),
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
    )


def _classifier(config):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.classification_c,
            max_iter=500,
            random_state=config.random_state,
        ),
    )
