"""Raw asynchronous sample containers and padding-aware batch collation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from chronaris.representation.contracts import (
    QUERY_POINT_COUNT,
    DualStreamObservationBatch,
    ObservationSchema,
    RepresentationContractError,
)


@dataclass(frozen=True, slots=True)
class ObservedDualStreamSample:
    """One unpadded 30-second observed-only context before tensor collation."""

    sample_id: str
    group_id: str
    schema: ObservationSchema
    physiology_values: NDArray[np.float32]
    physiology_timestamps_s: NDArray[np.float64]
    physiology_feature_mask: NDArray[np.bool_]
    vehicle_values: NDArray[np.float32]
    vehicle_timestamps_s: NDArray[np.float64]
    vehicle_feature_mask: NDArray[np.bool_]
    source_sample_hash: str
    context_duration_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise RepresentationContractError("sample_id and group_id are required")
        if self.context_duration_s <= 0:
            raise RepresentationContractError("context_duration_s must be positive")
        _validate_sample_stream(
            stream_name="physiology",
            values=self.physiology_values,
            timestamps=self.physiology_timestamps_s,
            feature_mask=self.physiology_feature_mask,
            expected_features=len(self.schema.physiology_feature_names),
            context_duration_s=self.context_duration_s,
        )
        _validate_sample_stream(
            stream_name="vehicle",
            values=self.vehicle_values,
            timestamps=self.vehicle_timestamps_s,
            feature_mask=self.vehicle_feature_mask,
            expected_features=len(self.schema.vehicle_feature_names),
            context_duration_s=self.context_duration_s,
        )
        if len(self.source_sample_hash) != 64:
            raise RepresentationContractError("source_sample_hash must be SHA-256")


def collate_observation_samples(
    samples: Sequence[ObservedDualStreamSample],
) -> DualStreamObservationBatch:
    """Pad asynchronous streams without interpolating or changing their timestamps."""

    if not samples:
        raise RepresentationContractError("cannot collate an empty sample sequence")
    schema_hash = samples[0].schema.schema_sha256
    duration_s = samples[0].context_duration_s
    for sample in samples:
        if sample.schema.schema_sha256 != schema_hash:
            raise RepresentationContractError("all collated samples must share one schema")
        if sample.context_duration_s != duration_s:
            raise RepresentationContractError(
                "all collated samples must share one context duration"
            )
    physiology = _pad_stream(
        [sample.physiology_values for sample in samples],
        [sample.physiology_timestamps_s for sample in samples],
        [sample.physiology_feature_mask for sample in samples],
    )
    vehicle = _pad_stream(
        [sample.vehicle_values for sample in samples],
        [sample.vehicle_timestamps_s for sample in samples],
        [sample.vehicle_feature_mask for sample in samples],
    )
    query = np.linspace(
        0.0,
        float(duration_s),
        num=QUERY_POINT_COUNT,
        endpoint=False,
        dtype=np.float64,
    )
    query_batch = np.broadcast_to(query, (len(samples), QUERY_POINT_COUNT)).copy()
    return DualStreamObservationBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        group_ids=tuple(sample.group_id for sample in samples),
        physiology_values=torch.from_numpy(physiology["values"]),
        physiology_timestamps_s=torch.from_numpy(physiology["timestamps"]),
        physiology_point_mask=torch.from_numpy(physiology["point_mask"]),
        physiology_feature_mask=torch.from_numpy(physiology["feature_mask"]),
        physiology_observation_age_s=torch.from_numpy(physiology["age"]),
        vehicle_values=torch.from_numpy(vehicle["values"]),
        vehicle_timestamps_s=torch.from_numpy(vehicle["timestamps"]),
        vehicle_point_mask=torch.from_numpy(vehicle["point_mask"]),
        vehicle_feature_mask=torch.from_numpy(vehicle["feature_mask"]),
        vehicle_observation_age_s=torch.from_numpy(vehicle["age"]),
        query_timestamps_s=torch.from_numpy(query_batch),
        source_sample_hashes=tuple(sample.source_sample_hash for sample in samples),
    )


def select_observation_batch(
    batch: DualStreamObservationBatch,
    sample_ids: Sequence[str],
) -> DualStreamObservationBatch:
    """Select rows in explicit order while retaining every tensor and lineage field."""

    requested = tuple(str(value) for value in sample_ids)
    if not requested or len(set(requested)) != len(requested):
        raise RepresentationContractError(
            "selected observation sample IDs must be non-empty and unique"
        )
    index_by_id = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
    missing = sorted(set(requested) - set(index_by_id))
    if missing:
        raise RepresentationContractError(
            f"selected samples are absent from observation batch: {missing[:5]}"
        )
    tensor_values = {
        field.name: getattr(batch, field.name).index_select(
            0,
            torch.tensor(
                [index_by_id[value] for value in requested],
                dtype=torch.long,
                device=getattr(batch, field.name).device,
            ),
        )
        for field in fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    }
    return DualStreamObservationBatch(
        sample_ids=requested,
        group_ids=tuple(batch.group_ids[index_by_id[value]] for value in requested),
        source_sample_hashes=tuple(
            batch.source_sample_hashes[index_by_id[value]] for value in requested
        ),
        **tensor_values,
    )


def stable_observed_sample_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _pad_stream(
    values: Sequence[NDArray[np.float32]],
    timestamps: Sequence[NDArray[np.float64]],
    feature_masks: Sequence[NDArray[np.bool_]],
) -> dict[str, np.ndarray]:
    batch_size = len(values)
    feature_count = int(values[0].shape[1])
    max_points = max(max(int(array.shape[0]), 1) for array in values)
    padded_values = np.zeros((batch_size, max_points, feature_count), dtype=np.float32)
    padded_times = np.zeros((batch_size, max_points), dtype=np.float64)
    padded_features = np.zeros((batch_size, max_points, feature_count), dtype=bool)
    padded_age = np.full(
        (batch_size, max_points, feature_count),
        np.inf,
        dtype=np.float32,
    )
    for row, (sample_values, sample_times, sample_mask) in enumerate(
        zip(values, timestamps, feature_masks, strict=True)
    ):
        count = len(sample_times)
        if count == 0:
            continue
        padded_values[row, :count] = sample_values
        padded_times[row, :count] = sample_times
        padded_features[row, :count] = sample_mask
        padded_age[row, :count] = _observation_age(sample_times, sample_mask)
    point_mask = padded_features.any(axis=-1)
    return {
        "values": padded_values,
        "timestamps": padded_times,
        "point_mask": point_mask,
        "feature_mask": padded_features,
        "age": padded_age,
    }


def _observation_age(
    timestamps: NDArray[np.float64],
    feature_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    age = np.full(feature_mask.shape, np.inf, dtype=np.float32)
    last_seen = np.full(feature_mask.shape[1], -np.inf, dtype=np.float64)
    for row, time_s in enumerate(timestamps):
        observed = feature_mask[row]
        last_seen[observed] = time_s
        available = np.isfinite(last_seen)
        age[row, available] = np.maximum(time_s - last_seen[available], 0.0).astype(
            np.float32
        )
    return age


def _validate_sample_stream(
    *,
    stream_name: str,
    values: np.ndarray,
    timestamps: np.ndarray,
    feature_mask: np.ndarray,
    expected_features: int,
    context_duration_s: float,
) -> None:
    if values.ndim != 2 or values.shape[1] != expected_features:
        raise RepresentationContractError(
            f"{stream_name} values must have shape [T,{expected_features}]"
        )
    if timestamps.ndim != 1 or len(timestamps) != len(values):
        raise RepresentationContractError(f"{stream_name} timestamp shape mismatch")
    if feature_mask.shape != values.shape or feature_mask.dtype != np.bool_:
        raise RepresentationContractError(f"{stream_name} feature mask shape/dtype mismatch")
    if len(timestamps) and (
        not np.isfinite(timestamps).all()
        or np.any(timestamps[1:] < timestamps[:-1])
        or float(timestamps.min()) < -1e-9
        or float(timestamps.max()) >= context_duration_s + 1e-9
    ):
        raise RepresentationContractError(
            f"{stream_name} timestamps must be finite, ordered and inside the context"
        )
    if not np.all(feature_mask.any(axis=1)):
        raise RepresentationContractError(
            f"{stream_name} sample rows without valid features must be removed"
        )
    if not np.isfinite(values[feature_mask]).all():
        raise RepresentationContractError(
            f"{stream_name} valid feature values must be finite"
        )
