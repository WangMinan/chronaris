"""Batching utilities for Stage E alignment models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chronaris.features.experiment_input import E0ExperimentSample


@dataclass(frozen=True, slots=True)
class AlignmentStreamBatch:
    """A padded batch representation for one stream."""

    values: np.ndarray
    mask: np.ndarray
    offsets_ms: np.ndarray
    point_counts: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AlignmentBatch:
    """Model-ready batch for dual-stream alignment experiments."""

    sample_ids: tuple[str, ...]
    physiology: AlignmentStreamBatch
    vehicle: AlignmentStreamBatch


def build_alignment_batch(samples: tuple[E0ExperimentSample, ...]) -> AlignmentBatch:
    """Convert E0 samples into a padded dual-stream batch."""

    if not samples:
        raise ValueError("At least one E0ExperimentSample is required to build an alignment batch.")

    physiology = _build_stream_batch(
        tuple(sample.physiology for sample in samples),
        fill_value=np.nan,
    )
    vehicle = _build_stream_batch(
        tuple(sample.vehicle for sample in samples),
        fill_value=np.nan,
    )

    return AlignmentBatch(
        sample_ids=tuple(sample.sample_id for sample in samples),
        physiology=physiology,
        vehicle=vehicle,
    )


def _build_stream_batch(streams, *, fill_value: float) -> AlignmentStreamBatch:
    feature_names = _resolve_shared_feature_names(streams)
    max_points = max(stream.point_count for stream in streams)
    batch_size = len(streams)
    feature_count = len(feature_names)

    values = np.full((batch_size, max_points, feature_count), fill_value, dtype=np.float64)
    mask = np.zeros((batch_size, max_points), dtype=bool)
    offsets_ms = np.full((batch_size, max_points), -1, dtype=np.int64)
    point_counts = np.zeros((batch_size,), dtype=np.int64)

    for sample_index, stream in enumerate(streams):
        point_counts[sample_index] = stream.point_count
        feature_index = {name: idx for idx, name in enumerate(feature_names)}
        for point_index, row in enumerate(stream.values):
            mask[sample_index, point_index] = True
            offsets_ms[sample_index, point_index] = stream.point_offsets_ms[point_index]
            row_map = {
                stream.feature_names[column_index]: value
                for column_index, value in enumerate(row)
            }
            for feature_name, value in row_map.items():
                values[sample_index, point_index, feature_index[feature_name]] = value

    return AlignmentStreamBatch(
        values=values,
        mask=mask,
        offsets_ms=offsets_ms,
        point_counts=point_counts,
        feature_names=feature_names,
    )


def _resolve_shared_feature_names(streams) -> tuple[str, ...]:
    names: set[str] = set()
    for stream in streams:
        names.update(stream.feature_names)
    return tuple(sorted(names))
