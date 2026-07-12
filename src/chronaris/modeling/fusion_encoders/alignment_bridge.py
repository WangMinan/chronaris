"""Bridge unified raw observation batches into the ODE-RNN tensor contract."""

from __future__ import annotations

import torch

from chronaris.models.alignment.torch_batch import (
    TorchAlignmentBatch,
    TorchAlignmentStreamBatch,
)
from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


def build_alignment_batch_from_observations(
    batch: DualStreamObservationBatch,
    *,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
) -> TorchAlignmentBatch:
    """Preserve raw irregular points while adapting names and delta times."""

    if len(physiology_feature_names) != batch.physiology_values.shape[-1]:
        raise RepresentationContractError("physiology feature-name dimension mismatch")
    if len(vehicle_feature_names) != batch.vehicle_values.shape[-1]:
        raise RepresentationContractError("vehicle feature-name dimension mismatch")
    return TorchAlignmentBatch(
        sample_ids=batch.sample_ids,
        physiology=_build_stream(
            values=batch.physiology_values,
            timestamps_s=batch.physiology_timestamps_s,
            point_mask=batch.physiology_point_mask,
            feature_mask=batch.physiology_feature_mask,
            observation_age_s=batch.physiology_observation_age_s,
            feature_names=physiology_feature_names,
        ),
        vehicle=_build_stream(
            values=batch.vehicle_values,
            timestamps_s=batch.vehicle_timestamps_s,
            point_mask=batch.vehicle_point_mask,
            feature_mask=batch.vehicle_feature_mask,
            observation_age_s=batch.vehicle_observation_age_s,
            feature_names=vehicle_feature_names,
        ),
    )


def _build_stream(
    *,
    values: torch.Tensor,
    timestamps_s: torch.Tensor,
    point_mask: torch.Tensor,
    feature_mask: torch.Tensor,
    observation_age_s: torch.Tensor,
    feature_names: tuple[str, ...],
) -> TorchAlignmentStreamBatch:
    if values.ndim != 3 or timestamps_s.shape != values.shape[:2]:
        raise RepresentationContractError("invalid raw stream tensor shape")
    if point_mask.shape != values.shape[:2] or feature_mask.shape != values.shape:
        raise RepresentationContractError("invalid raw stream mask shape")
    if not torch.equal(point_mask, feature_mask.any(dim=-1)):
        raise RepresentationContractError("point mask must equal any feature mask")
    values, timestamps_s, point_mask, feature_mask, observation_age_s = (
        _coalesce_simultaneous_observations(
            values=values,
            timestamps_s=timestamps_s,
            point_mask=point_mask,
            feature_mask=feature_mask,
            observation_age_s=observation_age_s,
        )
    )
    clean_values = torch.where(feature_mask, values, torch.zeros_like(values))
    clean_timestamps = torch.where(
        point_mask,
        timestamps_s.to(dtype=values.dtype),
        torch.zeros_like(timestamps_s, dtype=values.dtype),
    )
    delta_t_s = _stream_delta_t_seconds(clean_timestamps, point_mask)
    offsets_ms = torch.round(clean_timestamps * 1000.0).to(torch.int64)
    return TorchAlignmentStreamBatch(
        values=clean_values,
        mask=point_mask,
        feature_valid_mask=feature_mask,
        offsets_ms=offsets_ms,
        offsets_s=clean_timestamps,
        delta_t_s=delta_t_s,
        point_counts=point_mask.sum(dim=1).to(torch.int64),
        feature_names=feature_names,
        observation_age_s=observation_age_s,
    )


def _coalesce_simultaneous_observations(
    *,
    values: torch.Tensor,
    timestamps_s: torch.Tensor,
    point_mask: torch.Tensor,
    feature_mask: torch.Tensor,
    observation_age_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge long-table fields sharing one timestamp into one observation event."""
    sample_rows = []
    maximum_count = 0
    feature_count = values.shape[-1]
    for sample_index in range(values.shape[0]):
        indices = torch.nonzero(point_mask[sample_index], as_tuple=False).flatten()
        sample_times = timestamps_s[sample_index].index_select(0, indices)
        if sample_times.numel() > 1 and not bool(
            torch.all(sample_times[1:] >= sample_times[:-1])
        ):
            raise RepresentationContractError(
                "raw observation timestamps must be non-decreasing"
            )
        unique_times, inverse = torch.unique_consecutive(
            sample_times,
            return_inverse=True,
        )
        group_count = len(unique_times)
        sample_values = values[sample_index].index_select(0, indices)
        sample_features = feature_mask[sample_index].index_select(0, indices)
        sample_ages = observation_age_s[sample_index].index_select(0, indices)
        group_indices = inverse.unsqueeze(-1).expand(-1, feature_count)
        value_sums = values.new_zeros((group_count, feature_count))
        observation_counts = values.new_zeros((group_count, feature_count))
        age_sums = values.new_zeros((group_count, feature_count))
        value_sums.scatter_add_(
            0,
            group_indices,
            sample_values * sample_features.to(sample_values.dtype),
        )
        observation_counts.scatter_add_(
            0,
            group_indices,
            sample_features.to(values.dtype),
        )
        age_sums.scatter_add_(
            0,
            group_indices,
            sample_ages * sample_features.to(sample_ages.dtype),
        )
        coalesced_mask = observation_counts > 0
        coalesced_values = torch.where(
            coalesced_mask,
            value_sums / observation_counts.clamp_min(1.0),
            torch.zeros_like(value_sums),
        )
        coalesced_ages = torch.where(
            coalesced_mask,
            age_sums / observation_counts.clamp_min(1.0),
            torch.zeros_like(age_sums),
        )
        sample_rows.append(
            (coalesced_values, unique_times, coalesced_mask, coalesced_ages)
        )
        maximum_count = max(maximum_count, group_count)
    maximum_count = max(maximum_count, 1)
    coalesced_values = values.new_zeros(
        (values.shape[0], maximum_count, feature_count)
    )
    coalesced_times = timestamps_s.new_zeros((values.shape[0], maximum_count))
    coalesced_features = feature_mask.new_zeros(
        (values.shape[0], maximum_count, feature_count)
    )
    coalesced_ages = values.new_zeros(
        (values.shape[0], maximum_count, feature_count)
    )
    for sample_index, (
        sample_values,
        sample_times,
        sample_features,
        sample_ages,
    ) in enumerate(
        sample_rows
    ):
        count = len(sample_times)
        coalesced_values[sample_index, :count] = sample_values
        coalesced_times[sample_index, :count] = sample_times
        coalesced_features[sample_index, :count] = sample_features
        coalesced_ages[sample_index, :count] = sample_ages
    return (
        coalesced_values,
        coalesced_times,
        coalesced_features.any(dim=-1),
        coalesced_features,
        coalesced_ages,
    )


def _stream_delta_t_seconds(
    timestamps_s: torch.Tensor,
    point_mask: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(timestamps_s)
    for sample_index in range(timestamps_s.shape[0]):
        valid_indices = torch.nonzero(point_mask[sample_index], as_tuple=False).flatten()
        if valid_indices.numel() <= 1:
            continue
        valid_times = timestamps_s[sample_index].index_select(0, valid_indices)
        deltas = torch.clamp(valid_times[1:] - valid_times[:-1], min=0.0)
        result[sample_index, valid_indices[1:]] = deltas
    return result
