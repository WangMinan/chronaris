"""Method-independent causal event coalescing for sparse long-table streams."""

from __future__ import annotations

from dataclasses import replace

import torch

from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


def coalesce_observation_batch(
    batch: DualStreamObservationBatch,
    *,
    bin_width_s: float,
) -> DualStreamObservationBatch:
    """Merge events in fixed causal bins and timestamp each bin at its last event."""
    if bin_width_s <= 0:
        raise ValueError("observation coalescing bin width must be positive")
    physiology = _coalesce_stream(
        values=batch.physiology_values,
        timestamps=batch.physiology_timestamps_s,
        point_mask=batch.physiology_point_mask,
        feature_mask=batch.physiology_feature_mask,
        bin_width_s=bin_width_s,
    )
    vehicle = _coalesce_stream(
        values=batch.vehicle_values,
        timestamps=batch.vehicle_timestamps_s,
        point_mask=batch.vehicle_point_mask,
        feature_mask=batch.vehicle_feature_mask,
        bin_width_s=bin_width_s,
    )
    return replace(
        batch,
        physiology_values=physiology["values"],
        physiology_timestamps_s=physiology["timestamps"],
        physiology_point_mask=physiology["point_mask"],
        physiology_feature_mask=physiology["feature_mask"],
        physiology_observation_age_s=physiology["age"],
        vehicle_values=vehicle["values"],
        vehicle_timestamps_s=vehicle["timestamps"],
        vehicle_point_mask=vehicle["point_mask"],
        vehicle_feature_mask=vehicle["feature_mask"],
        vehicle_observation_age_s=vehicle["age"],
    )


def _coalesce_stream(*, values, timestamps, point_mask, feature_mask, bin_width_s):
    rows = []
    maximum_count = 1
    feature_count = values.shape[-1]
    for sample_index in range(values.shape[0]):
        indices = torch.nonzero(point_mask[sample_index], as_tuple=False).flatten()
        sample_times = timestamps[sample_index].index_select(0, indices)
        if sample_times.numel() > 1 and not bool(
            torch.all(sample_times[1:] >= sample_times[:-1])
        ):
            raise RepresentationContractError(
                "raw observation timestamps must be non-decreasing"
            )
        bin_ids = torch.floor(sample_times / bin_width_s).to(torch.int64)
        _unique_bins, inverse = torch.unique_consecutive(
            bin_ids,
            return_inverse=True,
        )
        group_count = int(inverse.max().item()) + 1 if inverse.numel() else 0
        sample_values = values[sample_index].index_select(0, indices)
        sample_features = feature_mask[sample_index].index_select(0, indices)
        group_indices = inverse.unsqueeze(-1).expand(-1, feature_count)
        value_sums = values.new_zeros((group_count, feature_count))
        observation_counts = values.new_zeros((group_count, feature_count))
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
        output_mask = observation_counts > 0
        output_values = torch.where(
            output_mask,
            value_sums / observation_counts.clamp_min(1.0),
            torch.zeros_like(value_sums),
        )
        output_times = timestamps.new_zeros(group_count)
        if group_count:
            output_times.scatter_reduce_(
                0,
                inverse,
                sample_times,
                reduce="amax",
                include_self=False,
            )
        rows.append((output_values, output_times, output_mask))
        maximum_count = max(maximum_count, group_count)
    output_values = values.new_zeros((values.shape[0], maximum_count, feature_count))
    output_times = timestamps.new_zeros((values.shape[0], maximum_count))
    output_features = feature_mask.new_zeros(
        (values.shape[0], maximum_count, feature_count)
    )
    output_age = values.new_full(
        (values.shape[0], maximum_count, feature_count),
        torch.inf,
    )
    for sample_index, (sample_values, sample_times, sample_features) in enumerate(rows):
        count = len(sample_times)
        output_values[sample_index, :count] = sample_values
        output_times[sample_index, :count] = sample_times
        output_features[sample_index, :count] = sample_features
        output_age[sample_index, :count] = _feature_age(
            sample_times,
            sample_features,
            dtype=values.dtype,
        )
    return {
        "values": output_values,
        "timestamps": output_times,
        "feature_mask": output_features,
        "point_mask": output_features.any(dim=-1),
        "age": output_age,
    }


def _feature_age(times, feature_mask, *, dtype):
    observed_times = torch.where(
        feature_mask,
        times.unsqueeze(-1),
        torch.full(
            feature_mask.shape,
            -torch.inf,
            dtype=times.dtype,
            device=times.device,
        ),
    )
    last_seen = torch.cummax(observed_times, dim=0).values
    available = torch.isfinite(last_seen)
    ages = (times.unsqueeze(-1) - last_seen).clamp_min(0).to(dtype)
    return torch.where(available, ages, torch.full_like(ages, torch.inf))
