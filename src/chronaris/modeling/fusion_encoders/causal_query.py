"""Feature-wise causal resampling from raw observations to the fixed query grid."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


@dataclass(frozen=True, slots=True)
class CausalQueryStream:
    values: torch.Tensor
    feature_mask: torch.Tensor
    modality_mask: torch.Tensor
    observation_age_s: torch.Tensor
    timestamps_s: torch.Tensor
    source_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise RepresentationContractError("causal query values must have shape [B,Q,F]")
        if self.feature_mask.shape != self.values.shape:
            raise RepresentationContractError("causal query feature mask shape mismatch")
        if self.observation_age_s.shape != self.values.shape:
            raise RepresentationContractError("causal query age shape mismatch")
        if self.modality_mask.shape != self.values.shape[:2]:
            raise RepresentationContractError("causal query modality mask shape mismatch")
        if self.timestamps_s.shape != self.values.shape[:2]:
            raise RepresentationContractError("causal query timestamps shape mismatch")
        if self.source_indices.shape != self.values.shape:
            raise RepresentationContractError("causal query source index shape mismatch")
        if self.source_indices.dtype != torch.int64:
            raise RepresentationContractError("causal query source indices must use int64")
        if not torch.equal(self.source_indices >= 0, self.feature_mask):
            raise RepresentationContractError(
                "causal query source availability must match feature mask"
            )
        if self.feature_mask.dtype != torch.bool or self.modality_mask.dtype != torch.bool:
            raise RepresentationContractError("causal query masks must use torch.bool")
        if not torch.equal(self.modality_mask, self.feature_mask.any(dim=-1)):
            raise RepresentationContractError(
                "causal query modality mask must equal any(feature mask)"
            )
        if not torch.isfinite(self.values[self.feature_mask]).all():
            raise RepresentationContractError("causal query valid values must be finite")
        if not torch.isfinite(self.observation_age_s[self.feature_mask]).all():
            raise RepresentationContractError("causal query valid ages must be finite")


def causal_query_stream(
    batch: DualStreamObservationBatch,
    *,
    stream_name: str,
) -> CausalQueryStream:
    """Forward-fill each feature from its latest observation at or before query time."""

    if stream_name == "physiology":
        values = batch.physiology_values
        timestamps = batch.physiology_timestamps_s
        feature_mask = batch.physiology_feature_mask
    elif stream_name == "vehicle":
        values = batch.vehicle_values
        timestamps = batch.vehicle_timestamps_s
        feature_mask = batch.vehicle_feature_mask
    else:
        raise ValueError("stream_name must be physiology or vehicle")
    queries = batch.query_timestamps_s.to(device=values.device, dtype=timestamps.dtype)
    batch_size, query_count = queries.shape
    feature_count = values.shape[-1]
    output_values = torch.zeros(
        (batch_size, query_count, feature_count),
        dtype=values.dtype,
        device=values.device,
    )
    output_mask = torch.zeros(
        (batch_size, query_count, feature_count),
        dtype=torch.bool,
        device=values.device,
    )
    output_age = torch.full(
        (batch_size, query_count, feature_count),
        torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    output_source_indices = torch.full(
        (batch_size, query_count, feature_count),
        -1,
        dtype=torch.int64,
        device=values.device,
    )
    for sample_index in range(batch_size):
        sample_queries = queries[sample_index]
        for feature_index in range(feature_count):
            observed = feature_mask[sample_index, :, feature_index]
            if not bool(observed.any()):
                continue
            feature_times = timestamps[sample_index][observed].to(queries.device)
            feature_values = values[sample_index, :, feature_index][observed]
            source_indices = torch.searchsorted(
                feature_times,
                sample_queries,
                right=True,
            ) - 1
            available = source_indices >= 0
            resolved_indices = source_indices.clamp_min(0)
            output_values[sample_index, available, feature_index] = feature_values[
                resolved_indices[available]
            ]
            output_mask[sample_index, available, feature_index] = True
            output_age[sample_index, available, feature_index] = (
                sample_queries[available] - feature_times[resolved_indices[available]]
            ).to(values.dtype)
            observed_indices = torch.nonzero(observed, as_tuple=False).flatten()
            output_source_indices[sample_index, available, feature_index] = (
                observed_indices[resolved_indices[available]]
            )
    return CausalQueryStream(
        values=output_values,
        feature_mask=output_mask,
        modality_mask=output_mask.any(dim=-1),
        observation_age_s=output_age,
        timestamps_s=batch.query_timestamps_s.to(values.device),
        source_indices=output_source_indices,
    )
