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
        point_mask = batch.physiology_point_mask
        feature_mask = batch.physiology_feature_mask
    elif stream_name == "vehicle":
        values = batch.vehicle_values
        timestamps = batch.vehicle_timestamps_s
        point_mask = batch.vehicle_point_mask
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
    feature_indices = torch.arange(feature_count, device=values.device).view(1, -1)
    for sample_index in range(batch_size):
        valid_rows = torch.nonzero(point_mask[sample_index], as_tuple=False).flatten()
        if len(valid_rows) == 0:
            continue
        compact_times = timestamps[sample_index].index_select(0, valid_rows)
        compact_masks = feature_mask[sample_index].index_select(0, valid_rows)
        original_row_indices = valid_rows.view(-1, 1).expand(-1, feature_count)
        latest_original_rows = torch.cummax(
            torch.where(
                compact_masks,
                original_row_indices,
                torch.full_like(original_row_indices, -1),
            ),
            dim=0,
        ).values
        query_rows = torch.searchsorted(
            compact_times,
            queries[sample_index],
            right=True,
        ) - 1
        query_has_any_row = query_rows >= 0
        source_indices = latest_original_rows.index_select(
            0,
            query_rows.clamp_min(0),
        )
        available = (source_indices >= 0) & query_has_any_row.unsqueeze(-1)
        safe_sources = source_indices.clamp_min(0)
        gathered_values = values[sample_index][safe_sources, feature_indices]
        gathered_times = timestamps[sample_index].index_select(
            0,
            safe_sources.reshape(-1),
        ).reshape(query_count, feature_count)
        output_values[sample_index] = torch.where(
            available,
            gathered_values,
            torch.zeros_like(gathered_values),
        )
        output_mask[sample_index] = available
        output_age[sample_index] = torch.where(
            available,
            queries[sample_index].unsqueeze(-1) - gathered_times,
            torch.full_like(gathered_times, torch.inf),
        ).to(values.dtype)
        output_source_indices[sample_index] = torch.where(
            available,
            source_indices,
            torch.full_like(source_indices, -1),
        )
    return CausalQueryStream(
        values=output_values,
        feature_mask=output_mask,
        modality_mask=output_mask.any(dim=-1),
        observation_age_s=output_age,
        timestamps_s=batch.query_timestamps_s.to(values.device),
        source_indices=output_source_indices,
    )
