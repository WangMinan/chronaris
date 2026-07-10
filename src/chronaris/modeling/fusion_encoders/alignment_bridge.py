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
            feature_names=physiology_feature_names,
        ),
        vehicle=_build_stream(
            values=batch.vehicle_values,
            timestamps_s=batch.vehicle_timestamps_s,
            point_mask=batch.vehicle_point_mask,
            feature_mask=batch.vehicle_feature_mask,
            feature_names=vehicle_feature_names,
        ),
    )


def _build_stream(
    *,
    values: torch.Tensor,
    timestamps_s: torch.Tensor,
    point_mask: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_names: tuple[str, ...],
) -> TorchAlignmentStreamBatch:
    if values.ndim != 3 or timestamps_s.shape != values.shape[:2]:
        raise RepresentationContractError("invalid raw stream tensor shape")
    if point_mask.shape != values.shape[:2] or feature_mask.shape != values.shape:
        raise RepresentationContractError("invalid raw stream mask shape")
    if not torch.equal(point_mask, feature_mask.any(dim=-1)):
        raise RepresentationContractError("point mask must equal any feature mask")
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
