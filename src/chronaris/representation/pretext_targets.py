"""Method-invariant targets for common self-supervised fusion pretraining."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from chronaris.representation.augmentation_apply import AppliedAugmentationBatch
from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


LAG_DISCRIMINATION_SHIFTS_S = (-10.0, -5.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class CommonPretextTargets:
    reconstruction_target: torch.Tensor
    reconstruction_mask: torch.Tensor
    next_query_target: torch.Tensor
    next_query_mask: torch.Tensor
    target_feature_count: int
    augmentation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reconstruction_target.ndim != 3:
            raise RepresentationContractError(
                "pretext targets must have shape [B,Q,F]"
            )
        if self.reconstruction_mask.shape != self.reconstruction_target.shape:
            raise RepresentationContractError("reconstruction mask shape mismatch")
        if self.next_query_target.shape != self.reconstruction_target.shape:
            raise RepresentationContractError("next-query target shape mismatch")
        if self.next_query_mask.shape != self.reconstruction_target.shape:
            raise RepresentationContractError("next-query mask shape mismatch")
        if self.reconstruction_mask.dtype != torch.bool:
            raise RepresentationContractError("reconstruction mask must be boolean")
        if self.next_query_mask.dtype != torch.bool:
            raise RepresentationContractError("next-query mask must be boolean")
        if self.target_feature_count != self.reconstruction_target.shape[-1]:
            raise RepresentationContractError("pretext target feature count mismatch")
        if len(self.augmentation_ids) != self.reconstruction_target.shape[0]:
            raise RepresentationContractError("pretext augmentation ID count mismatch")


@dataclass(frozen=True, slots=True)
class LagDiscriminationInputs:
    negative_batch: DualStreamObservationBatch
    shifts_s: torch.Tensor
    augmentation_ids: tuple[str, ...]


def build_common_pretext_targets(
    original_batch: DualStreamObservationBatch,
    augmented: AppliedAugmentationBatch,
) -> CommonPretextTargets:
    """Build targets before augmentation and masks from exact source replacement."""

    from chronaris.modeling.fusion_encoders.causal_query import (
        causal_query_stream,
    )

    if original_batch.sample_ids != augmented.batch.sample_ids:
        raise RepresentationContractError("pretext original/augmented sample mismatch")
    original_physiology = causal_query_stream(
        original_batch,
        stream_name="physiology",
    )
    original_vehicle = causal_query_stream(
        original_batch,
        stream_name="vehicle",
    )
    augmented_physiology = causal_query_stream(
        augmented.batch,
        stream_name="physiology",
    )
    augmented_vehicle = causal_query_stream(
        augmented.batch,
        stream_name="vehicle",
    )
    mapped_physiology_sources = map_augmented_query_sources_to_original(
        augmented_physiology.source_indices,
        augmented.physiology_provenance.original_point_indices,
    )
    mapped_vehicle_sources = map_augmented_query_sources_to_original(
        augmented_vehicle.source_indices,
        augmented.vehicle_provenance.original_point_indices,
    )
    target = torch.cat(
        (original_physiology.values, original_vehicle.values),
        dim=-1,
    )
    target_mask = torch.cat(
        (original_physiology.feature_mask, original_vehicle.feature_mask),
        dim=-1,
    )
    mapped_sources = torch.cat(
        (mapped_physiology_sources, mapped_vehicle_sources),
        dim=-1,
    )
    original_sources = torch.cat(
        (
            original_physiology.source_indices,
            original_vehicle.source_indices,
        ),
        dim=-1,
    )
    reconstruction_mask = target_mask & (mapped_sources != original_sources)
    next_target = torch.zeros_like(target)
    next_mask = torch.zeros_like(target_mask)
    next_target[:, :-1] = target[:, 1:]
    next_mask[:, :-1] = target_mask[:, 1:]
    return CommonPretextTargets(
        reconstruction_target=target,
        reconstruction_mask=reconstruction_mask,
        next_query_target=next_target,
        next_query_mask=next_mask,
        target_feature_count=target.shape[-1],
        augmentation_ids=augmented.augmentation_ids,
    )


def map_augmented_query_sources_to_original(
    augmented_source_indices: torch.Tensor,
    original_point_indices: torch.Tensor,
) -> torch.Tensor:
    """Resolve augmented padded row indices through retained-row provenance."""

    if augmented_source_indices.ndim != 3 or original_point_indices.ndim != 2:
        raise RepresentationContractError("pretext source provenance shape mismatch")
    if augmented_source_indices.shape[0] != original_point_indices.shape[0]:
        raise RepresentationContractError("pretext source provenance batch mismatch")
    safe = augmented_source_indices.clamp_min(0)
    expanded = original_point_indices.unsqueeze(1).expand(
        -1,
        augmented_source_indices.shape[1],
        -1,
    )
    mapped = torch.gather(expanded, 2, safe)
    return torch.where(
        augmented_source_indices >= 0,
        mapped,
        torch.full_like(mapped, -1),
    )


def build_lag_discrimination_inputs(
    batch: DualStreamObservationBatch,
    augmentation_ids: Sequence[str],
) -> LagDiscriminationInputs:
    """Shift vehicle timestamps by a frozen ID-derived offset without oracle access."""

    ids = tuple(str(value) for value in augmentation_ids)
    if len(ids) != len(batch.sample_ids):
        raise RepresentationContractError("lag discrimination ID count mismatch")
    shifts = torch.as_tensor(
        [
            LAG_DISCRIMINATION_SHIFTS_S[int(value[:8], 16) % len(LAG_DISCRIMINATION_SHIFTS_S)]
            for value in ids
        ],
        dtype=batch.vehicle_timestamps_s.dtype,
        device=batch.vehicle_timestamps_s.device,
    )
    duration_s = _context_duration_s(batch.query_timestamps_s)
    shifted = _shift_and_compact_vehicle(batch, shifts, duration_s=duration_s)
    return LagDiscriminationInputs(
        negative_batch=shifted,
        shifts_s=shifts,
        augmentation_ids=ids,
    )


def _shift_and_compact_vehicle(batch, shifts, *, duration_s):
    values = torch.zeros_like(batch.vehicle_values)
    times = torch.zeros_like(batch.vehicle_timestamps_s)
    features = torch.zeros_like(batch.vehicle_feature_mask)
    ages = torch.full_like(batch.vehicle_observation_age_s, torch.inf)
    for sample_index in range(len(batch.sample_ids)):
        original_indices = torch.nonzero(
            batch.vehicle_point_mask[sample_index],
            as_tuple=False,
        ).flatten()
        original_times = batch.vehicle_timestamps_s[sample_index].index_select(
            0,
            original_indices,
        )
        shifted_times = original_times + shifts[sample_index]
        keep = (shifted_times >= 0.0) & (shifted_times < duration_s)
        kept_indices = original_indices[keep]
        kept_times = shifted_times[keep]
        count = len(kept_indices)
        if count == 0:
            continue
        kept_values = batch.vehicle_values[sample_index].index_select(
            0,
            kept_indices,
        )
        kept_features = batch.vehicle_feature_mask[sample_index].index_select(
            0,
            kept_indices,
        )
        values[sample_index, :count] = kept_values
        times[sample_index, :count] = kept_times
        features[sample_index, :count] = kept_features
        ages[sample_index, :count] = _feature_age(
            kept_times,
            kept_features,
            dtype=values.dtype,
        )
    return replace(
        batch,
        vehicle_values=values,
        vehicle_timestamps_s=times,
        vehicle_point_mask=features.any(dim=-1),
        vehicle_feature_mask=features,
        vehicle_observation_age_s=ages,
    )


def _feature_age(times, feature_mask, *, dtype):
    result = torch.full(
        feature_mask.shape,
        torch.inf,
        dtype=dtype,
        device=feature_mask.device,
    )
    last_seen = torch.full(
        (feature_mask.shape[-1],),
        -torch.inf,
        dtype=times.dtype,
        device=times.device,
    )
    for point_index, time_s in enumerate(times):
        observed = feature_mask[point_index]
        last_seen = torch.where(observed, time_s, last_seen)
        available = torch.isfinite(last_seen)
        result[point_index, available] = (
            time_s - last_seen[available]
        ).clamp_min(0).to(dtype)
    return result


def _context_duration_s(query_timestamps_s: torch.Tensor) -> float:
    step = query_timestamps_s[:, 1] - query_timestamps_s[:, 0]
    duration = query_timestamps_s[:, -1] + step
    if not torch.allclose(duration, duration[:1], atol=1e-9, rtol=1e-9):
        raise RepresentationContractError("lag discrimination context mismatch")
    return float(duration[0].item())
