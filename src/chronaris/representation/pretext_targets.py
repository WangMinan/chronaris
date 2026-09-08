"""Method-invariant targets for common self-supervised fusion pretraining."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from chronaris.representation.augmentation_apply import AppliedAugmentationBatch
from chronaris.representation.temporal_coalescing import _feature_age
from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


LAG_DISCRIMINATION_SHIFTS_S = (-10.0, -5.0, 5.0, 10.0)
EXPLICIT_TIME_SHIFT_CLASSES_S = (-10.0, -5.0, 0.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class CommonPretextTargets:
    reconstruction_target: torch.Tensor
    reconstruction_mask: torch.Tensor
    next_query_target: torch.Tensor
    next_query_mask: torch.Tensor
    target_feature_count: int
    augmentation_ids: tuple[str, ...]
    prediction_horizons_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.reconstruction_target.ndim != 3:
            raise RepresentationContractError(
                "pretext targets must have shape [B,Q,F]"
            )
        if self.reconstruction_mask.shape != self.reconstruction_target.shape:
            raise RepresentationContractError("reconstruction mask shape mismatch")
        expected = (self.reconstruction_target.shape[:2] + (len(self.prediction_horizons_s), self.target_feature_count)
                    if self.prediction_horizons_s else self.reconstruction_target.shape)
        if self.prediction_horizons_s not in ((), (.5, 2., 5.)):
            raise RepresentationContractError("unsupported prediction horizons")
        if self.next_query_target.shape != expected:
            raise RepresentationContractError("next-query target shape mismatch")
        if self.next_query_mask.shape != expected:
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


@dataclass(frozen=True, slots=True)
class ExplicitTimeShiftInputs:
    shifted_batch: DualStreamObservationBatch
    shifts_s: torch.Tensor
    class_indices: torch.Tensor
    augmentation_ids: tuple[str, ...]


def move_common_pretext_targets(
    targets: CommonPretextTargets,
    *,
    device: str | torch.device,
) -> CommonPretextTargets:
    """Move only target tensors after deterministic augmentation stays on CPU."""
    return replace(
        targets,
        reconstruction_target=targets.reconstruction_target.to(device),
        reconstruction_mask=targets.reconstruction_mask.to(device),
        next_query_target=targets.next_query_target.to(device),
        next_query_mask=targets.next_query_mask.to(device),
    )


def build_common_pretext_targets(
    original_batch: DualStreamObservationBatch,
    augmented: AppliedAugmentationBatch,
    *, prediction_horizons_s: tuple[float, ...] = (),
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
    if prediction_horizons_s:
        future_targets, future_masks = [], []
        for horizon in prediction_horizons_s:
            queries = original_batch.query_timestamps_s + horizon
            teacher = replace(original_batch, query_timestamps_s=queries)
            streams = [causal_query_stream(teacher, stream_name=name) for name in ("physiology", "vehicle")]
            valid = torch.cat([stream.feature_mask for stream in streams], dim=-1)
            valid &= (queries < original_batch.context_durations_s[:, None])[..., None]
            future_masks.append(valid)
            future_targets.append(torch.cat([stream.values for stream in streams], dim=-1).masked_fill(~valid, 0))
        next_target, next_mask = torch.stack(future_targets, dim=2), torch.stack(future_masks, dim=2)
    return CommonPretextTargets(
        reconstruction_target=target,
        reconstruction_mask=reconstruction_mask,
        next_query_target=next_target,
        next_query_mask=next_mask,
        target_feature_count=target.shape[-1],
        augmentation_ids=augmented.augmentation_ids,
        prediction_horizons_s=prediction_horizons_s,
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
    duration_s = batch.context_durations_s
    shifted = _shift_and_compact_vehicle(batch, shifts, duration_s=duration_s)
    return LagDiscriminationInputs(
        negative_batch=shifted,
        shifts_s=shifts,
        augmentation_ids=ids,
    )


def build_explicit_time_shift_inputs(
    batch: DualStreamObservationBatch,
    augmentation_ids: Sequence[str],
    *,
    class_indices: Sequence[int] | None = None,
) -> ExplicitTimeShiftInputs:
    """Build deterministic five-class vehicle shifts including the zero-shift class."""

    ids = tuple(str(value) for value in augmentation_ids)
    if len(ids) != len(batch.sample_ids):
        raise RepresentationContractError("explicit time-shift ID count mismatch")
    resolved_class_indices = torch.as_tensor(
        (
            [int(value[8:16], 16) % len(EXPLICIT_TIME_SHIFT_CLASSES_S) for value in ids]
            if class_indices is None
            else tuple(int(value) for value in class_indices)
        ),
        dtype=torch.long,
        device=batch.vehicle_timestamps_s.device,
    )
    if resolved_class_indices.shape != (len(ids),) or bool(
        (
            (resolved_class_indices < 0)
            | (resolved_class_indices >= len(EXPLICIT_TIME_SHIFT_CLASSES_S))
        ).any()
    ):
        raise RepresentationContractError(
            "explicit time-shift classes must have one value in [0,4] per sample"
        )
    shifts = torch.as_tensor(
        EXPLICIT_TIME_SHIFT_CLASSES_S,
        dtype=batch.vehicle_timestamps_s.dtype,
        device=batch.vehicle_timestamps_s.device,
    ).index_select(0, resolved_class_indices)
    shifted = _shift_and_compact_vehicle(
        batch,
        shifts,
        duration_s=batch.context_durations_s,
    )
    return ExplicitTimeShiftInputs(
        shifted_batch=shifted,
        shifts_s=shifts,
        class_indices=resolved_class_indices,
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
        keep = (shifted_times >= 0.0) & (shifted_times < duration_s[sample_index])
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
