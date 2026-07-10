"""Apply method-invariant augmentation plans to raw asynchronous batches."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch

from chronaris.representation.augmentation import (
    AugmentationPolicy,
    AugmentationRealization,
)
from chronaris.representation.contracts import (
    DualStreamObservationBatch,
    RepresentationContractError,
)


@dataclass(frozen=True, slots=True)
class StreamAugmentationProvenance:
    """Map retained/padded augmented rows back to original padded row indices."""

    original_point_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.original_point_indices.ndim != 2:
            raise RepresentationContractError(
                "augmentation provenance must have shape [B,T]"
            )
        if self.original_point_indices.dtype != torch.int64:
            raise RepresentationContractError(
                "augmentation provenance must use int64"
            )


@dataclass(frozen=True, slots=True)
class AugmentationAuditRow:
    augmentation_id: str
    sample_id: str
    stream_name: str
    original_point_count: int
    retained_point_count: int
    removed_point_count: int
    original_valid_feature_count: int
    retained_valid_feature_count: int
    removed_valid_feature_count: int
    modality_dropped: bool
    block_start_s: float
    block_duration_s: float
    clock_offset_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class AppliedAugmentationBatch:
    batch: DualStreamObservationBatch
    physiology_provenance: StreamAugmentationProvenance
    vehicle_provenance: StreamAugmentationProvenance
    audit_rows: tuple[AugmentationAuditRow, ...]
    augmentation_ids: tuple[str, ...]


def apply_augmentation_realizations(
    batch: DualStreamObservationBatch,
    realizations: Sequence[AugmentationRealization],
    *,
    policy: AugmentationPolicy | None = None,
) -> AppliedAugmentationBatch:
    """Execute one shared plan per sample without accepting a method identity."""

    resolved_policy = policy or AugmentationPolicy()
    plans = tuple(realizations)
    if len(plans) != len(batch.sample_ids):
        raise RepresentationContractError(
            "augmentation realization count does not match batch size"
        )
    for sample_id, plan in zip(batch.sample_ids, plans, strict=True):
        if plan.sample_id != sample_id:
            raise RepresentationContractError(
                "augmentation realization sample order mismatch"
            )
        if plan.dropped_modality not in {None, "physiology", "vehicle"}:
            raise RepresentationContractError("invalid dropped modality")
    context_duration_s = _context_duration_s(batch.query_timestamps_s)
    physiology = _augment_stream(
        values=batch.physiology_values,
        timestamps=batch.physiology_timestamps_s,
        feature_mask=batch.physiology_feature_mask,
        point_mask=batch.physiology_point_mask,
        plans=plans,
        policy=resolved_policy,
        stream_name="physiology",
        context_duration_s=context_duration_s,
    )
    vehicle = _augment_stream(
        values=batch.vehicle_values,
        timestamps=batch.vehicle_timestamps_s,
        feature_mask=batch.vehicle_feature_mask,
        point_mask=batch.vehicle_point_mask,
        plans=plans,
        policy=resolved_policy,
        stream_name="vehicle",
        context_duration_s=context_duration_s,
    )
    augmented = replace(
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
    return AppliedAugmentationBatch(
        batch=augmented,
        physiology_provenance=StreamAugmentationProvenance(
            physiology["original_indices"]
        ),
        vehicle_provenance=StreamAugmentationProvenance(
            vehicle["original_indices"]
        ),
        audit_rows=tuple((*physiology["audit_rows"], *vehicle["audit_rows"])),
        augmentation_ids=tuple(plan.augmentation_id for plan in plans),
    )


def augmentation_executor_accepts_method_name() -> bool:
    """Support an explicit protocol test without source-string heuristics."""

    return "method_name" in inspect.signature(
        apply_augmentation_realizations
    ).parameters


def _augment_stream(
    *,
    values,
    timestamps,
    feature_mask,
    point_mask,
    plans,
    policy,
    stream_name,
    context_duration_s,
):
    output_values = torch.zeros_like(values)
    output_times = torch.zeros_like(timestamps)
    output_features = torch.zeros_like(feature_mask)
    output_age = torch.full_like(values, torch.inf)
    output_original_indices = torch.full(
        point_mask.shape,
        -1,
        dtype=torch.int64,
        device=point_mask.device,
    )
    audit_rows = []
    for sample_index, plan in enumerate(plans):
        valid_indices = torch.nonzero(
            point_mask[sample_index],
            as_tuple=False,
        ).flatten()
        sample_times = timestamps[sample_index].index_select(0, valid_indices)
        sample_values = values[sample_index].index_select(0, valid_indices)
        sample_features = feature_mask[sample_index].index_select(0, valid_indices)
        block_start = float(getattr(plan, f"{stream_name}_block_start_s"))
        block_duration = float(getattr(plan, f"{stream_name}_block_duration_s"))
        clock_offset = float(getattr(plan, f"{stream_name}_clock_offset_s"))
        jitter_seed = int(getattr(plan, f"{stream_name}_jitter_seed"))
        dropout_seed = int(getattr(plan, f"{stream_name}_point_dropout_seed"))
        modality_dropped = plan.dropped_modality == stream_name
        keep = torch.ones(
            len(valid_indices),
            dtype=torch.bool,
            device=point_mask.device,
        )
        if modality_dropped:
            keep.fill_(False)
        elif len(valid_indices):
            keep &= ~(
                (sample_times >= block_start)
                & (sample_times < block_start + block_duration)
            )
            generator = np.random.default_rng(dropout_seed)
            point_drop = torch.as_tensor(
                generator.random(len(valid_indices))
                < policy.point_dropout_probability,
                dtype=torch.bool,
                device=point_mask.device,
            )
            keep &= ~point_drop
        retained_indices = valid_indices[keep]
        retained_times = sample_times[keep]
        retained_values = sample_values[keep]
        retained_features = sample_features[keep]
        if len(retained_indices):
            jitter = np.random.default_rng(jitter_seed).normal(
                0.0,
                policy.timestamp_jitter_sigma_s,
                size=len(retained_indices),
            )
            transformed_times = retained_times + clock_offset + torch.as_tensor(
                jitter,
                dtype=retained_times.dtype,
                device=retained_times.device,
            )
            inside = (transformed_times >= 0.0) & (
                transformed_times < context_duration_s
            )
            retained_indices = retained_indices[inside]
            transformed_times = transformed_times[inside]
            retained_values = retained_values[inside]
            retained_features = retained_features[inside]
            order = _stable_time_order(transformed_times, retained_indices)
            retained_indices = retained_indices[order]
            transformed_times = transformed_times[order]
            retained_values = retained_values[order]
            retained_features = retained_features[order]
        else:
            transformed_times = retained_times
        retained_count = len(retained_indices)
        if retained_count:
            output_values[sample_index, :retained_count] = retained_values
            output_times[sample_index, :retained_count] = transformed_times
            output_features[sample_index, :retained_count] = retained_features
            output_original_indices[sample_index, :retained_count] = retained_indices
            output_age[sample_index, :retained_count] = _feature_age(
                transformed_times,
                retained_features,
                dtype=values.dtype,
            )
        original_feature_count = int(sample_features.sum().item())
        retained_feature_count = int(retained_features.sum().item())
        audit_rows.append(
            AugmentationAuditRow(
                augmentation_id=plan.augmentation_id,
                sample_id=plan.sample_id,
                stream_name=stream_name,
                original_point_count=len(valid_indices),
                retained_point_count=retained_count,
                removed_point_count=len(valid_indices) - retained_count,
                original_valid_feature_count=original_feature_count,
                retained_valid_feature_count=retained_feature_count,
                removed_valid_feature_count=(
                    original_feature_count - retained_feature_count
                ),
                modality_dropped=modality_dropped,
                block_start_s=block_start,
                block_duration_s=block_duration,
                clock_offset_s=clock_offset,
            )
        )
    output_point_mask = output_features.any(dim=-1)
    return {
        "values": output_values,
        "timestamps": output_times,
        "feature_mask": output_features,
        "point_mask": output_point_mask,
        "age": output_age,
        "original_indices": output_original_indices,
        "audit_rows": audit_rows,
    }


def _stable_time_order(
    timestamps: torch.Tensor,
    original_indices: torch.Tensor,
) -> torch.Tensor:
    order = np.lexsort(
        (
            original_indices.detach().cpu().numpy(),
            timestamps.detach().cpu().numpy(),
        )
    )
    return torch.as_tensor(order, dtype=torch.long, device=timestamps.device)


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
    if query_timestamps_s.shape[1] <= 1:
        raise RepresentationContractError("query axis is too short for augmentation")
    step = query_timestamps_s[:, 1:] - query_timestamps_s[:, :-1]
    if not torch.allclose(step, step[:, :1], atol=1e-9, rtol=1e-9):
        raise RepresentationContractError("augmentation requires a regular query grid")
    duration = query_timestamps_s[:, -1] + step[:, -1]
    if not torch.allclose(duration, duration[:1], atol=1e-9, rtol=1e-9):
        raise RepresentationContractError("batch context durations differ")
    return float(duration[0].item())
