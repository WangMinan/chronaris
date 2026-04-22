"""Pipeline helpers for Stage F physics-constraint context construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from chronaris.models.alignment.physics_features import (
    StageFPhysicsContext,
    StageFPhysiologyFeatureGroups,
    StageFVehicleFeatureGroups,
    build_physiology_feature_groups,
    build_vehicle_feature_groups,
)
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch


@dataclass(frozen=True, slots=True)
class StreamEnvelopeStats:
    """Per-feature envelope statistics derived from train partition values."""

    feature_names: tuple[str, ...]
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True, slots=True)
class AlignmentPhysicsConstraintStats:
    """Train-partition physics metadata shared across partitions in one run."""

    family: str
    mode: str
    field_labels: Mapping[str, str]
    vehicle_groups: StageFVehicleFeatureGroups
    physiology_groups: StageFPhysiologyFeatureGroups
    vehicle_envelope: StreamEnvelopeStats
    physiology_envelope: StreamEnvelopeStats


def build_physics_constraint_stats(
    train_batch: TorchAlignmentBatch,
    *,
    family: str,
    mode: str,
    vehicle_field_labels: Mapping[str, str],
    vehicle_envelope_quantile: float,
    physiology_envelope_quantile: float,
) -> AlignmentPhysicsConstraintStats:
    """Build train-derived physics stats before input normalization is applied."""

    return AlignmentPhysicsConstraintStats(
        family=family,
        mode=mode,
        field_labels=dict(vehicle_field_labels),
        vehicle_groups=build_vehicle_feature_groups(
            train_batch.vehicle.feature_names,
            field_labels=vehicle_field_labels,
        ),
        physiology_groups=build_physiology_feature_groups(train_batch.physiology.feature_names),
        vehicle_envelope=build_stream_envelope_stats(train_batch.vehicle, quantile=vehicle_envelope_quantile),
        physiology_envelope=build_stream_envelope_stats(train_batch.physiology, quantile=physiology_envelope_quantile),
    )


def build_batch_physics_context(
    batch: TorchAlignmentBatch,
    *,
    stats: AlignmentPhysicsConstraintStats,
    normalization_stats,
) -> StageFPhysicsContext:
    """Resolve train stats onto the current batch feature order."""

    vehicle_lower, vehicle_upper = resolve_stream_envelope_vectors(batch.vehicle, stats.vehicle_envelope)
    physiology_lower, physiology_upper = resolve_stream_envelope_vectors(batch.physiology, stats.physiology_envelope)
    vehicle_mean = vehicle_std = physiology_mean = physiology_std = None
    if normalization_stats is not None:
        vehicle_mean, vehicle_std = resolve_stream_normalization_vectors(batch.vehicle, normalization_stats.vehicle)
        physiology_mean, physiology_std = resolve_stream_normalization_vectors(batch.physiology, normalization_stats.physiology)

    return StageFPhysicsContext(
        vehicle_groups=stats.vehicle_groups,
        physiology_groups=stats.physiology_groups,
        vehicle_envelope_lower=vehicle_lower,
        vehicle_envelope_upper=vehicle_upper,
        physiology_envelope_lower=physiology_lower,
        physiology_envelope_upper=physiology_upper,
        vehicle_denormalize_mean=vehicle_mean,
        vehicle_denormalize_std=vehicle_std,
        physiology_denormalize_mean=physiology_mean,
        physiology_denormalize_std=physiology_std,
        field_labels=stats.field_labels,
    )


def build_stream_envelope_stats(
    stream_batch: TorchAlignmentStreamBatch,
    *,
    quantile: float,
) -> StreamEnvelopeStats:
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be between 0.5 and 1.0.")
    valid_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)
    values = stream_batch.values
    lower = torch.zeros((values.shape[-1],), dtype=values.dtype, device=values.device)
    upper = torch.zeros((values.shape[-1],), dtype=values.dtype, device=values.device)
    lower_q = 1.0 - quantile

    for feature_index in range(values.shape[-1]):
        feature_values = values[..., feature_index][valid_mask[..., feature_index]]
        if feature_values.numel() == 0:
            continue
        lower[feature_index] = torch.quantile(feature_values, q=lower_q)
        upper[feature_index] = torch.quantile(feature_values, q=quantile)

    return StreamEnvelopeStats(feature_names=stream_batch.feature_names, lower=lower.detach(), upper=upper.detach())


def resolve_stream_normalization_vectors(stream_batch: TorchAlignmentStreamBatch, stats) -> tuple[torch.Tensor, torch.Tensor]:
    if stream_batch.feature_names == stats.feature_names:
        return (
            stats.mean.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype),
            stats.std.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype),
        )
    mean = torch.zeros((len(stream_batch.feature_names),), dtype=stream_batch.values.dtype, device=stream_batch.values.device)
    std = torch.ones((len(stream_batch.feature_names),), dtype=stream_batch.values.dtype, device=stream_batch.values.device)
    source_index = {name: idx for idx, name in enumerate(stats.feature_names)}
    source_mean = stats.mean.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype)
    source_std = stats.std.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype)
    for feature_index, feature_name in enumerate(stream_batch.feature_names):
        mapped_index = source_index.get(feature_name)
        if mapped_index is not None:
            mean[feature_index] = source_mean[mapped_index]
            std[feature_index] = source_std[mapped_index]
    return mean, std


def resolve_stream_envelope_vectors(
    stream_batch: TorchAlignmentStreamBatch,
    stats: StreamEnvelopeStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stream_batch.feature_names == stats.feature_names:
        return (
            stats.lower.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype),
            stats.upper.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype),
        )
    lower = torch.zeros((len(stream_batch.feature_names),), dtype=stream_batch.values.dtype, device=stream_batch.values.device)
    upper = torch.zeros((len(stream_batch.feature_names),), dtype=stream_batch.values.dtype, device=stream_batch.values.device)
    source_index = {name: idx for idx, name in enumerate(stats.feature_names)}
    source_lower = stats.lower.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype)
    source_upper = stats.upper.to(device=stream_batch.values.device, dtype=stream_batch.values.dtype)
    for feature_index, feature_name in enumerate(stream_batch.feature_names):
        mapped_index = source_index.get(feature_name)
        if mapped_index is not None:
            lower[feature_index] = source_lower[mapped_index]
            upper[feature_index] = source_upper[mapped_index]
    return lower, upper
