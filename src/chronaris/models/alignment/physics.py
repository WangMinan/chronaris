"""Stage F physics-constraint loss families for alignment training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from chronaris.models.alignment.physics_features import (
    StageFPhysicsContext,
    StageFVehicleFeatureGroups,
    build_physiology_feature_groups,
    build_vehicle_feature_groups,
)
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch

if TYPE_CHECKING:
    from chronaris.models.alignment.prototype import DualStreamPrototypeOutput, StreamPrototypeOutput
else:
    DualStreamPrototypeOutput = Any
    StreamPrototypeOutput = Any

_ALLOWED_PHYSICS_CONSTRAINT_MODES = {
    "feature_first_with_latent_fallback",
    "feature_only",
    "latent_only",
}
_ALLOWED_PHYSICS_FAMILIES = {"minimal", "full"}


@dataclass(frozen=True, slots=True)
class PhysicsLossBreakdown:
    """Component-level Stage F physics-constraint loss summary."""

    vehicle_semantic: torch.Tensor
    vehicle_smoothness: torch.Tensor
    vehicle_envelope: torch.Tensor
    vehicle_latent: torch.Tensor
    physiology_smoothness: torch.Tensor
    physiology_envelope: torch.Tensor
    physiology_pairwise: torch.Tensor
    physiology_spo2_delta: torch.Tensor
    physiology_latent: torch.Tensor
    vehicle: torch.Tensor
    physiology: torch.Tensor
    total: torch.Tensor

    @classmethod
    def zeros(cls, reference: torch.Tensor) -> "PhysicsLossBreakdown":
        zero = reference.new_zeros(())
        return cls(
            vehicle_semantic=zero,
            vehicle_smoothness=zero,
            vehicle_envelope=zero,
            vehicle_latent=zero,
            physiology_smoothness=zero,
            physiology_envelope=zero,
            physiology_pairwise=zero,
            physiology_spo2_delta=zero,
            physiology_latent=zero,
            vehicle=zero,
            physiology=zero,
            total=zero,
        )

    def component_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "vehicle_semantic": self.vehicle_semantic,
            "vehicle_smoothness": self.vehicle_smoothness,
            "vehicle_envelope": self.vehicle_envelope,
            "vehicle_latent": self.vehicle_latent,
            "physiology_smoothness": self.physiology_smoothness,
            "physiology_envelope": self.physiology_envelope,
            "physiology_pairwise": self.physiology_pairwise,
            "physiology_spo2_delta": self.physiology_spo2_delta,
            "physiology_latent": self.physiology_latent,
        }


def vehicle_physics_consistency_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    huber_delta: float = 1.0,
    context: StageFPhysicsContext | None = None,
) -> torch.Tensor:
    """Compute the legacy vehicle-side Stage F(min) physics loss."""

    _validate_mode(mode)
    _validate_huber_delta(huber_delta)
    stream_batch = batch.vehicle
    stream_output = output.vehicle
    valid_feature_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)
    groups = context.vehicle_groups if context is not None else build_vehicle_feature_groups(stream_batch.feature_names)

    if mode in {"feature_first_with_latent_fallback", "feature_only"}:
        semantic = _vehicle_semantic_residual_loss(
            stream_output.reconstructions,
            stream_batch.offsets_s,
            valid_feature_mask,
            stream_batch.feature_names,
            groups,
            huber_delta=huber_delta,
        )
        if semantic is not None:
            return semantic
        if mode == "feature_only":
            return stream_batch.values.new_zeros(())

    latent_states, latent_times, latent_valid = _resolve_physics_stream_states(stream_output, stream_batch)
    return _second_derivative_smoothness_loss(latent_states, latent_times, latent_valid)


def physiology_physics_consistency_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    envelope_lower: torch.Tensor | None = None,
    envelope_upper: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the legacy physiology-side Stage F(min) physics loss."""

    _validate_mode(mode)
    stream_batch = batch.physiology
    stream_output = output.physiology
    valid_feature_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)

    if mode in {"feature_first_with_latent_fallback", "feature_only"}:
        smoothness = _second_derivative_smoothness_loss(
            stream_output.reconstructions,
            stream_batch.offsets_s,
            stream_batch.mask,
        )
        envelope = _feature_envelope_penalty(
            stream_output.reconstructions,
            valid_feature_mask,
            lower=envelope_lower,
            upper=envelope_upper,
        )
        return smoothness + envelope

    latent_states, latent_times, latent_valid = _resolve_physics_stream_states(stream_output, stream_batch)
    return _second_derivative_smoothness_loss(latent_states, latent_times, latent_valid)


def build_stage_f_physics_losses(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    family: str = "minimal",
    huber_delta: float = 1.0,
    context: StageFPhysicsContext | None = None,
    physiology_envelope_lower: torch.Tensor | None = None,
    physiology_envelope_upper: torch.Tensor | None = None,
) -> PhysicsLossBreakdown:
    """Build Stage F physics losses for either minimal or full family."""

    _validate_mode(mode)
    _validate_family(family)
    _validate_huber_delta(huber_delta)
    if family == "minimal":
        physiology_context_lower = context.physiology_envelope_lower if context is not None else physiology_envelope_lower
        physiology_context_upper = context.physiology_envelope_upper if context is not None else physiology_envelope_upper
        vehicle = vehicle_physics_consistency_loss(output, batch, mode=mode, huber_delta=huber_delta, context=context)
        physiology = physiology_physics_consistency_loss(
            output,
            batch,
            mode=mode,
            envelope_lower=physiology_context_lower,
            envelope_upper=physiology_context_upper,
        )
        zero = vehicle.new_zeros(())
        return PhysicsLossBreakdown(
            vehicle_semantic=vehicle,
            vehicle_smoothness=zero,
            vehicle_envelope=zero,
            vehicle_latent=zero,
            physiology_smoothness=physiology,
            physiology_envelope=zero,
            physiology_pairwise=zero,
            physiology_spo2_delta=zero,
            physiology_latent=zero,
            vehicle=vehicle,
            physiology=physiology,
            total=vehicle + physiology,
        )

    return _build_full_stage_f_physics_losses(output, batch, mode=mode, huber_delta=huber_delta, context=context)


def _build_full_stage_f_physics_losses(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str,
    huber_delta: float,
    context: StageFPhysicsContext | None,
) -> PhysicsLossBreakdown:
    reference = output.physiology.reconstructions
    if context is None:
        context = StageFPhysicsContext(
            vehicle_groups=build_vehicle_feature_groups(batch.vehicle.feature_names),
            physiology_groups=build_physiology_feature_groups(batch.physiology.feature_names),
        )

    vehicle_values = _denormalize_if_available(
        output.vehicle.reconstructions,
        context.vehicle_denormalize_mean,
        context.vehicle_denormalize_std,
    )
    physiology_values = _denormalize_if_available(
        output.physiology.reconstructions,
        context.physiology_denormalize_mean,
        context.physiology_denormalize_std,
    )
    vehicle_valid = batch.vehicle.feature_valid_mask & batch.vehicle.mask.unsqueeze(-1)
    physiology_valid = batch.physiology.feature_valid_mask & batch.physiology.mask.unsqueeze(-1)

    if mode == "latent_only":
        zero = reference.new_zeros(())
        vehicle_semantic = vehicle_smoothness = vehicle_envelope = zero
        physiology_smoothness = physiology_envelope = physiology_pairwise = physiology_spo2_delta = zero
    else:
        semantic_loss = _vehicle_semantic_residual_loss(
            vehicle_values,
            batch.vehicle.offsets_s,
            vehicle_valid,
            batch.vehicle.feature_names,
            context.vehicle_groups,
            huber_delta=huber_delta,
        )
        vehicle_semantic = semantic_loss if semantic_loss is not None else reference.new_zeros(())
        vehicle_smoothness = _second_derivative_smoothness_loss(
            output.vehicle.reconstructions,
            batch.vehicle.offsets_s,
            batch.vehicle.mask,
        )
        vehicle_envelope = _feature_envelope_penalty(
            vehicle_values,
            vehicle_valid,
            lower=context.vehicle_envelope_lower,
            upper=context.vehicle_envelope_upper,
        )
        physiology_smoothness = _selected_feature_smoothness_loss(
            output.physiology.reconstructions,
            batch.physiology.offsets_s,
            batch.physiology.mask,
            batch.physiology.feature_names,
            context.physiology_groups.eeg or batch.physiology.feature_names,
        )
        physiology_envelope = _feature_envelope_penalty(
            physiology_values,
            physiology_valid,
            lower=context.physiology_envelope_lower,
            upper=context.physiology_envelope_upper,
        )
        physiology_pairwise = _pairwise_feature_consistency_loss(
            output.physiology.reconstructions,
            physiology_valid,
            batch.physiology.feature_names,
            context.physiology_groups.eeg_pairs,
            huber_delta=huber_delta,
        )
        physiology_spo2_delta = _selected_first_derivative_loss(
            output.physiology.reconstructions,
            batch.physiology.offsets_s,
            physiology_valid,
            batch.physiology.feature_names,
            context.physiology_groups.spo2,
            huber_delta=huber_delta,
        )

    vehicle_latent = reference.new_zeros(())
    physiology_latent = reference.new_zeros(())
    if mode in {"feature_first_with_latent_fallback", "latent_only"}:
        vehicle_latent = _latent_fallback_loss(output.vehicle, batch.vehicle)
        physiology_latent = _latent_fallback_loss(output.physiology, batch.physiology)

    vehicle = vehicle_semantic + vehicle_smoothness + vehicle_envelope + vehicle_latent
    physiology = physiology_smoothness + physiology_envelope + physiology_pairwise + physiology_spo2_delta + physiology_latent
    return PhysicsLossBreakdown(
        vehicle_semantic=vehicle_semantic,
        vehicle_smoothness=vehicle_smoothness,
        vehicle_envelope=vehicle_envelope,
        vehicle_latent=vehicle_latent,
        physiology_smoothness=physiology_smoothness,
        physiology_envelope=physiology_envelope,
        physiology_pairwise=physiology_pairwise,
        physiology_spo2_delta=physiology_spo2_delta,
        physiology_latent=physiology_latent,
        vehicle=vehicle,
        physiology=physiology,
        total=vehicle + physiology,
    )


def _vehicle_semantic_residual_loss(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    groups: StageFVehicleFeatureGroups,
    *,
    huber_delta: float,
) -> torch.Tensor | None:
    losses: list[torch.Tensor] = []
    for source_features, target_features in (
        (groups.speed, groups.acceleration),
        (groups.altitude, groups.vertical_speed),
        (groups.attitude, groups.angular_rate),
    ):
        losses.extend(
            _derivative_residual_losses(
                values,
                times_s,
                valid_mask,
                feature_names,
                source_features=source_features,
                target_features=target_features,
                huber_delta=huber_delta,
            )
        )
    if not losses:
        return None
    return _mean_losses(losses)


def _derivative_residual_losses(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    *,
    source_features: tuple[str, ...],
    target_features: tuple[str, ...],
    huber_delta: float,
) -> tuple[torch.Tensor, ...]:
    source_indices = _indices_for_feature_names(feature_names, source_features)
    target_indices = _indices_for_feature_names(feature_names, target_features)
    if not source_indices or not target_indices:
        return ()
    source_series, source_valid = _aggregate_selected_features(values, valid_mask, source_indices)
    target_series, target_valid = _aggregate_selected_features(values, valid_mask, target_indices)
    derivative, derivative_valid = _first_derivative(source_series, times_s, source_valid)
    residual_valid = derivative_valid & target_valid[:, 1:]
    if not bool(torch.any(residual_valid)):
        return ()
    return (_masked_huber_loss(derivative - target_series[:, 1:], residual_valid, delta=huber_delta),)


def _selected_feature_smoothness_loss(
    values: torch.Tensor,
    times_s: torch.Tensor,
    point_valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    selected_features: tuple[str, ...],
) -> torch.Tensor:
    selected = _indices_for_feature_names(feature_names, selected_features)
    if not selected:
        return values.new_zeros(())
    return _second_derivative_smoothness_loss(values[..., selected], times_s, point_valid_mask)


def _selected_first_derivative_loss(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    selected_features: tuple[str, ...],
    *,
    huber_delta: float,
) -> torch.Tensor:
    selected = _indices_for_feature_names(feature_names, selected_features)
    if not selected:
        return values.new_zeros(())
    series, series_valid = _aggregate_selected_features(values, valid_mask, selected)
    derivative, derivative_valid = _first_derivative(series, times_s, series_valid)
    if not bool(torch.any(derivative_valid)):
        return values.new_zeros(())
    return _masked_huber_loss(derivative, derivative_valid, delta=huber_delta)


def _pairwise_feature_consistency_loss(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    pairs: tuple[tuple[str, str], ...],
    *,
    huber_delta: float,
) -> torch.Tensor:
    feature_index = {name: index for index, name in enumerate(feature_names)}
    losses: list[torch.Tensor] = []
    for left, right in pairs:
        left_index = feature_index.get(left)
        right_index = feature_index.get(right)
        if left_index is None or right_index is None:
            continue
        pair_valid = valid_mask[..., left_index] & valid_mask[..., right_index]
        if bool(torch.any(pair_valid)):
            losses.append(_masked_huber_loss(values[..., left_index] - values[..., right_index], pair_valid, delta=huber_delta))
    if not losses:
        return values.new_zeros(())
    return _mean_losses(losses)


def _feature_envelope_penalty(
    reconstructed_values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> torch.Tensor:
    if lower is None or upper is None:
        return reconstructed_values.new_zeros(())
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower/upper must be 1D vectors.")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")
    if lower.shape[0] != reconstructed_values.shape[-1]:
        raise ValueError("Envelope vector size must match feature dimension.")
    if torch.any(upper < lower):
        raise ValueError("Envelope upper bounds must be >= lower bounds.")
    lower_view = lower.to(device=reconstructed_values.device, dtype=reconstructed_values.dtype).view(1, 1, -1)
    upper_view = upper.to(device=reconstructed_values.device, dtype=reconstructed_values.dtype).view(1, 1, -1)
    envelope_violation = torch.relu(reconstructed_values - upper_view) + torch.relu(lower_view - reconstructed_values)
    weighted_mask = valid_mask.to(dtype=reconstructed_values.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return reconstructed_values.new_zeros(())
    return (envelope_violation * weighted_mask).sum() / valid_count


def _latent_fallback_loss(output: StreamPrototypeOutput, batch: TorchAlignmentStreamBatch) -> torch.Tensor:
    latent_states, latent_times, latent_valid = _resolve_physics_stream_states(output, batch)
    return _second_derivative_smoothness_loss(latent_states, latent_times, latent_valid)


def _resolve_physics_stream_states(
    output: StreamPrototypeOutput,
    stream_batch: TorchAlignmentStreamBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if output.reference_projected_states is not None and output.reference_offsets_s is not None:
        return (
            output.reference_projected_states,
            output.reference_offsets_s,
            torch.ones(output.reference_offsets_s.shape, dtype=torch.bool, device=output.reference_offsets_s.device),
        )
    return output.projected_states, stream_batch.offsets_s, stream_batch.mask


def _aggregate_selected_features(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, T, F].")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask must match values shape.")
    if not selected_indices:
        batch_size, point_count, _ = values.shape
        return values.new_zeros((batch_size, point_count)), torch.zeros((batch_size, point_count), dtype=torch.bool, device=values.device)
    selected_values = values[..., selected_indices]
    selected_valid = valid_mask[..., selected_indices]
    weights = selected_valid.to(dtype=values.dtype)
    valid_count = weights.sum(dim=-1)
    aggregated = (selected_values * weights).sum(dim=-1) / torch.clamp(valid_count, min=1.0)
    has_valid = valid_count > 0
    return torch.where(has_valid, aggregated, torch.zeros_like(aggregated)), has_valid


def _first_derivative(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("values must have shape [B, T].")
    if times_s.shape != values.shape:
        raise ValueError("times_s must match values shape.")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask must match values shape.")
    delta_value = values[:, 1:] - values[:, :-1]
    delta_time = times_s[:, 1:] - times_s[:, :-1]
    derivative = delta_value / torch.clamp(delta_time, min=epsilon)
    derivative_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & (delta_time > 0)
    return derivative, derivative_valid


def _second_derivative_smoothness_loss(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, T, C].")
    if times_s.ndim != 2:
        raise ValueError("times_s must have shape [B, T].")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [B, T].")
    if values.shape[:2] != times_s.shape or values.shape[:2] != valid_mask.shape:
        raise ValueError("values/times_s/valid_mask must share [B, T] dimensions.")
    if values.shape[1] < 3:
        return values.new_zeros(())
    dt_first = times_s[:, 1:] - times_s[:, :-1]
    velocity = (values[:, 1:, :] - values[:, :-1, :]) / torch.clamp(dt_first, min=epsilon).unsqueeze(-1)
    segment_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & (dt_first > 0)
    dt_second = times_s[:, 2:] - times_s[:, 1:-1]
    second_derivative = (velocity[:, 1:, :] - velocity[:, :-1, :]) / torch.clamp(dt_second, min=epsilon).unsqueeze(-1)
    second_valid = segment_valid[:, 1:] & segment_valid[:, :-1] & (dt_second > 0)
    expanded_valid = second_valid.unsqueeze(-1).expand_as(second_derivative)
    return _masked_mean_squared_error(second_derivative, torch.zeros_like(second_derivative), expanded_valid)


def _masked_mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if valid_mask.shape != predictions.shape:
        raise ValueError("valid_mask must match predictions shape.")
    weighted_mask = valid_mask.to(dtype=predictions.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return predictions.new_zeros(())
    return (((predictions - targets) ** 2) * weighted_mask).sum() / valid_count


def _masked_huber_loss(errors: torch.Tensor, valid_mask: torch.Tensor, *, delta: float) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if errors.shape != valid_mask.shape:
        raise ValueError("valid_mask must match errors shape.")
    weighted_mask = valid_mask.to(dtype=errors.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return errors.new_zeros(())
    abs_error = errors.abs()
    quadratic = torch.clamp(abs_error, max=delta)
    linear = abs_error - quadratic
    huber = 0.5 * (quadratic**2) + (delta * linear)
    return (huber * weighted_mask).sum() / valid_count


def _denormalize_if_available(values: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    if mean is None or std is None:
        return values
    if mean.shape != std.shape or mean.ndim != 1:
        raise ValueError("denormalization mean/std must be matching 1D vectors.")
    if mean.shape[0] != values.shape[-1]:
        raise ValueError("denormalization vectors must match feature dimension.")
    return values * std.to(device=values.device, dtype=values.dtype).view(1, 1, -1) + mean.to(
        device=values.device,
        dtype=values.dtype,
    ).view(1, 1, -1)


def _indices_for_feature_names(feature_names: tuple[str, ...], selected_names: tuple[str, ...]) -> tuple[int, ...]:
    wanted = set(selected_names)
    return tuple(index for index, feature_name in enumerate(feature_names) if feature_name in wanted)


def _mean_losses(losses: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not losses:
        raise ValueError("losses must not be empty.")
    return torch.stack(tuple(losses)).mean()


def _validate_mode(mode: str) -> None:
    if mode not in _ALLOWED_PHYSICS_CONSTRAINT_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_PHYSICS_CONSTRAINT_MODES)!r}.")


def _validate_family(family: str) -> None:
    if family not in _ALLOWED_PHYSICS_FAMILIES:
        raise ValueError(f"family must be one of {sorted(_ALLOWED_PHYSICS_FAMILIES)!r}.")


def _validate_huber_delta(huber_delta: float) -> None:
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive.")
