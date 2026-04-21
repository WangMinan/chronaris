"""Loss functions for Stage E baseline and Stage F(min) physics constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as F

from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch

if TYPE_CHECKING:
    from chronaris.models.alignment.prototype import DualStreamPrototypeOutput, StreamPrototypeOutput
else:
    DualStreamPrototypeOutput = Any
    StreamPrototypeOutput = Any


_VEHICLE_SPEED_TOKENS = (
    "speed",
    "groundspeed",
    "tas",
    "ias",
    "mach",
    "vx",
    "vy",
    "vz",
    "vel",
)
_VEHICLE_ACCEL_TOKENS = (
    "acc",
    "ax",
    "ay",
    "az",
    "nx",
    "ny",
    "nz",
)
_ALLOWED_PHYSICS_CONSTRAINT_MODES = {
    "feature_first_with_latent_fallback",
    "feature_only",
    "latent_only",
}


@dataclass(frozen=True, slots=True)
class ReconstructionLossBreakdown:
    """Per-stream reconstruction losses for the Stage E prototype."""

    physiology: torch.Tensor
    vehicle: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True, slots=True)
class AlignmentLossBreakdown:
    """Alignment loss summary on the shared reference grid."""

    alignment: torch.Tensor
    mode: str


@dataclass(frozen=True, slots=True)
class PhysicsLossBreakdown:
    """Physics-constraint loss summary for Stage F(min)."""

    vehicle: torch.Tensor
    physiology: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True, slots=True)
class StageEObjectiveBreakdown:
    """Combined reconstruction + alignment (+ optional physics) objective summary."""

    physiology_reconstruction: torch.Tensor
    vehicle_reconstruction: torch.Tensor
    reconstruction_total: torch.Tensor
    alignment: torch.Tensor
    vehicle_physics: torch.Tensor
    physiology_physics: torch.Tensor
    physics_total: torch.Tensor
    total: torch.Tensor


def masked_mean_squared_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only on positions marked valid."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")
    if valid_mask.shape != predictions.shape:
        raise ValueError("valid_mask must match predictions shape.")

    weighted_mask = valid_mask.to(dtype=predictions.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return predictions.new_zeros(())

    squared_error = (predictions - targets) ** 2
    return (squared_error * weighted_mask).sum() / valid_count


def _masked_mean_square(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean square magnitude on positions marked valid."""

    if values.shape != valid_mask.shape:
        raise ValueError("valid_mask must match values shape.")
    weighted_mask = valid_mask.to(dtype=values.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return values.new_zeros(())
    return ((values**2) * weighted_mask).sum() / valid_count


def _masked_huber_loss(
    errors: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Compute Huber loss on positions marked valid."""

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


def _select_feature_indices(feature_names: tuple[str, ...], tokens: tuple[str, ...]) -> tuple[int, ...]:
    normalized_tokens = tuple(token.lower() for token in tokens)
    selected: list[int] = []
    for feature_index, feature_name in enumerate(feature_names):
        normalized_feature = feature_name.lower()
        if any(token in normalized_feature for token in normalized_tokens):
            selected.append(feature_index)
    return tuple(selected)


def _aggregate_selected_features(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate selected feature channels into one scalar series."""

    if values.ndim != 3:
        raise ValueError("values must have shape [B, T, F].")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask must match values shape.")
    if not selected_indices:
        batch_size, point_count, _ = values.shape
        return values.new_zeros((batch_size, point_count)), torch.zeros(
            (batch_size, point_count),
            dtype=torch.bool,
            device=values.device,
        )

    selected_values = values[..., selected_indices]
    selected_valid = valid_mask[..., selected_indices]
    weights = selected_valid.to(dtype=values.dtype)
    valid_count = weights.sum(dim=-1)
    safe_count = torch.clamp(valid_count, min=1.0)
    aggregated = (selected_values * weights).sum(dim=-1) / safe_count
    has_valid = valid_count > 0
    aggregated = torch.where(has_valid, aggregated, torch.zeros_like(aggregated))
    return aggregated, has_valid


def _first_derivative(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute first derivative using adjacent finite differences."""

    if values.ndim != 2:
        raise ValueError("values must have shape [B, T].")
    if times_s.shape != values.shape:
        raise ValueError("times_s must match values shape.")
    if valid_mask.shape != values.shape:
        raise ValueError("valid_mask must match values shape.")

    delta_value = values[:, 1:] - values[:, :-1]
    delta_time = times_s[:, 1:] - times_s[:, :-1]
    delta_time_safe = torch.clamp(delta_time, min=epsilon)
    derivative = delta_value / delta_time_safe
    derivative_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & (delta_time > 0)
    return derivative, derivative_valid


def _second_derivative_smoothness_loss(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Penalize trajectory curvature with irregular-time finite differences."""

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
    dt_first_safe = torch.clamp(dt_first, min=epsilon)
    segment_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & (dt_first > 0)
    velocity = (values[:, 1:, :] - values[:, :-1, :]) / dt_first_safe.unsqueeze(-1)

    dt_second = times_s[:, 2:] - times_s[:, 1:-1]
    dt_second_safe = torch.clamp(dt_second, min=epsilon)
    second_derivative = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt_second_safe.unsqueeze(-1)
    second_valid = segment_valid[:, 1:] & segment_valid[:, :-1] & (dt_second > 0)
    expanded_valid = second_valid.unsqueeze(-1).expand_as(second_derivative)
    return masked_mean_squared_error(
        second_derivative,
        torch.zeros_like(second_derivative),
        expanded_valid,
    )


def _resolve_physics_stream_states(
    output: StreamPrototypeOutput,
    stream_batch: TorchAlignmentStreamBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve states/timestamps/valid mask for latent fallback constraints."""

    if output.reference_projected_states is not None and output.reference_offsets_s is not None:
        return (
            output.reference_projected_states,
            output.reference_offsets_s,
            torch.ones(output.reference_offsets_s.shape, dtype=torch.bool, device=output.reference_offsets_s.device),
        )
    return output.projected_states, stream_batch.offsets_s, stream_batch.mask


def _physiology_envelope_penalty(
    reconstructed_values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    lower: torch.Tensor | None,
    upper: torch.Tensor | None,
) -> torch.Tensor:
    """Penalize values outside train-derived physiology envelopes."""

    if lower is None or upper is None:
        return reconstructed_values.new_zeros(())
    if lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("lower/upper must be 1D vectors.")
    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")
    if lower.shape[0] != reconstructed_values.shape[-1]:
        raise ValueError("Envelope vector size must match physiology feature dimension.")
    if torch.any(upper < lower):
        raise ValueError("Envelope upper bounds must be >= lower bounds.")

    lower_view = lower.to(device=reconstructed_values.device, dtype=reconstructed_values.dtype).view(1, 1, -1)
    upper_view = upper.to(device=reconstructed_values.device, dtype=reconstructed_values.dtype).view(1, 1, -1)
    upper_violation = torch.relu(reconstructed_values - upper_view)
    lower_violation = torch.relu(lower_view - reconstructed_values)
    envelope_violation = upper_violation + lower_violation

    weighted_mask = valid_mask.to(dtype=reconstructed_values.dtype)
    valid_count = weighted_mask.sum()
    if torch.is_nonzero(valid_count <= 0):
        return reconstructed_values.new_zeros(())
    return (envelope_violation * weighted_mask).sum() / valid_count


def stream_reconstruction_loss(
    output: StreamPrototypeOutput,
    stream_batch: TorchAlignmentStreamBatch,
    *,
    mode: str = "mse",
    scale_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute reconstruction MSE for one stream."""

    if scale_epsilon <= 0:
        raise ValueError("scale_epsilon must be positive.")

    valid_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)
    reconstruction_mse = masked_mean_squared_error(
        output.reconstructions,
        stream_batch.values,
        valid_mask,
    )
    if mode == "mse":
        return reconstruction_mse
    if mode == "relative_mse":
        target_mean_square = _masked_mean_square(stream_batch.values, valid_mask)
        scale = torch.clamp(target_mean_square, min=scale_epsilon)
        return reconstruction_mse / scale
    raise ValueError("mode must be either 'mse' or 'relative_mse'.")


def dual_stream_reconstruction_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "mse",
    scale_epsilon: float = 1e-6,
) -> ReconstructionLossBreakdown:
    """Compute reconstruction losses for both streams."""

    physiology = stream_reconstruction_loss(
        output.physiology,
        batch.physiology,
        mode=mode,
        scale_epsilon=scale_epsilon,
    )
    vehicle = stream_reconstruction_loss(
        output.vehicle,
        batch.vehicle,
        mode=mode,
        scale_epsilon=scale_epsilon,
    )
    return ReconstructionLossBreakdown(
        physiology=physiology,
        vehicle=vehicle,
        total=physiology + vehicle,
    )


def projection_alignment_loss(
    physiology_projection: torch.Tensor,
    vehicle_projection: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    mode: str = "mse",
) -> torch.Tensor:
    """Compare two projected trajectories on the shared reference grid."""

    if physiology_projection.shape != vehicle_projection.shape:
        raise ValueError("Projection tensors must have the same shape.")
    if physiology_projection.ndim != 3:
        raise ValueError("Projection tensors must have shape [B, R, P].")
    if valid_mask is None:
        valid_mask = torch.ones(
            physiology_projection.shape[:-1],
            dtype=torch.bool,
            device=physiology_projection.device,
        )
    if valid_mask.shape != physiology_projection.shape[:-1]:
        raise ValueError("valid_mask must have shape [B, R].")

    expanded_mask = valid_mask.unsqueeze(-1).expand_as(physiology_projection)
    if mode == "mse":
        return masked_mean_squared_error(
            physiology_projection,
            vehicle_projection,
            expanded_mask,
        )
    if mode == "cosine":
        cosine = F.cosine_similarity(physiology_projection, vehicle_projection, dim=-1)
        weighted_mask = valid_mask.to(dtype=physiology_projection.dtype)
        valid_count = weighted_mask.sum()
        if torch.is_nonzero(valid_count <= 0):
            return physiology_projection.new_zeros(())
        return ((1.0 - cosine) * weighted_mask).sum() / valid_count
    raise ValueError("mode must be either 'mse' or 'cosine'.")


def dual_stream_alignment_loss(
    output: DualStreamPrototypeOutput,
    *,
    mode: str = "mse",
) -> AlignmentLossBreakdown:
    """Compute the shared-reference alignment loss for the dual-stream prototype."""

    physiology_projection = output.physiology.reference_projected_states
    vehicle_projection = output.vehicle.reference_projected_states
    if physiology_projection is None or vehicle_projection is None:
        raise ValueError("Dual-stream alignment loss requires reference_projected_states for both streams.")

    if output.physiology.reference_offsets_s is None or output.vehicle.reference_offsets_s is None:
        raise ValueError("Dual-stream alignment loss requires reference_offsets_s for both streams.")
    if not torch.allclose(
        output.physiology.reference_offsets_s,
        output.vehicle.reference_offsets_s,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise ValueError("Physiology and vehicle reference grids must match before computing alignment loss.")

    return AlignmentLossBreakdown(
        alignment=projection_alignment_loss(
            physiology_projection,
            vehicle_projection,
            mode=mode,
        ),
        mode=mode,
    )


def vehicle_physics_consistency_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute vehicle-side physics consistency loss."""

    if mode not in _ALLOWED_PHYSICS_CONSTRAINT_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_ALLOWED_PHYSICS_CONSTRAINT_MODES)!r}."
        )
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive.")

    stream_batch = batch.vehicle
    stream_output = output.vehicle
    valid_feature_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)

    use_feature_loss = mode in {"feature_first_with_latent_fallback", "feature_only"}
    if use_feature_loss:
        speed_indices = _select_feature_indices(stream_batch.feature_names, _VEHICLE_SPEED_TOKENS)
        accel_indices = _select_feature_indices(stream_batch.feature_names, _VEHICLE_ACCEL_TOKENS)
        if speed_indices and accel_indices:
            speed_series, speed_valid = _aggregate_selected_features(
                stream_output.reconstructions,
                valid_feature_mask,
                speed_indices,
            )
            accel_series, accel_valid = _aggregate_selected_features(
                stream_output.reconstructions,
                valid_feature_mask,
                accel_indices,
            )
            speed_derivative, derivative_valid = _first_derivative(
                speed_series,
                stream_batch.offsets_s,
                speed_valid,
            )
            accel_target = accel_series[:, 1:]
            residual_valid = derivative_valid & accel_valid[:, 1:]
            if bool(torch.any(residual_valid)):
                return _masked_huber_loss(
                    speed_derivative - accel_target,
                    residual_valid,
                    delta=huber_delta,
                )
            if mode == "feature_only":
                return stream_batch.values.new_zeros(())
        elif mode == "feature_only":
            return stream_batch.values.new_zeros(())

    latent_states, latent_times, latent_valid = _resolve_physics_stream_states(stream_output, stream_batch)
    return _second_derivative_smoothness_loss(
        latent_states,
        latent_times,
        latent_valid,
    )


def physiology_physics_consistency_loss(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    envelope_lower: torch.Tensor | None = None,
    envelope_upper: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute physiology-side smoothness/envelope consistency loss."""

    if mode not in _ALLOWED_PHYSICS_CONSTRAINT_MODES:
        raise ValueError(
            f"mode must be one of {sorted(_ALLOWED_PHYSICS_CONSTRAINT_MODES)!r}."
        )

    stream_batch = batch.physiology
    stream_output = output.physiology
    valid_feature_mask = stream_batch.feature_valid_mask & stream_batch.mask.unsqueeze(-1)

    use_feature_loss = mode in {"feature_first_with_latent_fallback", "feature_only"}
    if use_feature_loss:
        smoothness = _second_derivative_smoothness_loss(
            stream_output.reconstructions,
            stream_batch.offsets_s,
            stream_batch.mask,
        )
        envelope = _physiology_envelope_penalty(
            stream_output.reconstructions,
            valid_feature_mask,
            lower=envelope_lower,
            upper=envelope_upper,
        )
        return smoothness + envelope

    latent_states, latent_times, latent_valid = _resolve_physics_stream_states(stream_output, stream_batch)
    return _second_derivative_smoothness_loss(
        latent_states,
        latent_times,
        latent_valid,
    )


def build_stage_f_physics_losses(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    mode: str = "feature_first_with_latent_fallback",
    huber_delta: float = 1.0,
    physiology_envelope_lower: torch.Tensor | None = None,
    physiology_envelope_upper: torch.Tensor | None = None,
) -> PhysicsLossBreakdown:
    """Build minimal Stage F physics-constraint losses."""

    vehicle = vehicle_physics_consistency_loss(
        output,
        batch,
        mode=mode,
        huber_delta=huber_delta,
    )
    physiology = physiology_physics_consistency_loss(
        output,
        batch,
        mode=mode,
        envelope_lower=physiology_envelope_lower,
        envelope_upper=physiology_envelope_upper,
    )
    return PhysicsLossBreakdown(
        vehicle=vehicle,
        physiology=physiology,
        total=vehicle + physiology,
    )


def build_stage_e_objective(
    output: DualStreamPrototypeOutput,
    batch: TorchAlignmentBatch,
    *,
    reconstruction_mode: str = "mse",
    reconstruction_scale_epsilon: float = 1e-6,
    alignment_mode: str = "mse",
    physiology_weight: float = 1.0,
    vehicle_weight: float = 1.0,
    alignment_weight: float = 1.0,
    enable_physics_constraints: bool = False,
    physics_constraint_mode: str = "feature_first_with_latent_fallback",
    vehicle_physics_weight: float = 0.0,
    physiology_physics_weight: float = 0.0,
    physics_huber_delta: float = 1.0,
    physiology_envelope_lower: torch.Tensor | None = None,
    physiology_envelope_upper: torch.Tensor | None = None,
) -> StageEObjectiveBreakdown:
    """Combine reconstruction/alignment losses with optional Stage F(min) constraints."""

    if physiology_weight < 0 or vehicle_weight < 0 or alignment_weight < 0:
        raise ValueError("All objective weights must be non-negative.")
    if reconstruction_scale_epsilon <= 0:
        raise ValueError("reconstruction_scale_epsilon must be positive.")
    if vehicle_physics_weight < 0 or physiology_physics_weight < 0:
        raise ValueError("physics weights must be non-negative.")
    if physics_constraint_mode not in _ALLOWED_PHYSICS_CONSTRAINT_MODES:
        raise ValueError(
            f"physics_constraint_mode must be one of {sorted(_ALLOWED_PHYSICS_CONSTRAINT_MODES)!r}."
        )
    if physics_huber_delta <= 0:
        raise ValueError("physics_huber_delta must be positive.")

    reconstruction = dual_stream_reconstruction_loss(
        output,
        batch,
        mode=reconstruction_mode,
        scale_epsilon=reconstruction_scale_epsilon,
    )
    alignment = dual_stream_alignment_loss(output, mode=alignment_mode)
    if enable_physics_constraints:
        physics = build_stage_f_physics_losses(
            output,
            batch,
            mode=physics_constraint_mode,
            huber_delta=physics_huber_delta,
            physiology_envelope_lower=physiology_envelope_lower,
            physiology_envelope_upper=physiology_envelope_upper,
        )
    else:
        zero = reconstruction.total.new_zeros(())
        physics = PhysicsLossBreakdown(vehicle=zero, physiology=zero, total=zero)

    total = (
        (physiology_weight * reconstruction.physiology)
        + (vehicle_weight * reconstruction.vehicle)
        + (alignment_weight * alignment.alignment)
        + (vehicle_physics_weight * physics.vehicle)
        + (physiology_physics_weight * physics.physiology)
    )
    return StageEObjectiveBreakdown(
        physiology_reconstruction=reconstruction.physiology,
        vehicle_reconstruction=reconstruction.vehicle,
        reconstruction_total=reconstruction.total,
        alignment=alignment.alignment,
        vehicle_physics=physics.vehicle,
        physiology_physics=physics.physiology,
        physics_total=physics.total,
        total=total,
    )
