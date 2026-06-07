"""Rigid-body residual builders for Stage F vehicle constraints."""

from __future__ import annotations

import torch

from chronaris.models.alignment.physics_state_mapping import RigidBodyStateMapping


def build_rigid_body_vehicle_residuals(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    mapping: RigidBodyStateMapping,
    *,
    huber_delta: float,
) -> dict[str, torch.Tensor]:
    """Compute rigid-body vehicle residuals on feature-space trajectories."""

    translation = _derivative_residual(
        values,
        times_s,
        valid_mask,
        feature_names,
        source_features=mapping.speed,
        target_features=mapping.acceleration,
        huber_delta=huber_delta,
    )
    vertical = _derivative_residual(
        values,
        times_s,
        valid_mask,
        feature_names,
        source_features=mapping.altitude,
        target_features=mapping.vertical_speed,
        huber_delta=huber_delta,
    )
    rotation_losses = [
        _derivative_residual(
            values,
            times_s,
            valid_mask,
            feature_names,
            source_features=attitude,
            target_features=rate,
            huber_delta=huber_delta,
        )
        for attitude, rate in (
            (mapping.pitch, mapping.pitch_rate),
            (mapping.roll, mapping.roll_rate),
            (mapping.yaw, mapping.yaw_rate),
        )
    ]
    rotation = _mean_losses([loss for loss in rotation_losses if loss is not None], reference=values)
    return {
        "vehicle_rigid_body_translation": translation or values.new_zeros(()),
        "vehicle_rigid_body_vertical": vertical or values.new_zeros(()),
        "vehicle_rigid_body_rotation": rotation,
    }


def _derivative_residual(
    values: torch.Tensor,
    times_s: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_names: tuple[str, ...],
    *,
    source_features: tuple[str, ...],
    target_features: tuple[str, ...],
    huber_delta: float,
) -> torch.Tensor | None:
    source_indices = _indices_for_feature_names(feature_names, source_features)
    target_indices = _indices_for_feature_names(feature_names, target_features)
    if not source_indices or not target_indices:
        return None
    source_series, source_valid = _aggregate_selected_features(values, valid_mask, source_indices)
    target_series, target_valid = _aggregate_selected_features(values, valid_mask, target_indices)
    derivative, derivative_valid = _first_derivative(source_series, times_s, source_valid)
    residual_valid = derivative_valid & target_valid[:, 1:]
    if not bool(torch.any(residual_valid)):
        return None
    return _masked_huber_loss(derivative - target_series[:, 1:], residual_valid, delta=huber_delta)


def _indices_for_feature_names(
    feature_names: tuple[str, ...],
    selected_features: tuple[str, ...],
) -> tuple[int, ...]:
    feature_index = {name: index for index, name in enumerate(feature_names)}
    return tuple(feature_index[name] for name in selected_features if name in feature_index)


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


def _masked_huber_loss(
    errors: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
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


def _mean_losses(losses: list[torch.Tensor], *, reference: torch.Tensor) -> torch.Tensor:
    if not losses:
        return reference.new_zeros(())
    stacked = torch.stack(losses)
    return stacked.mean()
