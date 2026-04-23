"""Stage G minimal causal cross-modal fusion primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CausalFusionConfig:
    """Configuration for the minimal Stage G causal fusion module."""

    attention_temperature: float = 1.0
    event_bias_weight: float = 0.25
    causal_epsilon_s: float = 1e-6
    normalize_states: bool = True

    def __post_init__(self) -> None:
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        if self.event_bias_weight < 0:
            raise ValueError("event_bias_weight must be non-negative.")
        if self.causal_epsilon_s < 0:
            raise ValueError("causal_epsilon_s must be non-negative.")


@dataclass(frozen=True, slots=True)
class CausalFusionTensorInput:
    """Tensor inputs for physiology-query, vehicle-key/value causal fusion."""

    physiology_states: torch.Tensor
    vehicle_states: torch.Tensor
    physiology_offsets_s: torch.Tensor
    vehicle_offsets_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class CausalFusionTensorOutput:
    """Tensor outputs from the minimal Stage G causal fusion module."""

    fused_states: torch.Tensor
    attended_vehicle_states: torch.Tensor
    attention_weights: torch.Tensor
    causal_mask: torch.Tensor
    vehicle_event_scores: torch.Tensor


class CausalMaskedCrossModalFusion(nn.Module):
    """Asymmetric causal attention from vehicle events into physiology states."""

    def __init__(self, config: CausalFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalFusionConfig()

    def forward(self, inputs: CausalFusionTensorInput) -> CausalFusionTensorOutput:
        _validate_fusion_inputs(inputs)
        physiology_states = inputs.physiology_states
        vehicle_states = inputs.vehicle_states

        query_states = _maybe_l2_normalize(physiology_states, enabled=self.config.normalize_states)
        key_states = _maybe_l2_normalize(vehicle_states, enabled=self.config.normalize_states)
        event_scores = compute_vehicle_event_scores(vehicle_states)
        causal_mask = build_causal_attention_mask(
            inputs.physiology_offsets_s,
            inputs.vehicle_offsets_s,
            epsilon_s=self.config.causal_epsilon_s,
        )

        raw_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        raw_scores = raw_scores / self.config.attention_temperature
        if self.config.event_bias_weight:
            raw_scores = raw_scores + (event_scores.unsqueeze(1) * self.config.event_bias_weight)

        masked_scores = raw_scores.masked_fill(~causal_mask, torch.finfo(raw_scores.dtype).min)
        attention_weights = torch.softmax(masked_scores, dim=-1)
        attention_weights = attention_weights.masked_fill(~causal_mask, 0.0)
        attention_weights = _renormalize_attention_weights(attention_weights)

        attended_vehicle_states = torch.matmul(attention_weights, vehicle_states)
        fused_states = torch.cat(
            (
                physiology_states,
                attended_vehicle_states,
                physiology_states - attended_vehicle_states,
            ),
            dim=-1,
        )
        return CausalFusionTensorOutput(
            fused_states=fused_states,
            attended_vehicle_states=attended_vehicle_states,
            attention_weights=attention_weights,
            causal_mask=causal_mask,
            vehicle_event_scores=event_scores,
        )


def build_causal_attention_mask(
    physiology_offsets_s: torch.Tensor,
    vehicle_offsets_s: torch.Tensor,
    *,
    epsilon_s: float = 1e-6,
) -> torch.Tensor:
    """Build a mask where each physiology point sees only past/current vehicle states."""

    if physiology_offsets_s.ndim != 2 or vehicle_offsets_s.ndim != 2:
        raise ValueError("offset tensors must have shape [B, R].")
    if physiology_offsets_s.shape[0] != vehicle_offsets_s.shape[0]:
        raise ValueError("offset tensors must share the same batch size.")
    if epsilon_s < 0:
        raise ValueError("epsilon_s must be non-negative.")

    mask = vehicle_offsets_s.unsqueeze(1) <= (physiology_offsets_s.unsqueeze(-1) + epsilon_s)
    if not bool(mask.all(dim=-1).all()):
        empty_rows = ~mask.any(dim=-1)
        if bool(empty_rows.any()):
            first_vehicle = torch.zeros_like(mask)
            first_vehicle[:, :, 0] = True
            mask = torch.where(empty_rows.unsqueeze(-1), first_vehicle, mask)
    return mask


def compute_vehicle_event_scores(vehicle_states: torch.Tensor) -> torch.Tensor:
    """Compute event salience from adjacent vehicle reference-state changes."""

    if vehicle_states.ndim != 3:
        raise ValueError("vehicle_states must have shape [B, R, D].")
    batch_size, point_count, _ = vehicle_states.shape
    if point_count == 0:
        raise ValueError("vehicle_states must include at least one reference point.")

    event_scores = vehicle_states.new_zeros((batch_size, point_count))
    if point_count > 1:
        deltas = vehicle_states[:, 1:, :] - vehicle_states[:, :-1, :]
        event_scores[:, 1:] = torch.linalg.vector_norm(deltas, dim=-1)

    max_scores = event_scores.max(dim=-1, keepdim=True).values
    safe_max_scores = torch.clamp(max_scores, min=torch.finfo(vehicle_states.dtype).eps)
    return torch.where(max_scores > 0, event_scores / safe_max_scores, event_scores)


def attention_entropy(attention_weights: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
    """Return per-query normalized entropy over causally visible vehicle states."""

    if attention_weights.shape != causal_mask.shape:
        raise ValueError("attention_weights and causal_mask must have the same shape.")
    visible_count = causal_mask.sum(dim=-1).to(dtype=attention_weights.dtype)
    safe_visible_count = torch.clamp(visible_count, min=1.0)
    log_denominator = torch.log(safe_visible_count)
    raw_entropy = -(attention_weights * torch.log(torch.clamp(attention_weights, min=1e-12))).sum(dim=-1)
    normalized_entropy = torch.where(
        visible_count > 1,
        raw_entropy / torch.clamp(log_denominator, min=1e-12),
        torch.zeros_like(raw_entropy),
    )
    return normalized_entropy


def _validate_fusion_inputs(inputs: CausalFusionTensorInput) -> None:
    physiology = inputs.physiology_states
    vehicle = inputs.vehicle_states
    if physiology.ndim != 3 or vehicle.ndim != 3:
        raise ValueError("state tensors must have shape [B, R, D].")
    if physiology.shape[0] != vehicle.shape[0]:
        raise ValueError("state tensors must share the same batch size.")
    if physiology.shape[-1] != vehicle.shape[-1]:
        raise ValueError("state tensors must share the same feature dimension.")
    if physiology.shape[1] == 0 or vehicle.shape[1] == 0:
        raise ValueError("state tensors must include at least one reference point.")
    if inputs.physiology_offsets_s.shape != physiology.shape[:2]:
        raise ValueError("physiology_offsets_s must match physiology state shape [B, R].")
    if inputs.vehicle_offsets_s.shape != vehicle.shape[:2]:
        raise ValueError("vehicle_offsets_s must match vehicle state shape [B, R].")


def _maybe_l2_normalize(states: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return states
    return F.normalize(states, p=2, dim=-1, eps=1e-12)


def _renormalize_attention_weights(attention_weights: torch.Tensor) -> torch.Tensor:
    denominator = attention_weights.sum(dim=-1, keepdim=True)
    safe_denominator = torch.clamp(denominator, min=torch.finfo(attention_weights.dtype).eps)
    return attention_weights / safe_denominator
