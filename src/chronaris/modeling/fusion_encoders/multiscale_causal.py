"""Seconds-based multi-scale causal fusion for Chronaris continuous states."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_LAG_RANGES_S = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0))


@dataclass(frozen=True, slots=True)
class MultiScaleCausalFusionConfig:
    hidden_dim: int = 64
    output_dim: int = 64
    lag_ranges_s: tuple[tuple[float, float], ...] = DEFAULT_LAG_RANGES_S
    use_causal_mask: bool = True
    use_scale_gate: bool = True
    attention_temperature: float = 1.0
    boundary_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("fusion hidden/output dimensions must be positive")
        if not self.lag_ranges_s:
            raise ValueError("at least one lag range is required")
        previous_upper: float | None = None
        for lower, upper in self.lag_ranges_s:
            if lower < 0 or upper <= lower:
                raise ValueError("lag ranges must satisfy 0 <= lower < upper")
            if previous_upper is not None and abs(lower - previous_upper) > 1e-9:
                raise ValueError("lag ranges must be contiguous and ordered")
            previous_upper = upper
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if self.boundary_epsilon_s < 0:
            raise ValueError("boundary_epsilon_s must be non-negative")
        if not self.use_scale_gate and len(self.lag_ranges_s) != 1:
            raise ValueError("gate-free fusion requires exactly one lag range")


@dataclass(frozen=True, slots=True)
class MultiScaleCausalFusionInput:
    physiology_states: torch.Tensor
    vehicle_states: torch.Tensor
    physiology_valid_mask: torch.Tensor
    vehicle_valid_mask: torch.Tensor
    query_timestamps_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class MultiScaleCausalFusionOutput:
    sequence_embedding: torch.Tensor
    attended_vehicle_states: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...]
    lag_masks: tuple[torch.Tensor, ...]
    scale_available_mask: torch.Tensor
    scale_gate_weights: torch.Tensor
    modality_available_mask: torch.Tensor


class MultiScaleCausalLagFusion(nn.Module):
    """Fuse physiology queries with real-seconds vehicle history windows."""

    def __init__(self, config: MultiScaleCausalFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiScaleCausalFusionConfig()
        scale_count = len(self.config.lag_ranges_s)
        gate_input_dim = self.config.hidden_dim * (2 + scale_count)
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, scale_count),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim * 4),
            nn.Linear(self.config.hidden_dim * 4, self.config.output_dim),
        )

    def forward(
        self,
        inputs: MultiScaleCausalFusionInput,
    ) -> MultiScaleCausalFusionOutput:
        _validate_inputs(inputs, hidden_dim=self.config.hidden_dim)
        query_states = F.normalize(inputs.physiology_states, dim=-1, eps=1e-12)
        key_states = F.normalize(inputs.vehicle_states, dim=-1, eps=1e-12)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        scores = scores / (
            math.sqrt(self.config.hidden_dim) * self.config.attention_temperature
        )
        lag_masks = tuple(
            build_seconds_lag_mask(
                inputs.query_timestamps_s,
                inputs.query_timestamps_s,
                query_valid_mask=inputs.physiology_valid_mask,
                key_valid_mask=inputs.vehicle_valid_mask,
                lower_s=lower,
                upper_s=upper,
                range_index=index,
                use_causal_mask=self.config.use_causal_mask,
                epsilon_s=self.config.boundary_epsilon_s,
            )
            for index, (lower, upper) in enumerate(self.config.lag_ranges_s)
        )
        attention_weights = tuple(
            _masked_attention_weights(scores, mask) for mask in lag_masks
        )
        scale_contexts = torch.stack(
            tuple(
                torch.matmul(weights, inputs.vehicle_states)
                for weights in attention_weights
            ),
            dim=2,
        )
        scale_available = torch.stack(
            tuple(mask.any(dim=-1) for mask in lag_masks),
            dim=-1,
        )
        gate_inputs = torch.cat(
            (
                inputs.physiology_states,
                inputs.vehicle_states,
                *tuple(
                    scale_contexts[:, :, index]
                    for index in range(scale_contexts.shape[2])
                ),
            ),
            dim=-1,
        )
        if self.config.use_scale_gate:
            gate_logits = self.scale_gate(gate_inputs)
            scale_weights = _masked_scale_weights(gate_logits, scale_available)
        else:
            scale_weights = scale_available.to(scale_contexts.dtype)
        attended = (scale_contexts * scale_weights.unsqueeze(-1)).sum(dim=2)
        merged = torch.cat(
            (
                inputs.physiology_states,
                inputs.vehicle_states,
                attended,
                inputs.physiology_states - attended,
            ),
            dim=-1,
        )
        available = inputs.physiology_valid_mask | inputs.vehicle_valid_mask
        sequence = torch.nan_to_num(self.output_projection(merged))
        sequence = sequence * available.unsqueeze(-1).to(sequence.dtype)
        return MultiScaleCausalFusionOutput(
            sequence_embedding=sequence,
            attended_vehicle_states=attended,
            attention_weights=attention_weights,
            lag_masks=lag_masks,
            scale_available_mask=scale_available,
            scale_gate_weights=scale_weights,
            modality_available_mask=available,
        )


def build_seconds_lag_mask(
    query_timestamps_s: torch.Tensor,
    key_timestamps_s: torch.Tensor,
    *,
    query_valid_mask: torch.Tensor,
    key_valid_mask: torch.Tensor,
    lower_s: float,
    upper_s: float,
    range_index: int,
    use_causal_mask: bool,
    epsilon_s: float = 1e-6,
) -> torch.Tensor:
    """Assign each time difference to one non-overlapping seconds interval."""

    if lower_s < 0 or upper_s <= lower_s or range_index < 0 or epsilon_s < 0:
        raise ValueError("invalid seconds lag mask configuration")
    if query_timestamps_s.ndim != 2 or key_timestamps_s.ndim != 2:
        raise ValueError("timestamp tensors must have shape [B,T]")
    if query_timestamps_s.shape[0] != key_timestamps_s.shape[0]:
        raise ValueError("timestamp tensors must share batch size")
    if query_valid_mask.shape != query_timestamps_s.shape:
        raise ValueError("query valid mask shape mismatch")
    if key_valid_mask.shape != key_timestamps_s.shape:
        raise ValueError("key valid mask shape mismatch")
    signed_delta = query_timestamps_s.unsqueeze(-1) - key_timestamps_s.unsqueeze(1)
    delta = signed_delta if use_causal_mask else signed_delta.abs()
    lower_ok = (
        delta >= (lower_s - epsilon_s)
        if range_index == 0
        else delta > (lower_s + epsilon_s)
    )
    upper_ok = delta <= (upper_s + epsilon_s)
    temporal = lower_ok & upper_ok
    if use_causal_mask:
        temporal &= signed_delta >= -epsilon_s
    return temporal & query_valid_mask.unsqueeze(-1) & key_valid_mask.unsqueeze(1)


def _masked_attention_weights(
    scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    available = mask.any(dim=-1, keepdim=True)
    safe_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    safe_scores = torch.where(available, safe_scores, torch.zeros_like(safe_scores))
    weights = torch.softmax(safe_scores, dim=-1).masked_fill(~mask, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )


def _masked_scale_weights(
    logits: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    any_available = available.any(dim=-1, keepdim=True)
    masked = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    safe = torch.where(any_available, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=-1).masked_fill(~available, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )


def _validate_inputs(
    inputs: MultiScaleCausalFusionInput,
    *,
    hidden_dim: int,
) -> None:
    physiology = inputs.physiology_states
    vehicle = inputs.vehicle_states
    if physiology.ndim != 3 or vehicle.shape != physiology.shape:
        raise ValueError("continuous states must share shape [B,T,H]")
    if physiology.shape[-1] != hidden_dim:
        raise ValueError("continuous state hidden dimension mismatch")
    if inputs.physiology_valid_mask.shape != physiology.shape[:2]:
        raise ValueError("physiology valid mask shape mismatch")
    if inputs.vehicle_valid_mask.shape != vehicle.shape[:2]:
        raise ValueError("vehicle valid mask shape mismatch")
    if inputs.query_timestamps_s.shape != physiology.shape[:2]:
        raise ValueError("query timestamp shape mismatch")
    if (
        inputs.physiology_valid_mask.dtype != torch.bool
        or inputs.vehicle_valid_mask.dtype != torch.bool
    ):
        raise ValueError("continuous state masks must be boolean")
