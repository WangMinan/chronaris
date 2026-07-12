"""Learnable multi-head relative-time causal fusion for Chronaris v2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.multiscale_causal import (
    build_seconds_lag_mask,
)


FIVE_LAG_RANGES_S = (
    (0.0, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
    (20.0, 30.0),
)


@dataclass(frozen=True, slots=True)
class LearnedRelativeCausalFusionConfig:
    physiology_dim: int
    vehicle_dim: int
    attention_dim: int
    output_dim: int = 24
    num_heads: int = 4
    lag_mode: str = "fixed_five"
    lag_ranges_s: tuple[tuple[float, float], ...] = FIVE_LAG_RANGES_S
    dropout: float = 0.1
    maximum_lag_s: float = 30.0

    def __post_init__(self) -> None:
        if min(
            self.physiology_dim,
            self.vehicle_dim,
            self.attention_dim,
            self.output_dim,
            self.num_heads,
        ) <= 0:
            raise ValueError("learned fusion dimensions must be positive")
        if self.attention_dim % self.num_heads:
            raise ValueError("attention_dim must be divisible by num_heads")
        if self.lag_mode not in {"fixed_five", "continuous_basis"}:
            raise ValueError("unsupported lag_mode")
        if len(self.lag_ranges_s) != 5:
            raise ValueError("Chronaris v2 requires five lag components")
        if not 0 <= self.dropout < 1 or self.maximum_lag_s <= 0:
            raise ValueError("learned fusion dropout/lag configuration is invalid")


@dataclass(frozen=True, slots=True)
class LearnedRelativeCausalFusionInput:
    physiology_states: torch.Tensor
    vehicle_states: torch.Tensor
    physiology_valid_mask: torch.Tensor
    vehicle_valid_mask: torch.Tensor
    query_timestamps_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class LearnedRelativeCausalFusionOutput:
    shared_embedding: torch.Tensor
    attended_vehicle_states: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...]
    lag_masks: tuple[torch.Tensor, ...]
    scale_available_mask: torch.Tensor
    scale_gate_weights: torch.Tensor
    modality_available_mask: torch.Tensor
    learned_lag_centers_s: torch.Tensor
    learned_lag_bandwidths_s: torch.Tensor


class LearnedRelativeCausalFusion(nn.Module):
    """Physiology-query multi-head attention over historical vehicle states."""

    def __init__(self, config: LearnedRelativeCausalFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Linear(config.physiology_dim, config.attention_dim)
        self.key_projection = nn.Linear(config.vehicle_dim, config.attention_dim)
        self.value_projection = nn.Linear(config.vehicle_dim, config.attention_dim)
        self.current_vehicle_projection = nn.Linear(
            config.vehicle_dim,
            config.attention_dim,
        )
        self.current_physiology_projection = nn.Linear(
            config.physiology_dim,
            config.attention_dim,
        )
        self.physiology_missing_token = nn.Parameter(torch.zeros(config.physiology_dim))
        self.vehicle_missing_token = nn.Parameter(torch.zeros(config.vehicle_dim))
        self.relative_time_bias = nn.Sequential(
            nn.Linear(1, config.num_heads),
            nn.Tanh(),
            nn.Linear(config.num_heads, config.num_heads),
        )
        centers = torch.tensor((1.0, 3.5, 7.5, 15.0, 25.0))
        bandwidths = torch.tensor((1.0, 1.5, 2.5, 5.0, 5.0))
        self.lag_centers_unconstrained = nn.Parameter(_inverse_softplus(centers))
        self.lag_bandwidths_unconstrained = nn.Parameter(_inverse_softplus(bandwidths))
        gate_dim = config.attention_dim * (2 + len(config.lag_ranges_s))
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, len(config.lag_ranges_s)),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.attention_dim * 3),
            nn.Linear(config.attention_dim * 3, config.output_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        inputs: LearnedRelativeCausalFusionInput,
    ) -> LearnedRelativeCausalFusionOutput:
        _validate_inputs(inputs, self.config)
        query_valid = inputs.physiology_valid_mask | inputs.vehicle_valid_mask
        physiology = torch.where(
            inputs.physiology_valid_mask.unsqueeze(-1),
            inputs.physiology_states,
            self.physiology_missing_token.view(1, 1, -1),
        )
        vehicle = torch.where(
            inputs.vehicle_valid_mask.unsqueeze(-1),
            inputs.vehicle_states,
            self.vehicle_missing_token.view(1, 1, -1),
        )
        queries = self._split_heads(self.query_projection(physiology))
        keys = self._split_heads(self.key_projection(vehicle))
        values = self._split_heads(self.value_projection(vehicle))
        scores = torch.einsum("bhtd,bhsd->bhts", queries, keys)
        scores = scores / math.sqrt(self.config.attention_dim // self.config.num_heads)
        signed_delta = (
            inputs.query_timestamps_s.unsqueeze(-1)
            - inputs.query_timestamps_s.unsqueeze(1)
        )
        relative_input = (
            torch.log1p(signed_delta.clamp(min=0.0, max=self.config.maximum_lag_s))
            / math.log1p(self.config.maximum_lag_s)
        ).unsqueeze(-1)
        time_bias = self.relative_time_bias(relative_input).permute(0, 3, 1, 2)
        scores = scores + time_bias
        masks, basis_biases = self._lag_components(
            signed_delta,
            query_valid=query_valid,
            vehicle_valid=inputs.vehicle_valid_mask,
            timestamps=inputs.query_timestamps_s,
        )
        attention_weights = []
        contexts = []
        for mask, basis_bias in zip(masks, basis_biases, strict=True):
            weights = _masked_multihead_attention(
                scores + basis_bias.unsqueeze(1),
                mask,
            )
            attention_weights.append(weights.mean(dim=1))
            context = torch.einsum("bhts,bhsd->bhtd", weights, values)
            contexts.append(self._merge_heads(context))
        stacked_contexts = torch.stack(contexts, dim=2)
        scale_available = torch.stack(
            tuple(mask.any(dim=-1) for mask in masks),
            dim=-1,
        )
        current_physiology = self.current_physiology_projection(physiology)
        current_vehicle = self.current_vehicle_projection(vehicle)
        gate_input = torch.cat(
            (
                current_physiology,
                current_vehicle,
                *tuple(
                    stacked_contexts[:, :, index]
                    for index in range(stacked_contexts.shape[2])
                ),
            ),
            dim=-1,
        )
        gate_weights = _masked_scale_weights(
            self.scale_gate(gate_input),
            scale_available,
        )
        attended = (stacked_contexts * gate_weights.unsqueeze(-1)).sum(dim=2)
        shared = self.output_projection(
            torch.cat((current_physiology, current_vehicle, attended), dim=-1)
        )
        shared = self.dropout(shared) * query_valid.unsqueeze(-1).to(shared.dtype)
        centers, bandwidths = self._lag_parameters()
        return LearnedRelativeCausalFusionOutput(
            shared_embedding=shared,
            attended_vehicle_states=attended,
            attention_weights=tuple(attention_weights),
            lag_masks=tuple(masks),
            scale_available_mask=scale_available,
            scale_gate_weights=gate_weights,
            modality_available_mask=query_valid,
            learned_lag_centers_s=centers,
            learned_lag_bandwidths_s=bandwidths,
        )

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, points, _ = values.shape
        return values.reshape(
            batch,
            points,
            self.config.num_heads,
            self.config.attention_dim // self.config.num_heads,
        ).permute(0, 2, 1, 3)

    def _merge_heads(self, values: torch.Tensor) -> torch.Tensor:
        return values.permute(0, 2, 1, 3).reshape(
            values.shape[0],
            values.shape[2],
            self.config.attention_dim,
        )

    def _lag_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        centers = torch.nn.functional.softplus(self.lag_centers_unconstrained)
        bandwidths = torch.nn.functional.softplus(
            self.lag_bandwidths_unconstrained
        ).clamp_min(0.1)
        return centers, bandwidths

    def _lag_components(self, signed_delta, *, query_valid, vehicle_valid, timestamps):
        masks = []
        biases = []
        if self.config.lag_mode == "fixed_five":
            for index, (lower, upper) in enumerate(self.config.lag_ranges_s):
                mask = build_seconds_lag_mask(
                    timestamps,
                    timestamps,
                    query_valid_mask=query_valid,
                    key_valid_mask=vehicle_valid,
                    lower_s=lower,
                    upper_s=upper,
                    range_index=index,
                    use_causal_mask=True,
                )
                masks.append(mask)
                biases.append(torch.zeros_like(signed_delta))
        else:
            causal = (
                (signed_delta >= 0)
                & (signed_delta <= self.config.maximum_lag_s)
                & query_valid.unsqueeze(-1)
                & vehicle_valid.unsqueeze(1)
            )
            centers, bandwidths = self._lag_parameters()
            for center, bandwidth in zip(centers, bandwidths, strict=True):
                basis = torch.exp(
                    -0.5 * ((signed_delta - center) / bandwidth).square()
                )
                masks.append(causal)
                biases.append(basis.clamp_min(1e-8).log())
        return tuple(masks), tuple(biases)


def _masked_multihead_attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(1)
    available = expanded_mask.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~expanded_mask, torch.finfo(scores.dtype).min)
    safe = torch.where(available, masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=-1).masked_fill(~expanded_mask, 0.0)
    denominator = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )


def _masked_scale_weights(logits: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
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


def _validate_inputs(inputs, config) -> None:
    physiology = inputs.physiology_states
    vehicle = inputs.vehicle_states
    if physiology.ndim != 3 or vehicle.ndim != 3:
        raise ValueError("learned causal states must have shape [B,T,H]")
    if physiology.shape[:2] != vehicle.shape[:2]:
        raise ValueError("learned causal streams must share [B,T]")
    if physiology.shape[-1] != config.physiology_dim:
        raise ValueError("learned causal physiology dimension mismatch")
    if vehicle.shape[-1] != config.vehicle_dim:
        raise ValueError("learned causal vehicle dimension mismatch")
    if inputs.physiology_valid_mask.shape != physiology.shape[:2]:
        raise ValueError("learned causal physiology mask mismatch")
    if inputs.vehicle_valid_mask.shape != vehicle.shape[:2]:
        raise ValueError("learned causal vehicle mask mismatch")
    if inputs.query_timestamps_s.shape != physiology.shape[:2]:
        raise ValueError("learned causal timestamp shape mismatch")


def _inverse_softplus(values: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(values))
