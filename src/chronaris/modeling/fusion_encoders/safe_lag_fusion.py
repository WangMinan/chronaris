"""Safe lag-aware fusion for the Chronaris main line.

This module implements the representation-level "safe fusion" that the audit
(`docs/artifacts/runs/2026-07-22_safe-lag-aware-fusion-audit/report.md`) identified
as the fix for the catastrophic single-stream information loss of the original
``MultiScaleCausalLagFusion``:

    z_out = [z_phys_private, z_vehicle_private, gate * z_cross]

- ``z_phys_private`` / ``z_vehicle_private`` are private per-stream projections that
  carry strong single-stream information around the cross-modal bottleneck.
- ``z_cross`` is the lag-aware multi-scale attended vehicle context (reused from
  :mod:`chronaris.modeling.fusion_encoders.multiscale_causal`).
- ``gate`` is a per-timestep sigmoid initialized near 0, so the model starts at the
  safe fallback (private streams only) and only adds cross-modal information when the
  training signal justifies it. This eliminates forced negative transfer.

The seconds-lag attention, masking and helpers are reused verbatim from
:mod:`multiscale_causal` so the lag-awareness mechanism is unchanged; only the output
composition changes from a single ``Linear(4H->out)`` to a gated private+cross split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.modeling.fusion_encoders.multiscale_causal import (
    DEFAULT_LAG_RANGES_S,
    MultiScaleCausalFusionInput,
    _masked_attention_weights,
    _masked_scale_weights,
    _validate_inputs,
    build_seconds_lag_mask,
)

ATTENTION_KINDS = ("legacy_cosine", "cosine_temperature", "projected_dot_product")


@dataclass(frozen=True, slots=True)
class SafeLagAwareFusionConfig:
    """Configuration for the safe lag-aware fusion.

    The output dim is split into ``physiology_private_dim`` +
    ``vehicle_private_dim`` + ``cross_dim``. Defaults give 24 + 24 + 16 = 64.
    """

    hidden_dim: int = 64
    output_dim: int = 64
    physiology_private_dim: int = 24
    vehicle_private_dim: int = 24
    lag_ranges_s: tuple[tuple[float, float], ...] = DEFAULT_LAG_RANGES_S
    use_causal_mask: bool = True
    use_scale_gate: bool = True
    use_private_bypass: bool = True
    attention_temperature: float = 1.0
    attention_kind: str = "legacy_cosine"
    boundary_epsilon_s: float = 1e-6
    # A large negative bias makes the sigmoid gate start near 0 (safe fallback).
    cross_gate_init_bias: float = -4.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("fusion hidden/output dimensions must be positive")
        if self.physiology_private_dim <= 0 or self.vehicle_private_dim <= 0:
            raise ValueError("private dims must be positive")
        cross_dim = self.output_dim - (
            self.physiology_private_dim + self.vehicle_private_dim
        )
        if cross_dim <= 0:
            raise ValueError(
                "output_dim must exceed the sum of the two private dims to leave a "
                "positive cross-modal subspace"
            )
        if not self.lag_ranges_s:
            raise ValueError("at least one lag range is required")
        previous_upper: float | None = None
        for lower, upper in self.lag_ranges_s:
            if lower < 0 or upper <= lower:
                raise ValueError("lag ranges must satisfy 0 <= lower < upper")
            if previous_upper is not None and abs(lower - previous_upper) > 1e-9:
                raise ValueError("lag ranges must be contiguous and ordered")
            previous_upper = upper
        if not math.isfinite(self.attention_temperature) or self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if self.attention_kind not in ATTENTION_KINDS:
            raise ValueError("unsupported lag attention kind")
        if self.attention_kind == "cosine_temperature" and not .05 < self.attention_temperature < 2.:
            raise ValueError("learnable temperature initialization must be inside (0.05,2)")
        if self.attention_kind == "projected_dot_product" and self.attention_temperature != 1.:
            raise ValueError("projected attention uses standard square-root scaling only")
        if self.boundary_epsilon_s < 0:
            raise ValueError("boundary_epsilon_s must be non-negative")
        if not self.use_scale_gate and len(self.lag_ranges_s) != 1:
            raise ValueError("gate-free fusion requires exactly one lag range")

    @property
    def cross_dim(self) -> int:
        return self.output_dim - (self.physiology_private_dim + self.vehicle_private_dim)

    @property
    def scale_count(self) -> int:
        return len(self.lag_ranges_s)


@dataclass(frozen=True, slots=True)
class SafeLagAwareFusionOutput:
    sequence_embedding: torch.Tensor
    attended_vehicle_states: torch.Tensor
    attention_weights: tuple[torch.Tensor, ...]
    lag_masks: tuple[torch.Tensor, ...]
    scale_available_mask: torch.Tensor
    scale_gate_weights: torch.Tensor
    cross_gate: torch.Tensor
    modality_available_mask: torch.Tensor
    physiology_private: torch.Tensor
    vehicle_private: torch.Tensor
    cross_features: torch.Tensor


class SafeLagAwareFusion(nn.Module):
    """Lag-aware fusion with private single-stream bypass and a safe cross gate."""

    def __init__(self, config: SafeLagAwareFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or SafeLagAwareFusionConfig()
        hidden = self.config.hidden_dim
        scale_count = self.config.scale_count
        gate_input_dim = hidden * (2 + scale_count)

        # Multi-scale scale gate (kept for lag-window selection + anti-collapse reg).
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, scale_count),
        )

        # Private single-stream projections — the bypass that preserves strong signals.
        self.physiology_private_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.config.physiology_private_dim),
        )
        self.vehicle_private_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.config.vehicle_private_dim),
        )

        # Cross-modal projection fed by the lag-aware attended vehicle context.
        self.cross_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.config.cross_dim),
        )

        # Safe cross gate: scalar per timestep, initialized near 0 (safe fallback).
        self.cross_gate = nn.Linear(gate_input_dim, 1)
        with torch.no_grad():
            self.cross_gate.bias.fill_(self.config.cross_gate_init_bias)
        if self.config.attention_kind == "cosine_temperature":
            initial = self.config.attention_temperature
            self.temperature_logit = nn.Parameter(torch.tensor(math.log((initial - .05) / (2. - initial))))
        elif self.config.attention_kind == "projected_dot_product":
            self.query_projection = nn.Linear(hidden, hidden)
            self.key_projection = nn.Linear(hidden, hidden)
            self.value_projection = nn.Linear(hidden, hidden)

    @property
    def effective_attention_temperature(self):
        if self.config.attention_kind == "cosine_temperature":
            return .05 + 1.95 * self.temperature_logit.sigmoid()
        return self.config.attention_temperature

    def forward(self, inputs: MultiScaleCausalFusionInput) -> SafeLagAwareFusionOutput:
        _validate_inputs(inputs, hidden_dim=self.config.hidden_dim)
        value_states = inputs.vehicle_states
        if self.config.attention_kind == "projected_dot_product":
            query_states = self.query_projection(inputs.physiology_states)
            key_states = self.key_projection(inputs.vehicle_states)
            value_states = self.value_projection(inputs.vehicle_states)
            scale = math.sqrt(self.config.hidden_dim)
        else:
            query_states = F.normalize(inputs.physiology_states, dim=-1, eps=1e-12)
            key_states = F.normalize(inputs.vehicle_states, dim=-1, eps=1e-12)
            scale = self.effective_attention_temperature
            if self.config.attention_kind == "legacy_cosine":
                scale *= math.sqrt(self.config.hidden_dim)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / scale

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
                torch.matmul(weights, value_states)
                for weights in attention_weights
            ),
            dim=2,
        )
        scale_available = torch.stack(
            tuple(mask.any(dim=-1) for mask in lag_masks), dim=-1
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

        # Private bypass + safe gated cross residual.
        physiology_private = self.physiology_private_projection(inputs.physiology_states)
        vehicle_private = self.vehicle_private_projection(inputs.vehicle_states)
        physiology_private = physiology_private.masked_fill(~inputs.physiology_valid_mask.unsqueeze(-1), 0)
        vehicle_private = vehicle_private.masked_fill(~inputs.vehicle_valid_mask.unsqueeze(-1), 0)
        if not self.config.use_private_bypass:
            physiology_private = torch.zeros_like(physiology_private)
            vehicle_private = torch.zeros_like(vehicle_private)
        cross_available = inputs.physiology_valid_mask & scale_available.any(dim=-1)
        cross_features = self.cross_projection(attended).masked_fill(~cross_available.unsqueeze(-1), 0)
        cross_gate = torch.sigmoid(self.cross_gate(gate_inputs))
        cross_gate = cross_gate.masked_fill(~cross_available.unsqueeze(-1), 0)
        gated_cross = cross_gate * cross_features

        sequence = torch.cat(
            (physiology_private, vehicle_private, gated_cross), dim=-1
        )
        available = inputs.physiology_valid_mask | inputs.vehicle_valid_mask
        sequence = sequence.masked_fill(~available.unsqueeze(-1), 0)
        if not torch.isfinite(sequence).all():
            raise ValueError("safe-lag fusion produced non-finite valid states")

        return SafeLagAwareFusionOutput(
            sequence_embedding=sequence,
            attended_vehicle_states=attended,
            attention_weights=attention_weights,
            lag_masks=lag_masks,
            scale_available_mask=scale_available,
            scale_gate_weights=scale_weights,
            cross_gate=cross_gate,
            modality_available_mask=available,
            physiology_private=physiology_private,
            vehicle_private=vehicle_private,
            cross_features=cross_features,
        )


def scale_gate_entropy_regularization(
    scale_gate_weights: torch.Tensor,
    *,
    available_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative-entropy penalty that discourages scale-gate collapse to one window.

    ``scale_gate_weights`` has shape ``[B, T, S]``. Returns a scalar (mean over valid
    query points) that is *minimized* when the gate is uniform across available scales,
    i.e. it penalizes degenerate one-hot gates. Add it to the auxiliary loss with a
    small positive weight.
    """

    if scale_gate_weights.ndim != 3:
        raise ValueError("scale_gate_weights must have shape [B, T, S]")
    eps = torch.finfo(scale_gate_weights.dtype).eps
    safe = scale_gate_weights.clamp_min(eps)
    entropy = -(safe * safe.log()).sum(dim=-1)
    scale_count = available_mask.sum(dim=-1).clamp_min(1.0)
    # Normalized entropy in [0,1]; 1 == uniform, 0 == collapsed. Penalty = 1 - norm.
    max_entropy = scale_count.log()
    normalized = entropy / max_entropy.clamp_min(eps)
    valid = available_mask.any(dim=-1)
    return (1.0 - normalized)[valid].mean()
