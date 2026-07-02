"""Task-aware Stage I heads for optimized private proxy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class GateContributionSummary:
    """Detached diagnostics for gated task-head mixtures."""

    gate_mean: float
    gate_std: float
    vehicle_contribution: float
    fusion_contribution: float

    def to_jsonable(self) -> dict[str, float]:
        return {
            "gate_mean": self.gate_mean,
            "gate_std": self.gate_std,
            "vehicle_contribution": self.vehicle_contribution,
            "fusion_contribution": self.fusion_contribution,
        }


@dataclass(frozen=True, slots=True)
class VehicleAuxClassificationOutput:
    """Output of the T1 vehicle-dominant auxiliary classification head."""

    logits: torch.Tensor
    vehicle_logits: torch.Tensor
    fusion_logits: torch.Tensor
    gate: torch.Tensor

    def contribution_summary(self) -> GateContributionSummary:
        detached_gate = self.gate.detach().float()
        return GateContributionSummary(
            gate_mean=float(detached_gate.mean().item()),
            gate_std=float(detached_gate.std(unbiased=False).item()),
            vehicle_contribution=float(detached_gate.mean().item()),
            fusion_contribution=float((1.0 - detached_gate).mean().item()),
        )


@dataclass(frozen=True, slots=True)
class ResidualRegressionOutput:
    """Output of the T2 persistence/excitation/interaction residual head."""

    prediction: torch.Tensor
    persistence: torch.Tensor
    vehicle_excitation: torch.Tensor
    interaction: torch.Tensor
    uncertainty: torch.Tensor | None = None

    def decomposition_summary(self) -> dict[str, float]:
        return {
            "persistence_abs_mean": float(self.persistence.detach().abs().mean().item()),
            "vehicle_excitation_abs_mean": float(self.vehicle_excitation.detach().abs().mean().item()),
            "interaction_abs_mean": float(self.interaction.detach().abs().mean().item()),
        }


class VehicleDominantAuxiliaryClassificationHead(nn.Module):
    """T1 head that gates vehicle and fused representations."""

    def __init__(
        self,
        *,
        vehicle_dim: int,
        fused_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        initial_vehicle_bias: float = 0.75,
    ) -> None:
        super().__init__()
        _validate_positive("vehicle_dim", vehicle_dim)
        _validate_positive("fused_dim", fused_dim)
        _validate_positive("output_dim", output_dim)
        _validate_probability("initial_vehicle_bias", initial_vehicle_bias)
        self.vehicle_branch = _mlp(vehicle_dim, hidden_dim, output_dim, dropout)
        self.fusion_branch = _mlp(fused_dim, hidden_dim, output_dim, dropout)
        self.gate_network = nn.Sequential(
            nn.Linear(vehicle_dim + fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        bias = torch.logit(torch.tensor(float(initial_vehicle_bias)))
        nn.init.constant_(self.gate_network[-1].bias, float(bias.item()))

    def forward(
        self,
        *,
        vehicle_states: torch.Tensor,
        fused_states: torch.Tensor,
        physics_summary: torch.Tensor | None = None,
    ) -> VehicleAuxClassificationOutput:
        vehicle_features = _pool_sequence(vehicle_states)
        fused_features = _pool_sequence(fused_states)
        if physics_summary is not None:
            if physics_summary.ndim != 2 or physics_summary.shape[0] != vehicle_features.shape[0]:
                raise ValueError("physics_summary must have shape [B, D].")
            vehicle_features = vehicle_features + _match_last_dim(
                physics_summary,
                vehicle_features.shape[-1],
            )
        vehicle_logits = self.vehicle_branch(vehicle_features)
        fusion_logits = self.fusion_branch(fused_features)
        gate = torch.sigmoid(self.gate_network(torch.cat((vehicle_features, fused_features), dim=-1)))
        logits = gate * vehicle_logits + (1.0 - gate) * fusion_logits
        return VehicleAuxClassificationOutput(
            logits=logits,
            vehicle_logits=vehicle_logits,
            fusion_logits=fusion_logits,
            gate=gate,
        )


class PhysiologyResponseResidualRegressionHead(nn.Module):
    """T2 head: y_next = persistence + vehicle excitation + interaction."""

    def __init__(
        self,
        *,
        physiology_dim: int,
        vehicle_dim: int,
        fused_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive("physiology_dim", physiology_dim)
        _validate_positive("vehicle_dim", vehicle_dim)
        _validate_positive("fused_dim", fused_dim)
        _validate_positive("output_dim", output_dim)
        self.persistence_branch = _mlp(physiology_dim, hidden_dim, output_dim, dropout)
        self.vehicle_excitation_branch = _mlp(vehicle_dim, hidden_dim, output_dim, dropout)
        self.interaction_branch = _mlp(fused_dim, hidden_dim, output_dim, dropout)
        self.uncertainty_head = (
            _mlp(fused_dim, hidden_dim, output_dim, dropout) if predict_uncertainty else None
        )

    def forward(
        self,
        *,
        physiology_states: torch.Tensor,
        vehicle_states: torch.Tensor,
        fused_states: torch.Tensor,
        persistence_anchor: torch.Tensor | None = None,
    ) -> ResidualRegressionOutput:
        physiology_features = _pool_sequence(physiology_states)
        vehicle_features = _pool_sequence(vehicle_states)
        fused_features = _pool_sequence(fused_states)
        persistence = self.persistence_branch(physiology_features)
        if persistence_anchor is not None:
            persistence = persistence + persistence_anchor.to(device=persistence.device, dtype=persistence.dtype)
        vehicle_excitation = self.vehicle_excitation_branch(vehicle_features)
        interaction = self.interaction_branch(fused_features)
        prediction = persistence + vehicle_excitation + interaction
        uncertainty = None
        if self.uncertainty_head is not None:
            uncertainty = F.softplus(self.uncertainty_head(fused_features)) + 1e-6
        return ResidualRegressionOutput(
            prediction=prediction,
            persistence=persistence,
            vehicle_excitation=vehicle_excitation,
            interaction=interaction,
            uncertainty=uncertainty,
        )


class ContrastiveRetrievalProjectionHead(nn.Module):
    """T3 projection head for supervised contrastive retrieval."""

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive("input_dim", input_dim)
        _validate_positive("embedding_dim", embedding_dim)
        self.network = _mlp(input_dim, hidden_dim, embedding_dim, dropout)

    def forward(self, fused_states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(_pool_sequence(fused_states)), dim=-1, eps=1e-12)


def gate_regularization_loss(
    gate: torch.Tensor,
    *,
    target_vehicle_contribution: float = 0.65,
    collapse_margin: float = 0.05,
) -> torch.Tensor:
    """Keep T1 gates learnable while discouraging all-vehicle or all-fusion collapse."""

    _validate_probability("target_vehicle_contribution", target_vehicle_contribution)
    _validate_probability("collapse_margin", collapse_margin)
    mean_gate = gate.float().mean()
    target = gate.new_tensor(float(target_vehicle_contribution))
    balance = torch.square(mean_gate - target)
    lower = torch.relu(gate.new_tensor(collapse_margin) - mean_gate)
    upper = torch.relu(mean_gate - gate.new_tensor(1.0 - collapse_margin))
    return balance + lower + upper


def summarize_gate_contributions(
    rows: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    gate = rows["gate"].detach().float()
    return {
        "gate_mean": float(gate.mean().item()),
        "gate_std": float(gate.std(unbiased=False).item()),
        "vehicle_contribution": float(gate.mean().item()),
        "fusion_contribution": float((1.0 - gate).mean().item()),
    }


def _pool_sequence(states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 2:
        return states
    if states.ndim != 3:
        raise ValueError("states must have shape [B, D] or [B, T, D].")
    return states.mean(dim=1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(hidden_dim, output_dim),
    )


def _match_last_dim(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.shape[-1] == dim:
        return values
    if values.shape[-1] > dim:
        return values[..., :dim]
    pad = values.new_zeros((*values.shape[:-1], dim - values.shape[-1]))
    return torch.cat((values, pad), dim=-1)


def _validate_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
