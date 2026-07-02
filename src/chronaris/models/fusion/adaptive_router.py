"""Adaptive fusion router for stream-role-aware Stage I models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chronaris.models.fusion.stream_role import FusionRoute, StreamRole, normalize_stream_role


@dataclass(frozen=True, slots=True)
class FusionRouteDecision:
    """Routing gates and hard route selected for a batch."""

    route: FusionRoute
    route_logits: torch.Tensor
    lag_gate: torch.Tensor
    context_gate: torch.Tensor
    vehicle_gate: torch.Tensor
    causal_gate: torch.Tensor

    def gate_summary(self) -> dict[str, float | str]:
        return {
            "fusion_route": self.route.value,
            "lag_gate_mean": float(self.lag_gate.detach().float().mean().item()),
            "context_gate_mean": float(self.context_gate.detach().float().mean().item()),
            "vehicle_gate_mean": float(self.vehicle_gate.detach().float().mean().item()),
            "causal_gate_mean": float(self.causal_gate.detach().float().mean().item()),
        }


class StreamRoleEncoder(nn.Module):
    """Embedding layer for the finite stream-role vocabulary."""

    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.roles = tuple(StreamRole)
        self.role_to_index = {role: index for index, role in enumerate(self.roles)}
        self.embedding = nn.Embedding(len(self.roles), embedding_dim)

    def forward(self, roles: list[str | StreamRole] | tuple[str | StreamRole, ...]) -> torch.Tensor:
        indices = [
            self.role_to_index[normalize_stream_role(role)]
            for role in roles
        ]
        index_tensor = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        return self.embedding(index_tensor)


class AdaptiveFusionRouter(nn.Module):
    """Route second-stream fusion according to role and current hidden state."""

    route_order = (
        FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO,
        FusionRoute.ADAPTIVE_CONTEXT_GATE,
        FusionRoute.CONTEXT_ADAPTER_ONLY,
        FusionRoute.PHYSIOLOGY_ONLY,
    )

    def __init__(self, *, hidden_dim: int, role_embedding_dim: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.role_encoder = StreamRoleEncoder(role_embedding_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 2 + role_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.route_order)),
        )
        self.gates = nn.Sequential(
            nn.Linear(hidden_dim * 2 + role_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        *,
        physiology_summary: torch.Tensor,
        second_stream_summary: torch.Tensor,
        second_stream_roles: tuple[str | StreamRole, ...],
        hard_route: FusionRoute | None = None,
    ) -> FusionRouteDecision:
        if physiology_summary.shape != second_stream_summary.shape:
            raise ValueError("stream summaries must share shape [B, D].")
        role_embedding = self.role_encoder(second_stream_roles).to(physiology_summary.device)
        features = torch.cat((physiology_summary, second_stream_summary, role_embedding), dim=-1)
        route_logits = self.router(features)
        raw_gates = torch.sigmoid(self.gates(features))
        role = normalize_stream_role(second_stream_roles[0])
        route = hard_route or self._default_route_for_role(role)
        if route == FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO:
            lag_gate = raw_gates[:, 0:1] * 0.25 + 0.75
            context_gate = raw_gates[:, 1:2] * 0.25
            vehicle_gate = raw_gates[:, 2:3] * 0.25 + 0.75
            causal_gate = raw_gates[:, 3:4] * 0.25 + 0.75
        elif route == FusionRoute.CONTEXT_ADAPTER_ONLY:
            lag_gate = raw_gates[:, 0:1] * 0.1
            context_gate = raw_gates[:, 1:2] * 0.25 + 0.75
            vehicle_gate = raw_gates[:, 2:3] * 0.1
            causal_gate = raw_gates[:, 3:4] * 0.1
        elif route == FusionRoute.PHYSIOLOGY_ONLY:
            lag_gate = raw_gates[:, 0:1] * 0.1
            context_gate = raw_gates[:, 1:2] * 0.1
            vehicle_gate = raw_gates[:, 2:3] * 0.1
            causal_gate = raw_gates[:, 3:4] * 0.1
        else:
            lag_gate = raw_gates[:, 0:1] * 0.25
            context_gate = raw_gates[:, 1:2] * 0.25 + 0.65
            vehicle_gate = raw_gates[:, 2:3] * 0.35
            causal_gate = raw_gates[:, 3:4] * 0.25
        return FusionRouteDecision(
            route=route,
            route_logits=route_logits,
            lag_gate=lag_gate,
            context_gate=context_gate,
            vehicle_gate=vehicle_gate,
            causal_gate=causal_gate,
        )

    def _default_route_for_role(self, role: StreamRole) -> FusionRoute:
        if role == StreamRole.REAL_VEHICLE:
            return FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO
        if role in {StreamRole.TASK_CONTEXT_PROXY, StreamRole.SCENARIO_CONTEXT_PROXY, StreamRole.UNKNOWN_CONTEXT}:
            return FusionRoute.ADAPTIVE_CONTEXT_GATE
        return FusionRoute.PHYSIOLOGY_ONLY


def gate_regularization_for_route(decision: FusionRouteDecision) -> torch.Tensor:
    """Small route-aware gate regularizer."""

    if decision.route == FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO:
        target = decision.vehicle_gate.new_tensor(0.75)
        return (
            torch.square(decision.vehicle_gate.mean() - target)
            + torch.square(decision.causal_gate.mean() - target)
        )
    if decision.route == FusionRoute.ADAPTIVE_CONTEXT_GATE:
        return torch.relu(decision.context_gate.new_tensor(0.4) - decision.context_gate.mean())
    return decision.context_gate.new_tensor(0.0)
