"""Role-aware wrappers around the Stage G causal fusion module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chronaris.models.fusion.adaptive_router import AdaptiveFusionRouter, FusionRouteDecision
from chronaris.models.fusion.causal import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
)
from chronaris.models.fusion.stream_role import FusionRoute, StreamRole, StreamRoleMetadata


@dataclass(frozen=True, slots=True)
class RoleAwareFusionOutput:
    """Output of role-aware fusion."""

    fused_states: torch.Tensor
    attention_weights: torch.Tensor
    route_decision: FusionRouteDecision
    metadata: StreamRoleMetadata

    def gate_summary(self) -> dict[str, object]:
        summary = self.metadata.to_jsonable()
        summary.update(self.route_decision.gate_summary())
        return summary


class RoleAwareCausalFusion(nn.Module):
    """Use causal vehicle fusion for private streams and gated adapters for public proxies."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        causal_config: CausalFusionConfig | None = None,
    ) -> None:
        super().__init__()
        self.causal_fusion = CausalMaskedCrossModalFusion(causal_config or CausalFusionConfig(lag_window_points=16))
        self.router = AdaptiveFusionRouter(hidden_dim=hidden_dim)
        self.context_adapter = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        *,
        physiology_states: torch.Tensor,
        second_stream_states: torch.Tensor,
        physiology_offsets_s: torch.Tensor,
        second_stream_offsets_s: torch.Tensor,
        metadata: StreamRoleMetadata,
        variant: str = "chronaris_v3_stream_role",
    ) -> RoleAwareFusionOutput:
        if physiology_states.shape != second_stream_states.shape:
            raise ValueError("physiology and second-stream states must share shape [B, T, D].")
        hard_route = _variant_route(variant)
        roles = tuple(metadata.second_stream_role for _ in range(physiology_states.shape[0]))
        decision = self.router(
            physiology_summary=physiology_states.mean(dim=1),
            second_stream_summary=second_stream_states.mean(dim=1),
            second_stream_roles=roles,
            hard_route=hard_route,
        )
        if decision.route in {
            FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO,
            FusionRoute.FORCE_PRIVATE_CAUSAL,
        }:
            causal = self.causal_fusion(
                CausalFusionTensorInput(
                    physiology_states=physiology_states,
                    vehicle_states=second_stream_states,
                    physiology_offsets_s=physiology_offsets_s,
                    vehicle_offsets_s=second_stream_offsets_s,
                )
            )
            fused = causal.fused_states
            attention = causal.attention_weights
        elif decision.route == FusionRoute.PHYSIOLOGY_ONLY:
            zeros = torch.zeros_like(physiology_states)
            fused = torch.cat((physiology_states, zeros, physiology_states), dim=-1)
            attention = _identity_attention(physiology_states)
        else:
            adapted = self.context_adapter(torch.cat((physiology_states, second_stream_states), dim=-1))
            gate = decision.context_gate.view(-1, 1, 1)
            mixed = (1.0 - gate) * physiology_states + gate * adapted
            fused = torch.cat((physiology_states, mixed, physiology_states - mixed), dim=-1)
            attention = _identity_attention(physiology_states)
        return RoleAwareFusionOutput(
            fused_states=fused,
            attention_weights=attention,
            route_decision=decision,
            metadata=metadata,
        )


def _variant_route(variant: str) -> FusionRoute | None:
    normalized = variant.strip().lower()
    if normalized in {
        "v3_fixed_causal_lag",
        "fixed_causal_lag",
        "v3_force_private_causal",
        "force_private_causal",
    }:
        return FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO
    if normalized in {"v3_context_adapter_only", "context_adapter_only"}:
        return FusionRoute.CONTEXT_ADAPTER_ONLY
    if normalized in {
        "chronaris_v3_stream_role",
        "chronaris_v3_stream_role_fusion",
        "v3_stream_role",
        "v3_stream_role_fusion",
        "v3_stream_role_adaptive",
        "v3_no_role_gate",
        "no_role_gate",
    }:
        return None
    return None


def _identity_attention(states: torch.Tensor) -> torch.Tensor:
    batch_size, point_count, _ = states.shape
    return torch.eye(point_count, device=states.device, dtype=states.dtype).expand(
        batch_size,
        point_count,
        point_count,
    )


def private_stream_metadata() -> StreamRoleMetadata:
    return StreamRoleMetadata.private_real_vehicle()


def public_stream_metadata(dataset_id: str) -> StreamRoleMetadata:
    return StreamRoleMetadata.public_context_proxy(dataset_id=dataset_id)
