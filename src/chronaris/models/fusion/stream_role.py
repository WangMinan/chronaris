"""Stream-role metadata for private vehicle streams and public context proxies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StreamRole(str, Enum):
    """Supported task evaluation second-stream roles."""

    REAL_VEHICLE = "real_vehicle"
    TASK_CONTEXT_PROXY = "task_context_proxy"
    SCENARIO_CONTEXT_PROXY = "scenario_context_proxy"
    PHYSIOLOGY = "physiology"
    UNKNOWN_CONTEXT = "unknown_context"


class FusionRoute(str, Enum):
    """Role-aware fusion routes."""

    CAUSAL_LAGGED_VEHICLE_TO_PHYSIO = "causal_lagged_vehicle_to_physio"
    ADAPTIVE_CONTEXT_GATE = "adaptive_context_gate"
    CONTEXT_ADAPTER_ONLY = "context_adapter_only"
    PHYSIOLOGY_ONLY = "physiology_only"
    FORCE_PRIVATE_CAUSAL = "force_private_causal"


@dataclass(frozen=True, slots=True)
class StreamRoleMetadata:
    """Reader-facing metadata propagated through task evaluation fusion runs."""

    second_stream_role: StreamRole
    second_stream_is_real_vehicle: bool
    evidence_role: str
    fusion_route: FusionRoute
    lag_policy: str
    causal_policy: str

    @classmethod
    def private_real_vehicle(cls) -> "StreamRoleMetadata":
        return cls(
            second_stream_role=StreamRole.REAL_VEHICLE,
            second_stream_is_real_vehicle=True,
            evidence_role="private_real_dual_stream",
            fusion_route=FusionRoute.CAUSAL_LAGGED_VEHICLE_TO_PHYSIO,
            lag_policy="causal_lag_window_enabled",
            causal_policy="vehicle_to_physiology",
        )

    @classmethod
    def public_context_proxy(cls, *, dataset_id: str) -> "StreamRoleMetadata":
        role = (
            StreamRole.SCENARIO_CONTEXT_PROXY
            if dataset_id.strip().lower() == "nasa_csm"
            else StreamRole.TASK_CONTEXT_PROXY
        )
        return cls(
            second_stream_role=role,
            second_stream_is_real_vehicle=False,
            evidence_role="public_adapter_context_proxy_evidence",
            fusion_route=FusionRoute.ADAPTIVE_CONTEXT_GATE,
            lag_policy="no_lag_or_learned_context_gate_allowed",
            causal_policy="context_proxy_not_real_vehicle",
        )

    def to_jsonable(self) -> dict[str, object]:
        return {
            "second_stream_role": self.second_stream_role.value,
            "second_stream_is_real_vehicle": bool(self.second_stream_is_real_vehicle),
            "evidence_role": self.evidence_role,
            "fusion_route": self.fusion_route.value,
            "lag_policy": self.lag_policy,
            "causal_policy": self.causal_policy,
        }


def normalize_stream_role(value: str | StreamRole) -> StreamRole:
    if isinstance(value, StreamRole):
        return value
    try:
        return StreamRole(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"unsupported stream role: {value!r}") from exc
