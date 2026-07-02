"""Causal cross-modal fusion models."""

from chronaris.models.fusion.causal import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalFusionTensorOutput,
    CausalMaskedCrossModalFusion,
    attention_entropy,
    build_causal_attention_mask,
    compute_vehicle_event_scores,
)
from chronaris.models.fusion.adaptive_router import (
    AdaptiveFusionRouter,
    FusionRouteDecision,
    StreamRoleEncoder,
    gate_regularization_for_route,
)
from chronaris.models.fusion.role_aware_fusion import (
    RoleAwareCausalFusion,
    RoleAwareFusionOutput,
    private_stream_metadata,
    public_stream_metadata,
)
from chronaris.models.fusion.semantic_event import (
    CausalEventFusion,
    CausalEventFusionConfig,
    EventTokenExtractor,
    LLM_SEMANTIC_QUERY_RECIPE_WHITELIST,
    SemanticEventTensorInput,
    SemanticEventTensorOutput,
    SemanticQueryBank,
    SemanticQuerySpec,
    semantic_query_specs_from_llm_hints,
    semantic_query_entropy,
)
from chronaris.models.fusion.stream_role import (
    FusionRoute,
    StreamRole,
    StreamRoleMetadata,
    normalize_stream_role,
)

__all__ = [
    "AdaptiveFusionRouter",
    "CausalEventFusion",
    "CausalEventFusionConfig",
    "CausalFusionConfig",
    "CausalFusionTensorInput",
    "CausalFusionTensorOutput",
    "CausalMaskedCrossModalFusion",
    "EventTokenExtractor",
    "FusionRoute",
    "FusionRouteDecision",
    "LLM_SEMANTIC_QUERY_RECIPE_WHITELIST",
    "RoleAwareCausalFusion",
    "RoleAwareFusionOutput",
    "SemanticEventTensorInput",
    "SemanticEventTensorOutput",
    "SemanticQueryBank",
    "SemanticQuerySpec",
    "StreamRole",
    "StreamRoleEncoder",
    "StreamRoleMetadata",
    "attention_entropy",
    "build_causal_attention_mask",
    "compute_vehicle_event_scores",
    "gate_regularization_for_route",
    "normalize_stream_role",
    "private_stream_metadata",
    "public_stream_metadata",
    "semantic_query_specs_from_llm_hints",
    "semantic_query_entropy",
]
