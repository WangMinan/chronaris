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

__all__ = [
    "CausalEventFusion",
    "CausalEventFusionConfig",
    "CausalFusionConfig",
    "CausalFusionTensorInput",
    "CausalFusionTensorOutput",
    "CausalMaskedCrossModalFusion",
    "EventTokenExtractor",
    "LLM_SEMANTIC_QUERY_RECIPE_WHITELIST",
    "SemanticEventTensorInput",
    "SemanticEventTensorOutput",
    "SemanticQueryBank",
    "SemanticQuerySpec",
    "attention_entropy",
    "build_causal_attention_mask",
    "compute_vehicle_event_scores",
    "semantic_query_specs_from_llm_hints",
    "semantic_query_entropy",
]
