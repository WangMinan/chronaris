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

__all__ = [
    "CausalFusionConfig",
    "CausalFusionTensorInput",
    "CausalFusionTensorOutput",
    "CausalMaskedCrossModalFusion",
    "attention_entropy",
    "build_causal_attention_mask",
    "compute_vehicle_event_scores",
]
