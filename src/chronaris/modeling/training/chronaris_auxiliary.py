"""Differentiable method-specific regularizers for locked Chronaris retraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from chronaris.modeling.training.pretext import ChronarisAuxiliaryWeights


@dataclass(frozen=True, slots=True)
class ChronarisAuxiliaryLosses:
    total_loss: torch.Tensor
    continuous_alignment: torch.Tensor
    physical_consistency: torch.Tensor
    causal_direction: torch.Tensor
    alignment_count: int
    physics_component_count: int
    causal_count: int


def build_chronaris_auxiliary_losses(
    positive,
    negative,
    *,
    weights: ChronarisAuxiliaryWeights,
    causal_margin: float = 0.1,
) -> ChronarisAuxiliaryLosses:
    """Build three regularizers without contributing them to candidate ranking."""

    if positive.method_name != "chronaris" or negative.method_name != "chronaris":
        raise ValueError("Chronaris auxiliary losses require Chronaris encoder outputs")
    if causal_margin <= 0:
        raise ValueError("causal margin must be positive")
    positive_alignment = positive.auxiliary.get("alignment_output")
    negative_alignment = negative.auxiliary.get("alignment_output")
    if positive_alignment is None or negative_alignment is None:
        raise ValueError("Chronaris auxiliary losses require alignment diagnostics")
    positive_pair = _reference_pair(positive_alignment)
    negative_pair = _reference_pair(negative_alignment)
    valid = positive_pair[2] & negative_pair[2]
    if bool(valid.any()):
        positive_similarity = F.cosine_similarity(
            positive_pair[0],
            positive_pair[1],
            dim=-1,
        )
        negative_similarity = F.cosine_similarity(
            negative_pair[0],
            negative_pair[1],
            dim=-1,
        )
        continuous_alignment = (1.0 - positive_similarity[valid]).mean()
        causal_direction = F.relu(
            causal_margin - positive_similarity[valid] + negative_similarity[valid]
        ).mean()
        alignment_count = int(valid.sum().item())
        causal_count = alignment_count
    else:
        zero = positive.sequence_embedding.sum() * 0.0
        continuous_alignment = zero
        causal_direction = zero
        alignment_count = 0
        causal_count = 0
    physics_audit = positive.auxiliary.get("physical_consistency")
    raw_physics = (
        tuple(
            component.raw_value
            for component in physics_audit.components
            if component.raw_value is not None
        )
        if hasattr(physics_audit, "components")
        else ()
    )
    physical_consistency = (
        torch.stack(raw_physics).mean()
        if raw_physics
        else positive.sequence_embedding.sum() * 0.0
    )
    total = (
        continuous_alignment * weights.continuous_alignment
        + physical_consistency * weights.physical_consistency
        + causal_direction * weights.causal_direction
    )
    return ChronarisAuxiliaryLosses(
        total_loss=total,
        continuous_alignment=continuous_alignment,
        physical_consistency=physical_consistency,
        causal_direction=causal_direction,
        alignment_count=alignment_count,
        physics_component_count=len(raw_physics),
        causal_count=causal_count,
    )


def chronaris_auxiliary_losses_to_rows(losses, *, weights):
    return tuple(
        {
            "term_name": name,
            "weight": float(weight),
            "raw_loss": float(value.detach().cpu()),
            "weighted_loss": float((value * weight).detach().cpu()),
            "count": count,
            "status": "active" if weight > 0 and count > 0 else "scheduled_zero_or_unavailable",
        }
        for name, value, weight, count in (
            (
                "chronaris_continuous_alignment",
                losses.continuous_alignment,
                weights.continuous_alignment,
                losses.alignment_count,
            ),
            (
                "chronaris_physical_consistency",
                losses.physical_consistency,
                weights.physical_consistency,
                losses.physics_component_count,
            ),
            (
                "chronaris_causal_direction",
                losses.causal_direction,
                weights.causal_direction,
                losses.causal_count,
            ),
        )
    )


def _reference_pair(alignment):
    physiology = alignment.physiology.reference_projected_states
    vehicle = alignment.vehicle.reference_projected_states
    physiology_valid = alignment.physiology.reference_valid_mask
    vehicle_valid = alignment.vehicle.reference_valid_mask
    if (
        physiology is None
        or vehicle is None
        or physiology_valid is None
        or vehicle_valid is None
    ):
        raise ValueError("Chronaris alignment lacks reference-grid states")
    return physiology, vehicle, physiology_valid & vehicle_valid
