"""Differentiable method-specific regularizers for locked Chronaris retraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from chronaris.modeling.fusion_encoders.multiscale_causal import build_seconds_lag_mask
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
    observation_anchor: torch.Tensor | None = None
    observation_count: int = 0
    physical_effective_weight: float | None = None


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
            if component.active and component.raw_value is not None
        )
        if hasattr(physics_audit, "components")
        else ()
    )
    physical_consistency = (
        torch.stack(raw_physics).mean()
        if raw_physics
        else positive.sequence_embedding.sum() * 0.0
    )
    effective_physics_weight = weights.physical_consistency
    if getattr(physics_audit, "calibrated_residual", None) is not None:
        physical_consistency = physics_audit.calibrated_residual
        effective_physics_weight *= physics_audit.residual_weight / 0.1
    total = (
        continuous_alignment * weights.continuous_alignment
        + physical_consistency * effective_physics_weight
        + causal_direction * weights.causal_direction
    )
    anchor = getattr(physics_audit, "observation_anchor", None)
    if anchor is not None:
        total = total + anchor
    return ChronarisAuxiliaryLosses(
        total_loss=total,
        continuous_alignment=continuous_alignment,
        physical_consistency=physical_consistency,
        causal_direction=causal_direction,
        alignment_count=alignment_count,
        physics_component_count=len(raw_physics),
        causal_count=causal_count,
        observation_anchor=anchor,
        observation_count=getattr(physics_audit, "observation_count", 0),
        physical_effective_weight=effective_physics_weight,
    )


def chronaris_auxiliary_losses_to_rows(losses, *, weights):
    rows = tuple(
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
                losses.physical_effective_weight if losses.physical_effective_weight is not None else weights.physical_consistency,
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
    if losses.observation_anchor is not None:
        value = float(losses.observation_anchor.detach().cpu())
        rows += ({"term_name": "observation_anchor", "weight": 1.0,
                  "raw_loss": value, "weighted_loss": value, "count": losses.observation_count,
                  "status": "active" if losses.observation_count else "unavailable"},)
    return rows


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


@dataclass(frozen=True, slots=True)
class LagAwareAlignmentResult:
    loss: torch.Tensor
    count: int
    best_lag_index: torch.Tensor | None


def lag_aware_alignment_loss(
    alignment,
    *,
    min_lag_s: float = 0.0,
    max_lag_s: float = 15.0,
) -> LagAwareAlignmentResult:
    """Lag-tolerant causal alignment for the lag-response hypothesis (audit Q4 fix).

    The original ``continuous_alignment`` minimises ``1 - cos(phys_t, veh_t)`` at the
    *same* reference index, which contradicts physiology lagging vehicle inputs. Here,
    for each physiology reference point we take the *maximum* cosine similarity against
    vehicle reference points within a causal lag window ``[min_lag_s, max_lag_s]`` (i.e.
    vehicle history), then minimise ``1 - max_sim``. This lets physiology align with the
    best causal lag instead of being forced to zero lag.

    Returns a scalar loss plus the per-query best-lag index (for diagnostics). If no
    valid causal-lag pair exists, returns a zero loss tied to the physiology states.
    """

    if min_lag_s < 0 or max_lag_s <= min_lag_s:
        raise ValueError("lag window must satisfy 0 <= min_lag_s < max_lag_s")
    physiology, vehicle, valid = _reference_pair(alignment)
    query_times = alignment.physiology.reference_offsets_s
    if query_times is None:
        query_times = alignment.vehicle.reference_offsets_s
    if query_times is None:
        zero = physiology.sum() * 0.0
        return LagAwareAlignmentResult(loss=zero, count=0, best_lag_index=None)
    query_times = query_times.to(vehicle.device)
    lag_mask = build_seconds_lag_mask(
        query_times,
        query_times,
        query_valid_mask=valid,
        key_valid_mask=valid,
        lower_s=min_lag_s,
        upper_s=max_lag_s,
        range_index=0,
        use_causal_mask=True,
    )  # [B, T_query, T_key], True where key is a valid causal lag of query
    normalized_phys = F.normalize(physiology, dim=-1, eps=1e-12)
    normalized_veh = F.normalize(vehicle, dim=-1, eps=1e-12)
    similarity = torch.matmul(
        normalized_phys, normalized_veh.transpose(-1, -2)
    )  # [B, T_query, T_key]
    has_lag = lag_mask.any(dim=-1) & valid
    if not bool(has_lag.any()):
        zero = physiology.sum() * 0.0
        return LagAwareAlignmentResult(loss=zero, count=0, best_lag_index=None)
    masked_sim = similarity.masked_fill(~lag_mask, torch.finfo(similarity.dtype).min)
    best_sim, best_idx = masked_sim.max(dim=-1)
    loss = (1.0 - best_sim[has_lag]).mean()
    return LagAwareAlignmentResult(
        loss=loss, count=int(has_lag.sum().item()), best_lag_index=best_idx
    )
