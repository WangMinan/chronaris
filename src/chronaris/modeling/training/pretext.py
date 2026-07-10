"""Shared task-independent pretext heads, losses, and Chronaris weight schedule."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.representation.pretext_targets import CommonPretextTargets


@dataclass(frozen=True, slots=True)
class CommonPretextWeights:
    masked_reconstruction: float = 1.0
    short_horizon_prediction: float = 0.5
    lag_discrimination: float = 0.2

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.masked_reconstruction,
                self.short_horizon_prediction,
                self.lag_discrimination,
            )
        ):
            raise ValueError("common pretext weights must be non-negative")


@dataclass(frozen=True, slots=True)
class ChronarisAuxiliaryWeights:
    continuous_alignment: float
    physical_consistency: float
    causal_direction: float


@dataclass(frozen=True, slots=True)
class PretextLossTerm:
    term_name: str
    weight: float
    status: str
    count: int
    raw_loss: torch.Tensor | None
    weighted_loss: torch.Tensor | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CommonPretextLossOutput:
    total_loss: torch.Tensor
    terms: tuple[PretextLossTerm, ...]
    reconstruction_prediction: torch.Tensor
    next_query_prediction: torch.Tensor
    positive_lag_logits: torch.Tensor
    negative_lag_logits: torch.Tensor


class CommonPretextHeadBundle(nn.Module):
    """Identical public heads attached after each 64-dimensional encoder."""

    def __init__(self, *, representation_dim: int, target_feature_count: int) -> None:
        super().__init__()
        if representation_dim <= 0 or target_feature_count <= 0:
            raise ValueError("pretext head dimensions must be positive")
        self.representation_dim = representation_dim
        self.target_feature_count = target_feature_count
        self.reconstruction_head = nn.Linear(
            representation_dim,
            target_feature_count,
        )
        self.next_query_head = nn.Linear(
            representation_dim,
            target_feature_count,
        )
        self.lag_head = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, 1),
        )

    def forward(
        self,
        positive_sequence: torch.Tensor,
        negative_sequence: torch.Tensor,
        targets: CommonPretextTargets,
        *,
        weights: CommonPretextWeights | None = None,
    ) -> CommonPretextLossOutput:
        _validate_sequence_pair(
            positive_sequence,
            negative_sequence,
            representation_dim=self.representation_dim,
        )
        if positive_sequence.shape[:2] != targets.reconstruction_target.shape[:2]:
            raise ValueError("pretext sequence/target query shape mismatch")
        if targets.target_feature_count != self.target_feature_count:
            raise ValueError("pretext target feature dimension mismatch")
        resolved_weights = weights or CommonPretextWeights()
        reconstruction = self.reconstruction_head(positive_sequence)
        next_query = self.next_query_head(positive_sequence)
        positive_logits = self.lag_head(positive_sequence.mean(dim=1)).squeeze(-1)
        negative_logits = self.lag_head(negative_sequence.mean(dim=1)).squeeze(-1)
        terms = (
            _masked_huber_term(
                "masked_reconstruction",
                reconstruction,
                targets.reconstruction_target,
                targets.reconstruction_mask,
                weight=resolved_weights.masked_reconstruction,
            ),
            _masked_huber_term(
                "short_horizon_prediction",
                next_query,
                targets.next_query_target,
                targets.next_query_mask,
                weight=resolved_weights.short_horizon_prediction,
            ),
            _lag_discrimination_term(
                positive_logits,
                negative_logits,
                weight=resolved_weights.lag_discrimination,
            ),
        )
        active = [term.weighted_loss for term in terms if term.weighted_loss is not None]
        total = (
            torch.stack(active).sum()
            if active
            else positive_sequence.sum() * 0.0
        )
        return CommonPretextLossOutput(
            total_loss=total,
            terms=terms,
            reconstruction_prediction=reconstruction,
            next_query_prediction=next_query,
            positive_lag_logits=positive_logits,
            negative_lag_logits=negative_logits,
        )


def chronaris_auxiliary_weight_schedule(
    epoch: int,
) -> ChronarisAuxiliaryWeights:
    """Keep auxiliary weights zero for 10 epochs, then ramp through epoch 20."""

    if epoch <= 0:
        raise ValueError("training epoch must be one-based and positive")
    if epoch <= 10:
        fraction = 0.0
    elif epoch >= 20:
        fraction = 1.0
    else:
        fraction = (epoch - 10) / 10.0
    return ChronarisAuxiliaryWeights(
        continuous_alignment=0.2 * fraction,
        physical_consistency=0.1 * fraction,
        causal_direction=0.1 * fraction,
    )


def pretext_loss_terms_to_rows(
    terms: tuple[PretextLossTerm, ...],
) -> tuple[dict[str, object], ...]:
    rows = []
    for term in terms:
        rows.append(
            {
                "term_name": term.term_name,
                "weight": term.weight,
                "status": term.status,
                "count": term.count,
                "raw_loss": (
                    float(term.raw_loss.detach().cpu())
                    if term.raw_loss is not None
                    else None
                ),
                "weighted_loss": (
                    float(term.weighted_loss.detach().cpu())
                    if term.weighted_loss is not None
                    else None
                ),
                "reason": term.reason,
            }
        )
    return tuple(rows)


def _masked_huber_term(
    name,
    prediction,
    target,
    mask,
    *,
    weight,
) -> PretextLossTerm:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(f"{name} prediction/target/mask shape mismatch")
    count = int(mask.sum().item())
    if count == 0:
        return PretextLossTerm(
            term_name=name,
            weight=float(weight),
            status="unavailable",
            count=0,
            raw_loss=None,
            weighted_loss=None,
            reason="no_valid_target_positions",
        )
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    raw = element_loss[mask].mean()
    return PretextLossTerm(
        term_name=name,
        weight=float(weight),
        status="active",
        count=count,
        raw_loss=raw,
        weighted_loss=raw * weight,
        reason=None,
    )


def _lag_discrimination_term(
    positive_logits,
    negative_logits,
    *,
    weight,
) -> PretextLossTerm:
    logits = torch.cat((positive_logits, negative_logits), dim=0)
    labels = torch.cat(
        (torch.ones_like(positive_logits), torch.zeros_like(negative_logits)),
        dim=0,
    )
    if logits.numel() == 0:
        return PretextLossTerm(
            term_name="lag_discrimination",
            weight=float(weight),
            status="unavailable",
            count=0,
            raw_loss=None,
            weighted_loss=None,
            reason="empty_positive_negative_batch",
        )
    raw = F.binary_cross_entropy_with_logits(logits, labels)
    return PretextLossTerm(
        term_name="lag_discrimination",
        weight=float(weight),
        status="active",
        count=int(logits.numel()),
        raw_loss=raw,
        weighted_loss=raw * weight,
        reason=None,
    )


def _validate_sequence_pair(positive, negative, *, representation_dim):
    if positive.ndim != 3 or negative.shape != positive.shape:
        raise ValueError("pretext sequences must share shape [B,Q,D]")
    if positive.shape[-1] != representation_dim:
        raise ValueError("pretext representation dimension mismatch")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("pretext sequences contain non-finite values")
