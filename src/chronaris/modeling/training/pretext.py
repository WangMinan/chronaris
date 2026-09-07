"""Shared task-independent pretext heads, losses, and Chronaris weight schedule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    components: tuple[PretextLossTerm, ...] = ()


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

    def __init__(self, *, representation_dim: int, target_feature_count: int,
                 modality_feature_counts: tuple[int, int] | None = None,
                 input_streams: tuple[str, ...] = ("physiology", "vehicle")) -> None:
        super().__init__()
        if representation_dim <= 0 or target_feature_count <= 0:
            raise ValueError("pretext head dimensions must be positive")
        self.representation_dim = representation_dim
        self.target_feature_count = target_feature_count
        self.modality_feature_counts = tuple(modality_feature_counts) if modality_feature_counts is not None else None
        self.input_streams = tuple(input_streams)
        if (not self.input_streams or len(set(self.input_streams)) != len(self.input_streams)
            or not set(self.input_streams) <= {"physiology", "vehicle"}):
            raise ValueError("invalid pretext input modalities")
        if self.modality_feature_counts is not None and (
            len(self.modality_feature_counts) != 2 or min(self.modality_feature_counts) <= 0
            or sum(self.modality_feature_counts) != target_feature_count
        ):
            raise ValueError("pretext modality feature counts differ from target schema")
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

    @property
    def cross_stream_enabled(self):
        return self.modality_feature_counts is None or len(self.input_streams) == 2

    def objective_config(self):
        return {"modality_feature_counts": self.modality_feature_counts, "input_streams": self.input_streams}

    def forward(
        self,
        positive_sequence: torch.Tensor,
        negative_sequence: torch.Tensor,
        targets: CommonPretextTargets,
        *,
        weights: CommonPretextWeights | None = None,
        positive_valid_mask: torch.Tensor | None = None,
        negative_valid_mask: torch.Tensor | None = None,
        lag_valid_mask: torch.Tensor | None = None,
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
        if self.modality_feature_counts is not None and (positive_valid_mask is None or negative_valid_mask is None):
            raise ValueError("v4 pretext requires encoder validity masks")
        if self.cross_stream_enabled:
            positive_logits = self.lag_head(_pretext_pool(positive_sequence, positive_valid_mask)).squeeze(-1)
            negative_logits = self.lag_head(_pretext_pool(negative_sequence, negative_valid_mask)).squeeze(-1)
        else:
            positive_logits = positive_sequence.new_zeros(positive_sequence.shape[0])
            negative_logits = torch.zeros_like(positive_logits)
            lag_valid_mask = torch.zeros_like(positive_logits, dtype=torch.bool)
        terms = (
            self._reconstruction_term(
                "masked_reconstruction",
                reconstruction,
                targets.reconstruction_target,
                targets.reconstruction_mask,
                weight=resolved_weights.masked_reconstruction,
            ),
            self._reconstruction_term(
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
                valid_mask=lag_valid_mask,
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

    def _reconstruction_term(self, name, prediction, target, mask, *, weight):
        if self.modality_feature_counts is None:
            return _masked_huber_term(name, prediction, target, mask, weight=weight)
        first, second = self.modality_feature_counts
        slices = {"physiology": slice(0, first), "vehicle": slice(first, first + second)}
        components = tuple(_masked_huber_term(stream, prediction[..., slices[stream]],
            target[..., slices[stream]], mask[..., slices[stream]], weight=weight) for stream in self.input_streams)
        active = [term.raw_loss for term in components if term.count]
        raw = torch.stack(active).mean() if active else None
        return PretextLossTerm(name, weight, "active" if active else "unavailable",
            sum(term.count for term in components), raw, raw * weight if raw is not None else None,
            None if active else "no_observed_input_modality_targets", components)


def _pretext_pool(sequence, valid_mask):
    if valid_mask is None:
        return sequence.mean(dim=1)
    if valid_mask.shape != sequence.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("pretext pool validity mask is invalid")
    return sequence.masked_fill(~valid_mask[..., None], 0).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1)


class ExplicitTimeShiftHead(nn.Module):
    """Five-class shift head kept separate from legacy common checkpoints."""

    def __init__(self, representation_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, 5),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if sequence.ndim != 3 or valid_mask.shape != sequence.shape[:2]:
            raise ValueError("explicit time-shift head expects [B,Q,D] and [B,Q]")
        count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        pooled = (sequence * valid_mask.unsqueeze(-1).to(sequence.dtype)).sum(dim=1) / count
        return self.network(pooled)


def explicit_time_shift_loss_term(
    logits: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    weight: float,
) -> PretextLossTerm:
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError("explicit time-shift logits must have shape [B,5]")
    if class_indices.shape != logits.shape[:1]:
        raise ValueError("explicit time-shift class shape mismatch")
    raw = F.cross_entropy(logits, class_indices.to(logits.device))
    return PretextLossTerm(
        term_name="explicit_time_shift",
        weight=float(weight),
        status="active",
        count=len(class_indices),
        raw_loss=raw,
        weighted_loss=raw * weight,
        reason=None,
    )


def event_pair_contrastive_loss_term(
    semantic_output,
    group_ids: Sequence[str],
    *,
    temperature: float,
    weight: float,
) -> tuple[PretextLossTerm, dict[str, float | int | None]]:
    if temperature <= 0:
        raise ValueError("event-pair temperature must be positive")
    names = tuple(semantic_output.query_names)
    if "flight_event" not in names or "physiology_response" not in names:
        raise ValueError("event-pair loss requires flight_event and physiology_response queries")
    event_states = semantic_output.query_context_states[:, names.index("flight_event")]
    response_states = semantic_output.query_context_states[:, names.index("physiology_response")]
    if len(group_ids) != len(event_states):
        raise ValueError("event-pair group count mismatch")
    anchors: list[int] = []
    negatives: list[int] = []
    groups = tuple(str(value) for value in group_ids)
    for anchor, group in enumerate(groups):
        for offset in range(1, len(groups)):
            candidate = (anchor + offset) % len(groups)
            if groups[candidate] != group:
                anchors.append(anchor)
                negatives.append(candidate)
                break
    if not anchors:
        zero = event_states.sum() * 0.0
        return (
            PretextLossTerm(
                term_name="event_response_pairing",
                weight=float(weight),
                status="unavailable",
                count=0,
                raw_loss=None,
                weighted_loss=zero,
                reason="no_cross_group_negative",
            ),
            {
                "positive_similarity": None,
                "negative_similarity": None,
                "recall_at_1": None,
                "valid_pair_count": 0,
            },
        )
    anchor_index = torch.as_tensor(anchors, dtype=torch.long, device=event_states.device)
    negative_index = torch.as_tensor(negatives, dtype=torch.long, device=event_states.device)
    event = F.normalize(event_states.index_select(0, anchor_index), dim=-1, eps=1e-12)
    positive = F.normalize(response_states.index_select(0, anchor_index), dim=-1, eps=1e-12)
    negative = F.normalize(response_states.index_select(0, negative_index), dim=-1, eps=1e-12)
    positive_similarity = (event * positive).sum(dim=-1)
    negative_similarity = (event * negative).sum(dim=-1)
    logits = torch.stack((positive_similarity, negative_similarity), dim=-1) / temperature
    raw = F.cross_entropy(logits, torch.zeros(len(anchors), dtype=torch.long, device=logits.device))
    term = PretextLossTerm(
        term_name="event_response_pairing",
        weight=float(weight),
        status="active",
        count=len(anchors),
        raw_loss=raw,
        weighted_loss=raw * weight,
        reason=None,
    )
    return term, {
        "positive_similarity": float(positive_similarity.mean().detach().cpu()),
        "negative_similarity": float(negative_similarity.mean().detach().cpu()),
        "recall_at_1": float((positive_similarity > negative_similarity).float().mean().detach().cpu()),
        "valid_pair_count": len(anchors),
    }


def independent_window_pair_loss_term(pairing, group_ids, *, weight):
    """Symmetric window pairing against all valid other groups in the actual batch."""
    physiology, vehicle = pairing.physiology, pairing.vehicle
    size = len(physiology)
    if physiology.shape != vehicle.shape or physiology.ndim != 2 or physiology.shape[1] != 32 or len(group_ids) != size:
        raise ValueError("independent pairing vectors/groups have inconsistent dimensions")
    if not 0 <= weight < float("inf"):
        raise ValueError("independent pairing weight must be finite and non-negative")
    valid = pairing.physiology_valid & pairing.vehicle_valid
    if valid.shape != (size,) or valid.dtype != torch.bool:
        raise ValueError("independent pairing requires explicit window masks")
    if not torch.isfinite(physiology[valid]).all() or not torch.isfinite(vehicle[valid]).all():
        raise ValueError("valid independent pairing vectors must be finite")
    groups = tuple(str(value) for value in group_ids)
    if not all(groups):
        raise ValueError("independent pairing requires nonempty group identifiers")
    different = torch.tensor([[a != b for b in groups] for a in groups], dtype=torch.bool, device=physiology.device)
    negatives = different & valid[:, None] & valid[None, :]
    active = negatives.any(dim=1)
    count = int(active.sum())
    metrics = {"valid_pair_count": count, "negative_pair_count": int(negatives.sum()), "actual_batch_size": size,
               "positive_similarity": None, "negative_similarity": None, "recall_at_1": None}
    if not count:
        return PretextLossTerm("independent_window_pairing", float(weight), "unavailable", 0, None, None,
                               "no_valid_cross_group_negative"), metrics
    physiology = F.normalize(physiology.masked_fill(~valid[:, None], 0), dim=-1, eps=1e-12)
    vehicle = F.normalize(vehicle.masked_fill(~valid[:, None], 0), dim=-1, eps=1e-12)
    similarity = physiology @ vehicle.T
    allowed = negatives | torch.eye(size, dtype=torch.bool, device=physiology.device)
    forward = (similarity / .1).masked_fill(~allowed, -torch.inf)[active]
    reverse = (similarity.T / .1).masked_fill(~allowed, -torch.inf)[active]
    labels = torch.arange(size, device=physiology.device)[active]
    loss = (F.cross_entropy(forward, labels) + F.cross_entropy(reverse, labels)) / 2
    metrics.update(positive_similarity=float(similarity.diag()[active].mean().detach()),
        negative_similarity=float(similarity[negatives].mean().detach()),
        recall_at_1=float(((forward.argmax(-1) == labels).float().mean() + (reverse.argmax(-1) == labels).float().mean()).detach() / 2))
    return PretextLossTerm("independent_window_pairing", float(weight), "active" if weight else "disabled",
                           count, loss, loss * weight, None), metrics


def chronaris_auxiliary_weight_schedule(
    epoch: int,
    *, optimizer_updates: int | None = None, continuous_alignment_weight: float = 0.2,
) -> ChronarisAuxiliaryWeights:
    """Linearly warm mechanism losses from epoch one through epoch five."""

    if epoch <= 0:
        raise ValueError("training epoch must be one-based and positive")
    if not 0 <= continuous_alignment_weight < float("inf"):
        raise ValueError("continuous alignment weight must be finite and non-negative")
    if optimizer_updates is not None and optimizer_updates < 0:
        raise ValueError("completed optimizer updates must be non-negative")
    fraction = (
        min(epoch / 5.0, 1.0) if optimizer_updates is None
        else min(max((optimizer_updates - 50) / 150.0, 0.0), 1.0)
    )
    return ChronarisAuxiliaryWeights(
        continuous_alignment=continuous_alignment_weight * fraction,
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
                "modality_losses": {
                    component.term_name: {"count": component.count,
                        "raw_loss": float(component.raw_loss.detach()) if component.raw_loss is not None else None}
                    for component in term.components
                },
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
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("observed pretext prediction/target is non-finite")
    raw = F.smooth_l1_loss(prediction[mask], target[mask])
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
    valid_mask=None,
) -> PretextLossTerm:
    if valid_mask is not None:
        if valid_mask.shape != positive_logits.shape or valid_mask.dtype != torch.bool:
            raise ValueError("lag validity must identify available cross-stream samples")
        positive_logits, negative_logits = positive_logits[valid_mask], negative_logits[valid_mask]
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
