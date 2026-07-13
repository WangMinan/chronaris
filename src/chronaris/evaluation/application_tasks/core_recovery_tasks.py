"""Task-aware heads and losses for the Chronaris core-task recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from chronaris.modeling.fusion_encoders.observed_residual import (
    ObservedStateResidualOutput,
    masked_sequence_mean,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


@dataclass(frozen=True, slots=True)
class TaskAwareTargetBundle:
    sample_ids: tuple[str, ...]
    maneuver_class: torch.Tensor
    maneuver_score: torch.Tensor
    response_value: torch.Tensor
    high_response: torch.Tensor
    response_available: torch.Tensor
    field_deltas: torch.Tensor
    field_delta_mask: torch.Tensor
    maneuver_soft_targets: torch.Tensor | None = None
    maneuver_target_mode: str = "current_5s"

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if count == 0 or len(set(self.sample_ids)) != count:
            raise ValueError("task-aware target sample ids must be non-empty and unique")
        for name, values in (
            ("maneuver_class", self.maneuver_class),
            ("maneuver_score", self.maneuver_score),
            ("response_value", self.response_value),
            ("high_response", self.high_response),
            ("response_available", self.response_available),
        ):
            if tuple(values.shape) != (count,):
                raise ValueError(f"{name} must have shape [B]")
        if self.maneuver_class.dtype != torch.long:
            raise ValueError("maneuver_class must use torch.long")
        if self.response_available.dtype != torch.bool:
            raise ValueError("response_available must use torch.bool")
        if self.field_deltas.ndim != 2 or self.field_deltas.shape[0] != count:
            raise ValueError("field_deltas must have shape [B,F]")
        if self.field_delta_mask.shape != self.field_deltas.shape:
            raise ValueError("field_delta_mask must align with field_deltas")
        if self.field_delta_mask.dtype != torch.bool:
            raise ValueError("field_delta_mask must use torch.bool")
        if self.maneuver_soft_targets is not None:
            if tuple(self.maneuver_soft_targets.shape) != (count, 3):
                raise ValueError("maneuver_soft_targets must have shape [B,3]")
            sums = self.maneuver_soft_targets.sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
                raise ValueError("maneuver soft targets must sum to one")
        if bool((self.response_value[self.response_available] < 0).any()):
            raise ValueError("response values must be non-negative before log1p")
        if not bool(((self.maneuver_class >= 0) & (self.maneuver_class <= 2)).all()):
            raise ValueError("maneuver classes must be 0, 1, or 2")
        if self.maneuver_target_mode not in {"current_5s", "future_5s"}:
            raise ValueError("maneuver target mode is unsupported")


@dataclass(frozen=True, slots=True)
class CoreTaskHeadOutput:
    maneuver_logits: torch.Tensor
    maneuver_ordinal_logits: torch.Tensor
    maneuver_score: torch.Tensor
    response_log1p: torch.Tensor
    high_response_logit: torch.Tensor
    field_deltas: torch.Tensor
    maneuver_pooled: torch.Tensor
    response_pooled: torch.Tensor


@dataclass(frozen=True, slots=True)
class CoreTaskLossConfig:
    maneuver_classification_weight: float = 1.0
    maneuver_ordinal_weight: float = 0.5
    maneuver_score_weight: float = 0.25
    response_regression_weight: float = 1.0
    high_response_weight: float = 1.0
    field_delta_weight: float = 0.25
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.maneuver_classification_weight,
            self.maneuver_ordinal_weight,
            self.maneuver_score_weight,
            self.response_regression_weight,
            self.high_response_weight,
            self.field_delta_weight,
        )
        if any(value < 0 for value in values) or self.huber_delta <= 0:
            raise ValueError("core-task loss weights are invalid")


class ChronarisCoreTaskHeads(nn.Module):
    """One shared Chronaris backbone with maneuver and physiology-response heads."""

    def __init__(self, *, physiology_target_count: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if physiology_target_count <= 0 or hidden_dim <= 0:
            raise ValueError("task-head dimensions must be positive")
        self.physiology_target_count = int(physiology_target_count)
        summary_dim = FUSION_OUTPUT_DIM * 3
        self.maneuver_trunk = _task_trunk(summary_dim, hidden_dim)
        self.response_trunk = _task_trunk(summary_dim, hidden_dim)
        self.maneuver_classifier = nn.Linear(hidden_dim, 3)
        self.maneuver_ordinal = nn.Linear(hidden_dim, 2)
        self.maneuver_score = nn.Linear(hidden_dim, 1)
        self.response_log1p = nn.Linear(hidden_dim, 1)
        self.high_response = nn.Linear(hidden_dim, 1)
        self.field_delta = nn.Linear(hidden_dim, self.physiology_target_count)
        for layer in (
            self.maneuver_score,
            self.response_log1p,
            self.high_response,
            self.field_delta,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        representation: ObservedStateResidualOutput,
        *,
        residual_mode: str = "full",
    ) -> CoreTaskHeadOutput:
        maneuver_sequence = representation.sequence_for(
            "maneuver",
            mode=residual_mode,
        )
        response_sequence = representation.sequence_for(
            "physiology_response",
            mode=residual_mode,
        )
        return self.forward_sequences(
            maneuver_sequence=maneuver_sequence,
            response_sequence=response_sequence,
            valid_mask=representation.valid_mask,
        )

    def forward_sequences(
        self,
        *,
        maneuver_sequence: torch.Tensor,
        response_sequence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> CoreTaskHeadOutput:
        maneuver_pooled = _temporal_summary(
            maneuver_sequence,
            valid_mask,
            tail_points=16,
        )
        response_pooled = _temporal_summary(
            response_sequence,
            valid_mask,
            tail_points=None,
        )
        maneuver_hidden = self.maneuver_trunk(maneuver_pooled)
        response_hidden = self.response_trunk(response_pooled)
        return CoreTaskHeadOutput(
            maneuver_logits=self.maneuver_classifier(maneuver_hidden),
            maneuver_ordinal_logits=self.maneuver_ordinal(maneuver_hidden),
            maneuver_score=self.maneuver_score(maneuver_hidden).squeeze(-1),
            response_log1p=self.response_log1p(response_hidden).squeeze(-1),
            high_response_logit=self.high_response(response_hidden).squeeze(-1),
            field_deltas=self.field_delta(response_hidden),
            maneuver_pooled=maneuver_pooled,
            response_pooled=response_pooled,
        )


def core_task_losses(
    output: CoreTaskHeadOutput,
    targets: TaskAwareTargetBundle,
    *,
    config: CoreTaskLossConfig | None = None,
) -> Mapping[str, torch.Tensor]:
    resolved = config or CoreTaskLossConfig()
    if output.maneuver_logits.shape != (len(targets.sample_ids), 3):
        raise RepresentationContractError("maneuver logits and targets do not align")
    class_weights = _balanced_class_weights(targets.maneuver_class, class_count=3)
    if targets.maneuver_soft_targets is None:
        classification = F.cross_entropy(
            output.maneuver_logits,
            targets.maneuver_class,
            weight=class_weights,
        )
    else:
        per_sample = -(targets.maneuver_soft_targets * F.log_softmax(
            output.maneuver_logits,
            dim=-1,
        )).sum(dim=-1)
        classification = (
            per_sample * class_weights.index_select(0, targets.maneuver_class)
        ).mean()
    ordinal_targets = torch.stack(
        (
            (targets.maneuver_class > 0).to(output.maneuver_logits.dtype),
            (targets.maneuver_class > 1).to(output.maneuver_logits.dtype),
        ),
        dim=-1,
    )
    ordinal = F.binary_cross_entropy_with_logits(
        output.maneuver_ordinal_logits,
        ordinal_targets,
    )
    score = F.huber_loss(
        output.maneuver_score,
        targets.maneuver_score,
        delta=resolved.huber_delta,
    )
    response_mask = targets.response_available
    if not bool(response_mask.any()):
        raise RepresentationContractError("a task batch must contain a response target")
    response = F.huber_loss(
        output.response_log1p[response_mask],
        torch.log1p(targets.response_value[response_mask]),
        delta=resolved.huber_delta,
    )
    high = F.binary_cross_entropy_with_logits(
        output.high_response_logit[response_mask],
        targets.high_response[response_mask].to(output.high_response_logit.dtype),
        pos_weight=_positive_class_weight(targets.high_response[response_mask]),
    )
    field_mask = targets.field_delta_mask & response_mask.unsqueeze(-1)
    if bool(field_mask.any()):
        field_delta = F.huber_loss(
            output.field_deltas[field_mask],
            targets.field_deltas[field_mask],
            delta=resolved.huber_delta,
        )
    else:
        field_delta = output.field_deltas.sum() * 0
    weighted = {
        "maneuver_classification": classification * resolved.maneuver_classification_weight,
        "maneuver_ordinal": ordinal * resolved.maneuver_ordinal_weight,
        "maneuver_score": score * resolved.maneuver_score_weight,
        "response_regression": response * resolved.response_regression_weight,
        "high_response": high * resolved.high_response_weight,
        "field_delta": field_delta * resolved.field_delta_weight,
    }
    return {**weighted, "total": sum(weighted.values())}


def select_task_aware_targets(
    targets: TaskAwareTargetBundle,
    sample_ids: tuple[str, ...],
    *,
    device: str | torch.device | None = None,
) -> TaskAwareTargetBundle:
    positions = {sample_id: index for index, sample_id in enumerate(targets.sample_ids)}
    try:
        indices = torch.tensor([positions[value] for value in sample_ids], dtype=torch.long)
    except KeyError as error:
        raise KeyError(f"unknown task-aware target sample: {error.args[0]}") from error
    destination = torch.device(device) if device is not None else targets.maneuver_class.device

    def take(values: torch.Tensor) -> torch.Tensor:
        return values.index_select(0, indices.to(values.device)).to(destination)

    return TaskAwareTargetBundle(
        sample_ids=sample_ids,
        maneuver_class=take(targets.maneuver_class),
        maneuver_score=take(targets.maneuver_score),
        response_value=take(targets.response_value),
        high_response=take(targets.high_response),
        response_available=take(targets.response_available),
        field_deltas=take(targets.field_deltas),
        field_delta_mask=take(targets.field_delta_mask),
        maneuver_soft_targets=(
            None
            if targets.maneuver_soft_targets is None
            else take(targets.maneuver_soft_targets)
        ),
        maneuver_target_mode=targets.maneuver_target_mode,
    )


def initialize_task_head_biases(
    heads: ChronarisCoreTaskHeads,
    targets: TaskAwareTargetBundle,
) -> Mapping[str, float]:
    """Initialize output priors from inner-train labels only."""

    with torch.no_grad():
        class_counts = torch.bincount(targets.maneuver_class, minlength=3).to(torch.float32)
        class_probabilities = (class_counts + 1) / (class_counts.sum() + 3)
        heads.maneuver_classifier.bias.copy_(class_probabilities.log())
        heads.maneuver_score.bias.fill_(float(targets.maneuver_score.mean()))
        ordinal_probabilities = torch.stack(
            (
                (targets.maneuver_class > 0).to(torch.float32).mean(),
                (targets.maneuver_class > 1).to(torch.float32).mean(),
            )
        ).clamp(1e-4, 1 - 1e-4)
        heads.maneuver_ordinal.bias.copy_(torch.logit(ordinal_probabilities))

        response_mask = targets.response_available
        response_values = targets.response_value[response_mask]
        response_prior = torch.log1p(response_values).mean()
        heads.response_log1p.bias.fill_(float(response_prior))
        high_prior = targets.high_response[response_mask].to(torch.float32).mean().clamp(
            1e-4,
            1 - 1e-4,
        )
        heads.high_response.bias.fill_(float(torch.logit(high_prior)))
        field_mask = targets.field_delta_mask & response_mask.unsqueeze(-1)
        counts = field_mask.sum(dim=0).clamp_min(1)
        field_prior = (targets.field_deltas * field_mask).sum(dim=0) / counts
        heads.field_delta.bias.copy_(field_prior)
    return {
        "maneuver_score": float(targets.maneuver_score.mean()),
        "response_log1p": float(response_prior),
        "high_response_probability": float(high_prior),
    }


def _task_trunk(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
    )


def _temporal_summary(
    sequence: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tail_points: int | None,
) -> torch.Tensor:
    """Retain level, endpoint and trend without exposing target-horizon data."""

    if tail_points is not None:
        if tail_points <= 0:
            raise ValueError("tail_points must be positive")
        sequence = sequence[:, -tail_points:]
        valid_mask = valid_mask[:, -tail_points:]
    mean = masked_sequence_mean(sequence, valid_mask)
    batch_positions = torch.arange(sequence.shape[0], device=sequence.device)
    valid_positions = torch.arange(sequence.shape[1], device=sequence.device).view(1, -1)
    first_indices = torch.where(
        valid_mask,
        valid_positions,
        torch.full_like(valid_positions, sequence.shape[1]),
    ).min(dim=1).values
    last_indices = torch.where(
        valid_mask,
        valid_positions,
        torch.full_like(valid_positions, -1),
    ).max(dim=1).values
    if bool((first_indices >= sequence.shape[1]).any()) or bool((last_indices < 0).any()):
        raise RepresentationContractError("temporal summary has no valid query")
    first = sequence[batch_positions, first_indices]
    last = sequence[batch_positions, last_indices]
    return torch.cat((mean, last, last - first), dim=-1)


def _balanced_class_weights(labels: torch.Tensor, *, class_count: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=class_count).to(torch.float32)
    weights = torch.where(
        counts > 0,
        counts.sum() / (class_count * counts.clamp_min(1)),
        torch.zeros_like(counts),
    )
    return weights.to(labels.device)


def _positive_class_weight(labels: torch.Tensor) -> torch.Tensor:
    values = labels.to(torch.float32)
    positives = values.sum()
    negatives = values.numel() - positives
    if positives <= 0 or negatives <= 0:
        return torch.ones((), dtype=torch.float32, device=labels.device)
    return (negatives / positives).to(labels.device)
