"""Contracts and OOF teacher targets for Dingxin residual activation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    SafeAnchorPredictions,
)


TASK_NAMES = ("maneuver", "maneuver_score", "response", "high_response")


@dataclass(frozen=True, slots=True)
class ResidualActivationCandidate:
    """One bounded stage-3A candidate."""

    candidate_id: str
    gate_mode: str
    initial_gate: float
    objective_mode: str
    adapter_mode: str
    use_distillation: bool
    learning_rate: float = 5e-3
    distillation_weight: float = 0.35
    warmup_epochs: int = 8

    def __post_init__(self) -> None:
        if self.gate_mode not in {"fixed", "global", "conditional"}:
            raise ValueError("stage-3A gate mode is invalid")
        if self.objective_mode not in {"full", "explicit_residual"}:
            raise ValueError("stage-3A objective mode is invalid")
        if self.adapter_mode not in {"shared", "task_specific"}:
            raise ValueError("stage-3A adapter mode is invalid")
        if not 0 < self.initial_gate < 0.5:
            raise ValueError("stage-3A initial gate must be inside (0, 0.5)")
        if min(self.learning_rate, self.warmup_epochs) <= 0:
            raise ValueError("stage-3A candidate schedule must be positive")


CANDIDATES = (
    ResidualActivationCandidate(
        "fixed015_full_shared",
        gate_mode="fixed",
        initial_gate=0.15,
        objective_mode="full",
        adapter_mode="shared",
        use_distillation=False,
    ),
    ResidualActivationCandidate(
        "fixed025_residual_task",
        gate_mode="fixed",
        initial_gate=0.25,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=False,
    ),
    ResidualActivationCandidate(
        "global015_residual_task",
        gate_mode="global",
        initial_gate=0.15,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=False,
    ),
    ResidualActivationCandidate(
        "conditional015_residual_task",
        gate_mode="conditional",
        initial_gate=0.15,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=False,
    ),
    ResidualActivationCandidate(
        "conditional025_residual_task",
        gate_mode="conditional",
        initial_gate=0.25,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=False,
    ),
    ResidualActivationCandidate(
        "conditional015_residual_task_distill",
        gate_mode="conditional",
        initial_gate=0.15,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=True,
    ),
    ResidualActivationCandidate(
        "conditional025_residual_task_distill",
        gate_mode="conditional",
        initial_gate=0.25,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=True,
    ),
    ResidualActivationCandidate(
        "global025_residual_task_distill",
        gate_mode="global",
        initial_gate=0.25,
        objective_mode="explicit_residual",
        adapter_mode="task_specific",
        use_distillation=True,
    ),
)


@dataclass(frozen=True, slots=True)
class SelectiveTeacherTargets:
    """OOF teacher targets and per-sample selection weights."""

    maneuver_probability: np.ndarray
    maneuver_weight: np.ndarray
    maneuver_score: np.ndarray
    maneuver_score_weight: np.ndarray
    response: np.ndarray
    response_weight: np.ndarray
    high_response_probability: np.ndarray
    high_response_weight: np.ndarray

    def summary(self) -> Mapping[str, float]:
        return {
            "maneuver_selected_fraction": float(np.mean(self.maneuver_weight > 0)),
            "maneuver_score_selected_fraction": float(
                np.mean(self.maneuver_score_weight > 0)
            ),
            "response_selected_fraction": float(np.mean(self.response_weight > 0)),
            "high_response_selected_fraction": float(
                np.mean(self.high_response_weight > 0)
            ),
            "mean_maneuver_weight": float(np.mean(self.maneuver_weight)),
            "mean_response_weight": float(np.mean(self.response_weight)),
            "mean_high_response_weight": float(
                np.mean(self.high_response_weight)
            ),
        }


@dataclass(frozen=True, slots=True)
class ActivationTrainingConfig:
    max_epochs: int = 80
    patience: int = 12
    hidden_dim: int = 64
    dropout: float = 0.1
    weight_decay: float = 1e-4
    trust_region_weight: float = 0.05
    released_gate_regularization: float = 1e-3
    score_loss_weight: float = 0.2
    gradient_clip_norm: float = 5.0
    maximum_gate: float = 0.5
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(
            self.max_epochs,
            self.patience,
            self.hidden_dim,
            self.gradient_clip_norm,
        ) <= 0:
            raise ValueError("stage-3A training configuration must be positive")
        if not 0 <= self.dropout < 1 or self.weight_decay < 0:
            raise ValueError("stage-3A regularization configuration is invalid")
        if not 0 < self.maximum_gate <= 1:
            raise ValueError("stage-3A maximum gate is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("stage-3A device is invalid")


@dataclass(frozen=True, slots=True)
class ActivationTrainingResult:
    model_state_dict: Mapping[str, object]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    epoch_rows: tuple[Mapping[str, object], ...]
    direct_metrics: Mapping[str, Mapping[str, float]]
    full_metrics: Mapping[str, Mapping[str, float]]
    prediction_diagnostics: Mapping[str, float]
    gate_summary: Mapping[str, Mapping[str, float]]
    teacher_summary: Mapping[str, float]
    best_epoch: int
    task_best_epochs: Mapping[str, int]
    safety_passed: bool


def build_selective_teacher_targets(
    *,
    method_anchors: Mapping[str, SafeAnchorPredictions],
    safe_anchor: SafeAnchorPredictions,
    targets: FoldTargets,
) -> SelectiveTeacherTargets:
    """Create selective OOF soft targets without using validation labels."""

    maneuver_methods = ("vehicle_only", "contiformer")
    response_methods = ("mult", "contiformer", "naive_time_sync")
    risk_methods = ("vehicle_only", "mult")
    missing = sorted(
        set(maneuver_methods + response_methods + risk_methods) - set(method_anchors)
    )
    if missing:
        raise KeyError(f"stage-3A teachers missing: {missing}")
    maneuver_probabilities = np.stack(
        [_softmax(method_anchors[name].maneuver_logits.train) for name in maneuver_methods]
    )
    maneuver_probability = maneuver_probabilities.mean(axis=0)
    anchor_probability = _softmax(safe_anchor.maneuver_logits.train)
    row = np.arange(targets.train_maneuver.shape[0])
    maneuver_advantage = (
        maneuver_probability[row, targets.train_maneuver]
        - anchor_probability[row, targets.train_maneuver]
    )
    maneuver_consensus = 1.0 - _normalized_entropy(maneuver_probability)
    maneuver_weight = maneuver_consensus * (
        maneuver_advantage > 0.01
    ).astype(np.float32)
    teacher_prediction = maneuver_probability.argmax(axis=1)
    anchor_prediction = anchor_probability.argmax(axis=1)
    correction = (teacher_prediction == targets.train_maneuver) & (
        anchor_prediction != targets.train_maneuver
    )
    maneuver_weight = np.maximum(maneuver_weight, correction.astype(np.float32))

    maneuver_scores = np.stack(
        [method_anchors[name].maneuver_score.train for name in maneuver_methods]
    )
    maneuver_score = maneuver_scores.mean(axis=0)
    score_scale = max(float(np.std(targets.train_maneuver_score)), 1e-6)
    score_agreement = np.exp(-maneuver_scores.std(axis=0) / score_scale)
    score_improved = (
        np.abs(maneuver_score - targets.train_maneuver_score)
        < np.abs(safe_anchor.maneuver_score.train - targets.train_maneuver_score)
    )
    maneuver_score_weight = score_agreement * score_improved.astype(np.float32)

    responses = np.stack(
        [method_anchors[name].response.train for name in response_methods]
    )
    response = responses.mean(axis=0)
    response_scale = max(float(np.std(targets.train_response)), 1e-6)
    response_agreement = np.exp(-responses.std(axis=0) / response_scale)
    response_improved = (
        np.abs(response - targets.train_response)
        < np.abs(safe_anchor.response.train - targets.train_response)
    )
    response_weight = response_agreement * response_improved.astype(np.float32)

    risk_probabilities = np.stack(
        [
            _sigmoid(method_anchors[name].high_response_logit.train)
            for name in risk_methods
        ]
    )
    high_response_probability = risk_probabilities.mean(axis=0)
    anchor_risk_probability = _sigmoid(safe_anchor.high_response_logit.train)
    risk_target = targets.train_high_response.astype(np.float32)
    risk_improved = (
        np.abs(high_response_probability - risk_target)
        < np.abs(anchor_risk_probability - risk_target)
    )
    risk_consensus = 1.0 - _binary_entropy(high_response_probability)
    high_response_weight = risk_consensus * risk_improved.astype(np.float32)
    return SelectiveTeacherTargets(
        maneuver_probability=maneuver_probability.astype(np.float32),
        maneuver_weight=maneuver_weight.astype(np.float32),
        maneuver_score=maneuver_score.astype(np.float32),
        maneuver_score_weight=maneuver_score_weight.astype(np.float32),
        response=response.astype(np.float32),
        response_weight=response_weight.astype(np.float32),
        high_response_probability=high_response_probability.astype(np.float32),
        high_response_weight=high_response_weight.astype(np.float32),
    )


def _softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(np.clip(values, -30, 30))
    return exponent / exponent.sum(axis=1, keepdims=True)


def _sigmoid(logits):
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _normalized_entropy(probability):
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0)
    return -(values * np.log(values)).sum(axis=1) / math.log(values.shape[1])


def _binary_entropy(probability):
    values = np.clip(np.asarray(probability, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return -(values * np.log(values) + (1.0 - values) * np.log(1.0 - values)) / math.log(2.0)
