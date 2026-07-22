"""Partially unfreeze Chronaris behind immutable task-safe anchors."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    SafeAnchorPredictions,
    TaskSafeResidualHead,
    evaluate_task_predictions,
    gate_is_non_degenerate,
    safe_against_anchor,
    summarize_sequence_tensor,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training import TrainableFusionEncoder
from chronaris.representation import DualStreamObservationBatch, TrainOnlyRobustNormalizer


PARTIAL_UNFREEZE_PATTERNS = (
    "continuous_backbone.physiology_stream.encoder",
    "continuous_backbone.physiology_stream.ode_rnn_cell.observation_update",
    "continuous_backbone.vehicle_stream.encoder",
    "continuous_backbone.vehicle_stream.ode_rnn_cell.observation_update",
    "causal_fusion.scale_gate",
    "causal_fusion.output_projection",
)


@dataclass(frozen=True, slots=True)
class PartialUnfreezeConfig:
    head_learning_rate: float = 5e-3
    backbone_learning_rate_ratio: float = 0.02
    weight_decay: float = 1e-4
    max_epochs: int = 24
    patience: int = 6
    gradient_clip_norm: float = 2.0
    score_loss_weight: float = 0.2
    gate_penalty: float = 1e-2
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(
            self.head_learning_rate,
            self.backbone_learning_rate_ratio,
            self.max_epochs,
            self.patience,
            self.gradient_clip_norm,
        ) <= 0:
            raise ValueError("partial-unfreeze configuration must be positive")
        if self.weight_decay < 0 or self.device not in {"cpu", "cuda"}:
            raise ValueError("partial-unfreeze optimizer or device is invalid")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("partial-unfreeze requested unavailable CUDA device")


@dataclass(frozen=True, slots=True)
class PartialUnfreezeResult:
    model_state_dict: Mapping[str, torch.Tensor]
    epoch_rows: tuple[Mapping[str, object], ...]
    frozen_metrics: Mapping[str, Mapping[str, float]]
    partial_metrics: Mapping[str, Mapping[str, float]]
    gate_values: Mapping[str, tuple[float, ...]]
    best_epoch: int
    trainable_encoder_parameter_names: tuple[str, ...]
    safety_passed: bool


class PartiallyTrainableTaskChronaris(nn.Module):
    """Chronaris encoder plus the pre-screened task residual head."""

    def __init__(
        self,
        *,
        encoder: TrainableFusionEncoder,
        head: TaskSafeResidualHead,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.register_buffer(
            "feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(feature_scale, dtype=torch.float32)
        )

    def features(self, batch: DualStreamObservationBatch) -> torch.Tensor:
        output = self.encoder(batch)
        valid_mask = torch.ones(
            output.sequence_embedding.shape[:2],
            dtype=torch.bool,
            device=output.sequence_embedding.device,
        )
        summary = summarize_sequence_tensor(output.sequence_embedding, valid_mask)
        return (summary - self.feature_mean) / self.feature_scale


def configure_partial_unfreeze(
    encoder: TrainableFusionEncoder,
    patterns: Sequence[str] = PARTIAL_UNFREEZE_PATTERNS,
) -> tuple[str, ...]:
    """Freeze the encoder except the pre-registered observation/fusion modules."""

    selected = []
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = any(pattern in name for pattern in patterns)
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError("partial-unfreeze patterns matched no encoder parameters")
    return tuple(selected)


def train_partially_unfrozen_chronaris(
    *,
    encoder: TrainableFusionEncoder,
    normalizer: TrainOnlyRobustNormalizer,
    train_maneuver_batch: DualStreamObservationBatch,
    validation_maneuver_batch: DualStreamObservationBatch,
    train_response_batch: DualStreamObservationBatch,
    validation_response_batch: DualStreamObservationBatch,
    anchors: SafeAnchorPredictions,
    targets: FoldTargets,
    residual_state_dict: Mapping[str, torch.Tensor],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    gate_mode: str,
    config: PartialUnfreezeConfig,
) -> PartialUnfreezeResult:
    """Fine-tune only the bounded Chronaris subset while preserving task anchors."""

    torch.manual_seed(config.seed)
    trainable_names = configure_partial_unfreeze(encoder)
    head = TaskSafeResidualHead(
        int(np.asarray(feature_mean).shape[0]),
        gate_mode=gate_mode,
    )
    head.load_state_dict(residual_state_dict, strict=True)
    model = PartiallyTrainableTaskChronaris(
        encoder=encoder,
        head=head,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    ).to(config.device)
    batches = {
        "train_maneuver": _normalized_batch(
            normalizer, train_maneuver_batch, config.device
        ),
        "validation_maneuver": _normalized_batch(
            normalizer, validation_maneuver_batch, config.device
        ),
        "train_response": _normalized_batch(
            normalizer, train_response_batch, config.device
        ),
        "validation_response": _normalized_batch(
            normalizer, validation_response_batch, config.device
        ),
    }
    tensors = _task_tensors(anchors=anchors, targets=targets, device=config.device)
    score_scale = max(float(np.std(targets.train_maneuver_score)), 1e-6)
    response_scale = max(float(np.std(targets.train_response)), 1e-6)
    head_parameters = list(model.head.parameters())
    backbone_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        (
            {"params": head_parameters, "lr": config.head_learning_rate},
            {
                "params": backbone_parameters,
                "lr": config.head_learning_rate * config.backbone_learning_rate_ratio,
            },
        ),
        weight_decay=config.weight_decay,
    )
    frozen_metrics = _evaluate(
        model,
        batches=batches,
        tensors=tensors,
        targets=targets,
        score_scale=score_scale,
        response_scale=response_scale,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = _selection_score(frozen_metrics)
    best_epoch = 0
    stale = 0
    epoch_rows = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        maneuver_features = model.features(batches["train_maneuver"])
        response_features = model.features(batches["train_response"])
        maneuver = model.head.forward_maneuver(
            maneuver_features,
            tensors["train_maneuver_anchor_logits"],
            tensors["train_maneuver_anchor_score"],
            score_scale=score_scale,
        )
        response = model.head.forward_response(
            response_features,
            tensors["train_response_anchor"],
            tensors["train_risk_anchor_logit"],
            response_scale=response_scale,
        )
        losses = _losses(
            maneuver=maneuver,
            response=response,
            tensors=tensors,
            score_scale=score_scale,
            response_scale=response_scale,
        )
        gate_penalty = sum(
            torch.sigmoid(parameter).mean()
            for name, parameter in model.head.named_parameters()
            if name.endswith("gate_logit")
        )
        total = (
            losses["maneuver"]
            + config.score_loss_weight * losses["score"]
            + losses["response"]
            + losses["risk"]
            + config.gate_penalty * gate_penalty
        )
        total.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        )
        optimizer.step()
        metrics = _evaluate(
            model,
            batches=batches,
            tensors=tensors,
            targets=targets,
            score_scale=score_scale,
            response_scale=response_scale,
        )
        safe = safe_against_anchor(metrics, frozen_metrics)
        score = _selection_score(metrics)
        improved = bool(safe and score < best_score - 1e-9)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_score = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                **{
                    f"train_{name}_loss": float(value.detach())
                    for name, value in losses.items()
                },
                "train_total_loss": float(total.detach()),
                "gradient_norm_before_clip": gradient_norm,
                "validation_selection_score": score,
                "validation_safe": safe,
                "improved": improved,
                "gate_maneuver": float(
                    np.mean(model.head.gate_values()["maneuver"])
                ),
                "gate_response": float(model.head.gate_values()["response"][0]),
                "gate_high_response": float(
                    model.head.gate_values()["high_response"][0]
                ),
            }
        )
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    partial_metrics = _evaluate(
        model,
        batches=batches,
        tensors=tensors,
        targets=targets,
        score_scale=score_scale,
        response_scale=response_scale,
    )
    gates = model.head.gate_values()
    return PartialUnfreezeResult(
        model_state_dict={
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        epoch_rows=tuple(epoch_rows),
        frozen_metrics=frozen_metrics,
        partial_metrics=partial_metrics,
        gate_values=gates,
        best_epoch=best_epoch,
        trainable_encoder_parameter_names=trainable_names,
        safety_passed=bool(
            safe_against_anchor(partial_metrics, frozen_metrics)
            and gate_is_non_degenerate(gates)
        ),
    )


def _normalized_batch(normalizer, batch, device):
    return move_observation_batch(normalizer.transform(batch), device=device)


def _task_tensors(*, anchors, targets, device):
    def floating(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    return {
        "train_maneuver_anchor_logits": floating(anchors.maneuver_logits.train),
        "validation_maneuver_anchor_logits": floating(
            anchors.maneuver_logits.validation
        ),
        "train_maneuver_anchor_score": floating(anchors.maneuver_score.train),
        "validation_maneuver_anchor_score": floating(
            anchors.maneuver_score.validation
        ),
        "train_response_anchor": floating(anchors.response.train),
        "validation_response_anchor": floating(anchors.response.validation),
        "train_risk_anchor_logit": floating(anchors.high_response_logit.train),
        "validation_risk_anchor_logit": floating(
            anchors.high_response_logit.validation
        ),
        "train_maneuver_target": torch.as_tensor(
            targets.train_maneuver, dtype=torch.long, device=device
        ),
        "train_score_target": floating(targets.train_maneuver_score),
        "train_response_target": floating(targets.train_response),
        "train_high_target": floating(targets.train_high_response),
    }


def _losses(*, maneuver, response, tensors, score_scale, response_scale):
    labels = tensors["train_maneuver_target"]
    counts = torch.bincount(labels, minlength=3).float()
    class_weights = torch.where(
        counts > 0,
        counts.sum() / (3 * counts.clamp_min(1.0)),
        torch.zeros_like(counts),
    )
    high = tensors["train_high_target"]
    positive_weight = (high.numel() - high.sum()).clamp_min(1.0) / high.sum().clamp_min(1.0)
    return {
        "maneuver": F.cross_entropy(
            maneuver["logits"], labels, weight=class_weights
        ),
        "score": F.smooth_l1_loss(
            (maneuver["score"] - tensors["train_score_target"]) / score_scale,
            torch.zeros_like(tensors["train_score_target"]),
        ),
        "response": F.smooth_l1_loss(
            (response["response"] - tensors["train_response_target"])
            / response_scale,
            torch.zeros_like(tensors["train_response_target"]),
        ),
        "risk": F.binary_cross_entropy_with_logits(
            response["risk_logit"], high, pos_weight=positive_weight
        ),
    }


def _evaluate(model, *, batches, tensors, targets, score_scale, response_scale):
    model.eval()
    with torch.inference_mode():
        maneuver_features = model.features(batches["validation_maneuver"])
        response_features = model.features(batches["validation_response"])
        maneuver = model.head.forward_maneuver(
            maneuver_features,
            tensors["validation_maneuver_anchor_logits"],
            tensors["validation_maneuver_anchor_score"],
            score_scale=score_scale,
        )
        response = model.head.forward_response(
            response_features,
            tensors["validation_response_anchor"],
            tensors["validation_risk_anchor_logit"],
            response_scale=response_scale,
        )
    return evaluate_task_predictions(
        maneuver_logits=maneuver["logits"].cpu().numpy(),
        maneuver_score=maneuver["score"].cpu().numpy(),
        response=response["response"].cpu().numpy(),
        risk_logit=response["risk_logit"].cpu().numpy(),
        targets=targets,
    )


def _selection_score(metrics):
    return float(
        (1.0 - metrics["maneuver"]["macro_f1"])
        + metrics["response"]["rmse_ratio"]
        + (1.0 - metrics["high_response"]["normalized_ap"])
    )
