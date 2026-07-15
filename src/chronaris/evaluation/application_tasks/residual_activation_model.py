"""Activated, task-decoupled residual heads for Dingxin stage 3A."""

from __future__ import annotations

import copy
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.residual_activation_contracts import (
    ActivationTrainingConfig,
    ActivationTrainingResult,
    ResidualActivationCandidate,
    SelectiveTeacherTargets,
    TASK_NAMES,
)
from chronaris.evaluation.application_tasks.residual_activation_optimization import (
    compose_task_checkpoint,
    shared_gradient_cosines,
    update_task_best_states,
)
from chronaris.evaluation.application_tasks.task_aware_safe_residual_model import (
    SafeAnchorPredictions,
    evaluate_task_predictions,
    safe_against_anchor,
)


class _Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ActivatedResidualHead(nn.Module):
    """Task-decoupled residual head with fixed, global, or conditional gates."""

    def __init__(
        self,
        input_dim: int,
        *,
        candidate: ResidualActivationCandidate,
        hidden_dim: int,
        dropout: float,
        maximum_gate: float,
    ) -> None:
        super().__init__()
        self.candidate = candidate
        self.maximum_gate = float(maximum_gate)
        adapter_keys = ("shared",) if candidate.adapter_mode == "shared" else TASK_NAMES
        self.adapters = nn.ModuleDict(
            {
                key: _Adapter(input_dim, hidden_dim, dropout)
                for key in adapter_keys
            }
        )
        self.delta_layers = nn.ModuleDict(
            {
                "maneuver": nn.Linear(hidden_dim, 3),
                "maneuver_score": nn.Linear(hidden_dim, 1),
                "response": nn.Linear(hidden_dim, 1),
                "high_response": nn.Linear(hidden_dim, 1),
            }
        )
        normalized_initial = candidate.initial_gate / self.maximum_gate
        initial_logit = math.log(normalized_initial / (1.0 - normalized_initial))
        self.global_gate_logits = nn.ParameterDict()
        self.conditional_gates = nn.ModuleDict()
        if candidate.gate_mode == "global":
            for task in TASK_NAMES:
                self.global_gate_logits[task] = nn.Parameter(
                    torch.tensor([initial_logit], dtype=torch.float32)
                )
        elif candidate.gate_mode == "conditional":
            for task in TASK_NAMES:
                layer = nn.Linear(hidden_dim + 2, 1)
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, initial_logit)
                self.conditional_gates[task] = layer
        for layer in self.delta_layers.values():
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)

    def _hidden(self, task: str, features: torch.Tensor) -> torch.Tensor:
        key = "shared" if self.candidate.adapter_mode == "shared" else task
        return self.adapters[key](features)

    def _gate(
        self,
        task: str,
        hidden: torch.Tensor,
        signal: torch.Tensor,
        *,
        release_gate: bool,
    ) -> torch.Tensor:
        if self.candidate.gate_mode == "fixed" or not release_gate:
            return torch.full(
                (hidden.shape[0], 1),
                self.candidate.initial_gate,
                dtype=hidden.dtype,
                device=hidden.device,
            )
        if self.candidate.gate_mode == "global":
            value = self.maximum_gate * torch.sigmoid(self.global_gate_logits[task])
            return value.reshape(1, 1).expand(hidden.shape[0], 1)
        return self.maximum_gate * torch.sigmoid(
            self.conditional_gates[task](torch.cat((hidden, signal), dim=-1))
        )

    def forward_maneuver(
        self,
        features: torch.Tensor,
        anchor_logits: torch.Tensor,
        anchor_score: torch.Tensor,
        *,
        score_scale: float,
        release_gate: bool,
    ) -> Mapping[str, torch.Tensor]:
        probabilities = torch.softmax(anchor_logits, dim=-1)
        top = probabilities.topk(k=2, dim=-1).values
        classification_signal = torch.stack((top[:, 0], top[:, 0] - top[:, 1]), dim=-1)
        score_signal = torch.stack(
            (
                anchor_score / score_scale,
                anchor_score.abs() / score_scale,
            ),
            dim=-1,
        )
        maneuver_hidden = self._hidden("maneuver", features)
        score_hidden = self._hidden("maneuver_score", features)
        maneuver_gate = self._gate(
            "maneuver",
            maneuver_hidden,
            classification_signal,
            release_gate=release_gate,
        )
        score_gate = self._gate(
            "maneuver_score",
            score_hidden,
            score_signal,
            release_gate=release_gate,
        )
        raw_maneuver_delta = self.delta_layers["maneuver"](maneuver_hidden)
        raw_score_delta = self.delta_layers["maneuver_score"](score_hidden).squeeze(-1)
        maneuver_correction = maneuver_gate * raw_maneuver_delta
        score_correction = score_gate.squeeze(-1) * raw_score_delta * score_scale
        return {
            "logits": anchor_logits + maneuver_correction,
            "score": anchor_score + score_correction,
            "raw_maneuver_delta": raw_maneuver_delta,
            "raw_score_delta": raw_score_delta,
            "maneuver_correction": maneuver_correction,
            "score_correction": score_correction,
            "maneuver_gate": maneuver_gate.squeeze(-1),
            "score_gate": score_gate.squeeze(-1),
        }

    def forward_response(
        self,
        features: torch.Tensor,
        anchor_response: torch.Tensor,
        anchor_risk_logit: torch.Tensor,
        *,
        response_scale: float,
        release_gate: bool,
    ) -> Mapping[str, torch.Tensor]:
        risk_probability = torch.sigmoid(anchor_risk_logit)
        response_signal = torch.stack(
            (
                anchor_response / response_scale,
                anchor_response.abs() / response_scale,
            ),
            dim=-1,
        )
        risk_signal = torch.stack(
            (risk_probability, (risk_probability - 0.5).abs() * 2.0), dim=-1
        )
        response_hidden = self._hidden("response", features)
        risk_hidden = self._hidden("high_response", features)
        response_gate = self._gate(
            "response",
            response_hidden,
            response_signal,
            release_gate=release_gate,
        )
        risk_gate = self._gate(
            "high_response",
            risk_hidden,
            risk_signal,
            release_gate=release_gate,
        )
        raw_response_delta = self.delta_layers["response"](response_hidden).squeeze(-1)
        raw_risk_delta = self.delta_layers["high_response"](risk_hidden).squeeze(-1)
        response_correction = (
            response_gate.squeeze(-1) * raw_response_delta * response_scale
        )
        risk_correction = risk_gate.squeeze(-1) * raw_risk_delta
        return {
            "response": anchor_response + response_correction,
            "risk_logit": anchor_risk_logit + risk_correction,
            "raw_response_delta": raw_response_delta,
            "raw_risk_delta": raw_risk_delta,
            "response_correction": response_correction,
            "risk_correction": risk_correction,
            "response_gate": response_gate.squeeze(-1),
            "risk_gate": risk_gate.squeeze(-1),
        }


def train_activated_residual(
    *,
    chronaris_train_features: Mapping[str, np.ndarray],
    chronaris_validation_features: Mapping[str, np.ndarray],
    anchors: SafeAnchorPredictions,
    teachers: SelectiveTeacherTargets,
    targets: FoldTargets,
    candidate: ResidualActivationCandidate,
    config: ActivationTrainingConfig,
) -> ActivationTrainingResult:
    """Train one activated residual candidate on a single development support."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    feature_mean, feature_scale = _feature_transform(
        chronaris_train_features["all"]
    )
    train_maneuver = _standardize(
        chronaris_train_features["maneuver"], feature_mean, feature_scale
    )
    validation_maneuver = _standardize(
        chronaris_validation_features["maneuver"], feature_mean, feature_scale
    )
    train_response = _standardize(
        chronaris_train_features["response"], feature_mean, feature_scale
    )
    validation_response = _standardize(
        chronaris_validation_features["response"], feature_mean, feature_scale
    )
    device = torch.device(config.device)
    model = ActivatedResidualHead(
        train_maneuver.shape[1],
        candidate=candidate,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        maximum_gate=config.maximum_gate,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=config.weight_decay,
    )
    score_scale = max(float(np.std(targets.train_maneuver_score)), 1e-6)
    response_scale = max(float(np.std(targets.train_response)), 1e-6)
    tensors = _training_tensors(
        train_maneuver=train_maneuver,
        validation_maneuver=validation_maneuver,
        train_response=train_response,
        validation_response=validation_response,
        anchors=anchors,
        teachers=teachers,
        targets=targets,
        device=device,
    )
    direct_metrics = evaluate_task_predictions(
        maneuver_logits=anchors.maneuver_logits.validation,
        maneuver_score=anchors.maneuver_score.validation,
        response=anchors.response.validation,
        risk_logit=anchors.high_response_logit.validation,
        targets=targets,
    )
    initial_metrics, initial_diagnostics, initial_gates = _evaluate(
        model,
        tensors=tensors,
        targets=targets,
        score_scale=score_scale,
        response_scale=response_scale,
        release_gate=False,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = _selection_score(initial_metrics)
    best_diagnostics = initial_diagnostics
    best_gates = initial_gates
    task_best_states = {
        task: copy.deepcopy(model.state_dict())
        for task in ("maneuver", "response", "high_response")
    }
    task_best_values = {
        "maneuver": float(initial_metrics["maneuver"]["macro_f1"]),
        "response": float(initial_metrics["response"]["rmse"]),
        "high_response": float(initial_metrics["high_response"]["normalized_ap"]),
    }
    task_best_epochs = {task: 0 for task in task_best_states}
    stale = 0
    epoch_rows: list[Mapping[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        release_gate = epoch > candidate.warmup_epochs
        model.train()
        optimizer.zero_grad(set_to_none=True)
        maneuver = model.forward_maneuver(
            tensors["train_maneuver"],
            tensors["train_maneuver_anchor_logits"],
            tensors["train_maneuver_anchor_score"],
            score_scale=score_scale,
            release_gate=release_gate,
        )
        response = model.forward_response(
            tensors["train_response"],
            tensors["train_response_anchor"],
            tensors["train_risk_anchor_logit"],
            response_scale=response_scale,
            release_gate=release_gate,
        )
        losses = _task_losses(
            maneuver=maneuver,
            response=response,
            tensors=tensors,
            score_scale=score_scale,
            response_scale=response_scale,
            candidate=candidate,
        )
        distillation = _distillation_loss(
            maneuver=maneuver,
            response=response,
            tensors=tensors,
            score_scale=score_scale,
            response_scale=response_scale,
        )
        trust_region = _trust_region_loss(
            maneuver=maneuver,
            response=response,
            score_scale=score_scale,
            response_scale=response_scale,
        )
        gate_regularization = _gate_regularization(maneuver, response)
        task_groups = {
            "maneuver": losses["maneuver"]
            + config.score_loss_weight * losses["score"],
            "response": losses["response"],
            "high_response": losses["risk"],
        }
        gradient_cosines = shared_gradient_cosines(model, task_groups)
        total = (
            sum(task_groups.values())
            + config.trust_region_weight * trust_region
            + (
                candidate.distillation_weight * distillation
                if candidate.use_distillation
                else torch.zeros((), device=device)
            )
            + (
                config.released_gate_regularization * gate_regularization
                if release_gate
                else torch.zeros((), device=device)
            )
        )
        total.backward()
        gradient_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        )
        optimizer.step()
        metrics, diagnostics, gate_summary = _evaluate(
            model,
            tensors=tensors,
            targets=targets,
            score_scale=score_scale,
            response_scale=response_scale,
            release_gate=release_gate,
        )
        score = _selection_score(metrics)
        safe = safe_against_anchor(metrics, direct_metrics)
        task_improvements = update_task_best_states(
            model=model,
            metrics=metrics,
            direct_metrics=direct_metrics,
            task_best_states=task_best_states,
            task_best_values=task_best_values,
            task_best_epochs=task_best_epochs,
            epoch=epoch,
        )
        improved = bool(safe and score < best_score - 1e-9)
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_score = score
            best_diagnostics = diagnostics
            best_gates = gate_summary
            stale = 0
        elif any(task_improvements.values()):
            stale = 0
        else:
            stale += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "gate_released": release_gate,
                **{f"train_{name}_loss": float(value.detach()) for name, value in losses.items()},
                "train_distillation_loss": float(distillation.detach()),
                "train_trust_region_loss": float(trust_region.detach()),
                "train_gate_regularization": float(gate_regularization.detach()),
                "train_total_loss": float(total.detach()),
                "gradient_norm_before_clip": gradient_norm,
                **gradient_cosines,
                "validation_selection_score": score,
                "validation_safe": safe,
                "validation_corrected_sample_fraction": diagnostics[
                    "corrected_sample_fraction"
                ],
                "validation_median_contribution_ratio": diagnostics[
                    "median_contribution_ratio"
                ],
                "validation_maneuver_macro_f1": metrics["maneuver"]["macro_f1"],
                "validation_response_rmse": metrics["response"]["rmse"],
                "validation_response_skill": metrics["response"]["response_skill"],
                "validation_high_response_auprc": metrics["high_response"]["auprc"],
                "validation_high_response_normalized_ap": metrics["high_response"][
                    "normalized_ap"
                ],
                **{
                    f"{task}_checkpoint_improved": value
                    for task, value in task_improvements.items()
                },
                "improved": improved,
            }
        )
        if stale >= config.patience and epoch > candidate.warmup_epochs:
            break
    if candidate.adapter_mode == "task_specific":
        best_state = compose_task_checkpoint(
            base_state=best_state,
            task_states=task_best_states,
        )
        best_epoch = max(task_best_epochs.values())
    else:
        task_best_epochs = {
            task: best_epoch for task in task_best_epochs
        }
    model.load_state_dict(best_state)
    full_metrics, final_diagnostics, final_gates = _evaluate(
        model,
        tensors=tensors,
        targets=targets,
        score_scale=score_scale,
        response_scale=response_scale,
        release_gate=best_epoch > candidate.warmup_epochs,
    )
    if best_epoch == 0:
        final_diagnostics = best_diagnostics
        final_gates = best_gates
    return ActivationTrainingResult(
        model_state_dict={
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        epoch_rows=tuple(epoch_rows),
        direct_metrics=direct_metrics,
        full_metrics=full_metrics,
        prediction_diagnostics=final_diagnostics,
        gate_summary=final_gates,
        teacher_summary=teachers.summary(),
        best_epoch=best_epoch,
        task_best_epochs=task_best_epochs,
        safety_passed=safe_against_anchor(full_metrics, direct_metrics),
    )


def _task_losses(
    *, maneuver, response, tensors, score_scale, response_scale, candidate
):
    labels = tensors["train_maneuver_target"]
    class_weights = _class_weights(labels, 3)
    anchor_probabilities = torch.softmax(
        tensors["train_maneuver_anchor_logits"], dim=-1
    )
    anchor_prediction = anchor_probabilities.argmax(dim=-1)
    anchor_confidence = anchor_probabilities.max(dim=-1).values
    maneuver_focus = 0.25 + 0.75 * (
        (anchor_prediction != labels) | (anchor_confidence < 0.70)
    ).float()
    maneuver_loss = F.cross_entropy(
        maneuver["logits"], labels, weight=class_weights, reduction="none"
    )
    maneuver_loss = _weighted_mean(maneuver_loss, maneuver_focus)
    score_error = (
        tensors["train_score_target"] - tensors["train_maneuver_anchor_score"]
    ) / score_scale
    score_prediction = maneuver["score_correction"] / score_scale
    score_weight = (
        1.0 + score_error.abs().clamp(max=3.0)
        if candidate.objective_mode == "explicit_residual"
        else torch.ones_like(score_error)
    )
    score_loss = _weighted_mean(
        F.smooth_l1_loss(score_prediction, score_error, reduction="none"),
        score_weight,
    )
    response_error = (
        tensors["train_response_target"] - tensors["train_response_anchor"]
    ) / response_scale
    response_prediction = response["response_correction"] / response_scale
    response_weight = (
        1.0 + response_error.abs().clamp(max=3.0)
        if candidate.objective_mode == "explicit_residual"
        else torch.ones_like(response_error)
    )
    response_loss = _weighted_mean(
        F.smooth_l1_loss(
            response_prediction, response_error, reduction="none"
        ),
        response_weight,
    )
    high = tensors["train_high_target"]
    risk_probability = torch.sigmoid(tensors["train_risk_anchor_logit"])
    risk_focus = 0.25 + 0.75 * (
        (risk_probability - high).abs() >= 0.35
    ).float()
    risk_loss = F.binary_cross_entropy_with_logits(
        response["risk_logit"],
        high,
        pos_weight=_positive_weight(high),
        reduction="none",
    )
    return {
        "maneuver": maneuver_loss,
        "score": score_loss,
        "response": response_loss,
        "risk": _weighted_mean(risk_loss, risk_focus),
    }


def _distillation_loss(
    *, maneuver, response, tensors, score_scale, response_scale
):
    maneuver_kl = F.kl_div(
        F.log_softmax(maneuver["logits"], dim=-1),
        tensors["teacher_maneuver_probability"],
        reduction="none",
    ).sum(dim=-1)
    score_loss = F.smooth_l1_loss(
        maneuver["score"] / score_scale,
        tensors["teacher_maneuver_score"] / score_scale,
        reduction="none",
    )
    response_loss = F.smooth_l1_loss(
        response["response"] / response_scale,
        tensors["teacher_response"] / response_scale,
        reduction="none",
    )
    risk_loss = F.binary_cross_entropy_with_logits(
        response["risk_logit"],
        tensors["teacher_high_probability"],
        reduction="none",
    )
    return (
        _weighted_mean(maneuver_kl, tensors["teacher_maneuver_weight"])
        + 0.2
        * _weighted_mean(score_loss, tensors["teacher_maneuver_score_weight"])
        + _weighted_mean(response_loss, tensors["teacher_response_weight"])
        + _weighted_mean(risk_loss, tensors["teacher_high_weight"])
    )


def _trust_region_loss(*, maneuver, response, score_scale, response_scale):
    normalized = (
        maneuver["maneuver_correction"].reshape(-1),
        maneuver["score_correction"] / score_scale,
        response["response_correction"] / response_scale,
        response["risk_correction"],
    )
    return sum(torch.relu(values.abs() - 0.30).square().mean() for values in normalized)


def _gate_regularization(maneuver, response):
    return torch.stack(
        (
            maneuver["maneuver_gate"].mean(),
            maneuver["score_gate"].mean(),
            response["response_gate"].mean(),
            response["risk_gate"].mean(),
        )
    ).mean()


def _evaluate(
    model,
    *,
    tensors,
    targets,
    score_scale,
    response_scale,
    release_gate,
):
    model.eval()
    with torch.inference_mode():
        maneuver = model.forward_maneuver(
            tensors["validation_maneuver"],
            tensors["validation_maneuver_anchor_logits"],
            tensors["validation_maneuver_anchor_score"],
            score_scale=score_scale,
            release_gate=release_gate,
        )
        response = model.forward_response(
            tensors["validation_response"],
            tensors["validation_response_anchor"],
            tensors["validation_risk_anchor_logit"],
            response_scale=response_scale,
            release_gate=release_gate,
        )
    metrics = evaluate_task_predictions(
        maneuver_logits=maneuver["logits"].cpu().numpy(),
        maneuver_score=maneuver["score"].cpu().numpy(),
        response=response["response"].cpu().numpy(),
        risk_logit=response["risk_logit"].cpu().numpy(),
        targets=targets,
    )
    diagnostics = _prediction_diagnostics(
        maneuver=maneuver,
        response=response,
        tensors=tensors,
        score_scale=score_scale,
        response_scale=response_scale,
    )
    gates = {
        "maneuver": _summarize_values(maneuver["maneuver_gate"]),
        "maneuver_score": _summarize_values(maneuver["score_gate"]),
        "response": _summarize_values(response["response_gate"]),
        "high_response": _summarize_values(response["risk_gate"]),
    }
    return metrics, diagnostics, gates


def _prediction_diagnostics(
    *, maneuver, response, tensors, score_scale, response_scale
):
    ratios = torch.cat(
        (
            maneuver["maneuver_correction"].norm(dim=-1)
            / tensors["validation_maneuver_anchor_logits"].norm(dim=-1).clamp_min(1.0),
            maneuver["score_correction"].abs()
            / tensors["validation_maneuver_anchor_score"].abs().clamp_min(score_scale),
            response["response_correction"].abs()
            / tensors["validation_response_anchor"].abs().clamp_min(response_scale),
            response["risk_correction"].abs()
            / tensors["validation_risk_anchor_logit"].abs().clamp_min(1.0),
        )
    )
    changed = ratios >= 0.01
    maximum_difference = max(
        float(maneuver["maneuver_correction"].abs().max().cpu()),
        float(maneuver["score_correction"].abs().max().cpu()),
        float(response["response_correction"].abs().max().cpu()),
        float(response["risk_correction"].abs().max().cpu()),
    )
    return {
        "corrected_sample_fraction": float(changed.float().mean().cpu()),
        "median_contribution_ratio": float(ratios.median().cpu()),
        "mean_contribution_ratio": float(ratios.mean().cpu()),
        "maximum_prediction_difference": maximum_difference,
    }


def _training_tensors(
    *,
    train_maneuver,
    validation_maneuver,
    train_response,
    validation_response,
    anchors,
    teachers,
    targets,
    device,
):
    def floating(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    return {
        "train_maneuver": floating(train_maneuver),
        "validation_maneuver": floating(validation_maneuver),
        "train_response": floating(train_response),
        "validation_response": floating(validation_response),
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
        "teacher_maneuver_probability": floating(teachers.maneuver_probability),
        "teacher_maneuver_weight": floating(teachers.maneuver_weight),
        "teacher_maneuver_score": floating(teachers.maneuver_score),
        "teacher_maneuver_score_weight": floating(
            teachers.maneuver_score_weight
        ),
        "teacher_response": floating(teachers.response),
        "teacher_response_weight": floating(teachers.response_weight),
        "teacher_high_probability": floating(
            teachers.high_response_probability
        ),
        "teacher_high_weight": floating(teachers.high_response_weight),
    }


def _selection_score(metrics):
    return float(
        (1.0 - metrics["maneuver"]["macro_f1"])
        + metrics["response"]["rmse_ratio"]
        + (1.0 - metrics["high_response"]["normalized_ap"])
    )


def _feature_transform(train):
    values = np.asarray(train, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(values, mean, scale):
    output = (np.asarray(values, dtype=np.float32) - mean) / scale
    return np.nan_to_num(output).astype(np.float32)


def _class_weights(labels, class_count):
    counts = torch.bincount(labels, minlength=class_count).float()
    return torch.where(
        counts > 0,
        counts.sum() / (class_count * counts.clamp_min(1.0)),
        torch.zeros_like(counts),
    )


def _positive_weight(labels):
    positives = labels.sum().clamp_min(1.0)
    negatives = (labels.numel() - labels.sum()).clamp_min(1.0)
    return negatives / positives


def _weighted_mean(values, weights):
    denominator = weights.sum()
    if float(denominator.detach()) <= 1e-12:
        return values.sum() * 0.0
    return (values * weights).sum() / denominator


def _summarize_values(values):
    data = values.detach().cpu().numpy().astype(np.float64)
    return {
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "maximum": float(np.max(data)),
        "lower_saturation_fraction": float(np.mean(data <= 0.005)),
        "upper_saturation_fraction": float(np.mean(data >= 0.495)),
    }
