"""Task-conditioned safe residual heads for Dingxin core tasks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from chronaris.evaluation.application_tasks.core_feasibility_models import FoldTargets
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    high_response_metrics,
    maneuver_metrics,
    response_metrics,
)
from chronaris.representation import FusionStreamBatch


@dataclass(frozen=True, slots=True)
class PredictionPair:
    """Cross-fitted train predictions and full-train validation predictions."""

    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True, slots=True)
class SafeAnchorPredictions:
    """Task-specific safe anchor predictions for one development split."""

    maneuver_logits: PredictionPair
    maneuver_score: PredictionPair
    response: PredictionPair
    high_response_logit: PredictionPair


@dataclass(frozen=True, slots=True)
class ResidualTrainingConfig:
    learning_rate: float = 1e-2
    weight_decay: float = 1e-3
    max_epochs: int = 80
    patience: int = 12
    hidden_dim: int = 64
    dropout: float = 0.1
    gate_mode: str = "scalar"
    initial_gate_logit: float = -3.0
    gate_penalty: float = 1e-2
    score_loss_weight: float = 0.2
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.gate_mode not in {"scalar", "channel"}:
            raise ValueError("gate_mode must be scalar or channel")
        if min(self.learning_rate, self.max_epochs, self.patience, self.hidden_dim) <= 0:
            raise ValueError("residual training configuration must be positive")
        if not 0 <= self.dropout < 1 or self.weight_decay < 0:
            raise ValueError("residual regularization is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("residual training device is invalid")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA residual training requested without CUDA")


@dataclass(frozen=True, slots=True)
class ResidualTrainingResult:
    model_state_dict: Mapping[str, torch.Tensor]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    epoch_rows: tuple[Mapping[str, object], ...]
    direct_metrics: Mapping[str, Mapping[str, float]]
    full_metrics: Mapping[str, Mapping[str, float]]
    gate_values: Mapping[str, tuple[float, ...]]
    best_epoch: int
    safety_passed: bool


def summarize_fusion_batch(
    batch: FusionStreamBatch,
    sample_ids: Sequence[str],
) -> np.ndarray:
    """Build train-safe multi-scale summaries from a 64-D sequence."""

    positions = {sample_id: index for index, sample_id in enumerate(batch.sample_ids)}
    indices = [positions[str(sample_id)] for sample_id in sample_ids]
    sequence = batch.sequence_embedding[indices].detach().cpu()
    mask = batch.valid_mask[indices].detach().cpu().bool()
    weight = mask.unsqueeze(-1).to(sequence.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (sequence * weight).sum(dim=1) / count
    variance = ((sequence - mean.unsqueeze(1)).square() * weight).sum(dim=1) / count
    std = variance.clamp_min(0.0).sqrt()
    first = _first_valid(sequence, mask)
    last = _last_valid(sequence, mask)
    safe_max = sequence.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
    safe_min = sequence.masked_fill(~mask.unsqueeze(-1), torch.inf).amin(dim=1)
    amplitude = torch.nan_to_num(safe_max - safe_min)
    summary = torch.cat((mean, std, last, last - first, amplitude), dim=-1)
    if not torch.isfinite(summary).all():
        raise ValueError("fusion summary contains non-finite values")
    return summary.numpy().astype(np.float32)


def summarize_sequence_tensor(
    sequence: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable counterpart of :func:`summarize_fusion_batch`."""

    mask = valid_mask.bool()
    weight = mask.unsqueeze(-1).to(sequence.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (sequence * weight).sum(dim=1) / count
    variance = ((sequence - mean.unsqueeze(1)).square() * weight).sum(dim=1) / count
    std = variance.clamp_min(0.0).sqrt()
    first = _first_valid(sequence, mask)
    last = _last_valid(sequence, mask)
    finite_max = torch.finfo(sequence.dtype).max
    safe_max = sequence.masked_fill(~mask.unsqueeze(-1), -finite_max).amax(dim=1)
    safe_min = sequence.masked_fill(~mask.unsqueeze(-1), finite_max).amin(dim=1)
    return torch.nan_to_num(torch.cat((mean, std, last, last - first, safe_max - safe_min), dim=-1))


def fit_safe_anchor_predictions(
    *,
    maneuver_train_features: np.ndarray,
    maneuver_validation_features: np.ndarray,
    response_train_features: np.ndarray,
    response_validation_features: np.ndarray,
    targets: FoldTargets,
    seed: int,
) -> SafeAnchorPredictions:
    """Fit task anchors with train-only cross-fitted predictions."""

    maneuver_logits = _cross_fitted_multiclass(
        maneuver_train_features,
        maneuver_validation_features,
        targets.train_maneuver,
        seed=seed,
    )
    maneuver_score = _cross_fitted_regression(
        maneuver_train_features,
        maneuver_validation_features,
        targets.train_maneuver_score,
        seed=seed,
        alpha=10.0,
    )
    response = _cross_fitted_regression(
        response_train_features,
        response_validation_features,
        targets.train_response,
        seed=seed,
        alpha=10.0,
    )
    high_response_logit = _cross_fitted_binary(
        response_train_features,
        response_validation_features,
        targets.train_high_response,
        seed=seed,
    )
    return SafeAnchorPredictions(
        maneuver_logits=maneuver_logits,
        maneuver_score=maneuver_score,
        response=response,
        high_response_logit=high_response_logit,
    )


class TaskSafeResidualHead(nn.Module):
    """Add a bounded Chronaris residual to immutable task anchor outputs."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        gate_mode: str = "scalar",
        initial_gate_logit: float = -3.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"scalar", "channel"}:
            raise ValueError("gate_mode must be scalar or channel")
        self.gate_mode = gate_mode
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.maneuver_delta = nn.Linear(hidden_dim, 3)
        self.score_delta = nn.Linear(hidden_dim, 1)
        self.response_delta = nn.Linear(hidden_dim, 1)
        self.risk_delta = nn.Linear(hidden_dim, 1)
        maneuver_gate_dim = 3 if gate_mode == "channel" else 1
        self.maneuver_gate_logit = nn.Parameter(
            torch.full((maneuver_gate_dim,), initial_gate_logit)
        )
        self.score_gate_logit = nn.Parameter(torch.tensor([initial_gate_logit]))
        self.response_gate_logit = nn.Parameter(torch.tensor([initial_gate_logit]))
        self.risk_gate_logit = nn.Parameter(torch.tensor([initial_gate_logit]))
        for layer in (
            self.maneuver_delta,
            self.score_delta,
            self.response_delta,
            self.risk_delta,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward_maneuver(
        self,
        features: torch.Tensor,
        anchor_logits: torch.Tensor,
        anchor_score: torch.Tensor,
        *,
        score_scale: float,
    ) -> Mapping[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "logits": anchor_logits
            + torch.sigmoid(self.maneuver_gate_logit) * self.maneuver_delta(hidden),
            "score": anchor_score
            + torch.sigmoid(self.score_gate_logit)
            * self.score_delta(hidden).squeeze(-1)
            * score_scale,
        }

    def forward_response(
        self,
        features: torch.Tensor,
        anchor_response: torch.Tensor,
        anchor_risk_logit: torch.Tensor,
        *,
        response_scale: float,
    ) -> Mapping[str, torch.Tensor]:
        hidden = self.shared(features)
        return {
            "response": anchor_response
            + torch.sigmoid(self.response_gate_logit)
            * self.response_delta(hidden).squeeze(-1)
            * response_scale,
            "risk_logit": anchor_risk_logit
            + torch.sigmoid(self.risk_gate_logit)
            * self.risk_delta(hidden).squeeze(-1),
        }

    def gate_values(self) -> Mapping[str, tuple[float, ...]]:
        return {
            "maneuver": tuple(torch.sigmoid(self.maneuver_gate_logit).detach().cpu().tolist()),
            "maneuver_score": tuple(torch.sigmoid(self.score_gate_logit).detach().cpu().tolist()),
            "response": tuple(torch.sigmoid(self.response_gate_logit).detach().cpu().tolist()),
            "high_response": tuple(torch.sigmoid(self.risk_gate_logit).detach().cpu().tolist()),
        }


def train_frozen_safe_residual(
    *,
    chronaris_train_features: Mapping[str, np.ndarray],
    chronaris_validation_features: Mapping[str, np.ndarray],
    anchors: SafeAnchorPredictions,
    targets: FoldTargets,
    config: ResidualTrainingConfig,
) -> ResidualTrainingResult:
    """Train residual heads while keeping both task anchors and backbone frozen."""

    torch.manual_seed(config.seed)
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
    model = TaskSafeResidualHead(
        train_maneuver.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        gate_mode=config.gate_mode,
        initial_gate_logit=config.initial_gate_logit,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
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
        targets=targets,
        device=device,
    )
    direct_metrics = _evaluate_direct(anchors, targets)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = _selection_score(direct_metrics)
    stale = 0
    epoch_rows: list[Mapping[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        maneuver_output = model.forward_maneuver(
            tensors["train_maneuver"],
            tensors["train_maneuver_anchor_logits"],
            tensors["train_maneuver_anchor_score"],
            score_scale=score_scale,
        )
        response_output = model.forward_response(
            tensors["train_response"],
            tensors["train_response_anchor"],
            tensors["train_risk_anchor_logit"],
            response_scale=response_scale,
        )
        class_weights = _class_weights(tensors["train_maneuver_target"], 3)
        risk_weight = _positive_weight(tensors["train_high_target"])
        losses = {
            "maneuver": F.cross_entropy(
                maneuver_output["logits"],
                tensors["train_maneuver_target"],
                weight=class_weights,
            ),
            "score": F.smooth_l1_loss(
                (maneuver_output["score"] - tensors["train_score_target"])
                / score_scale,
                torch.zeros_like(tensors["train_score_target"]),
            ),
            "response": F.smooth_l1_loss(
                (response_output["response"] - tensors["train_response_target"])
                / response_scale,
                torch.zeros_like(tensors["train_response_target"]),
            ),
            "risk": F.binary_cross_entropy_with_logits(
                response_output["risk_logit"],
                tensors["train_high_target"].float(),
                pos_weight=risk_weight,
            ),
        }
        gate_penalty = sum(
            value.mean() for value in _gate_tensors(model).values()
        )
        total = (
            losses["maneuver"]
            + config.score_loss_weight * losses["score"]
            + losses["response"]
            + losses["risk"]
            + config.gate_penalty * gate_penalty
        )
        total.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        metrics = _evaluate_model(
            model,
            tensors=tensors,
            targets=targets,
            score_scale=score_scale,
            response_scale=response_scale,
        )
        score = _selection_score(metrics)
        safe = safe_against_anchor(metrics, direct_metrics)
        improved = bool(safe and score < best_score - 1e-9)
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                **{f"train_{key}_loss": float(value.detach()) for key, value in losses.items()},
                "train_total_loss": float(total.detach()),
                "gradient_norm_before_clip": gradient_norm,
                "validation_selection_score": score,
                "validation_safe": safe,
                "improved": improved,
                "gate_maneuver": float(np.mean(model.gate_values()["maneuver"])),
                "gate_response": float(model.gate_values()["response"][0]),
                "gate_high_response": float(model.gate_values()["high_response"][0]),
            }
        )
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    final_metrics = _evaluate_model(
        model,
        tensors=tensors,
        targets=targets,
        score_scale=score_scale,
        response_scale=response_scale,
    )
    return ResidualTrainingResult(
        model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        epoch_rows=tuple(epoch_rows),
        direct_metrics=direct_metrics,
        full_metrics=final_metrics,
        gate_values=model.gate_values(),
        best_epoch=best_epoch,
        safety_passed=safe_against_anchor(final_metrics, direct_metrics),
    )


def safe_against_anchor(
    full: Mapping[str, Mapping[str, float]],
    direct: Mapping[str, Mapping[str, float]],
) -> bool:
    return bool(
        full["maneuver"]["macro_f1"] >= direct["maneuver"]["macro_f1"] - 0.005
        and full["response"]["rmse"] <= direct["response"]["rmse"] * 1.01
        and full["high_response"]["auprc"]
        >= direct["high_response"]["auprc"] - 0.005
    )


def gate_is_non_degenerate(values: Mapping[str, Sequence[float]]) -> bool:
    flattened = np.asarray([item for group in values.values() for item in group])
    return bool(np.all(flattened > 1e-5) and np.all(flattened < 0.99999))


def evaluate_anchor_predictions(
    anchors: SafeAnchorPredictions,
    targets: FoldTargets,
) -> Mapping[str, Mapping[str, float]]:
    """Evaluate the immutable task anchors with the stage-2 metric contract."""

    return _evaluate_direct(anchors, targets)


def evaluate_task_predictions(
    *,
    maneuver_logits: np.ndarray,
    maneuver_score: np.ndarray,
    response: np.ndarray,
    risk_logit: np.ndarray,
    targets: FoldTargets,
) -> Mapping[str, Mapping[str, float]]:
    """Evaluate task predictions with the shared stage-2 metric contract."""

    return _metrics_from_predictions(
        maneuver_logits=maneuver_logits,
        maneuver_score=maneuver_score,
        response=response,
        risk_logit=risk_logit,
        targets=targets,
    )


def _evaluate_direct(anchors: SafeAnchorPredictions, targets: FoldTargets):
    return _metrics_from_predictions(
        maneuver_logits=anchors.maneuver_logits.validation,
        maneuver_score=anchors.maneuver_score.validation,
        response=anchors.response.validation,
        risk_logit=anchors.high_response_logit.validation,
        targets=targets,
    )


def _evaluate_model(model, *, tensors, targets, score_scale, response_scale):
    model.eval()
    with torch.inference_mode():
        maneuver = model.forward_maneuver(
            tensors["validation_maneuver"],
            tensors["validation_maneuver_anchor_logits"],
            tensors["validation_maneuver_anchor_score"],
            score_scale=score_scale,
        )
        response = model.forward_response(
            tensors["validation_response"],
            tensors["validation_response_anchor"],
            tensors["validation_risk_anchor_logit"],
            response_scale=response_scale,
        )
    return _metrics_from_predictions(
        maneuver_logits=maneuver["logits"].cpu().numpy(),
        maneuver_score=maneuver["score"].cpu().numpy(),
        response=response["response"].cpu().numpy(),
        risk_logit=response["risk_logit"].cpu().numpy(),
        targets=targets,
    )


def _metrics_from_predictions(
    *, maneuver_logits, maneuver_score, response, risk_logit, targets
):
    maneuver_prediction = np.asarray(maneuver_logits).argmax(axis=1)
    risk_probability = 1.0 / (1.0 + np.exp(-np.clip(risk_logit, -30, 30)))
    return {
        "maneuver": maneuver_metrics(
            train_labels=targets.train_maneuver,
            validation_labels=targets.validation_maneuver,
            prediction=maneuver_prediction,
            validation_score=targets.validation_maneuver_score,
            score_prediction=maneuver_score,
        ),
        "response": response_metrics(
            train_target=targets.train_response,
            validation_target=targets.validation_response,
            prediction=np.clip(response, 0.0, 10.0),
        ),
        "high_response": high_response_metrics(
            validation_target=targets.validation_high_response,
            probability=risk_probability,
        ),
    }


def _selection_score(metrics: Mapping[str, Mapping[str, float]]) -> float:
    return float(
        (1.0 - metrics["maneuver"]["macro_f1"])
        + metrics["response"]["rmse_ratio"]
        + (1.0 - metrics["high_response"]["normalized_ap"])
    )


def _training_tensors(**values):
    device = values["device"]
    anchors = values["anchors"]
    targets = values["targets"]
    output = {
        key: torch.as_tensor(values[key], dtype=torch.float32, device=device)
        for key in (
            "train_maneuver",
            "validation_maneuver",
            "train_response",
            "validation_response",
        )
    }
    output.update(
        {
            "train_maneuver_anchor_logits": _float_tensor(
                anchors.maneuver_logits.train, device
            ),
            "validation_maneuver_anchor_logits": _float_tensor(
                anchors.maneuver_logits.validation, device
            ),
            "train_maneuver_anchor_score": _float_tensor(
                anchors.maneuver_score.train, device
            ),
            "validation_maneuver_anchor_score": _float_tensor(
                anchors.maneuver_score.validation, device
            ),
            "train_response_anchor": _float_tensor(anchors.response.train, device),
            "validation_response_anchor": _float_tensor(
                anchors.response.validation, device
            ),
            "train_risk_anchor_logit": _float_tensor(
                anchors.high_response_logit.train, device
            ),
            "validation_risk_anchor_logit": _float_tensor(
                anchors.high_response_logit.validation, device
            ),
            "train_maneuver_target": torch.as_tensor(
                targets.train_maneuver, dtype=torch.long, device=device
            ),
            "train_score_target": _float_tensor(
                targets.train_maneuver_score, device
            ),
            "train_response_target": _float_tensor(targets.train_response, device),
            "train_high_target": torch.as_tensor(
                targets.train_high_response, dtype=torch.long, device=device
            ),
        }
    )
    return output


def _cross_fitted_multiclass(train, validation, labels, *, seed):
    labels = np.asarray(labels, dtype=np.int64)
    oof = np.zeros((len(labels), 3), dtype=np.float64)
    splitter = _stratified_splitter(labels, seed)
    for fit, held in splitter.split(train, labels):
        model = _multiclass_model(seed).fit(train[fit], labels[fit])
        oof[held] = _aligned_multiclass_logits(model, train[held])
    model = _multiclass_model(seed).fit(train, labels)
    return PredictionPair(oof, _aligned_multiclass_logits(model, validation))


def _cross_fitted_binary(train, validation, labels, *, seed):
    labels = np.asarray(labels, dtype=np.int64)
    oof = np.zeros(len(labels), dtype=np.float64)
    splitter = _stratified_splitter(labels, seed)
    for fit, held in splitter.split(train, labels):
        model = _binary_model(seed).fit(train[fit], labels[fit])
        oof[held] = _binary_logits(model, train[held])
    model = _binary_model(seed).fit(train, labels)
    return PredictionPair(oof, _binary_logits(model, validation))


def _cross_fitted_regression(train, validation, target, *, seed, alpha):
    target = np.asarray(target, dtype=np.float64)
    split_count = min(3, len(target))
    splitter = KFold(n_splits=split_count, shuffle=True, random_state=seed)
    oof = np.zeros(len(target), dtype=np.float64)
    for fit, held in splitter.split(train):
        model = _regression_model(alpha).fit(train[fit], target[fit])
        oof[held] = model.predict(train[held])
    model = _regression_model(alpha).fit(train, target)
    return PredictionPair(oof, np.asarray(model.predict(validation)).reshape(-1))


def _stratified_splitter(labels, seed):
    counts = np.bincount(labels)
    positive = counts[counts > 0]
    split_count = min(3, int(positive.min()))
    if split_count < 2:
        raise ValueError("cross-fitted classification requires two samples per class")
    return StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)


def _multiclass_model(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2_000,
            random_state=seed,
        ),
    )


def _binary_model(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2_000,
            random_state=seed,
        ),
    )


def _regression_model(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _aligned_multiclass_logits(model, features):
    classifier = model[-1]
    raw = np.asarray(model.decision_function(features))
    output = np.full((len(features), 3), -20.0, dtype=np.float64)
    output[:, classifier.classes_.astype(int)] = raw
    return output


def _binary_logits(model, features):
    probability = np.clip(model.predict_proba(features)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _feature_transform(train):
    mean = np.asarray(train, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train, dtype=np.float64).std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def _standardize(values, mean, scale):
    return ((np.asarray(values) - mean) / scale).astype(np.float32)


def _first_valid(sequence, mask):
    index = mask.to(torch.int64).argmax(dim=1)
    rows = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[rows, index]


def _last_valid(sequence, mask):
    reverse_index = mask.flip(dims=(1,)).to(torch.int64).argmax(dim=1)
    index = mask.shape[1] - reverse_index - 1
    rows = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[rows, index]


def _float_tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _class_weights(labels, class_count):
    counts = torch.bincount(labels, minlength=class_count).float()
    return torch.where(
        counts > 0,
        counts.sum() / (class_count * counts.clamp_min(1.0)),
        torch.zeros_like(counts),
    )


def _positive_weight(labels):
    positives = labels.sum().float().clamp_min(1.0)
    negatives = (labels.numel() - labels.sum()).float().clamp_min(1.0)
    return negatives / positives


def _gate_tensors(model):
    return {
        "maneuver": torch.sigmoid(model.maneuver_gate_logit),
        "maneuver_score": torch.sigmoid(model.score_gate_logit),
        "response": torch.sigmoid(model.response_gate_logit),
        "high_response": torch.sigmoid(model.risk_gate_logit),
    }
