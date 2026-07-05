"""Loss helpers for task evaluation optimized task heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


RegressionLossName = Literal["mse", "smooth_l1", "huber"]


@dataclass(frozen=True, slots=True)
class TrainFoldTargetTransform:
    """Train-fold-only scalar target standardization."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_train_targets(cls, targets: torch.Tensor) -> "TrainFoldTargetTransform":
        if targets.numel() == 0:
            raise ValueError("train targets must be non-empty.")
        finite = torch.isfinite(targets)
        if not bool(finite.any()):
            raise ValueError("train targets must contain at least one finite value.")
        selected = targets[finite].float()
        std = torch.clamp(selected.std(unbiased=False), min=1e-6)
        return cls(mean=selected.mean(), std=std)

    def transform(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean.to(targets.device)) / self.std.to(targets.device)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values.device) + self.mean.to(values.device)


def class_balanced_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_count: int | None = None,
) -> torch.Tensor:
    """Cross entropy with weights estimated from the current train fold labels."""

    if targets.ndim != 1:
        targets = targets.view(-1)
    output_dim = int(class_count or logits.shape[-1])
    counts = torch.bincount(targets.to(dtype=torch.long), minlength=output_dim).float()
    weights = torch.where(counts > 0, counts.sum() / torch.clamp(counts, min=1.0), torch.zeros_like(counts))
    if bool((weights > 0).any()):
        weights = weights / torch.clamp(weights[weights > 0].mean(), min=1e-6)
    return F.cross_entropy(logits, targets.to(dtype=torch.long), weight=weights.to(logits.device))


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal cross entropy for imbalanced T1 folds."""

    ce = F.cross_entropy(logits, targets.to(dtype=torch.long), reduction="none")
    p_t = torch.exp(-ce)
    return (((1.0 - p_t) ** float(gamma)) * ce).mean()


def residual_regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: RegressionLossName = "smooth_l1",
    uncertainty: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regression loss with optional aleatoric uncertainty weighting."""

    if loss_name == "mse":
        base = F.mse_loss(predictions, targets, reduction="none")
    elif loss_name in {"smooth_l1", "huber"}:
        base = F.smooth_l1_loss(predictions, targets, reduction="none")
    else:
        raise ValueError(f"unsupported regression loss: {loss_name}")
    if uncertainty is None:
        return base.mean()
    variance = torch.clamp(torch.square(uncertainty), min=1e-6)
    return (0.5 * base / variance + torch.log(torch.sqrt(variance))).mean()


def persistence_improvement_rate(
    predictions: torch.Tensor,
    persistence: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Fractional MSE improvement over persistence; positive is better."""

    model_mse = F.mse_loss(predictions, targets)
    persistence_mse = F.mse_loss(persistence, targets)
    return (persistence_mse - model_mse) / torch.clamp(persistence_mse, min=1e-12)
