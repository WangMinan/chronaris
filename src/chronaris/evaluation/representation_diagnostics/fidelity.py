"""Linear task-independent probes for single-stream information retention."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from chronaris.representation.contracts import FusionStreamBatch


@dataclass(frozen=True, slots=True)
class FidelityProbeResult:
    source_method: str
    target_method: str
    train_sample_count: int
    validation_sample_count: int
    ridge_alpha: float
    variance_weighted_r2: float
    normalized_rmse: float
    target_variance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureRecoveryProbeResult:
    source_method: str
    target_name: str
    target_dimension: int
    train_sample_count: int
    validation_sample_count: int
    ridge_alpha: float
    variance_weighted_r2: float
    normalized_rmse: float
    target_variance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fit_fidelity_probe(
    train_source: FusionStreamBatch,
    train_target: FusionStreamBatch,
    validation_source: FusionStreamBatch,
    validation_target: FusionStreamBatch,
    *,
    ridge_alpha: float = 1e-3,
) -> FidelityProbeResult:
    """Fit a ridge map from one frozen representation to another."""

    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    train_x, train_y = _aligned_pooled(train_source, train_target)
    validation_x, validation_y = _aligned_pooled(
        validation_source,
        validation_target,
    )
    metrics = _fit_ridge_metrics(
        train_x,
        train_y,
        validation_x,
        validation_y,
        ridge_alpha=ridge_alpha,
    )
    return FidelityProbeResult(
        source_method=train_source.method_name,
        target_method=train_target.method_name,
        train_sample_count=train_x.shape[0],
        validation_sample_count=validation_x.shape[0],
        ridge_alpha=float(ridge_alpha),
        **metrics,
    )


def fit_feature_recovery_probe(
    train_source: FusionStreamBatch,
    train_target: torch.Tensor,
    validation_source: FusionStreamBatch,
    validation_target: torch.Tensor,
    *,
    target_name: str,
    ridge_alpha: float = 1e-3,
) -> FeatureRecoveryProbeResult:
    """Fit the same ridge probe to a task-independent observed-feature target."""

    if not target_name or ridge_alpha < 0:
        raise ValueError("feature recovery target/ridge configuration is invalid")
    if train_target.ndim != 2 or validation_target.ndim != 2:
        raise ValueError("feature recovery targets must have shape [N,D]")
    if train_target.shape[0] != len(train_source.sample_ids):
        raise ValueError("feature recovery train sample count changed")
    if validation_target.shape[0] != len(validation_source.sample_ids):
        raise ValueError("feature recovery validation sample count changed")
    if train_target.shape[1] != validation_target.shape[1]:
        raise ValueError("feature recovery target dimension changed")
    metrics = _fit_ridge_metrics(
        train_source.pooled_embedding,
        train_target,
        validation_source.pooled_embedding,
        validation_target,
        ridge_alpha=ridge_alpha,
    )
    return FeatureRecoveryProbeResult(
        source_method=train_source.method_name,
        target_name=target_name,
        target_dimension=train_target.shape[1],
        train_sample_count=len(train_source.sample_ids),
        validation_sample_count=len(validation_source.sample_ids),
        ridge_alpha=float(ridge_alpha),
        **metrics,
    )


def _fit_ridge_metrics(train_x, train_y, validation_x, validation_y, *, ridge_alpha):
    device = train_x.device
    train_x = train_x.to(device=device, dtype=torch.float64)
    train_y = train_y.to(device=device, dtype=torch.float64)
    validation_x = validation_x.to(device=device, dtype=torch.float64)
    validation_y = validation_y.to(device=device, dtype=torch.float64)
    x_mean = train_x.mean(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    x_centered = train_x - x_mean
    y_centered = train_y - y_mean
    identity = torch.eye(
        x_centered.shape[1],
        dtype=torch.float64,
        device=device,
    )
    weights = torch.linalg.solve(
        x_centered.transpose(0, 1) @ x_centered + ridge_alpha * identity,
        x_centered.transpose(0, 1) @ y_centered,
    )
    prediction = (validation_x - x_mean) @ weights + y_mean
    residual_sum = (validation_y - prediction).square().sum()
    target_centered = validation_y - validation_y.mean(dim=0, keepdim=True)
    target_sum = target_centered.square().sum().clamp_min(1e-15)
    target_variance = validation_y.var(dim=0, unbiased=False).mean().clamp_min(1e-15)
    rmse = (validation_y - prediction).square().mean().sqrt()
    return {
        "variance_weighted_r2": float(1.0 - residual_sum / target_sum),
        "normalized_rmse": float(rmse / target_variance.sqrt()),
        "target_variance": float(target_variance),
    }


def _aligned_pooled(
    source: FusionStreamBatch,
    target: FusionStreamBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.fold_id != target.fold_id:
        raise ValueError("fidelity representations must share fold_id")
    target_index = {sample_id: index for index, sample_id in enumerate(target.sample_ids)}
    if set(source.sample_ids) != set(target.sample_ids):
        raise ValueError("fidelity representations must contain identical sample IDs")
    order = torch.tensor(
        [target_index[sample_id] for sample_id in source.sample_ids],
        dtype=torch.long,
        device=target.pooled_embedding.device,
    )
    if tuple(source.source_sample_hashes) != tuple(
        target.source_sample_hashes[index] for index in order.tolist()
    ):
        raise ValueError("fidelity source hashes differ")
    return source.pooled_embedding, target.pooled_embedding.index_select(0, order)
