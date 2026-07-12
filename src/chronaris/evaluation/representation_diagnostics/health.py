"""Representation collapse and temporal-variation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from chronaris.representation.contracts import FusionStreamBatch


@dataclass(frozen=True, slots=True)
class RepresentationHealth:
    method_name: str
    fold_id: str
    sample_count: int
    valid_vector_count: int
    effective_rank: float
    stable_rank: float
    near_zero_variance_fraction: float
    mean_dimension_variance: float
    mean_temporal_total_variation: float
    mean_pairwise_pooled_cosine: float
    maximum_pairwise_pooled_cosine: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def representation_health(
    batch: FusionStreamBatch,
    *,
    variance_epsilon: float = 1e-8,
) -> RepresentationHealth:
    """Summarize rank, variance, smoothness and sample collapse."""

    if variance_epsilon <= 0:
        raise ValueError("variance_epsilon must be positive")
    valid_vectors = batch.sequence_embedding[batch.valid_mask].to(torch.float64)
    if valid_vectors.shape[0] < 2:
        raise ValueError("representation health requires at least two valid vectors")
    centered = valid_vectors - valid_vectors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    singular_sum = singular_values.sum()
    if float(singular_sum) > 0:
        probabilities = singular_values / singular_sum
        entropy = -(probabilities * probabilities.clamp_min(1e-15).log()).sum()
        effective_rank = float(entropy.exp())
        stable_rank = float(
            singular_values.square().sum()
            / singular_values.square().max().clamp_min(1e-15)
        )
    else:
        effective_rank = 0.0
        stable_rank = 0.0
    variances = valid_vectors.var(dim=0, unbiased=False)
    temporal_differences = batch.sequence_embedding[:, 1:] - batch.sequence_embedding[:, :-1]
    temporal_valid = batch.valid_mask[:, 1:] & batch.valid_mask[:, :-1]
    temporal_norm = torch.linalg.vector_norm(temporal_differences, dim=-1)
    mean_temporal = (
        float(temporal_norm[temporal_valid].mean())
        if bool(temporal_valid.any())
        else 0.0
    )
    pooled = F.normalize(batch.pooled_embedding.to(torch.float64), dim=-1, eps=1e-12)
    similarity = pooled @ pooled.transpose(0, 1)
    upper = torch.triu(
        torch.ones_like(similarity, dtype=torch.bool),
        diagonal=1,
    )
    pairwise = similarity[upper]
    return RepresentationHealth(
        method_name=batch.method_name,
        fold_id=batch.fold_id,
        sample_count=len(batch.sample_ids),
        valid_vector_count=int(valid_vectors.shape[0]),
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        near_zero_variance_fraction=float((variances <= variance_epsilon).float().mean()),
        mean_dimension_variance=float(variances.mean()),
        mean_temporal_total_variation=mean_temporal,
        mean_pairwise_pooled_cosine=float(pairwise.mean()) if pairwise.numel() else 1.0,
        maximum_pairwise_pooled_cosine=float(pairwise.max()) if pairwise.numel() else 1.0,
    )
