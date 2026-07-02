"""Contrastive retrieval utilities for Stage I T3 optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Metadata for one T3 retrieval candidate pair."""

    anchor_sample_id: str
    candidate_sample_id: str
    same_sortie_cross_pilot: bool
    is_positive: bool
    negative_kind: str = "easy"


def validate_same_sortie_cross_pilot_policy(candidates: Sequence[RetrievalCandidate]) -> None:
    """Ensure all positives preserve the private T3 cross-pilot policy."""

    for candidate in candidates:
        if candidate.is_positive and not candidate.same_sortie_cross_pilot:
            raise ValueError(
                "positive T3 candidates must be same_sortie_cross_pilot; "
                f"got {candidate.anchor_sample_id}->{candidate.candidate_sample_id}."
            )


def info_nce_loss(
    anchor_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    temperature: float | torch.Tensor = 0.07,
) -> torch.Tensor:
    """InfoNCE loss where each anchor has one positive candidate index."""

    if anchor_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N, D].")
    if anchor_embeddings.shape[-1] != candidate_embeddings.shape[-1]:
        raise ValueError("anchor and candidate embeddings must share embedding dim.")
    if positive_indices.shape[0] != anchor_embeddings.shape[0]:
        raise ValueError("positive_indices must align with anchors.")
    temp = torch.as_tensor(temperature, device=anchor_embeddings.device, dtype=anchor_embeddings.dtype)
    temp = torch.clamp(temp, min=1e-6)
    logits = torch.matmul(
        F.normalize(anchor_embeddings, dim=-1, eps=1e-12),
        F.normalize(candidate_embeddings, dim=-1, eps=1e-12).transpose(0, 1),
    ) / temp
    return F.cross_entropy(logits, positive_indices.to(device=logits.device, dtype=torch.long))


def retrieval_metrics_from_scores(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    topk: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Compute top-k, MRR, margin, and collapse diagnostics from score matrix."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [anchors, candidates].")
    positives = positive_indices.to(device=scores.device, dtype=torch.long)
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    positive_ranks = (sorted_indices == positives.unsqueeze(-1)).nonzero(as_tuple=False)[:, 1] + 1
    metrics: dict[str, float] = {}
    for k in topk:
        metrics[f"top{k}"] = float((positive_ranks <= int(k)).float().mean().item())
    metrics["mrr"] = float((1.0 / positive_ranks.float()).mean().item())
    positive_scores = scores.gather(1, positives.unsqueeze(-1)).squeeze(-1)
    masked = scores.clone()
    masked.scatter_(1, positives.unsqueeze(-1), torch.finfo(scores.dtype).min)
    hardest_negative = masked.max(dim=-1).values
    metrics["positive_negative_margin"] = float((positive_scores - hardest_negative).mean().item())
    metrics["score_std"] = float(scores.detach().float().std(unbiased=False).item())
    return metrics


def build_positive_index_tensor(
    anchor_ids: Sequence[str],
    candidate_ids: Sequence[str],
    positive_pairs: Mapping[str, str],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map anchor ids to positive candidate positions for InfoNCE training."""

    candidate_lookup = {sample_id: index for index, sample_id in enumerate(candidate_ids)}
    indices: list[int] = []
    for anchor_id in anchor_ids:
        positive_id = positive_pairs.get(anchor_id)
        if positive_id is None or positive_id not in candidate_lookup:
            raise ValueError(f"missing positive candidate for anchor {anchor_id!r}.")
        indices.append(candidate_lookup[positive_id])
    return torch.as_tensor(indices, dtype=torch.long, device=device)
