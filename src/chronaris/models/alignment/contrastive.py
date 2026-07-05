"""Contrastive retrieval utilities for task evaluation T3 optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True, slots=True)
class HardNegativeSamplerConfig:
    """Sampling policy for P37 T3 same-sortie temporal negatives."""

    near_window_radius: int = 2
    max_per_kind: int = 2
    include_easy_different_sortie: bool = True


@dataclass(frozen=True, slots=True)
class HardNegativeSample:
    """One sampled negative candidate for a retrieval anchor."""

    anchor_index: int
    candidate_index: int
    negative_kind: str
    temporal_offset: int | None


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
    candidate_weights: torch.Tensor | None = None,
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
    if candidate_weights is not None:
        if candidate_weights.shape != logits.shape:
            raise ValueError("candidate_weights must match the logits shape.")
        weights = torch.clamp(candidate_weights.to(device=logits.device, dtype=logits.dtype), min=1e-6)
        logits = logits + torch.log(weights)
    return F.cross_entropy(logits, positive_indices.to(device=logits.device, dtype=torch.long))


def supervised_contrastive_margin_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    margin: float = 0.1,
    negative_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Margin loss that separates each positive from the hardest negative."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [anchors, candidates].")
    positives = positive_indices.to(device=scores.device, dtype=torch.long)
    positive_scores = scores.gather(1, positives.unsqueeze(-1)).squeeze(-1)
    negative_scores = scores.clone()
    negative_scores.scatter_(1, positives.unsqueeze(-1), torch.finfo(scores.dtype).min)
    if negative_weights is not None:
        if negative_weights.shape != scores.shape:
            raise ValueError("negative_weights must match scores.")
        negative_scores = negative_scores + torch.log(torch.clamp(negative_weights.to(scores), min=1e-6))
    hardest_negative = negative_scores.max(dim=-1).values
    return F.relu(scores.new_tensor(float(margin)) - positive_scores + hardest_negative).mean()


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


def stratified_hard_negative_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    anchor_index: int,
    positive_index: int,
    pool_indices: Sequence[int],
    config: HardNegativeSamplerConfig | None = None,
) -> tuple[HardNegativeSample, ...]:
    """Select near-window, far-window, and different-sortie negatives for one anchor."""

    policy = config or HardNegativeSamplerConfig()
    anchor = rows[int(anchor_index)]
    positive = rows[int(positive_index)]
    if not _same_sortie_cross_pilot(anchor, positive):
        raise ValueError("positive candidate violates same_sortie_cross_pilot policy.")

    buckets: dict[str, list[HardNegativeSample]] = {
        "hard_negative_same_sortie_near_window": [],
        "hard_negative_same_sortie_far_window": [],
        "easy_negative_different_sortie": [],
    }
    anchor_window = _optional_int(anchor.get("window_index"))
    for candidate_index in pool_indices:
        index = int(candidate_index)
        if index == int(anchor_index) or index == int(positive_index):
            continue
        candidate = rows[index]
        if _same_sortie_cross_pilot(anchor, candidate):
            offset = _temporal_offset(anchor_window, _optional_int(candidate.get("window_index")))
            kind = (
                "hard_negative_same_sortie_near_window"
                if offset is not None and offset <= max(int(policy.near_window_radius), 0)
                else "hard_negative_same_sortie_far_window"
            )
            buckets[kind].append(HardNegativeSample(int(anchor_index), index, kind, offset))
        elif policy.include_easy_different_sortie and str(candidate.get("sortie_id")) != str(anchor.get("sortie_id")):
            buckets["easy_negative_different_sortie"].append(
                HardNegativeSample(int(anchor_index), index, "easy_negative_different_sortie", None)
            )
    samples: list[HardNegativeSample] = []
    for kind in (
        "hard_negative_same_sortie_near_window",
        "hard_negative_same_sortie_far_window",
        "easy_negative_different_sortie",
    ):
        samples.extend(buckets[kind][: max(int(policy.max_per_kind), 0)])
    return tuple(samples)


def _same_sortie_cross_pilot(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return str(left.get("sortie_id")) == str(right.get("sortie_id")) and str(left.get("pilot_id")) != str(right.get("pilot_id"))


def _temporal_offset(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return int(abs(left - right))


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
