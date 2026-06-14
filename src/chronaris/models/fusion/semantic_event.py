"""Semantic event-token fusion on top of Stage G causal attention."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

LLM_SEMANTIC_QUERY_RECIPE_WHITELIST = frozenset(
    {
        "coordination_gap",
        "gap_plus_event",
        "physiology_plus_gap",
        "vehicle_plus_event",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticQuerySpec:
    """One deterministic semantic query recipe."""

    name: str
    recipe: str


@dataclass(frozen=True, slots=True)
class CausalEventFusionConfig:
    """Controls event-token extraction and query-to-event attribution."""

    attention_temperature: float = 1.0
    event_score_bias_weight: float = 0.25
    event_top_k: int = 4
    event_score_quantile: float = 0.75
    query_specs: tuple[SemanticQuerySpec, ...] = field(
        default_factory=lambda: (
            SemanticQuerySpec(name="risk_proxy", recipe="gap_plus_event"),
            SemanticQuerySpec(name="workload_proxy", recipe="physiology_plus_gap"),
            SemanticQuerySpec(name="event_replay_tag", recipe="vehicle_plus_event"),
        )
    )

    def __post_init__(self) -> None:
        if self.attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive.")
        if self.event_score_bias_weight < 0:
            raise ValueError("event_score_bias_weight must be non-negative.")
        if self.event_top_k <= 0:
            raise ValueError("event_top_k must be positive.")
        if not 0.0 < self.event_score_quantile <= 1.0:
            raise ValueError("event_score_quantile must be in (0, 1].")
        if not self.query_specs:
            raise ValueError("query_specs must not be empty.")


def semantic_query_specs_from_llm_hints(
    hints: object,
    *,
    include_defaults: bool = True,
) -> tuple[SemanticQuerySpec, ...]:
    """Convert audited LLM query hints into deterministic whitelisted specs."""

    specs: list[SemanticQuerySpec] = []
    seen_names: set[str] = set()
    if include_defaults:
        for spec in CausalEventFusionConfig().query_specs:
            specs.append(spec)
            seen_names.add(spec.name)
    if not isinstance(hints, (list, tuple)):
        return tuple(specs)
    for row in hints:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("query_name") or "").strip()
        recipe = str(row.get("recipe") or "").strip()
        if not name or name in seen_names:
            continue
        if recipe not in LLM_SEMANTIC_QUERY_RECIPE_WHITELIST:
            continue
        specs.append(SemanticQuerySpec(name=name, recipe=recipe))
        seen_names.add(name)
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class SemanticEventTensorInput:
    """Inputs required for deterministic event-token fusion."""

    physiology_states: torch.Tensor
    vehicle_states: torch.Tensor
    attention_weights: torch.Tensor
    vehicle_event_scores: torch.Tensor
    vehicle_offsets_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class SemanticEventTensorOutput:
    """Query-to-event attribution tensors."""

    query_names: tuple[str, ...]
    query_states: torch.Tensor
    query_context_states: torch.Tensor
    event_token_states: torch.Tensor
    event_token_scores: torch.Tensor
    event_token_center_offsets_s: torch.Tensor
    event_token_start_offsets_s: torch.Tensor
    event_token_end_offsets_s: torch.Tensor
    event_token_mask: torch.Tensor
    query_to_event_attention: torch.Tensor
    event_attribution_scores: torch.Tensor
    query_attribution_scores: torch.Tensor


class EventTokenExtractor(nn.Module):
    """Aggregate time-step events into a compact token set."""

    def __init__(self, config: CausalEventFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalEventFusionConfig()

    def forward(self, inputs: SemanticEventTensorInput) -> SemanticEventTensorOutput:
        raise RuntimeError("EventTokenExtractor cannot be used directly; use CausalEventFusion.")

    def extract(
        self,
        inputs: SemanticEventTensorInput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_semantic_inputs(inputs)
        contribution_scores = inputs.attention_weights.sum(dim=1) * inputs.vehicle_event_scores
        batch_size, point_count, state_dim = inputs.vehicle_states.shape
        top_k = self.config.event_top_k
        token_states = inputs.vehicle_states.new_zeros((batch_size, top_k, state_dim))
        token_scores = inputs.vehicle_states.new_zeros((batch_size, top_k))
        center_offsets = inputs.vehicle_states.new_zeros((batch_size, top_k))
        start_offsets = inputs.vehicle_states.new_zeros((batch_size, top_k))
        end_offsets = inputs.vehicle_states.new_zeros((batch_size, top_k))
        token_mask = torch.zeros((batch_size, top_k), dtype=torch.bool, device=inputs.vehicle_states.device)

        for sample_index in range(batch_size):
            scores = contribution_scores[sample_index]
            if point_count == 0:
                continue
            positive_scores = scores[scores > 0]
            if positive_scores.numel() > 0:
                threshold = torch.quantile(
                    positive_scores,
                    q=min(self.config.event_score_quantile, 1.0),
                )
                active = scores >= threshold
            else:
                active = torch.zeros_like(scores, dtype=torch.bool)
                active[int(torch.argmax(inputs.vehicle_event_scores[sample_index]).item())] = True
            segments = _collect_segments(active)
            if not segments:
                anchor_index = int(torch.argmax(inputs.vehicle_event_scores[sample_index]).item())
                segments = [(anchor_index, anchor_index)]
            ranked_segments = sorted(
                segments,
                key=lambda item: float(scores[item[0] : item[1] + 1].max().detach().cpu()),
                reverse=True,
            )[:top_k]
            for token_index, (start_index, end_index) in enumerate(ranked_segments):
                span = slice(start_index, end_index + 1)
                span_scores = scores[span]
                if not bool(torch.any(span_scores > 0)):
                    span_scores = inputs.vehicle_event_scores[sample_index, span]
                safe_scores = torch.clamp(span_scores, min=torch.finfo(span_scores.dtype).eps)
                weight_sum = safe_scores.sum()
                normalized = safe_scores / torch.clamp(weight_sum, min=torch.finfo(safe_scores.dtype).eps)
                token_states[sample_index, token_index] = (
                    inputs.vehicle_states[sample_index, span, :] * normalized.unsqueeze(-1)
                ).sum(dim=0)
                token_scores[sample_index, token_index] = span_scores.max()
                offsets = inputs.vehicle_offsets_s[sample_index, span]
                center_offsets[sample_index, token_index] = (offsets * normalized).sum()
                start_offsets[sample_index, token_index] = offsets[0]
                end_offsets[sample_index, token_index] = offsets[-1]
                token_mask[sample_index, token_index] = True

        return token_states, token_scores, center_offsets, start_offsets, end_offsets, token_mask


class SemanticQueryBank(nn.Module):
    """Build deterministic semantic queries from fused physiology/vehicle context."""

    def __init__(self, config: CausalEventFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalEventFusionConfig()

    def build_queries(
        self,
        *,
        physiology_states: torch.Tensor,
        vehicle_states: torch.Tensor,
        event_token_states: torch.Tensor,
        event_token_scores: torch.Tensor,
        event_token_mask: torch.Tensor,
    ) -> tuple[tuple[str, ...], torch.Tensor]:
        physiology_pool = physiology_states.mean(dim=1)
        vehicle_pool = vehicle_states.mean(dim=1)
        gap_pool = torch.abs(physiology_pool - vehicle_pool)
        weighted_event = _masked_weighted_pool(
            event_token_states,
            event_token_scores,
            event_token_mask,
        )
        query_rows: list[torch.Tensor] = []
        query_names: list[str] = []
        for spec in self.config.query_specs:
            if spec.recipe == "gap_plus_event":
                query = gap_pool + weighted_event
            elif spec.recipe == "physiology_plus_gap":
                query = physiology_pool + gap_pool
            elif spec.recipe == "vehicle_plus_event":
                query = vehicle_pool + weighted_event
            elif spec.recipe == "coordination_gap":
                query = gap_pool
            else:
                raise ValueError(f"unsupported semantic query recipe: {spec.recipe}")
            query_rows.append(F.normalize(query, p=2, dim=-1, eps=1e-12))
            query_names.append(spec.name)
        return tuple(query_names), torch.stack(query_rows, dim=1)


class CausalEventFusion(nn.Module):
    """Fuse semantic queries with event tokens for event-level attribution."""

    def __init__(self, config: CausalEventFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalEventFusionConfig()
        self.extractor = EventTokenExtractor(self.config)
        self.query_bank = SemanticQueryBank(self.config)

    def forward(self, inputs: SemanticEventTensorInput) -> SemanticEventTensorOutput:
        (
            event_token_states,
            event_token_scores,
            center_offsets,
            start_offsets,
            end_offsets,
            token_mask,
        ) = self.extractor.extract(inputs)
        query_names, query_states = self.query_bank.build_queries(
            physiology_states=inputs.physiology_states,
            vehicle_states=inputs.vehicle_states,
            event_token_states=event_token_states,
            event_token_scores=event_token_scores,
            event_token_mask=token_mask,
        )
        raw_scores = torch.matmul(query_states, event_token_states.transpose(-1, -2))
        raw_scores = raw_scores / self.config.attention_temperature
        raw_scores = raw_scores + (event_token_scores.unsqueeze(1) * self.config.event_score_bias_weight)
        token_mask_3d = token_mask.unsqueeze(1).expand_as(raw_scores)
        masked_scores = raw_scores.masked_fill(~token_mask_3d, torch.finfo(raw_scores.dtype).min)
        query_to_event_attention = torch.softmax(masked_scores, dim=-1)
        query_to_event_attention = query_to_event_attention.masked_fill(~token_mask_3d, 0.0)
        query_to_event_attention = _renormalize_attention(query_to_event_attention)
        query_context_states = torch.matmul(query_to_event_attention, event_token_states)
        event_attribution_scores = query_to_event_attention.sum(dim=1) * event_token_scores
        query_attribution_scores = (query_to_event_attention * event_token_scores.unsqueeze(1)).max(dim=-1).values
        return SemanticEventTensorOutput(
            query_names=query_names,
            query_states=query_states,
            query_context_states=query_context_states,
            event_token_states=event_token_states,
            event_token_scores=event_token_scores,
            event_token_center_offsets_s=center_offsets,
            event_token_start_offsets_s=start_offsets,
            event_token_end_offsets_s=end_offsets,
            event_token_mask=token_mask,
            query_to_event_attention=query_to_event_attention,
            event_attribution_scores=event_attribution_scores,
            query_attribution_scores=query_attribution_scores,
        )


def semantic_query_entropy(
    attention_weights: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Return normalized entropy for query-to-event attention."""

    if attention_weights.ndim != 3:
        raise ValueError("attention_weights must have shape [B, Q, K].")
    if token_mask.shape != attention_weights.shape[:1] + attention_weights.shape[2:]:
        raise ValueError("token_mask must have shape [B, K].")
    visible_count = token_mask.sum(dim=-1, keepdim=True).to(dtype=attention_weights.dtype)
    safe_visible = torch.clamp(visible_count, min=1.0)
    raw_entropy = -(attention_weights * torch.log(torch.clamp(attention_weights, min=1e-12))).sum(dim=-1)
    return torch.where(
        visible_count > 1,
        raw_entropy / torch.clamp(torch.log(safe_visible), min=1e-12),
        torch.zeros_like(raw_entropy),
    )


def _validate_semantic_inputs(inputs: SemanticEventTensorInput) -> None:
    if inputs.physiology_states.ndim != 3 or inputs.vehicle_states.ndim != 3:
        raise ValueError("state tensors must have shape [B, R, D].")
    if inputs.attention_weights.ndim != 3:
        raise ValueError("attention_weights must have shape [B, Q, R].")
    if inputs.vehicle_event_scores.shape != inputs.vehicle_states.shape[:2]:
        raise ValueError("vehicle_event_scores must match vehicle state shape [B, R].")
    if inputs.vehicle_offsets_s.shape != inputs.vehicle_states.shape[:2]:
        raise ValueError("vehicle_offsets_s must match vehicle state shape [B, R].")
    if inputs.attention_weights.shape[0] != inputs.vehicle_states.shape[0]:
        raise ValueError("attention_weights batch dimension must match vehicle states.")
    if inputs.attention_weights.shape[-1] != inputs.vehicle_states.shape[1]:
        raise ValueError("attention_weights key dimension must match vehicle point count.")


def _collect_segments(active: torch.Tensor) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, is_active in enumerate(active.tolist()):
        if is_active and start_index is None:
            start_index = index
        elif not is_active and start_index is not None:
            segments.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        segments.append((start_index, active.shape[0] - 1))
    return segments


def _masked_weighted_pool(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    safe_weights = torch.where(mask, weights, torch.zeros_like(weights))
    denominator = safe_weights.sum(dim=-1, keepdim=True)
    normalized = safe_weights / torch.clamp(denominator, min=torch.finfo(weights.dtype).eps)
    return (values * normalized.unsqueeze(-1)).sum(dim=1)


def _renormalize_attention(attention_weights: torch.Tensor) -> torch.Tensor:
    denominator = attention_weights.sum(dim=-1, keepdim=True)
    return attention_weights / torch.clamp(denominator, min=torch.finfo(attention_weights.dtype).eps)
