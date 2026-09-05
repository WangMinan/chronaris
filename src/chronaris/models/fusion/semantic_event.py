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

    state_dim: int = 64
    learnable_queries: bool = False
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
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
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
    physiology_valid_mask: torch.Tensor | None = None
    vehicle_valid_mask: torch.Tensor | None = None


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
        batch_size, point_count, state_dim = inputs.vehicle_states.shape
        if point_count == 0:
            raise ValueError("event extraction requires reference points")
        top_k = self.config.event_top_k
        valid = (inputs.vehicle_valid_mask if inputs.vehicle_valid_mask is not None
                 else torch.ones_like(inputs.vehicle_event_scores, dtype=torch.bool))
        scores = (inputs.attention_weights.sum(dim=1) * inputs.vehicle_event_scores).masked_fill(~valid, 0)
        # Selection is discrete in the scalar reference too. Stable sorting keeps
        # the earliest segment when salience ties; pooling below remains differentiable.
        with torch.no_grad():
            positive = (scores > 0) & valid
            count = positive.sum(dim=-1)
            ordered = scores.masked_fill(~positive, torch.inf).sort(dim=-1).values
            ordered = ordered.masked_fill((count == 0).unsqueeze(-1), 0)
            rank = (count - 1).clamp_min(0).to(scores.dtype) * self.config.event_score_quantile
            lower, upper = rank.floor().long(), rank.ceil().long()
            threshold = torch.lerp(ordered.gather(1, lower[:, None]),
                                   ordered.gather(1, upper[:, None]), (rank - lower)[:, None])
            active = positive & (scores >= threshold)
            if inputs.vehicle_valid_mask is None:
                fallback = F.one_hot(inputs.vehicle_event_scores.argmax(-1), point_count).bool()
                active = torch.where((count > 0)[:, None], active, fallback)
            starts = active & ~F.pad(active[:, :-1], (1, 0), value=False)
            segment_ids = starts.long().cumsum(-1) - 1
            segment_count = starts.sum(-1)
            membership = (segment_ids[:, None, :] == torch.arange(point_count, device=scores.device)[None, :, None]) & active[:, None, :]
            maxima = scores[:, None, :].expand_as(membership).masked_fill(~membership, -torch.inf).amax(-1)
            selected = maxima.argsort(dim=-1, descending=True, stable=True)[:, :top_k]
            selected = F.pad(selected, (0, max(top_k - point_count, 0)))
            token_mask = torch.arange(top_k, device=scores.device)[None] < segment_count[:, None]
            membership = (segment_ids[:, None, :] == selected[:, :, None]) & active[:, None, :] & token_mask[:, :, None]
        contribution = scores[:, None, :].expand(-1, top_k, -1)
        use_contribution = (membership & (contribution > 0)).any(-1, keepdim=True)
        span_scores = torch.where(use_contribution, contribution, inputs.vehicle_event_scores[:, None, :])
        weights = span_scores.clamp_min(torch.finfo(scores.dtype).eps).masked_fill(~membership, 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).eps)
        states = inputs.vehicle_states.masked_fill(~valid.unsqueeze(-1), 0)
        token_states = torch.bmm(weights, states)
        token_scores = span_scores.expand_as(membership).masked_fill(~membership, -torch.inf).amax(-1).masked_fill(~token_mask, 0)
        offsets = inputs.vehicle_offsets_s[:, None, :].expand_as(membership)
        center = (weights * offsets).sum(-1).masked_fill(~token_mask, 0).to(states.dtype)
        start = offsets.masked_fill(~membership, torch.inf).amin(-1).masked_fill(~token_mask, 0).to(states.dtype)
        end = offsets.masked_fill(~membership, -torch.inf).amax(-1).masked_fill(~token_mask, 0).to(states.dtype)
        return token_states, token_scores, center, start, end, token_mask


class SemanticQueryBank(nn.Module):
    """Build deterministic or residual-learned queries from dual-stream context."""

    def __init__(self, config: CausalEventFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalEventFusionConfig()
        if self.config.learnable_queries:
            self.query_residual = nn.Parameter(
                torch.zeros(len(self.config.query_specs), self.config.state_dim)
            )
        else:
            self.register_parameter("query_residual", None)

    def build_queries(
        self,
        *,
        physiology_states: torch.Tensor,
        vehicle_states: torch.Tensor,
        event_token_states: torch.Tensor,
        event_token_scores: torch.Tensor,
        event_token_mask: torch.Tensor,
        physiology_valid_mask: torch.Tensor | None = None,
        vehicle_valid_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[str, ...], torch.Tensor]:
        physiology_pool = _masked_mean(physiology_states, physiology_valid_mask)
        vehicle_pool = _masked_mean(vehicle_states, vehicle_valid_mask)
        gap_pool = torch.abs(physiology_pool - vehicle_pool)
        weighted_event = _masked_weighted_pool(
            event_token_states,
            event_token_scores,
            event_token_mask,
        )
        query_rows: list[torch.Tensor] = []
        query_names: list[str] = []
        if self.query_residual is not None and physiology_states.shape[-1] != self.config.state_dim:
            raise ValueError("learnable semantic query state dimension does not match inputs")
        for query_index, spec in enumerate(self.config.query_specs):
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
            if self.query_residual is not None:
                query = query + self.query_residual[query_index]
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
            physiology_valid_mask=inputs.physiology_valid_mask,
            vehicle_valid_mask=inputs.vehicle_valid_mask,
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
    for stream in ("physiology", "vehicle"):
        mask = getattr(inputs, f"{stream}_valid_mask")
        if mask is not None and (mask.dtype != torch.bool or mask.shape != getattr(inputs, f"{stream}_states").shape[:2]):
            raise ValueError("semantic validity must be boolean [B,R]")


def _masked_mean(states: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return states.mean(dim=1)
    return states.masked_fill(~mask.unsqueeze(-1), 0).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


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
