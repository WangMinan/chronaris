"""Frozen scalar reference for vectorized event extraction checks."""

from types import SimpleNamespace
import torch
from chronaris.models.fusion.semantic_event import _validate_semantic_inputs

def reference_extract(inputs, config):
    self = SimpleNamespace(config=config)
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
        visible = (inputs.vehicle_valid_mask[sample_index]
                   if inputs.vehicle_valid_mask is not None else torch.ones_like(scores, dtype=torch.bool))
        positive_scores = scores[visible & (scores > 0)]
        if positive_scores.numel() > 0:
            threshold = torch.quantile(
                positive_scores,
                q=min(self.config.event_score_quantile, 1.0),
            )
            active = visible & (scores >= threshold)
        else:
            if inputs.vehicle_valid_mask is not None:
                continue
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
