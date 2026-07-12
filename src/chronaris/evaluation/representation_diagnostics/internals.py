"""Attention and scale-gate summaries for Chronaris fusion outputs."""

from __future__ import annotations

import torch


def fusion_internal_rows(fusion_output) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for scale_index, (weights, mask) in enumerate(
        zip(fusion_output.attention_weights, fusion_output.lag_masks, strict=True)
    ):
        visible_count = mask.sum(dim=-1)
        available = visible_count > 0
        entropy = -(weights.clamp_min(1e-15) * weights.clamp_min(1e-15).log()).sum(dim=-1)
        maximum = visible_count.clamp_min(1).to(weights.dtype).log()
        normalized = torch.where(maximum > 0, entropy / maximum, torch.zeros_like(entropy))
        valid_entropy = normalized[available]
        gate = fusion_output.scale_gate_weights[..., scale_index]
        rows.append(
            {
                "scale_index": scale_index,
                "available_query_count": int(available.sum()),
                "normalized_attention_entropy_mean": (
                    float(valid_entropy.mean()) if valid_entropy.numel() else None
                ),
                "normalized_attention_entropy_std": (
                    float(valid_entropy.std(unbiased=False)) if valid_entropy.numel() else None
                ),
                "near_uniform_attention_fraction": (
                    float((valid_entropy >= 0.95).float().mean())
                    if valid_entropy.numel()
                    else None
                ),
                "scale_gate_mean": float(gate[available].mean()) if bool(available.any()) else None,
                "scale_gate_std": (
                    float(gate[available].std(unbiased=False)) if bool(available.any()) else None
                ),
            }
        )
    return tuple(rows)
