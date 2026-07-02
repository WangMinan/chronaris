"""Chronaris v3 role-aware public fusion wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.mult.modules.transformer import TransformerEncoder
from chronaris.models.fusion import (
    RoleAwareCausalFusion,
    private_stream_metadata,
    public_stream_metadata,
)


@dataclass(frozen=True, slots=True)
class DeepForwardResult:
    pooled_embedding: torch.Tensor
    sequence_embedding: torch.Tensor
    attention_map: torch.Tensor
    logits: torch.Tensor | None


class ChronarisRoleAwareFusionWrapper(nn.Module):
    """Chronaris v3 wrapper with explicit stream-role-aware routing."""

    def __init__(
        self,
        *,
        ordered_modalities: Sequence[str],
        modality_input_dims: Mapping[str, int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        output_dim: int | None = None,
        dataset_id: str = "nasa_csm",
        variant: str = "chronaris_v3_stream_role",
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if len(self.ordered_modalities) != 2:
            raise ValueError("ChronarisRoleAwareFusionWrapper expects exactly two modalities.")
        self.dataset_id = dataset_id
        self.metadata = (
            private_stream_metadata()
            if dataset_id == "private_stage_h"
            else public_stream_metadata(dataset_id)
        )
        self.variant = variant
        self.projections = nn.ModuleDict(
            {
                modality_name: nn.Linear(modality_input_dims[modality_name] + 2, hidden_dim)
                for modality_name in self.ordered_modalities
            }
        )
        self.temporal_blocks = nn.ModuleDict(
            {
                modality_name: TransformerEncoder(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    layers=max(layers, 1),
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                )
                for modality_name in self.ordered_modalities
            }
        )
        self.role_aware_fusion = RoleAwareCausalFusion(hidden_dim=hidden_dim)
        self.output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, output_dim),
            )
            if output_dim is not None
            else None
        )

    def forward(
        self,
        modality_arrays: Mapping[str, torch.Tensor],
        *,
        time_axis: torch.Tensor,
        modality_masks: Mapping[str, torch.Tensor],
    ) -> DeepForwardResult:
        first_name, second_name = self.ordered_modalities
        first = self._encode(first_name, modality_arrays[first_name], time_axis, modality_masks[first_name])
        second = self._encode(second_name, modality_arrays[second_name], time_axis, modality_masks[second_name])
        combined_mask = torch.maximum(modality_masks[first_name], modality_masks[second_name])
        fusion_output = self.role_aware_fusion(
            physiology_states=first,
            second_stream_states=second,
            physiology_offsets_s=time_axis,
            second_stream_offsets_s=time_axis,
            metadata=self.metadata,
            variant=self.variant,
        )
        pooled = _masked_mean_pool(fusion_output.fused_states, combined_mask)
        logits = self.output_head(pooled) if self.output_head is not None else None
        return DeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=fusion_output.fused_states,
            attention_map=fusion_output.attention_weights,
            logits=logits,
        )

    def _encode(
        self,
        modality_name: str,
        values: torch.Tensor,
        time_axis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.projections[modality_name](_append_time_features(values, time_axis))
        projected = projected * mask.unsqueeze(-1)
        return self.temporal_blocks[modality_name](projected.transpose(0, 1)).transpose(0, 1)


def _masked_mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(dtype=values.dtype)
    denominator = weight.sum(dim=1).clamp_min(1.0)
    return (values * weight).sum(dim=1) / denominator


def _append_time_features(values: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
    deltas = torch.zeros_like(time_axis)
    deltas[:, 1:] = time_axis[:, 1:] - time_axis[:, :-1]
    return torch.cat((values, time_axis.unsqueeze(-1), deltas.unsqueeze(-1)), dim=-1)
