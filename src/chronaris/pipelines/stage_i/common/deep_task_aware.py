"""Chronaris v2 task-aware deep wrapper for private Stage I tasks."""

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
from chronaris.models.alignment.task_heads_v2 import (
    ContrastiveRetrievalProjectionHead,
    PhysiologyResponseResidualRegressionHead,
    VehicleDominantAuxiliaryClassificationHead,
)
from chronaris.models.fusion import CausalFusionConfig, CausalFusionTensorInput, CausalMaskedCrossModalFusion


@dataclass(frozen=True, slots=True)
class DeepForwardResult:
    pooled_embedding: torch.Tensor
    sequence_embedding: torch.Tensor
    attention_map: torch.Tensor
    logits: torch.Tensor | None
    auxiliary_outputs: Mapping[str, torch.Tensor] | None = None


class ChronarisPrivateTaskAwareWrapper(nn.Module):
    """Private Chronaris v2 wrapper with task-aware heads."""

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
        variant: str = "chronaris_v2_task_heads",
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if len(self.ordered_modalities) != 2:
            raise ValueError("ChronarisPrivateTaskAwareWrapper expects exactly two modalities.")
        self.variant = variant
        self.task_type = "retrieval" if output_dim is None else "classification" if output_dim > 1 else "regression"
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
        self.causal_fusion = CausalMaskedCrossModalFusion(
            CausalFusionConfig(event_bias_weight=0.25, lag_window_points=16)
        )
        self.task_head: nn.Module | None
        self.output_head: nn.Module | None
        if self.task_type == "classification":
            if "no_vehicle_aux" in variant:
                self.task_head = None
                self.output_head = _fused_output_head(hidden_dim * 3, hidden_dim, int(output_dim), dropout)
            else:
                self.task_head = VehicleDominantAuxiliaryClassificationHead(
                    vehicle_dim=hidden_dim,
                    fused_dim=hidden_dim * 3,
                    output_dim=int(output_dim),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    initial_vehicle_bias=_vehicle_bias_from_variant(variant),
                    vehicle_skip_weight=0.15 if "vehicle_skip" in variant else 0.0,
                )
                self.output_head = None
        elif self.task_type == "regression":
            if "no_residual_t2" in variant:
                self.task_head = None
                self.output_head = _fused_output_head(hidden_dim * 3, hidden_dim, int(output_dim), dropout)
            else:
                self.task_head = PhysiologyResponseResidualRegressionHead(
                    physiology_dim=hidden_dim,
                    vehicle_dim=hidden_dim,
                    fused_dim=hidden_dim * 3,
                    output_dim=int(output_dim),
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                self.output_head = None
        else:
            self.task_head = ContrastiveRetrievalProjectionHead(
                input_dim=hidden_dim * 3,
                embedding_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            self.output_head = None

    def forward(
        self,
        modality_arrays: Mapping[str, torch.Tensor],
        *,
        time_axis: torch.Tensor,
        modality_masks: Mapping[str, torch.Tensor],
    ) -> DeepForwardResult:
        physiology_name, vehicle_name = self.ordered_modalities
        physiology = self._encode(physiology_name, modality_arrays[physiology_name], time_axis, modality_masks[physiology_name])
        vehicle = self._encode(vehicle_name, modality_arrays[vehicle_name], time_axis, modality_masks[vehicle_name])
        fusion = self.causal_fusion(
            CausalFusionTensorInput(
                physiology_states=physiology,
                vehicle_states=vehicle,
                physiology_offsets_s=time_axis,
                vehicle_offsets_s=time_axis,
            )
        )
        mask = torch.maximum(modality_masks[physiology_name], modality_masks[vehicle_name])
        pooled = _masked_mean_pool(fusion.fused_states, mask)
        logits = None
        auxiliary_outputs = None
        if self.task_type == "classification":
            if self.task_head is None:
                logits = self.output_head(pooled)
            else:
                head_output = self.task_head(vehicle_states=vehicle, fused_states=fusion.fused_states)
                logits = head_output.logits
                auxiliary_outputs = {
                    "gate": head_output.gate,
                    "vehicle_logits": head_output.vehicle_logits,
                    "fusion_logits": head_output.fusion_logits,
                }
        elif self.task_type == "regression":
            logits = (
                self.output_head(pooled)
                if self.task_head is None
                else self.task_head(
                    physiology_states=physiology,
                    vehicle_states=vehicle,
                    fused_states=fusion.fused_states,
                ).prediction
            )
        else:
            pooled = self.task_head(fusion.fused_states)
        return DeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=fusion.fused_states,
            attention_map=fusion.attention_weights,
            logits=logits,
            auxiliary_outputs=auxiliary_outputs,
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


def _fused_output_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _vehicle_bias_from_variant(variant: str) -> float:
    normalized = variant.lower()
    for token, value in (
        ("gate0p55", 0.55),
        ("gate0p65", 0.65),
        ("gate0p75", 0.75),
        ("gate0p85", 0.85),
    ):
        if token in normalized:
            return value
    return 0.75
