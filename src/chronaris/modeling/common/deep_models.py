"""Deep baseline model wrappers for task evaluation sequence experiments."""

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

from third_party.contiformer.physiopro.network import ContiFormerEncoder
from third_party.mult.modules.transformer import TransformerEncoder
from chronaris.models.alignment.task_heads_v2 import (
    ContrastiveRetrievalProjectionHead,
    PhysiologyResponseResidualRegressionHead,
    VehicleDominantAuxiliaryClassificationHead,
)
from chronaris.models.fusion.causal import (
    CausalFusionConfig,
    CausalFusionTensorInput,
    CausalMaskedCrossModalFusion,
)
from chronaris.models.fusion.role_aware_fusion import (
    RoleAwareCausalFusion,
    private_stream_metadata,
    public_stream_metadata,
)

PUBLIC_ADAPTER_EVIDENCE_ROLE = "public_adapter_evidence"
PUBLIC_CONTEXT_PROXY_ROLE = "context_proxy"


@dataclass(frozen=True, slots=True)
class StageIDeepForwardResult:
    """Common forward output for task evaluation deep baseline wrappers."""

    pooled_embedding: torch.Tensor
    sequence_embedding: torch.Tensor
    attention_map: torch.Tensor
    logits: torch.Tensor | None


class ChronarisMulTWrapper(nn.Module):
    """Bi-modal MulT-style wrapper over the vendored transformer blocks."""

    def __init__(
        self,
        *,
        ordered_modalities: Sequence[str],
        modality_input_dims: Mapping[str, int],
        hidden_dim: int = 32,
        num_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if len(ordered_modalities) != 2:
            raise ValueError("ChronarisMulTWrapper expects exactly two modalities.")
        first_name, second_name = tuple(ordered_modalities)
        self.ordered_modalities = (first_name, second_name)
        self.projections = nn.ModuleDict(
            {
                modality_name: nn.Linear(
                    modality_input_dims[modality_name] + 2,
                    hidden_dim,
                )
                for modality_name in self.ordered_modalities
            },
        )
        self.cross_encoders = nn.ModuleDict(
            {
                f"{first_name}_with_{second_name}": TransformerEncoder(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    layers=layers,
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                ),
                f"{second_name}_with_{first_name}": TransformerEncoder(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    layers=layers,
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                ),
            },
        )
        self.memory_blocks = nn.ModuleDict(
            {
                modality_name: TransformerEncoder(
                    embed_dim=hidden_dim * 2,
                    num_heads=num_heads,
                    layers=max(layers, 2),
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                )
                for modality_name in self.ordered_modalities
            },
        )
        self.output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
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
    ) -> StageIDeepForwardResult:
        first_name, second_name = self.ordered_modalities
        first = self.projections[first_name](
            _append_time_features(modality_arrays[first_name], time_axis),
        )
        second = self.projections[second_name](
            _append_time_features(modality_arrays[second_name], time_axis),
        )
        first = first * modality_masks[first_name].unsqueeze(-1)
        second = second * modality_masks[second_name].unsqueeze(-1)
        cross_first = self._cross_encode(
            query=first,
            key_value=second,
            encoder=self.cross_encoders[f"{first_name}_with_{second_name}"],
        )
        cross_second = self._cross_encode(
            query=second,
            key_value=first,
            encoder=self.cross_encoders[f"{second_name}_with_{first_name}"],
        )
        merged_first = self._self_encode(
            modality_name=first_name,
            values=torch.cat((first, cross_first), dim=-1),
        )
        merged_second = self._self_encode(
            modality_name=second_name,
            values=torch.cat((second, cross_second), dim=-1),
        )
        pooled_first = masked_mean_pool(merged_first, modality_masks[first_name])
        pooled_second = masked_mean_pool(merged_second, modality_masks[second_name])
        pooled = torch.cat((pooled_first, pooled_second), dim=-1)
        logits = self.output_head(pooled) if self.output_head is not None else None
        attention_map = _scaled_attention_map(
            first,
            second,
            modality_masks[first_name],
            modality_masks[second_name],
        )
        sequence_embedding = torch.cat((merged_first, merged_second), dim=-1)
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=sequence_embedding,
            attention_map=attention_map,
            logits=logits,
        )

    def _cross_encode(
        self,
        *,
        query: torch.Tensor,
        key_value: torch.Tensor,
        encoder: TransformerEncoder,
    ) -> torch.Tensor:
        query_seq = query.transpose(0, 1)
        key_seq = key_value.transpose(0, 1)
        return encoder(query_seq, key_seq, key_seq).transpose(0, 1)

    def _self_encode(self, *, modality_name: str, values: torch.Tensor) -> torch.Tensor:
        sequence = values.transpose(0, 1)
        return self.memory_blocks[modality_name](sequence).transpose(0, 1)


class ChronarisContiFormerWrapper(nn.Module):
    """Continuous-time wrapper built on the minimal vendored ContiFormer subset."""

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
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if len(self.ordered_modalities) != 2:
            raise ValueError("ChronarisContiFormerWrapper expects exactly two modalities.")
        combined_input_dim = (
            sum(modality_input_dims[name] for name in self.ordered_modalities)
            + len(self.ordered_modalities)
        )
        self.encoder = ContiFormerEncoder(
            input_dim=combined_input_dim,
            model_dim=hidden_dim,
            num_heads=num_heads,
            depth=layers,
            dropout=dropout,
        )
        self.output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
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
    ) -> StageIDeepForwardResult:
        merged_values = torch.cat(
            [
                modality_arrays[name]
                for name in self.ordered_modalities
            ]
            + [
                modality_masks[name].unsqueeze(-1)
                for name in self.ordered_modalities
            ],
            dim=-1,
        )
        combined_mask = torch.zeros_like(next(iter(modality_masks.values())))
        for mask in modality_masks.values():
            combined_mask = torch.maximum(combined_mask, mask)
        encoded, attention_stack = self.encoder(
            merged_values,
            time_axis=time_axis,
            mask=combined_mask,
        )
        pooled = masked_mean_pool(encoded, combined_mask)
        logits = self.output_head(pooled) if self.output_head is not None else None
        attention_map = attention_stack[-1] if attention_stack else _empty_attention(encoded)
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=encoded,
            attention_map=attention_map,
            logits=logits,
        )


class ChronarisPublicFusionWrapper(nn.Module):
    """Public Chronaris wrapper over physiology + context-proxy adapter streams."""

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
        fusion_event_bias_weight: float = 0.25,
        fusion_lag_window_points: int | None = None,
        fusion_normalize_states: bool = True,
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if len(self.ordered_modalities) != 2:
            raise ValueError("ChronarisPublicFusionWrapper expects exactly two modalities.")
        first_name, second_name = self.ordered_modalities
        self.thesis_facing_contract = {
            "evidence_role": PUBLIC_ADAPTER_EVIDENCE_ROLE,
            "first_stream_name": first_name,
            "first_stream_role": "physiology",
            "second_stream_name": second_name,
            "second_stream_role": PUBLIC_CONTEXT_PROXY_ROLE,
            "second_stream_is_real_vehicle": False,
            "fusion_vehicle_slot_semantics": "context_proxy_adapter_reuse_only",
        }
        self.projections = nn.ModuleDict(
            {
                modality_name: nn.Linear(
                    modality_input_dims[modality_name] + 2,
                    hidden_dim,
                )
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
            CausalFusionConfig(
                event_bias_weight=fusion_event_bias_weight,
                lag_window_points=fusion_lag_window_points,
                normalize_states=fusion_normalize_states,
            )
        )
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
    ) -> StageIDeepForwardResult:
        first_name, second_name = self.ordered_modalities
        first = self.projections[first_name](
            _append_time_features(modality_arrays[first_name], time_axis),
        )
        context_proxy = self.projections[second_name](
            _append_time_features(modality_arrays[second_name], time_axis),
        )
        first = first * modality_masks[first_name].unsqueeze(-1)
        context_proxy = context_proxy * modality_masks[second_name].unsqueeze(-1)
        encoded_first = self._self_encode(first_name, first)
        encoded_context_proxy = self._self_encode(second_name, context_proxy)
        combined_mask = torch.maximum(modality_masks[first_name], modality_masks[second_name])
        # The Stage G fusion core still exposes a `vehicle_states` slot. On the public
        # branch we reuse that slot for the adapter/context proxy stream only.
        fusion_output = self.causal_fusion(
            CausalFusionTensorInput(
                physiology_states=encoded_first,
                vehicle_states=encoded_context_proxy,
                physiology_offsets_s=time_axis,
                vehicle_offsets_s=time_axis,
            )
        )
        pooled = masked_mean_pool(fusion_output.fused_states, combined_mask)
        logits = self.output_head(pooled) if self.output_head is not None else None
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=fusion_output.fused_states,
            attention_map=fusion_output.attention_weights,
            logits=logits,
        )

    def _self_encode(self, modality_name: str, values: torch.Tensor) -> torch.Tensor:
        sequence = values.transpose(0, 1)
        return self.temporal_blocks[modality_name](sequence).transpose(0, 1)


class ChronarisSingleStreamWrapper(nn.Module):
    """Single-stream public ablation wrapper using one prepared modality only."""

    def __init__(
        self,
        *,
        ordered_modalities: Sequence[str],
        modality_input_dims: Mapping[str, int],
        stream_index: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if not self.ordered_modalities:
            raise ValueError("ChronarisSingleStreamWrapper requires at least one modality.")
        if stream_index >= len(self.ordered_modalities):
            raise ValueError(
                f"stream_index={stream_index} is out of range for {self.ordered_modalities}."
            )
        self.stream_name = self.ordered_modalities[stream_index]
        self.projection = nn.Linear(
            modality_input_dims[self.stream_name] + 2,
            hidden_dim,
        )
        self.encoder = TransformerEncoder(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            layers=max(layers, 1),
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=dropout,
            attn_mask=False,
        )
        self.output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
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
    ) -> StageIDeepForwardResult:
        mask = modality_masks[self.stream_name]
        values = self.projection(
            _append_time_features(modality_arrays[self.stream_name], time_axis),
        )
        values = values * mask.unsqueeze(-1)
        encoded = self.encoder(values.transpose(0, 1)).transpose(0, 1)
        pooled = masked_mean_pool(encoded, mask)
        logits = self.output_head(pooled) if self.output_head is not None else None
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=encoded,
            attention_map=_empty_attention(encoded),
            logits=logits,
        )


class ChronarisSimpleDualStreamConcatWrapper(nn.Module):
    """Dual-stream ablation with temporal encoders and late concat, no causal fusion."""

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
    ) -> None:
        super().__init__()
        self.ordered_modalities = tuple(ordered_modalities)
        if len(self.ordered_modalities) != 2:
            raise ValueError(
                "ChronarisSimpleDualStreamConcatWrapper expects exactly two modalities."
            )
        self.projections = nn.ModuleDict(
            {
                modality_name: nn.Linear(
                    modality_input_dims[modality_name] + 2,
                    hidden_dim,
                )
                for modality_name in self.ordered_modalities
            }
        )
        self.encoders = nn.ModuleDict(
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
        self.output_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
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
    ) -> StageIDeepForwardResult:
        pooled_parts = []
        encoded_parts = []
        for modality_name in self.ordered_modalities:
            mask = modality_masks[modality_name]
            values = self.projections[modality_name](
                _append_time_features(modality_arrays[modality_name], time_axis),
            )
            values = values * mask.unsqueeze(-1)
            encoded = self.encoders[modality_name](values.transpose(0, 1)).transpose(0, 1)
            encoded_parts.append(encoded)
            pooled_parts.append(masked_mean_pool(encoded, mask))
        pooled = torch.cat(pooled_parts, dim=-1)
        sequence_embedding = torch.cat(encoded_parts, dim=-1)
        logits = self.output_head(pooled) if self.output_head is not None else None
        attention_map = _scaled_attention_map(
            encoded_parts[0],
            encoded_parts[1],
            modality_masks[self.ordered_modalities[0]],
            modality_masks[self.ordered_modalities[1]],
        )
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=sequence_embedding,
            attention_map=attention_map,
            logits=logits,
        )


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
                name: nn.Linear(modality_input_dims[name] + 2, hidden_dim)
                for name in self.ordered_modalities
            }
        )
        self.temporal_blocks = nn.ModuleDict(
            {
                name: TransformerEncoder(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    layers=max(layers, 1),
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                )
                for name in self.ordered_modalities
            }
        )
        self.causal_fusion = CausalMaskedCrossModalFusion(CausalFusionConfig(event_bias_weight=0.25, lag_window_points=16))
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
    ) -> StageIDeepForwardResult:
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
        pooled = masked_mean_pool(fusion.fused_states, mask)
        logits = None
        if self.task_type == "classification":
            if self.task_head is None:
                logits = self.output_head(pooled)
            else:
                logits = self.task_head(vehicle_states=vehicle, fused_states=fusion.fused_states).logits
        elif self.task_type == "regression":
            logits = (
                self.output_head(pooled)
                if self.task_head is None
                else self.task_head(physiology_states=physiology, vehicle_states=vehicle, fused_states=fusion.fused_states).prediction
            )
        else:
            pooled = self.task_head(fusion.fused_states)
        return StageIDeepForwardResult(
            pooled_embedding=pooled,
            sequence_embedding=fusion.fused_states,
            attention_map=fusion.attention_weights,
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
        self.metadata = private_stream_metadata() if dataset_id == "private_feature_export" else public_stream_metadata(dataset_id)
        self.variant = variant
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(modality_input_dims[name] + 2, hidden_dim)
                for name in self.ordered_modalities
            }
        )
        self.temporal_blocks = nn.ModuleDict(
            {
                name: TransformerEncoder(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    layers=max(layers, 1),
                    attn_dropout=dropout,
                    relu_dropout=dropout,
                    res_dropout=dropout,
                    embed_dropout=dropout,
                    attn_mask=False,
                )
                for name in self.ordered_modalities
            }
        )
        self.role_aware_fusion = RoleAwareCausalFusion(
            hidden_dim=hidden_dim,
            context_adapter_hidden_multiplier=2 if "cap2x" in variant else 1,
            context_adapter_dropout=0.2 if "do0p2" in variant else 0.1 if "do0p1" in variant else 0.0,
        )
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
    ) -> StageIDeepForwardResult:
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
        pooled = masked_mean_pool(fusion_output.fused_states, combined_mask)
        logits = self.output_head(pooled) if self.output_head is not None else None
        return StageIDeepForwardResult(
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


def build_task_eval_deep_model(
    *,
    model_name: str,
    ordered_modalities: Sequence[str],
    modality_input_dims: Mapping[str, int],
    output_dim: int | None,
    hidden_dim: int = 64,
    num_heads: int = 4,
    layers: int = 2,
    dropout: float = 0.1,
    fusion_event_bias_weight: float = 0.25,
    fusion_lag_window_points: int | None = None,
    fusion_normalize_states: bool = True,
    dataset_id: str | None = None,
) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "mult":
        return ChronarisMulTWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=max(hidden_dim // 2, 16),
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
        )
    if normalized == "contiformer":
        return ChronarisContiFormerWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
        )
    if normalized == "chronaris_public_fusion":
        return ChronarisPublicFusionWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
            fusion_event_bias_weight=fusion_event_bias_weight,
            fusion_lag_window_points=fusion_lag_window_points,
            fusion_normalize_states=fusion_normalize_states,
        )
    if normalized in {
        "chronaris_v2_task_heads",
        "v2_no_vehicle_aux_head",
        "v2_no_vehicle_aux",
        "v2_no_residual_t2_head",
        "v2_no_residual_t2",
        "v2_no_contrastive_t3_loss",
    } or normalized.startswith("p37_t1_") or normalized.startswith("p37_t3_"):
        return ChronarisPrivateTaskAwareWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
            variant=normalized,
        )
    if normalized in {
        "chronaris_v3_stream_role",
        "chronaris_v3_stream_role_fusion",
        "v3_stream_role",
        "v3_stream_role_fusion",
        "v3_stream_role_adaptive",
        "v3_no_role_gate",
        "v3_fixed_causal_lag",
        "v3_force_private_causal",
        "v3_context_adapter_only",
    } or normalized.startswith("p37_public_"):
        return ChronarisRoleAwareFusionWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
            dataset_id=dataset_id or "nasa_csm",
            variant=normalized,
        )
    if normalized in {"chronaris_public_fusion_physiology_only", "physiology_only"}:
        return ChronarisSingleStreamWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            stream_index=0,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
        )
    if normalized in {"chronaris_public_fusion_context_only", "context_only"}:
        return ChronarisSingleStreamWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            stream_index=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
        )
    if normalized in {
        "chronaris_public_fusion_simple_concat",
        "simple_dual_stream_concat",
        "late_concat_no_causal_fusion",
    }:
        return ChronarisSimpleDualStreamConcatWrapper(
            ordered_modalities=ordered_modalities,
            modality_input_dims=modality_input_dims,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layers=layers,
            dropout=dropout,
            output_dim=output_dim,
        )
    raise ValueError(f"unsupported deep baseline model: {model_name}")


def masked_mean_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(dtype=values.dtype)
    denominator = weight.sum(dim=1).clamp_min(1.0)
    return (values * weight).sum(dim=1) / denominator


def _append_time_features(values: torch.Tensor, time_axis: torch.Tensor) -> torch.Tensor:
    deltas = torch.zeros_like(time_axis)
    deltas[:, 1:] = time_axis[:, 1:] - time_axis[:, :-1]
    return torch.cat(
        (
            values,
            time_axis.unsqueeze(-1),
            deltas.unsqueeze(-1),
        ),
        dim=-1,
    )


def _scaled_attention_map(
    first: torch.Tensor,
    second: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
) -> torch.Tensor:
    scores = torch.matmul(first, second.transpose(1, 2)) / max(first.shape[-1], 1) ** 0.5
    valid = first_mask.unsqueeze(-1) * second_mask.unsqueeze(1)
    scores = scores.masked_fill(valid <= 0, torch.finfo(scores.dtype).min)
    return torch.softmax(scores, dim=-1)


def _empty_attention(encoded: torch.Tensor) -> torch.Tensor:
    batch_size, time_steps = encoded.shape[:2]
    return torch.zeros(
        (batch_size, time_steps, time_steps),
        dtype=encoded.dtype,
        device=encoded.device,
    )


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
