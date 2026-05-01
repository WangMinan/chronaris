"""Deep baseline model wrappers for Stage I sequence experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.contiformer.physiopro.network import ContiFormerEncoder
from third_party.mult.modules.transformer import TransformerEncoder


@dataclass(frozen=True, slots=True)
class StageIDeepForwardResult:
    """Common forward output for Stage I deep baseline wrappers."""

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


def build_stage_i_deep_model(
    *,
    model_name: str,
    ordered_modalities: Sequence[str],
    modality_input_dims: Mapping[str, int],
    output_dim: int | None,
    hidden_dim: int = 64,
    num_heads: int = 4,
    layers: int = 2,
    dropout: float = 0.1,
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
