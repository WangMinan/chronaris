"""Minimal continuous-time encoder subset for Chronaris ContiFormer wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class AttrDict(dict):
    """Dictionary with attribute access to mimic the upstream helper."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as error:  # pragma: no cover - parity helper
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value):
        self[key] = value


class ContinuousTimeEncoding(nn.Module):
    """Project absolute time and delta time into the model space."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, time_axis: torch.Tensor) -> torch.Tensor:
        if time_axis.ndim != 2:
            raise ValueError("time_axis must have shape [batch, time].")
        deltas = torch.zeros_like(time_axis)
        deltas[:, 1:] = time_axis[:, 1:] - time_axis[:, :-1]
        features = torch.stack((time_axis, deltas), dim=-1)
        return self.proj(features)


class EncoderLayer(nn.Module):
    """One lightweight continuous-time self-attention layer."""

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_encoding = ContinuousTimeEncoding(model_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        time_axis: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = values + self.time_encoding(time_axis)
        normed = self.norm1(encoded)
        attn_output, attn_weights = self.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        hidden = encoded + self.dropout(attn_output)
        hidden = hidden + self.dropout(self.ffn(self.norm2(hidden)))
        attention = attn_weights.mean(dim=1)
        return hidden, attention


@dataclass(frozen=True, slots=True)
class _EncoderConfig:
    input_dim: int
    model_dim: int
    num_heads: int
    depth: int
    dropout: float


class ContiFormerEncoder(nn.Module):
    """Small continuous-time encoder used by the Chronaris wrapper."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = _EncoderConfig(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
        )
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.layers = nn.ModuleList(
            EncoderLayer(model_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: torch.Tensor,
        *,
        time_axis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden = self.input_proj(values)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.to(dtype=torch.bool)
        attention_maps: list[torch.Tensor] = []
        for layer in self.layers:
            hidden, attention = layer(
                hidden,
                time_axis=time_axis,
                key_padding_mask=key_padding_mask,
            )
            attention_maps.append(attention)
        return self.output_norm(hidden), tuple(attention_maps)
