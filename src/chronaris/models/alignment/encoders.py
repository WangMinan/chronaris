"""Observation encoders for Stage E alignment prototypes."""

from __future__ import annotations

import torch
from torch import nn

from chronaris.models.alignment._torch_utils import build_activation_module


class ObservationEncoder(nn.Module):
    """Encode irregular observations into latent embeddings."""

    def __init__(
        self,
        feature_dim: int,
        *,
        embedding_dim: int,
        hidden_dim: int,
        activation: str = "gelu",
        use_feature_valid_mask: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.use_feature_valid_mask = use_feature_valid_mask
        input_dim = feature_dim * 2 if use_feature_valid_mask else feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            build_activation_module(activation),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        feature_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of observation matrices."""

        if values.ndim != 3:
            raise ValueError("values must have shape [B, T, F].")
        if feature_valid_mask.shape != values.shape:
            raise ValueError("feature_valid_mask must match values shape.")
        if values.shape[-1] != self.feature_dim:
            raise ValueError("Input feature dimension does not match encoder feature_dim.")

        if self.use_feature_valid_mask:
            encoder_input = torch.cat((values, feature_valid_mask.to(dtype=values.dtype)), dim=-1)
        else:
            encoder_input = values

        batch_size, point_count, _ = encoder_input.shape
        encoded = self.network(encoder_input.reshape(batch_size * point_count, -1))
        return encoded.reshape(batch_size, point_count, self.embedding_dim)
