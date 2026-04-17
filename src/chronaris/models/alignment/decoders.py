"""Decoder and projection heads for Stage E alignment prototypes."""

from __future__ import annotations

import torch
from torch import nn

from chronaris.models.alignment._torch_utils import build_activation_module


class ObservationDecoder(nn.Module):
    """Decode latent states back into observation space."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        output_dim: int,
        projection_hidden_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, projection_hidden_dim),
            nn.LayerNorm(projection_hidden_dim),
            build_activation_module(activation),
            nn.Linear(projection_hidden_dim, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode hidden states with shape [B, T, H] or [B, H]."""

        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError("Input hidden dimension does not match decoder hidden_dim.")

        trailing_shape = hidden_states.shape[:-1]
        decoded = self.network(hidden_states.reshape(-1, self.hidden_dim))
        return decoded.reshape(*trailing_shape, self.output_dim)


class AlignmentProjectionHead(nn.Module):
    """Project hidden states into a shared alignment space."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        projection_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")

        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            build_activation_module(activation),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states with shape [B, T, H] or [B, H]."""

        if hidden_states.shape[-1] != self.hidden_dim:
            raise ValueError("Input hidden dimension does not match projection head hidden_dim.")

        trailing_shape = hidden_states.shape[:-1]
        projected = self.network(hidden_states.reshape(-1, self.hidden_dim))
        return projected.reshape(*trailing_shape, self.projection_dim)
