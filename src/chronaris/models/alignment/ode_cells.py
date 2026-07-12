"""Continuous-time state update cells for Stage E prototypes."""

from __future__ import annotations

import torch
from torch import nn
from torchdiffeq import odeint

from chronaris.models.alignment._torch_utils import build_activation_module


class HiddenStateODEFunc(nn.Module):
    """A minimal hidden-state dynamics function dh/dt = f(h, t)."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        dynamics_hidden_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if dynamics_hidden_dim <= 0:
            raise ValueError("dynamics_hidden_dim must be positive.")

        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, dynamics_hidden_dim),
            nn.LayerNorm(dynamics_hidden_dim),
            build_activation_module(activation),
            nn.Linear(dynamics_hidden_dim, hidden_dim),
        )

    def forward(self, _time: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute the hidden-state derivative."""

        if hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError("Input hidden dimension does not match dynamics hidden_dim.")
        return self.network(hidden_state)


class ODERNNCell(nn.Module):
    """A minimal ODE evolve + GRU update cell for irregular observations."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dim: int,
        dynamics_hidden_dim: int,
        activation: str = "gelu",
        ode_method: str = "rk4",
        ode_rtol: float = 1e-3,
        ode_atol: float = 1e-4,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.ode_method = ode_method
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol
        self.ode_func = HiddenStateODEFunc(
            hidden_dim,
            dynamics_hidden_dim=dynamics_hidden_dim,
            activation=activation,
        )
        self.observation_update = nn.GRUCell(embedding_dim, hidden_dim)

    def evolve_hidden_state(
        self,
        hidden_state: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> torch.Tensor:
        """Evolve one batch of hidden states forward by per-sample delta_t."""

        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must have shape [B, H].")
        if delta_t_s.shape != (hidden_state.shape[0],):
            raise ValueError("delta_t_s must have shape [B].")

        clamped_delta_t = torch.clamp(
            delta_t_s.to(dtype=hidden_state.dtype),
            min=0.0,
        )
        if self.ode_method == "euler":
            derivative = self.ode_func(hidden_state.new_zeros(()), hidden_state)
            return hidden_state + (clamped_delta_t.unsqueeze(-1) * derivative)

        if self.ode_method == "rk4":
            # torchdiffeq's fixed-grid RK4 uses one alternative 3/8-rule step
            # when only [0, delta_t] is supplied. The dynamics are autonomous,
            # so every row can use its own delta while sharing four batched
            # network evaluations instead of launching one odeint per sample.
            delta = clamped_delta_t.unsqueeze(-1)
            zero_time = hidden_state.new_zeros(())
            k1 = self.ode_func(zero_time, hidden_state)
            k2 = self.ode_func(
                zero_time,
                hidden_state + delta * k1 / 3.0,
            )
            k3 = self.ode_func(
                zero_time,
                hidden_state + delta * (k2 - k1 / 3.0),
            )
            k4 = self.ode_func(
                zero_time,
                hidden_state + delta * (k1 - k2 + k3),
            )
            return hidden_state + delta * (k1 + 3.0 * (k2 + k3) + k4) / 8.0

        evolved_rows: list[torch.Tensor] = []
        zero_time = hidden_state.new_zeros(())

        for sample_index in range(hidden_state.shape[0]):
            current_hidden = hidden_state[sample_index]
            current_delta_t = clamped_delta_t[sample_index]
            if torch.is_nonzero(current_delta_t <= 0):
                evolved_rows.append(current_hidden)
                continue

            integration_times = torch.stack((zero_time, current_delta_t))
            trajectory = odeint(
                self.ode_func,
                current_hidden.unsqueeze(0),
                integration_times,
                method=self.ode_method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
            )
            evolved_rows.append(trajectory[-1, 0])

        return torch.stack(evolved_rows, dim=0)

    def update_hidden_state(
        self,
        hidden_state: torch.Tensor,
        observation_embedding: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GRU-style observation updates only where the point is valid."""

        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must have shape [B, H].")
        if observation_embedding.shape != (hidden_state.shape[0], self.embedding_dim):
            raise ValueError("observation_embedding must have shape [B, E].")
        if observation_mask.shape != (hidden_state.shape[0],):
            raise ValueError("observation_mask must have shape [B].")

        updated_state = self.observation_update(
            observation_embedding,
            hidden_state,
        )
        return torch.where(
            observation_mask.unsqueeze(-1),
            updated_state,
            hidden_state,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        delta_t_s: torch.Tensor,
        observation_embedding: torch.Tensor,
        observation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply continuous evolution first, then observation updates."""

        evolved_state = self.evolve_hidden_state(hidden_state, delta_t_s)
        updated_state = self.update_hidden_state(evolved_state, observation_embedding, observation_mask)
        return evolved_state, updated_state
