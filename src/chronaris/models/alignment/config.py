"""Configuration objects for Stage E alignment prototypes."""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_ACTIVATIONS = {"gelu", "relu", "tanh"}
_ALLOWED_ODE_METHODS = {"euler", "midpoint", "rk4", "dopri5"}


@dataclass(frozen=True, slots=True)
class AlignmentPrototypeConfig:
    """Controls the minimal deterministic ODE-RNN prototype for Stage E."""

    hidden_dim: int = 32
    embedding_dim: int = 32
    encoder_hidden_dim: int = 64
    decoder_hidden_dim: int = 64
    dynamics_hidden_dim: int = 64
    projection_dim: int = 16
    activation: str = "gelu"
    ode_method: str = "rk4"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    use_feature_valid_mask: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "hidden_dim",
            "embedding_dim",
            "encoder_hidden_dim",
            "decoder_hidden_dim",
            "dynamics_hidden_dim",
            "projection_dim",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.activation not in _ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {self.activation!r}. Expected one of {sorted(_ALLOWED_ACTIVATIONS)!r}."
            )
        if self.ode_method not in _ALLOWED_ODE_METHODS:
            raise ValueError(
                f"Unsupported ode_method {self.ode_method!r}. Expected one of {sorted(_ALLOWED_ODE_METHODS)!r}."
            )
        if self.ode_rtol <= 0:
            raise ValueError("ode_rtol must be positive.")
        if self.ode_atol <= 0:
            raise ValueError("ode_atol must be positive.")
