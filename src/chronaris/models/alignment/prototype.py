"""Minimal deterministic dual-stream ODE-RNN prototype for Stage E."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chronaris.models.alignment.config import AlignmentPrototypeConfig
from chronaris.models.alignment.decoders import AlignmentProjectionHead, ObservationDecoder
from chronaris.models.alignment.encoders import ObservationEncoder
from chronaris.models.alignment.ode_cells import ODERNNCell
from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch


@dataclass(frozen=True, slots=True)
class StreamPathTrace:
    """Auditable execution counts for one irregular stream forward pass."""

    continuous_evolution_enabled: bool
    observation_update_count: int
    observation_positive_evolution_count: int
    reference_query_count: int
    reference_positive_evolution_count: int
    maximum_positive_delta_t_s: float


@dataclass(frozen=True, slots=True)
class StreamPrototypeOutput:
    """Forward outputs and intermediate states for one stream."""

    feature_names: tuple[str, ...]
    observation_embeddings: torch.Tensor
    evolved_hidden_states: torch.Tensor
    updated_hidden_states: torch.Tensor
    reconstructions: torch.Tensor
    projected_states: torch.Tensor
    mask: torch.Tensor
    feature_valid_mask: torch.Tensor
    offsets_s: torch.Tensor
    delta_t_s: torch.Tensor
    point_counts: torch.Tensor
    terminal_hidden_state: torch.Tensor
    reference_offsets_s: torch.Tensor | None = None
    reference_hidden_states: torch.Tensor | None = None
    reference_projected_states: torch.Tensor | None = None
    reference_valid_mask: torch.Tensor | None = None
    path_trace: StreamPathTrace | None = None


@dataclass(frozen=True, slots=True)
class DualStreamPrototypeOutput:
    """Forward outputs for the dual-stream deterministic prototype."""

    sample_ids: tuple[str, ...]
    physiology: StreamPrototypeOutput
    vehicle: StreamPrototypeOutput


class SingleStreamODERNNPrototype(nn.Module):
    """A minimal deterministic ODE-RNN for one irregularly sampled stream."""

    def __init__(
        self,
        feature_dim: int,
        *,
        config: AlignmentPrototypeConfig | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")

        self.feature_dim = feature_dim
        self.config = config or AlignmentPrototypeConfig()
        self.encoder = ObservationEncoder(
            feature_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.encoder_hidden_dim,
            activation=self.config.activation,
            use_feature_valid_mask=self.config.use_feature_valid_mask,
        )
        self.ode_rnn_cell = ODERNNCell(
            self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dynamics_hidden_dim=self.config.dynamics_hidden_dim,
            activation=self.config.activation,
            ode_method=self.config.ode_method,
            ode_rtol=self.config.ode_rtol,
            ode_atol=self.config.ode_atol,
            max_ode_step_s=self.config.max_ode_step_s,
        )
        self.decoder = ObservationDecoder(
            self.config.hidden_dim,
            output_dim=feature_dim,
            projection_hidden_dim=self.config.decoder_hidden_dim,
            activation=self.config.activation,
        )
        self.projection_head = AlignmentProjectionHead(
            self.config.hidden_dim,
            projection_dim=self.config.projection_dim,
            activation=self.config.activation,
        )

    def forward(
        self,
        stream: TorchAlignmentStreamBatch,
        *,
        reference_offsets_s: torch.Tensor | None = None,
        include_observation_diagnostics: bool = True,
    ) -> StreamPrototypeOutput:
        """Run the minimal deterministic ODE-RNN forward pass for one stream."""

        if stream.values.ndim != 3:
            raise ValueError("stream.values must have shape [B, T, F].")
        if stream.values.shape[-1] != self.feature_dim:
            raise ValueError("Input feature dimension does not match the prototype feature_dim.")

        batch_size, point_count, _ = stream.values.shape
        value_dtype = stream.values.dtype
        point_mask = stream.mask.to(dtype=value_dtype).unsqueeze(-1)
        observation_embeddings = self.encoder(stream.values, stream.feature_valid_mask) * point_mask

        hidden_state = stream.values.new_zeros((batch_size, self.config.hidden_dim))
        evolved_hidden_steps: list[torch.Tensor] = []
        updated_hidden_steps: list[torch.Tensor] = []

        for point_index in range(point_count):
            valid_mask = stream.mask[:, point_index]
            valid_mask_float = valid_mask.to(dtype=value_dtype).unsqueeze(-1)
            evolution_delta_t_s = (
                stream.delta_t_s[:, point_index]
                if self.config.enable_continuous_evolution
                else torch.zeros_like(stream.delta_t_s[:, point_index])
            )
            evolved_state, hidden_state = self.ode_rnn_cell(
                hidden_state,
                evolution_delta_t_s,
                observation_embeddings[:, point_index],
                valid_mask,
            )
            if include_observation_diagnostics:
                evolved_hidden_steps.append(evolved_state * valid_mask_float)
            updated_hidden_steps.append(hidden_state * valid_mask_float)

        reference_hidden_states: torch.Tensor | None = None
        reference_projected_states: torch.Tensor | None = None
        reference_valid_mask: torch.Tensor | None = None
        reference_positive_evolution_count = 0
        reference_maximum_delta_t_s = 0.0
        updated_hidden_tensor = torch.stack(updated_hidden_steps, dim=1)
        if reference_offsets_s is not None:
            resolved_reference_offsets = _resolve_reference_offsets_s(
                reference_offsets_s,
                batch_size=batch_size,
                device=stream.values.device,
                dtype=stream.offsets_s.dtype,
            )
            (
                reference_hidden_states,
                reference_valid_mask,
                reference_positive_evolution_count,
                reference_maximum_delta_t_s,
            ) = self._sample_reference_hidden_states(
                stream,
                updated_hidden_tensor,
                resolved_reference_offsets,
            )
            reference_projected_states = self.projection_head(reference_hidden_states)
        else:
            resolved_reference_offsets = None

        if include_observation_diagnostics:
            evolved_hidden_tensor = torch.stack(evolved_hidden_steps, dim=1)
            reconstruction_tensor = self.decoder(updated_hidden_tensor) * point_mask
            projection_tensor = self.projection_head(updated_hidden_tensor) * point_mask
        else:
            evolved_hidden_tensor = stream.values.new_zeros(
                (batch_size, point_count, self.config.hidden_dim)
            )
            reconstruction_tensor = stream.values.new_zeros(
                (batch_size, point_count, self.feature_dim)
            )
            projection_tensor = stream.values.new_zeros(
                (batch_size, point_count, self.config.projection_dim)
            )

        return StreamPrototypeOutput(
            feature_names=stream.feature_names,
            observation_embeddings=observation_embeddings,
            evolved_hidden_states=evolved_hidden_tensor,
            updated_hidden_states=updated_hidden_tensor,
            reconstructions=reconstruction_tensor,
            projected_states=projection_tensor,
            mask=stream.mask,
            feature_valid_mask=stream.feature_valid_mask,
            offsets_s=stream.offsets_s,
            delta_t_s=stream.delta_t_s,
            point_counts=stream.point_counts,
            terminal_hidden_state=hidden_state,
            reference_offsets_s=resolved_reference_offsets,
            reference_hidden_states=reference_hidden_states,
            reference_projected_states=reference_projected_states,
            reference_valid_mask=reference_valid_mask,
            path_trace=StreamPathTrace(
                continuous_evolution_enabled=self.config.enable_continuous_evolution,
                observation_update_count=int(stream.mask.sum().item()),
                observation_positive_evolution_count=(
                    int(((stream.delta_t_s > 0) & stream.mask).sum().item())
                    if self.config.enable_continuous_evolution
                    else 0
                ),
                reference_query_count=(
                    int(resolved_reference_offsets.numel())
                    if resolved_reference_offsets is not None
                    else 0
                ),
                reference_positive_evolution_count=reference_positive_evolution_count,
                maximum_positive_delta_t_s=max(
                    float(
                        stream.delta_t_s[stream.mask].max().item()
                        if bool(stream.mask.any())
                        else 0.0
                    )
                    if self.config.enable_continuous_evolution
                    else 0.0,
                    reference_maximum_delta_t_s,
                ),
            ),
        )

    def _sample_reference_hidden_states(
        self,
        stream: TorchAlignmentStreamBatch,
        updated_hidden_states: torch.Tensor,
        reference_offsets_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, float]:
        """Replay one stream and sample hidden states on a shared reference grid."""

        if reference_offsets_s.ndim != 2:
            raise ValueError("reference_offsets_s must have shape [B, R].")
        if reference_offsets_s.shape[0] != stream.values.shape[0]:
            raise ValueError("reference_offsets_s batch dimension must match the stream batch size.")
        if reference_offsets_s.shape[1] == 0:
            raise ValueError("reference_offsets_s must include at least one reference point.")
        if reference_offsets_s.shape[1] > 1 and not bool(
            torch.all(reference_offsets_s[:, 1:] >= reference_offsets_s[:, :-1])
        ):
            raise ValueError("reference_offsets_s must be monotonically non-decreasing within each sample.")

        batch_size, point_count = stream.mask.shape
        reference_count = reference_offsets_s.shape[1]
        observation_indices = torch.arange(
            point_count,
            device=stream.values.device,
        ).view(1, point_count, 1)
        eligible = stream.mask.unsqueeze(-1) & (
            stream.offsets_s.unsqueeze(-1) <= reference_offsets_s.unsqueeze(1)
        )
        source_indices = torch.where(
            eligible,
            observation_indices,
            torch.full_like(observation_indices, -1),
        ).amax(dim=1)
        reference_valid_mask = source_indices >= 0
        safe_indices = source_indices.clamp_min(0)
        gathered_states = torch.gather(
            updated_hidden_states,
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_dim),
        )
        source_offsets = torch.gather(stream.offsets_s, 1, safe_indices)
        delta_to_reference = torch.clamp(
            reference_offsets_s - source_offsets,
            min=0.0,
        )
        if self.config.enable_continuous_evolution:
            sampled_states = self.ode_rnn_cell.evolve_hidden_state(
                gathered_states.reshape(batch_size * reference_count, -1),
                delta_to_reference.reshape(batch_size * reference_count),
            ).reshape(batch_size, reference_count, -1)
            observation_positive = (stream.delta_t_s > 0) & stream.mask
            reference_positive = (delta_to_reference > 0) & reference_valid_mask
            positive_evolution_count = int(observation_positive.sum().item()) + int(
                reference_positive.sum().item()
            )
            maximum_positive_delta_t_s = max(
                float(
                    stream.delta_t_s[stream.mask].max().item()
                    if bool(stream.mask.any())
                    else 0.0
                ),
                float(
                    delta_to_reference[reference_valid_mask].max().item()
                    if bool(reference_valid_mask.any())
                    else 0.0
                ),
            )
        else:
            sampled_states = gathered_states
            positive_evolution_count = 0
            maximum_positive_delta_t_s = 0.0
        sampled_states = torch.where(
            reference_valid_mask.unsqueeze(-1),
            sampled_states,
            torch.zeros_like(sampled_states),
        )
        return (
            sampled_states,
            reference_valid_mask,
            positive_evolution_count,
            maximum_positive_delta_t_s,
        )


class DualStreamODERNNPrototype(nn.Module):
    """A minimal deterministic dual-stream ODE-RNN prototype."""

    def __init__(
        self,
        physiology_feature_dim: int,
        vehicle_feature_dim: int,
        *,
        config: AlignmentPrototypeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AlignmentPrototypeConfig()
        self.physiology_stream = SingleStreamODERNNPrototype(
            physiology_feature_dim,
            config=self.config,
        )
        self.vehicle_stream = SingleStreamODERNNPrototype(
            vehicle_feature_dim,
            config=self.config,
        )

    @classmethod
    def from_torch_alignment_batch(
        cls,
        batch: TorchAlignmentBatch,
        *,
        config: AlignmentPrototypeConfig | None = None,
    ) -> "DualStreamODERNNPrototype":
        """Build one prototype instance using feature dimensions from a torch batch."""

        return cls(
            len(batch.physiology.feature_names),
            len(batch.vehicle.feature_names),
            config=config,
        )

    def forward(
        self,
        batch: TorchAlignmentBatch,
        *,
        reference_offsets_s: torch.Tensor | None = None,
        include_observation_diagnostics: bool = True,
    ) -> DualStreamPrototypeOutput:
        """Run the deterministic dual-stream forward pass."""

        return DualStreamPrototypeOutput(
            sample_ids=batch.sample_ids,
            physiology=self.physiology_stream(
                batch.physiology,
                reference_offsets_s=reference_offsets_s,
                include_observation_diagnostics=include_observation_diagnostics,
            ),
            vehicle=self.vehicle_stream(
                batch.vehicle,
                reference_offsets_s=reference_offsets_s,
                include_observation_diagnostics=include_observation_diagnostics,
            ),
        )


def _resolve_reference_offsets_s(
    reference_offsets_s: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize reference offsets into shape [B, R] on the target device."""

    if reference_offsets_s.ndim == 1:
        return reference_offsets_s.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1)
    if reference_offsets_s.ndim == 2:
        if reference_offsets_s.shape[0] != batch_size:
            raise ValueError("reference_offsets_s batch dimension must match the stream batch size.")
        return reference_offsets_s.to(device=device, dtype=dtype)
    raise ValueError("reference_offsets_s must have shape [R] or [B, R].")
