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
    final_hidden_state: torch.Tensor
    reference_offsets_s: torch.Tensor | None = None
    reference_hidden_states: torch.Tensor | None = None
    reference_projected_states: torch.Tensor | None = None


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
        reconstruction_steps: list[torch.Tensor] = []
        projection_steps: list[torch.Tensor] = []

        for point_index in range(point_count):
            valid_mask = stream.mask[:, point_index]
            valid_mask_float = valid_mask.to(dtype=value_dtype).unsqueeze(-1)
            evolved_state, hidden_state = self.ode_rnn_cell(
                hidden_state,
                stream.delta_t_s[:, point_index],
                observation_embeddings[:, point_index],
                valid_mask,
            )
            evolved_hidden_steps.append(evolved_state * valid_mask_float)
            updated_hidden_steps.append(hidden_state * valid_mask_float)
            reconstruction_steps.append(self.decoder(hidden_state) * valid_mask_float)
            projection_steps.append(self.projection_head(hidden_state) * valid_mask_float)

        reference_hidden_states: torch.Tensor | None = None
        reference_projected_states: torch.Tensor | None = None
        if reference_offsets_s is not None:
            resolved_reference_offsets = _resolve_reference_offsets_s(
                reference_offsets_s,
                batch_size=batch_size,
                device=stream.values.device,
                dtype=value_dtype,
            )
            reference_hidden_states = self._sample_reference_hidden_states(
                stream,
                observation_embeddings,
                resolved_reference_offsets,
            )
            reference_projected_states = self.projection_head(reference_hidden_states)
        else:
            resolved_reference_offsets = None

        return StreamPrototypeOutput(
            feature_names=stream.feature_names,
            observation_embeddings=observation_embeddings,
            evolved_hidden_states=torch.stack(evolved_hidden_steps, dim=1),
            updated_hidden_states=torch.stack(updated_hidden_steps, dim=1),
            reconstructions=torch.stack(reconstruction_steps, dim=1),
            projected_states=torch.stack(projection_steps, dim=1),
            mask=stream.mask,
            feature_valid_mask=stream.feature_valid_mask,
            offsets_s=stream.offsets_s,
            delta_t_s=stream.delta_t_s,
            point_counts=stream.point_counts,
            final_hidden_state=hidden_state,
            reference_offsets_s=resolved_reference_offsets,
            reference_hidden_states=reference_hidden_states,
            reference_projected_states=reference_projected_states,
        )

    def _sample_reference_hidden_states(
        self,
        stream: TorchAlignmentStreamBatch,
        observation_embeddings: torch.Tensor,
        reference_offsets_s: torch.Tensor,
    ) -> torch.Tensor:
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

        reference_rows: list[torch.Tensor] = []
        hidden_dtype = stream.values.dtype
        device = stream.values.device

        for sample_index in range(stream.values.shape[0]):
            sample_hidden = stream.values.new_zeros((1, self.config.hidden_dim))
            sample_current_time = stream.values.new_zeros(())
            sample_reference_states: list[torch.Tensor] = []
            observation_count = int(stream.point_counts[sample_index].item())
            observation_index = 0

            while observation_index < observation_count and not bool(stream.mask[sample_index, observation_index]):
                observation_index += 1

            for reference_time in reference_offsets_s[sample_index]:
                resolved_reference_time = torch.clamp(reference_time.to(dtype=hidden_dtype), min=0.0)

                while observation_index < observation_count:
                    if not bool(stream.mask[sample_index, observation_index]):
                        observation_index += 1
                        continue

                    observation_time = stream.offsets_s[sample_index, observation_index].to(dtype=hidden_dtype)
                    if bool(observation_time > resolved_reference_time):
                        break

                    delta_to_observation = torch.clamp(observation_time - sample_current_time, min=0.0)
                    sample_hidden = self.ode_rnn_cell.evolve_hidden_state(
                        sample_hidden,
                        delta_to_observation.reshape(1),
                    )
                    sample_hidden = self.ode_rnn_cell.update_hidden_state(
                        sample_hidden,
                        observation_embeddings[sample_index, observation_index].unsqueeze(0),
                        torch.ones((1,), dtype=torch.bool, device=device),
                    )
                    sample_current_time = observation_time
                    observation_index += 1

                delta_to_reference = torch.clamp(resolved_reference_time - sample_current_time, min=0.0)
                sampled_hidden = self.ode_rnn_cell.evolve_hidden_state(
                    sample_hidden,
                    delta_to_reference.reshape(1),
                )[0]
                sample_reference_states.append(sampled_hidden)

            reference_rows.append(torch.stack(sample_reference_states, dim=0))

        return torch.stack(reference_rows, dim=0)


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
    ) -> DualStreamPrototypeOutput:
        """Run the deterministic dual-stream forward pass."""

        return DualStreamPrototypeOutput(
            sample_ids=batch.sample_ids,
            physiology=self.physiology_stream(
                batch.physiology,
                reference_offsets_s=reference_offsets_s,
            ),
            vehicle=self.vehicle_stream(
                batch.vehicle,
                reference_offsets_s=reference_offsets_s,
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
