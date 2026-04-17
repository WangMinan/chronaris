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

    def forward(self, stream: TorchAlignmentStreamBatch) -> StreamPrototypeOutput:
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

    def forward(self, batch: TorchAlignmentBatch) -> DualStreamPrototypeOutput:
        """Run the deterministic dual-stream forward pass."""

        return DualStreamPrototypeOutput(
            sample_ids=batch.sample_ids,
            physiology=self.physiology_stream(batch.physiology),
            vehicle=self.vehicle_stream(batch.vehicle),
        )
