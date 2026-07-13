"""Observed-state anchor and task-specific residual enhancement for Chronaris."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from chronaris.modeling.fusion_encoders.chronaris_continuous import (
    ChronarisContinuousEncoding,
    ChronarisContinuousFusionEncoder,
)
from chronaris.modeling.fusion_encoders.naive_sync import (
    build_causal_observed_features,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    RepresentationContractError,
)
from chronaris.representation.normalization import TrainOnlyPCAProjector


CORE_TASK_SLUGS = ("maneuver", "physiology_response")
RESIDUAL_MODES = ("direct_only", "continuous_only", "full")


@dataclass(frozen=True, slots=True)
class ObservedStateResidualConfig:
    output_dim: int = FUSION_OUTPUT_DIM
    hidden_dim: int = 128
    maximum_age_s: float = 30.0
    initial_gate: float = 0.05
    task_slugs: tuple[str, ...] = CORE_TASK_SLUGS

    def __post_init__(self) -> None:
        if self.output_dim != FUSION_OUTPUT_DIM:
            raise ValueError("observed-state residual output must remain 64-dimensional")
        if self.hidden_dim <= 0 or self.maximum_age_s <= 0:
            raise ValueError("observed-state residual dimensions and age must be positive")
        if not 0 < self.initial_gate < 1:
            raise ValueError("initial task gate must lie strictly between zero and one")
        if not self.task_slugs or len(set(self.task_slugs)) != len(self.task_slugs):
            raise ValueError("task slugs must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class ObservedStateResidualOutput:
    observed_sequence: torch.Tensor
    maneuver_observed_sequence: torch.Tensor
    continuous_sequence: torch.Tensor
    delta_sequence: torch.Tensor
    task_sequences: Mapping[str, torch.Tensor]
    task_gate_weights: Mapping[str, torch.Tensor]
    valid_mask: torch.Tensor
    continuous_encoding: ChronarisContinuousEncoding | None

    def sequence_for(self, task_slug: str, *, mode: str = "full") -> torch.Tensor:
        if mode not in RESIDUAL_MODES:
            raise ValueError(f"unsupported residual mode: {mode}")
        if mode == "direct_only":
            return (
                self.maneuver_observed_sequence
                if task_slug == "maneuver"
                else self.observed_sequence
            )
        if mode == "continuous_only":
            return self.continuous_sequence
        try:
            return self.task_sequences[task_slug]
        except KeyError as error:
            raise KeyError(f"unknown task slug: {task_slug}") from error


class ObservedStateResidual(nn.Module):
    """Keep causal direct observations and learn only a task-specific correction."""

    def __init__(
        self,
        *,
        continuous_encoder: ChronarisContinuousFusionEncoder,
        projector: TrainOnlyPCAProjector,
        maneuver_projector: TrainOnlyPCAProjector | None = None,
        config: ObservedStateResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ObservedStateResidualConfig()
        state = projector.state_dict()
        center = _as_tensor(state["center"])
        components = _as_tensor(state["components"])
        if center.ndim != 1 or components.ndim != 2:
            raise RepresentationContractError("observed-state PCA tensors are malformed")
        if components.shape[1] != center.shape[0]:
            raise RepresentationContractError("observed-state PCA input dimensions disagree")
        if components.shape[0] > self.config.output_dim:
            raise RepresentationContractError("observed-state PCA exceeds the 64-D contract")
        self.continuous_encoder = continuous_encoder
        self.register_buffer("projector_center", center.to(torch.float32))
        self.register_buffer("projector_components", components.to(torch.float32))
        self.has_maneuver_projector = maneuver_projector is not None
        maneuver_state = (maneuver_projector or projector).state_dict()
        self.register_buffer(
            "maneuver_projector_center",
            _as_tensor(maneuver_state["center"]).to(torch.float32),
        )
        self.register_buffer(
            "maneuver_projector_components",
            _as_tensor(maneuver_state["components"]).to(torch.float32),
        )
        self.projector_fit_sample_ids = tuple(str(value) for value in state["fit_sample_ids"])
        self.projector_fit_sample_hash = str(state["fit_sample_hash"])
        residual_input_dim = self.config.output_dim * 3
        self.delta_projection = nn.Sequential(
            nn.Linear(residual_input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )
        final = self.delta_projection[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        gate_logit = math.log(self.config.initial_gate / (1 - self.config.initial_gate))
        self.task_gate_logits = nn.ParameterDict(
            {
                task_slug: nn.Parameter(torch.full((self.config.output_dim,), gate_logit))
                for task_slug in self.config.task_slugs
            }
        )

    def forward(
        self,
        batch: DualStreamObservationBatch,
        *,
        compute_diagnostics: bool = False,
    ) -> ObservedStateResidualOutput:
        (
            observed_sequence,
            maneuver_observed_sequence,
            continuous_sequence,
            valid_mask,
            continuous,
        ) = self.encode_paths(
            batch,
            compute_diagnostics=compute_diagnostics,
        )
        return self.fuse_paths(
            observed_sequence=observed_sequence,
            maneuver_observed_sequence=maneuver_observed_sequence,
            continuous_sequence=continuous_sequence,
            valid_mask=valid_mask,
            continuous_encoding=continuous,
        )

    def encode_paths(
        self,
        batch: DualStreamObservationBatch,
        *,
        compute_diagnostics: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        ChronarisContinuousEncoding,
    ]:
        """Encode expensive direct and continuous paths before cheap task fusion."""

        observed_features, valid_mask = build_causal_observed_features(
            batch,
            maximum_age_s=self.config.maximum_age_s,
        )
        observed_sequence = self._project_observed(observed_features)
        if self.has_maneuver_projector:
            maneuver_features, maneuver_valid = build_causal_observed_features(
                batch,
                maximum_age_s=self.config.maximum_age_s,
                active_stream="vehicle",
            )
            maneuver_observed_sequence = self._project_maneuver_observed(
                maneuver_features
            )
        else:
            maneuver_observed_sequence = observed_sequence
            maneuver_valid = valid_mask
        continuous = self.continuous_encoder(
            batch,
            compute_diagnostics=compute_diagnostics,
        )
        continuous_sequence = continuous.sequence_embedding
        if continuous_sequence.shape != observed_sequence.shape:
            raise RepresentationContractError(
                "observed and continuous Chronaris paths must share [B,T,64]"
            )
        valid_float = valid_mask.unsqueeze(-1).to(observed_sequence.dtype)
        return (
            observed_sequence * valid_float,
            maneuver_observed_sequence
            * maneuver_valid.unsqueeze(-1).to(maneuver_observed_sequence.dtype),
            continuous_sequence * valid_float,
            valid_mask,
            continuous,
        )

    def fuse_paths(
        self,
        *,
        observed_sequence: torch.Tensor,
        maneuver_observed_sequence: torch.Tensor | None = None,
        continuous_sequence: torch.Tensor,
        valid_mask: torch.Tensor,
        continuous_encoding: ChronarisContinuousEncoding | None = None,
    ) -> ObservedStateResidualOutput:
        """Apply trainable task gates and residual projection to encoded paths."""

        if observed_sequence.shape != continuous_sequence.shape:
            raise RepresentationContractError("residual path tensors do not align")
        if maneuver_observed_sequence is None:
            maneuver_observed_sequence = observed_sequence
        if maneuver_observed_sequence.shape != observed_sequence.shape:
            raise RepresentationContractError("maneuver observation anchor does not align")
        if valid_mask.shape != observed_sequence.shape[:2]:
            raise RepresentationContractError("residual path mask does not align")
        residual_inputs = torch.cat(
            (
                observed_sequence,
                continuous_sequence,
                continuous_sequence - observed_sequence,
            ),
            dim=-1,
        )
        delta = self.delta_projection(residual_inputs)
        valid_float = valid_mask.unsqueeze(-1).to(delta.dtype)
        delta = delta * valid_float
        gate_weights = {
            task_slug: torch.sigmoid(logits)
            for task_slug, logits in self.task_gate_logits.items()
        }
        task_sequences = {
            task_slug: (
                (
                    maneuver_observed_sequence
                    if task_slug == "maneuver"
                    else observed_sequence
                )
                + gate.view(1, 1, -1) * delta
            )
            * valid_float
            for task_slug, gate in gate_weights.items()
        }
        return ObservedStateResidualOutput(
            observed_sequence=observed_sequence,
            maneuver_observed_sequence=maneuver_observed_sequence,
            continuous_sequence=continuous_sequence,
            delta_sequence=delta,
            task_sequences=task_sequences,
            task_gate_weights=gate_weights,
            valid_mask=valid_mask,
            continuous_encoding=continuous_encoding,
        )

    def _project_observed(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.projector_center.shape[0]:
            raise RepresentationContractError("observed-state PCA feature dimension changed")
        center = self.projector_center.to(device=features.device, dtype=features.dtype)
        components = self.projector_components.to(
            device=features.device,
            dtype=features.dtype,
        )
        projected = torch.matmul(features - center, components.transpose(0, 1))
        if projected.shape[-1] == self.config.output_dim:
            return projected
        padding = torch.zeros(
            (*projected.shape[:-1], self.config.output_dim - projected.shape[-1]),
            dtype=projected.dtype,
            device=projected.device,
        )
        return torch.cat((projected, padding), dim=-1)

    def _project_maneuver_observed(self, features: torch.Tensor) -> torch.Tensor:
        return _project_features(
            features,
            center=self.maneuver_projector_center,
            components=self.maneuver_projector_components,
            output_dim=self.config.output_dim,
        )

    def to_manifest(self) -> Mapping[str, object]:
        return {
            "format": "chronaris.observed_state_residual.v1",
            "config": asdict(self.config),
            "projector_fit_sample_ids": list(self.projector_fit_sample_ids),
            "projector_fit_sample_hash": self.projector_fit_sample_hash,
            "projector_input_dim": int(self.projector_center.shape[0]),
            "projector_component_count": int(self.projector_components.shape[0]),
            "maneuver_projector_input_dim": int(
                self.maneuver_projector_center.shape[0]
            ),
            "maneuver_projector_active_stream": (
                "vehicle" if self.has_maneuver_projector else "both"
            ),
            "delta_last_layer_zero_initialized": True,
            "reader_visible_model_name": "Chronaris",
            "label_used_for_encoder_training": True,
        }


def fit_observed_state_projector(
    batch: DualStreamObservationBatch,
    *,
    train_sample_ids: Sequence[str],
    held_out_sample_ids: Sequence[str],
    maximum_age_s: float = 30.0,
    absolute_query_keys: Mapping[str, Sequence[str]] | None = None,
    random_state: int = 17,
    active_stream: str = "both",
) -> TrainOnlyPCAProjector:
    """Fit the direct-observation projection on train-owned unique query rows."""

    features, _valid = build_causal_observed_features(
        batch,
        maximum_age_s=maximum_age_s,
        active_stream=active_stream,
    )
    rows = features.detach().cpu().numpy().reshape(-1, features.shape[-1])
    row_sample_ids = np.asarray(
        [sample_id for sample_id in batch.sample_ids for _ in range(features.shape[1])],
        dtype=str,
    )
    if absolute_query_keys is not None:
        keys = []
        for sample_id in batch.sample_ids:
            sample_keys = tuple(str(value) for value in absolute_query_keys[sample_id])
            if len(sample_keys) != features.shape[1]:
                raise ValueError("absolute query key count must match query points")
            keys.extend(sample_keys)
        keep = _first_unique_positions(keys)
        rows = rows[keep]
        row_sample_ids = row_sample_ids[keep]
    projector = TrainOnlyPCAProjector(
        output_dim=FUSION_OUTPUT_DIM,
        solver="randomized",
        random_state=random_state,
    )
    return projector.fit(
        rows,
        row_sample_ids=tuple(row_sample_ids.tolist()),
        train_sample_ids=train_sample_ids,
        held_out_sample_ids=held_out_sample_ids,
    )


def masked_sequence_mean(sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3 or valid_mask.shape != sequence.shape[:2]:
        raise ValueError("masked sequence mean expects [B,T,D] and [B,T]")
    if valid_mask.dtype != torch.bool:
        raise ValueError("masked sequence mean requires a boolean mask")
    counts = valid_mask.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise RepresentationContractError("cannot pool a sample with no valid query")
    return (
        sequence * valid_mask.unsqueeze(-1).to(sequence.dtype)
    ).sum(dim=1) / counts.to(sequence.dtype)


def _first_unique_positions(values: Sequence[str]) -> np.ndarray:
    seen = set()
    positions = []
    for index, value in enumerate(values):
        if value in seen:
            continue
        seen.add(value)
        positions.append(index)
    return np.asarray(positions, dtype=np.int64)


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value)


def _project_features(
    features: torch.Tensor,
    *,
    center: torch.Tensor,
    components: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    if features.shape[-1] != center.shape[0]:
        raise RepresentationContractError("PCA feature dimension changed")
    center = center.to(device=features.device, dtype=features.dtype)
    components = components.to(device=features.device, dtype=features.dtype)
    projected = torch.matmul(features - center, components.transpose(0, 1))
    if projected.shape[-1] == output_dim:
        return projected
    padding = torch.zeros(
        (*projected.shape[:-1], output_dim - projected.shape[-1]),
        dtype=projected.dtype,
        device=projected.device,
    )
    return torch.cat((projected, padding), dim=-1)
