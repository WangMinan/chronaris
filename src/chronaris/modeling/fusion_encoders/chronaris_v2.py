"""Versioned Chronaris v2 semantic-group continuous fusion encoder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.alignment_bridge import (
    build_alignment_batch_from_observations,
)
from chronaris.modeling.fusion_encoders.causal_query import causal_query_stream
from chronaris.modeling.fusion_encoders.chronaris_physics import (
    ChronarisPhysicsAudit,
    build_chronaris_physics_audit,
    build_skipped_chronaris_physics_audit,
)
from chronaris.modeling.fusion_encoders.learned_causal import (
    LearnedRelativeCausalFusion,
    LearnedRelativeCausalFusionConfig,
    LearnedRelativeCausalFusionInput,
    LearnedRelativeCausalFusionOutput,
)
from chronaris.modeling.fusion_encoders.multiscale_causal import (
    MultiScaleCausalFusionConfig,
    MultiScaleCausalFusionInput,
    MultiScaleCausalFusionOutput,
    MultiScaleCausalLagFusion,
)
from chronaris.modeling.fusion_encoders.semantic_groups import (
    SemanticGroupedObservationEncoder,
    VehicleSemanticGroupMap,
    build_vehicle_semantic_group_map,
)
from chronaris.models.alignment.config import AlignmentPrototypeConfig
from chronaris.models.alignment.prototype import (
    DualStreamPrototypeOutput,
    SingleStreamODERNNPrototype,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    RepresentationContractError,
)
from chronaris.representation.normalization import TrainOnlyRobustNormalizer


V2_SUBSPACE_SLICES = {
    "vehicle_private": (0, 24),
    "physiology_private": (24, 40),
    "causal_shared": (40, 64),
}


@dataclass(frozen=True, slots=True)
class ChronarisV2EncoderConfig:
    physiology_feature_names: tuple[str, ...]
    vehicle_feature_names: tuple[str, ...]
    field_labels: tuple[tuple[str, str], ...] = ()
    architecture_version: str = "v2"
    internal_hidden_dim: int = 64
    physiology_hidden_dim: int = 32
    vehicle_hidden_dim: int = 64
    num_heads: int = 4
    lag_mode: str = "fixed_five"
    ode_method: str = "euler"
    dropout: float = 0.1
    physics_enabled: bool = True
    physics_weight: float = 0.1
    physics_huber_delta: float = 1.0
    learned_causal_attention: bool = True
    private_shared_subspaces: bool = True
    corrected_physics: bool = True
    physiology_residual_mode: str = "learned"

    def __post_init__(self) -> None:
        if self.architecture_version != "v2":
            raise ValueError("ChronarisV2EncoderConfig requires architecture_version=v2")
        if not self.physiology_feature_names or not self.vehicle_feature_names:
            raise ValueError("Chronaris v2 feature names must be non-empty")
        if len(set(self.physiology_feature_names)) != len(self.physiology_feature_names):
            raise ValueError("Chronaris v2 physiology feature names must be unique")
        if len(set(self.vehicle_feature_names)) != len(self.vehicle_feature_names):
            raise ValueError("Chronaris v2 vehicle feature names must be unique")
        if min(
            self.internal_hidden_dim,
            self.physiology_hidden_dim,
            self.vehicle_hidden_dim,
            self.num_heads,
        ) <= 0:
            raise ValueError("Chronaris v2 dimensions must be positive")
        if self.internal_hidden_dim % self.num_heads:
            raise ValueError("Chronaris v2 hidden dimension must divide num_heads")
        if self.lag_mode not in {"fixed_five", "continuous_basis"}:
            raise ValueError("Chronaris v2 lag mode is invalid")
        if self.ode_method not in {"euler", "rk4"}:
            raise ValueError("Chronaris v2 screen only supports euler or rk4")
        if not 0 <= self.dropout < 1:
            raise ValueError("Chronaris v2 dropout is invalid")
        if self.physiology_residual_mode not in {
            "learned",
            "direct_causal_query",
        }:
            raise ValueError("Chronaris v2 physiology residual mode is invalid")
        if (
            self.physiology_residual_mode == "direct_causal_query"
            and len(self.physiology_feature_names) > 16
        ):
            raise ValueError(
                "direct physiology residual requires at most 16 observed features"
            )
        if self.physics_weight < 0 or self.physics_huber_delta <= 0:
            raise ValueError("Chronaris v2 physics configuration is invalid")
        if not self.learned_causal_attention and (
            self.physiology_hidden_dim != self.vehicle_hidden_dim
        ):
            raise ValueError("legacy causal fusion requires equal stream hidden dimensions")

    @property
    def field_label_mapping(self) -> Mapping[str, str]:
        return dict(self.field_labels)

    def to_checkpoint_dict(self) -> Mapping[str, object]:
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "ChronarisV2EncoderConfig":
        resolved = dict(payload)
        resolved["physiology_feature_names"] = tuple(
            str(value) for value in resolved["physiology_feature_names"]
        )
        resolved["vehicle_feature_names"] = tuple(
            str(value) for value in resolved["vehicle_feature_names"]
        )
        resolved["field_labels"] = tuple(
            (str(key), str(value)) for key, value in resolved.get("field_labels", ())
        )
        return cls(**resolved)


@dataclass(frozen=True, slots=True)
class ChronarisV2Encoding:
    sequence_embedding: torch.Tensor
    modality_available_mask: torch.Tensor
    vehicle_private: torch.Tensor
    physiology_private: torch.Tensor
    causal_shared: torch.Tensor
    alignment_output: DualStreamPrototypeOutput
    fusion_output: LearnedRelativeCausalFusionOutput | MultiScaleCausalFusionOutput
    physics_audit: ChronarisPhysicsAudit
    subspace_slices: Mapping[str, tuple[int, int]]


class ChronarisV2FusionEncoder(nn.Module):
    """Raw asynchronous streams -> stream-specific ODE-RNN -> v2 fusion."""

    def __init__(self, config: ChronarisV2EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "physiology_denormalize_center",
            torch.zeros(len(config.physiology_feature_names)),
        )
        self.register_buffer(
            "physiology_denormalize_scale",
            torch.ones(len(config.physiology_feature_names)),
        )
        self.register_buffer(
            "vehicle_denormalize_center",
            torch.zeros(len(config.vehicle_feature_names)),
        )
        self.register_buffer(
            "vehicle_denormalize_scale",
            torch.ones(len(config.vehicle_feature_names)),
        )
        self.register_buffer(
            "physics_normalization_attached",
            torch.tensor(False, dtype=torch.bool),
        )
        self.semantic_group_map: VehicleSemanticGroupMap = (
            build_vehicle_semantic_group_map(
                config.vehicle_feature_names,
                field_labels=config.field_label_mapping,
            )
        )
        physiology_config = AlignmentPrototypeConfig(
            hidden_dim=config.physiology_hidden_dim,
            embedding_dim=config.physiology_hidden_dim,
            encoder_hidden_dim=config.internal_hidden_dim,
            decoder_hidden_dim=config.internal_hidden_dim,
            dynamics_hidden_dim=config.physiology_hidden_dim,
            projection_dim=config.physiology_hidden_dim,
            ode_method=config.ode_method,
        )
        vehicle_config = AlignmentPrototypeConfig(
            hidden_dim=config.vehicle_hidden_dim,
            embedding_dim=config.vehicle_hidden_dim,
            encoder_hidden_dim=config.internal_hidden_dim,
            decoder_hidden_dim=config.internal_hidden_dim,
            dynamics_hidden_dim=config.vehicle_hidden_dim,
            projection_dim=config.vehicle_hidden_dim,
            ode_method=config.ode_method,
        )
        vehicle_observation_encoder = SemanticGroupedObservationEncoder(
            self.semantic_group_map,
            embedding_dim=config.vehicle_hidden_dim,
            group_hidden_dim=max(16, config.internal_hidden_dim // 2),
            dropout=config.dropout,
        )
        self.physiology_stream = SingleStreamODERNNPrototype(
            len(config.physiology_feature_names),
            config=physiology_config,
        )
        self.vehicle_stream = SingleStreamODERNNPrototype(
            len(config.vehicle_feature_names),
            config=vehicle_config,
            observation_encoder=vehicle_observation_encoder,
        )
        self.vehicle_private_projection = _subspace_projection(
            config.vehicle_hidden_dim,
            24,
        )
        self.physiology_private_projection = _subspace_projection(
            config.physiology_hidden_dim,
            16,
        )
        self.vehicle_private_missing = nn.Parameter(torch.zeros(24))
        self.physiology_private_missing = nn.Parameter(torch.zeros(16))
        if config.learned_causal_attention:
            self.causal_fusion = LearnedRelativeCausalFusion(
                LearnedRelativeCausalFusionConfig(
                    physiology_dim=config.physiology_hidden_dim,
                    vehicle_dim=config.vehicle_hidden_dim,
                    attention_dim=config.internal_hidden_dim,
                    output_dim=24,
                    num_heads=config.num_heads,
                    lag_mode=config.lag_mode,
                    dropout=config.dropout,
                )
            )
        else:
            self.causal_fusion = MultiScaleCausalLagFusion(
                MultiScaleCausalFusionConfig(
                    hidden_dim=config.vehicle_hidden_dim,
                    output_dim=24,
                )
            )
        self.mixed_output_projection = (
            nn.Sequential(
                nn.LayerNorm(
                    config.vehicle_hidden_dim
                    + config.physiology_hidden_dim
                    + 24
                ),
                nn.Linear(
                    config.vehicle_hidden_dim
                    + config.physiology_hidden_dim
                    + 24,
                    FUSION_OUTPUT_DIM,
                ),
            )
            if not config.private_shared_subspaces
            else None
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        batch: DualStreamObservationBatch,
        *,
        compute_diagnostics: bool = True,
    ) -> ChronarisV2Encoding:
        alignment_batch = build_alignment_batch_from_observations(
            batch,
            physiology_feature_names=self.config.physiology_feature_names,
            vehicle_feature_names=self.config.vehicle_feature_names,
        )
        query_times = batch.query_timestamps_s.to(
            device=batch.physiology_values.device,
            dtype=batch.physiology_values.dtype,
        )
        physiology_output = self.physiology_stream(
            alignment_batch.physiology,
            reference_offsets_s=query_times,
            include_observation_diagnostics=compute_diagnostics,
        )
        vehicle_output = self.vehicle_stream(
            alignment_batch.vehicle,
            reference_offsets_s=query_times,
            include_observation_diagnostics=compute_diagnostics,
        )
        alignment = DualStreamPrototypeOutput(
            sample_ids=batch.sample_ids,
            physiology=physiology_output,
            vehicle=vehicle_output,
        )
        physiology_states = physiology_output.reference_hidden_states
        vehicle_states = vehicle_output.reference_hidden_states
        physiology_valid = physiology_output.reference_valid_mask
        vehicle_valid = vehicle_output.reference_valid_mask
        if (
            physiology_states is None
            or vehicle_states is None
            or physiology_valid is None
            or vehicle_valid is None
        ):
            raise RepresentationContractError(
                "Chronaris v2 continuous streams did not produce query states"
            )
        vehicle_private = self.vehicle_private_projection(vehicle_states)
        vehicle_private = torch.where(
            vehicle_valid.unsqueeze(-1),
            vehicle_private,
            self.vehicle_private_missing.view(1, 1, -1),
        )
        physiology_private = self.physiology_private_projection(physiology_states)
        if self.config.physiology_residual_mode == "direct_causal_query":
            queried_physiology = causal_query_stream(
                batch,
                stream_name="physiology",
            )
            direct_count = queried_physiology.values.shape[-1]
            direct_values = torch.where(
                queried_physiology.feature_mask,
                queried_physiology.values,
                torch.zeros_like(queried_physiology.values),
            )
            physiology_private = torch.cat(
                (
                    direct_values,
                    physiology_private[..., direct_count:],
                ),
                dim=-1,
            )
        physiology_private = torch.where(
            physiology_valid.unsqueeze(-1),
            physiology_private,
            self.physiology_private_missing.view(1, 1, -1),
        )
        if self.config.learned_causal_attention:
            fusion = self.causal_fusion(
                LearnedRelativeCausalFusionInput(
                    physiology_states=physiology_states,
                    vehicle_states=vehicle_states,
                    physiology_valid_mask=physiology_valid,
                    vehicle_valid_mask=vehicle_valid,
                    query_timestamps_s=query_times,
                )
            )
            causal_shared = fusion.shared_embedding
        else:
            fusion = self.causal_fusion(
                MultiScaleCausalFusionInput(
                    physiology_states=physiology_states,
                    vehicle_states=vehicle_states,
                    physiology_valid_mask=physiology_valid,
                    vehicle_valid_mask=vehicle_valid,
                    query_timestamps_s=query_times,
                )
            )
            causal_shared = fusion.sequence_embedding
        sequence = (
            torch.cat(
                (vehicle_private, physiology_private, causal_shared),
                dim=-1,
            )
            if self.config.private_shared_subspaces
            else self._mixed_output(
                vehicle_states,
                physiology_states,
                causal_shared,
            )
        )
        if sequence.shape[-1] != FUSION_OUTPUT_DIM:
            raise RepresentationContractError("Chronaris v2 violated 64-D contract")
        sequence = self.output_dropout(sequence)
        sequence = sequence * fusion.modality_available_mask.unsqueeze(-1).to(
            sequence.dtype
        )
        physics = (
            build_chronaris_physics_audit(
                alignment,
                alignment_batch,
                field_labels=self.config.field_label_mapping,
                enabled=self.config.physics_enabled,
                weight=self.config.physics_weight,
                huber_delta=self.config.physics_huber_delta,
                vehicle_denormalize_center=self.vehicle_denormalize_center,
                vehicle_denormalize_scale=self.vehicle_denormalize_scale,
                physiology_denormalize_center=self.physiology_denormalize_center,
                physiology_denormalize_scale=self.physiology_denormalize_scale,
                strict_axis_pairs=True,
            )
            if compute_diagnostics and bool(self.physics_normalization_attached)
            else build_skipped_chronaris_physics_audit(
                sequence,
                reason=(
                    "physics_normalization_unavailable"
                    if compute_diagnostics
                    else "task_independent_pretext_fast_path"
                ),
            )
        )
        return ChronarisV2Encoding(
            sequence_embedding=sequence,
            modality_available_mask=fusion.modality_available_mask,
            vehicle_private=vehicle_private,
            physiology_private=physiology_private,
            causal_shared=causal_shared,
            alignment_output=alignment,
            fusion_output=fusion,
            physics_audit=physics,
            subspace_slices=V2_SUBSPACE_SLICES,
        )

    def _mixed_output(self, vehicle_states, physiology_states, causal_shared):
        if self.mixed_output_projection is None:
            raise RepresentationContractError("mixed v2 output projection is unavailable")
        return self.mixed_output_projection(
            torch.cat((vehicle_states, physiology_states, causal_shared), dim=-1)
        )

    def effective_mechanisms(self) -> Mapping[str, object]:
        return {
            "architecture_version": "v2",
            "semantic_group_vehicle_encoder": True,
            "learned_multihead_qkv": self.config.learned_causal_attention,
            "relative_time_bias": self.config.learned_causal_attention,
            "lag_mode": self.config.lag_mode,
            "private_shared_subspaces": self.config.private_shared_subspaces,
            "subspace_slices": (
                dict(V2_SUBSPACE_SLICES)
                if self.config.private_shared_subspaces
                else "diagnostic_heads_only"
            ),
            "semantic_group_mapping_sha256": self.semantic_group_map.mapping_sha256,
            "physics_enabled": self.config.physics_enabled,
            "physics_requires_inverse_normalization": self.config.corrected_physics,
            "strict_axis_pairs": self.config.corrected_physics,
        }

    @property
    def physics_mapping_sha256(self) -> str:
        return _physics_mapping_sha256(self.config)

    def lag_config_manifest(self) -> Mapping[str, object]:
        return {
            "mode": self.config.lag_mode,
            "ranges_s": [
                list(bounds) for bounds in self.causal_fusion.config.lag_ranges_s
            ],
            "learned_causal_attention": self.config.learned_causal_attention,
        }

    def attach_normalizer(self, normalizer: TrainOnlyRobustNormalizer) -> None:
        physiology, vehicle = normalizer._require_fitted()
        if physiology.center.shape != self.physiology_denormalize_center.shape:
            raise RepresentationContractError(
                "Chronaris v2 physiology normalizer dimension changed"
            )
        if vehicle.center.shape != self.vehicle_denormalize_center.shape:
            raise RepresentationContractError(
                "Chronaris v2 vehicle normalizer dimension changed"
            )
        with torch.no_grad():
            self.physiology_denormalize_center.copy_(
                physiology.center.to(self.physiology_denormalize_center)
            )
            self.physiology_denormalize_scale.copy_(
                physiology.scale.to(self.physiology_denormalize_scale)
            )
            self.vehicle_denormalize_center.copy_(
                vehicle.center.to(self.vehicle_denormalize_center)
            )
            self.vehicle_denormalize_scale.copy_(
                vehicle.scale.to(self.vehicle_denormalize_scale)
            )
            self.physics_normalization_attached.fill_(True)


def save_chronaris_v2_checkpoint(
    path: str | Path,
    *,
    backbone: ChronarisV2FusionEncoder,
    normalizer: TrainOnlyRobustNormalizer,
    seed: int,
) -> Path:
    backbone.attach_normalizer(normalizer)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    payload = {
        "format": "chronaris.continuous_fusion_encoder.v2",
        "architecture_version": "v2",
        "config": dict(backbone.config.to_checkpoint_dict()),
        "model_state_dict": backbone.state_dict(),
        "normalizer": dict(normalizer.to_manifest()),
        "seed": int(seed),
        "subspace_slices": {
            name: list(bounds) for name, bounds in V2_SUBSPACE_SLICES.items()
        },
        "semantic_group_mapping_sha256": (
            backbone.semantic_group_map.mapping_sha256
        ),
        "physics_mapping_sha256": backbone.physics_mapping_sha256,
        "lag_mode": backbone.config.lag_mode,
        "lag_ranges_s": [
            list(bounds) for bounds in backbone.causal_fusion.config.lag_ranges_s
        ],
        "sequence_source": "task_head_free_continuous_fusion_v2",
        "label_used_for_encoder_training": False,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def load_chronaris_v2_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    ChronarisV2FusionEncoder,
    TrainOnlyRobustNormalizer,
    Mapping[str, object],
]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.continuous_fusion_encoder.v2":
        raise RepresentationContractError("unsupported Chronaris v2 checkpoint format")
    if payload.get("architecture_version") != "v2":
        raise RepresentationContractError("Chronaris v2 checkpoint version mismatch")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("Chronaris v2 checkpoint used downstream labels")
    if payload.get("sequence_source") != "task_head_free_continuous_fusion_v2":
        raise RepresentationContractError("Chronaris v2 checkpoint is task-head dependent")
    config = ChronarisV2EncoderConfig.from_checkpoint_dict(payload["config"])
    backbone = ChronarisV2FusionEncoder(config).to(device)
    if payload.get("semantic_group_mapping_sha256") != (
        backbone.semantic_group_map.mapping_sha256
    ):
        raise RepresentationContractError("Chronaris v2 semantic group mapping changed")
    if payload.get("physics_mapping_sha256") != backbone.physics_mapping_sha256:
        raise RepresentationContractError("Chronaris v2 physics mapping changed")
    expected_slices = {
        name: list(bounds) for name, bounds in V2_SUBSPACE_SLICES.items()
    }
    if payload.get("subspace_slices") != expected_slices:
        raise RepresentationContractError("Chronaris v2 subspace slices changed")
    backbone.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    metadata = {
        key: payload[key]
        for key in (
            "format",
            "architecture_version",
            "seed",
            "subspace_slices",
            "semantic_group_mapping_sha256",
            "physics_mapping_sha256",
            "lag_mode",
            "lag_ranges_s",
            "sequence_source",
            "label_used_for_encoder_training",
        )
    }
    return backbone, normalizer, metadata


def _subspace_projection(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim),
        nn.LayerNorm(output_dim),
    )


def _physics_mapping_sha256(config: ChronarisV2EncoderConfig) -> str:
    payload = json.dumps(
        {
            "vehicle_feature_names": list(config.vehicle_feature_names),
            "field_labels": [list(value) for value in config.field_labels],
            "physics_enabled": config.physics_enabled,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
