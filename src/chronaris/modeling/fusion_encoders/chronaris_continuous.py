"""Task-independent Chronaris continuous dual-stream production encoder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from chronaris.modeling.fusion_encoders.alignment_bridge import (
    build_alignment_batch_from_observations,
)
from chronaris.modeling.fusion_encoders.chronaris_physics import (
    ChronarisPhysicsAudit,
    build_chronaris_physics_audit,
    build_skipped_chronaris_physics_audit,
    physics_audit_to_rows,
)
from chronaris.modeling.fusion_encoders.multiscale_causal import (
    DEFAULT_LAG_RANGES_S,
    MultiScaleCausalFusionConfig,
    MultiScaleCausalFusionInput,
    MultiScaleCausalFusionOutput,
    MultiScaleCausalLagFusion,
)
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.models.alignment.config import AlignmentPrototypeConfig
from chronaris.models.alignment.prototype import (
    DualStreamODERNNPrototype,
    DualStreamPrototypeOutput,
)
from chronaris.representation.contracts import (
    FUSION_OUTPUT_DIM,
    DualStreamObservationBatch,
    FusionStreamBatch,
    RepresentationContractError,
)
from chronaris.representation.normalization import TrainOnlyRobustNormalizer


CHRONARIS_VARIANTS = (
    "full",
    "no_continuous_evolution",
    "no_physics",
    "no_causal_mask",
    "single_scale_lag",
)
ABLATION_TARGET_FIELDS = {
    "no_continuous_evolution": frozenset({"continuous_evolution_enabled"}),
    "no_physics": frozenset({"physics_enabled"}),
    "no_causal_mask": frozenset({"causal_mask_enabled"}),
    "single_scale_lag": frozenset({"lag_ranges_s", "scale_gate_enabled"}),
}


@dataclass(frozen=True, slots=True)
class ChronarisContinuousEncoderConfig:
    physiology_feature_names: tuple[str, ...]
    vehicle_feature_names: tuple[str, ...]
    field_labels: tuple[tuple[str, str], ...] = ()
    variant: str = "full"
    hidden_dim: int = FUSION_OUTPUT_DIM
    embedding_dim: int = FUSION_OUTPUT_DIM
    encoder_hidden_dim: int = FUSION_OUTPUT_DIM
    decoder_hidden_dim: int = FUSION_OUTPUT_DIM
    dynamics_hidden_dim: int = FUSION_OUTPUT_DIM
    ode_method: str = "euler"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    physics_weight: float = 0.1
    physics_huber_delta: float = 1.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not self.physiology_feature_names or not self.vehicle_feature_names:
            raise ValueError("Chronaris feature names must be non-empty")
        if len(set(self.physiology_feature_names)) != len(
            self.physiology_feature_names
        ):
            raise ValueError("physiology feature names must be unique")
        if len(set(self.vehicle_feature_names)) != len(self.vehicle_feature_names):
            raise ValueError("vehicle feature names must be unique")
        if self.variant not in CHRONARIS_VARIANTS:
            raise ValueError(f"unsupported Chronaris variant: {self.variant}")
        dimensions = (
            self.hidden_dim,
            self.embedding_dim,
            self.encoder_hidden_dim,
            self.decoder_hidden_dim,
            self.dynamics_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Chronaris dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("Chronaris dropout is invalid")
        if self.physics_weight < 0 or self.physics_huber_delta <= 0:
            raise ValueError("Chronaris physics configuration is invalid")
        labels = dict(self.field_labels)
        if len(labels) != len(self.field_labels):
            raise ValueError("Chronaris field label keys must be unique")

    @property
    def continuous_evolution_enabled(self) -> bool:
        return self.variant != "no_continuous_evolution"

    @property
    def physics_enabled(self) -> bool:
        return self.variant != "no_physics"

    @property
    def causal_mask_enabled(self) -> bool:
        return self.variant != "no_causal_mask"

    @property
    def lag_ranges_s(self) -> tuple[tuple[float, float], ...]:
        return (
            ((0.0, 30.0),)
            if self.variant == "single_scale_lag"
            else DEFAULT_LAG_RANGES_S
        )

    @property
    def scale_gate_enabled(self) -> bool:
        return self.variant != "single_scale_lag"

    @property
    def field_label_mapping(self) -> Mapping[str, str]:
        return dict(self.field_labels)

    def alignment_config(self) -> AlignmentPrototypeConfig:
        return AlignmentPrototypeConfig(
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            encoder_hidden_dim=self.encoder_hidden_dim,
            decoder_hidden_dim=self.decoder_hidden_dim,
            dynamics_hidden_dim=self.dynamics_hidden_dim,
            projection_dim=self.hidden_dim,
            ode_method=self.ode_method,
            ode_rtol=self.ode_rtol,
            ode_atol=self.ode_atol,
            enable_continuous_evolution=self.continuous_evolution_enabled,
        )

    def effective_mechanisms(self) -> Mapping[str, object]:
        return {
            "continuous_evolution_enabled": self.continuous_evolution_enabled,
            "physics_enabled": self.physics_enabled,
            "causal_mask_enabled": self.causal_mask_enabled,
            "lag_ranges_s": [list(value) for value in self.lag_ranges_s],
            "scale_gate_enabled": self.scale_gate_enabled,
        }

    def to_checkpoint_dict(self) -> Mapping[str, object]:
        return asdict(self)

    @classmethod
    def from_checkpoint_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "ChronarisContinuousEncoderConfig":
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
class ChronarisContinuousEncoding:
    sequence_embedding: torch.Tensor
    modality_available_mask: torch.Tensor
    alignment_output: DualStreamPrototypeOutput
    fusion_output: MultiScaleCausalFusionOutput
    physics_audit: ChronarisPhysicsAudit


class ChronarisContinuousFusionEncoder(nn.Module):
    """Raw asynchronous streams -> dual ODE-RNN -> seconds-based causal fusion."""

    def __init__(self, config: ChronarisContinuousEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.continuous_backbone = DualStreamODERNNPrototype(
            len(config.physiology_feature_names),
            len(config.vehicle_feature_names),
            config=config.alignment_config(),
        )
        self.causal_fusion = MultiScaleCausalLagFusion(
            MultiScaleCausalFusionConfig(
                hidden_dim=config.hidden_dim,
                output_dim=FUSION_OUTPUT_DIM,
                lag_ranges_s=config.lag_ranges_s,
                use_causal_mask=config.causal_mask_enabled,
                use_scale_gate=config.scale_gate_enabled,
            )
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        batch: DualStreamObservationBatch,
        *,
        compute_diagnostics: bool = True,
    ) -> ChronarisContinuousEncoding:
        alignment_batch = build_alignment_batch_from_observations(
            batch,
            physiology_feature_names=self.config.physiology_feature_names,
            vehicle_feature_names=self.config.vehicle_feature_names,
        )
        query_times = batch.query_timestamps_s.to(
            device=batch.physiology_values.device,
            dtype=batch.physiology_values.dtype,
        )
        alignment = self.continuous_backbone(
            alignment_batch,
            reference_offsets_s=query_times,
            include_observation_diagnostics=compute_diagnostics,
        )
        physiology_states = alignment.physiology.reference_hidden_states
        vehicle_states = alignment.vehicle.reference_hidden_states
        physiology_valid = alignment.physiology.reference_valid_mask
        vehicle_valid = alignment.vehicle.reference_valid_mask
        if (
            physiology_states is None
            or vehicle_states is None
            or physiology_valid is None
            or vehicle_valid is None
        ):
            raise RepresentationContractError(
                "continuous backbone did not produce reference-grid states"
            )
        fusion = self.causal_fusion(
            MultiScaleCausalFusionInput(
                physiology_states=physiology_states,
                vehicle_states=vehicle_states,
                physiology_valid_mask=physiology_valid,
                vehicle_valid_mask=vehicle_valid,
                query_timestamps_s=query_times,
            )
        )
        physics = (
            build_chronaris_physics_audit(
                alignment,
                alignment_batch,
                field_labels=self.config.field_label_mapping,
                enabled=self.config.physics_enabled,
                weight=self.config.physics_weight,
                huber_delta=self.config.physics_huber_delta,
            )
            if compute_diagnostics
            else build_skipped_chronaris_physics_audit(fusion.sequence_embedding)
        )
        return ChronarisContinuousEncoding(
            sequence_embedding=self.output_dropout(fusion.sequence_embedding),
            modality_available_mask=fusion.modality_available_mask,
            alignment_output=alignment,
            fusion_output=fusion,
            physics_audit=physics,
        )


class ChronarisContinuousFusionAdapter:
    method_name = "chronaris"
    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        backbone: ChronarisContinuousFusionEncoder,
        normalizer: TrainOnlyRobustNormalizer,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.backbone = backbone
        self.normalizer = normalizer
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.last_encoding: ChronarisContinuousEncoding | None = None

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        device = next(self.backbone.parameters()).device
        normalized = self.normalizer.transform(
            move_observation_batch(batch, device=device)
        )
        self.backbone.eval()
        with torch.inference_mode():
            encoded = self.backbone(normalized)
        self.last_encoding = encoded
        sequence = encoded.sequence_embedding
        query_valid = torch.ones(
            sequence.shape[:2],
            dtype=torch.bool,
            device=sequence.device,
        )
        pooled = sequence.mean(dim=1)
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s.to(device),
            sequence_embedding=sequence,
            valid_mask=query_valid,
            pooled_embedding=pooled,
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.backbone.parameters())

    def to_manifest(self) -> Mapping[str, object]:
        physics_rows = (
            physics_audit_to_rows(self.last_encoding.physics_audit)
            if self.last_encoding is not None
            else ()
        )
        return {
            "method_name": self.method_name,
            "backbone_class": type(self.backbone).__name__,
            "backbone_config": dict(self.backbone.config.to_checkpoint_dict()),
            "effective_mechanisms": dict(
                self.backbone.config.effective_mechanisms()
            ),
            "parameter_count": self.parameter_count,
            "normalizer": self.normalizer.to_manifest(),
            "fold_id": self.fold_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "physics_components": list(physics_rows),
            "sequence_source": "task_head_free_continuous_fusion",
            "label_used_for_encoder_training": False,
        }


def build_chronaris_ablation_configs(
    full_config: ChronarisContinuousEncoderConfig,
) -> tuple[ChronarisContinuousEncoderConfig, ...]:
    if full_config.variant != "full":
        raise ValueError("ablation matrix requires a full Chronaris base config")
    return (full_config,) + tuple(
        replace(full_config, variant=variant)
        for variant in CHRONARIS_VARIANTS
        if variant != "full"
    )


def chronaris_ablation_diff(
    full_config: ChronarisContinuousEncoderConfig,
    variant_config: ChronarisContinuousEncoderConfig,
) -> Mapping[str, tuple[object, object]]:
    full = full_config.effective_mechanisms()
    variant = variant_config.effective_mechanisms()
    return {
        key: (full[key], variant[key])
        for key in full
        if full[key] != variant[key]
    }


def validate_chronaris_ablation_diff(
    full_config: ChronarisContinuousEncoderConfig,
    variant_config: ChronarisContinuousEncoderConfig,
) -> Mapping[str, tuple[object, object]]:
    if variant_config.variant not in ABLATION_TARGET_FIELDS:
        raise ValueError("variant is not one of the four fixed ablations")
    diff = chronaris_ablation_diff(full_config, variant_config)
    expected = ABLATION_TARGET_FIELDS[variant_config.variant]
    if frozenset(diff) != expected:
        raise RepresentationContractError(
            f"ablation {variant_config.variant} changed {sorted(diff)}; "
            f"expected {sorted(expected)}"
        )
    return diff


def save_chronaris_continuous_checkpoint(
    path: str | Path,
    *,
    backbone: ChronarisContinuousFusionEncoder,
    normalizer: TrainOnlyRobustNormalizer,
    seed: int,
) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(resolved.name + ".tmp")
    payload = {
        "format": "chronaris.continuous_fusion_encoder.v1",
        "config": dict(backbone.config.to_checkpoint_dict()),
        "model_state_dict": backbone.state_dict(),
        "normalizer": dict(normalizer.to_manifest()),
        "seed": int(seed),
        "sequence_source": "task_head_free_continuous_fusion",
        "label_used_for_encoder_training": False,
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(resolved)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return resolved


def load_chronaris_continuous_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    ChronarisContinuousFusionEncoder,
    TrainOnlyRobustNormalizer,
    Mapping[str, object],
]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "chronaris.continuous_fusion_encoder.v1":
        raise RepresentationContractError("unsupported Chronaris checkpoint format")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("Chronaris checkpoint used downstream labels")
    if payload.get("sequence_source") != "task_head_free_continuous_fusion":
        raise RepresentationContractError("Chronaris checkpoint is task-head dependent")
    config = ChronarisContinuousEncoderConfig.from_checkpoint_dict(payload["config"])
    backbone = ChronarisContinuousFusionEncoder(config).to(device)
    backbone.load_state_dict(payload["model_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    metadata = {
        "format": payload["format"],
        "seed": int(payload["seed"]),
        "sequence_source": payload["sequence_source"],
        "label_used_for_encoder_training": False,
    }
    return backbone, normalizer, metadata
