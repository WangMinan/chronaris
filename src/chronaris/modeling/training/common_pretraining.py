"""Resumable common-pretext training for the five trainable fusion encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.training.candidate_screen import (
    CandidateScreenConfig,
    train_pretext_candidate,
)
from chronaris.modeling.training.pretext import CommonPretextHeadBundle
from chronaris.modeling.training.pretraining_encoders import (
    ENCODER_SCREEN_CANDIDATES,
    TRAINABLE_FUSION_METHODS,
    EncoderCandidateConfig,
    TrainableFusionEncoder,
    build_trainable_fusion_encoder,
)
from chronaris.representation import (
    AugmentationPolicy,
    DualStreamObservationBatch,
    FoldLineage,
    FusionStreamBatch,
    TrainOnlyRobustNormalizer,
)
from chronaris.representation.contracts import FUSION_OUTPUT_DIM, RepresentationContractError


@dataclass(frozen=True, slots=True)
class CommonPretrainingConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"
    deterministic: bool = True
    max_ode_step_s: float | None = None
    ode_method: str = "euler"
    semantic_event_enabled: bool = False
    learnable_semantic_queries: bool = False
    heartbeat_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("pretraining epoch/batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("pretraining optimizer configuration is invalid")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient clip norm must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("pretraining device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("pretraining requested unavailable CUDA device")
        if self.max_ode_step_s is not None and (
            not math.isfinite(self.max_ode_step_s) or self.max_ode_step_s <= 0
        ):
            raise ValueError("max_ode_step_s must be finite and positive when set")
        if self.ode_method not in {"euler", "midpoint", "rk4", "dopri5"}:
            raise ValueError("unsupported pretraining Chronaris ODE method")
        if self.learnable_semantic_queries and not self.semantic_event_enabled:
            raise ValueError("learnable semantic queries require semantic_event_enabled")
        if not 0 < self.heartbeat_interval_s <= 60:
            raise ValueError("heartbeat_interval_s must be in (0,60]")


@dataclass(frozen=True, slots=True)
class CommonPretrainingResult:
    method_name: str
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    training_elapsed_s: float
    parameter_count: int
    head_parameter_count: int
    step_count: int
    training_rows: tuple[Mapping[str, object], ...]
    augmentation_rows: tuple[Mapping[str, object], ...]


class TrainedFusionAdapter:
    output_dim = FUSION_OUTPUT_DIM

    def __init__(
        self,
        *,
        encoder: TrainableFusionEncoder,
        normalizer: TrainOnlyRobustNormalizer,
        fold_id: str,
        checkpoint_sha256: str,
    ) -> None:
        self.encoder = encoder
        self.normalizer = normalizer
        self.fold_id = fold_id
        self.checkpoint_sha256 = checkpoint_sha256
        self.method_name = encoder.method_name

    def __call__(self, batch: DualStreamObservationBatch) -> FusionStreamBatch:
        device = next(self.encoder.parameters()).device
        normalized = move_observation_batch(
            self.normalizer.transform(batch),
            device=device,
        )
        self.encoder.eval()
        with torch.inference_mode():
            encoded = self.encoder(normalized)
        sequence = encoded.sequence_embedding
        valid = encoded.modality_available_mask.to(device=sequence.device)
        sequence = sequence.masked_fill(~valid.any(dim=1)[:, None, None], 0)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        pooled = (sequence * valid.unsqueeze(-1).to(sequence.dtype)).sum(dim=1) / count
        return FusionStreamBatch(
            sample_ids=batch.sample_ids,
            timestamps_s=batch.query_timestamps_s.to(device),
            sequence_embedding=sequence,
            valid_mask=valid,
            pooled_embedding=pooled,
            method_name=self.method_name,
            fold_id=self.fold_id,
            checkpoint_sha256=self.checkpoint_sha256,
            source_sample_hashes=batch.source_sample_hashes,
        )


def train_common_pretext_method(
    method_name: str,
    *,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: CommonPretrainingConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    candidate_config: EncoderCandidateConfig | None = None,
    chronaris_fusion_kind: str = "multiscale",
    chronaris_lag_aware_weight: float = 0.0,
    chronaris_mechanism_enabled: bool = False,
    chronaris_explicit_shift_enabled: bool = False,
    chronaris_explicit_shift_weight: float = 0.0,
    chronaris_event_pair_weight: float = 0.0,
    resume: bool = True,
) -> CommonPretrainingResult:
    """Compatibility wrapper around the validation-backed candidate trainer."""

    if method_name not in TRAINABLE_FUSION_METHODS:
        raise ValueError(f"unsupported trainable method: {method_name}")
    resolved_config = config or CommonPretrainingConfig()
    resolved_policy = augmentation_policy or AugmentationPolicy()
    resolved_candidate = replace(
        candidate_config or ENCODER_SCREEN_CANDIDATES[0],
        learning_rate=resolved_config.learning_rate,
    )
    candidate_result = train_pretext_candidate(
        method_name=method_name,
        candidate=resolved_candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        normalizer=normalizer,
        output_root=output_root,
        config=CandidateScreenConfig(
            max_epochs=resolved_config.epochs,
            batch_size=resolved_config.batch_size,
            patience=resolved_config.epochs,
            weight_decay=resolved_config.weight_decay,
            gradient_clip_norm=resolved_config.gradient_clip_norm,
            seed=resolved_config.seed,
            device=resolved_config.device,
            deterministic=resolved_config.deterministic,
            max_ode_step_s=resolved_config.max_ode_step_s,
            ode_method=resolved_config.ode_method,
            semantic_event_enabled=resolved_config.semantic_event_enabled,
            learnable_semantic_queries=resolved_config.learnable_semantic_queries,
            heartbeat_interval_s=resolved_config.heartbeat_interval_s,
        ),
        augmentation_policy=resolved_policy,
        batch_provider=batch_provider,
        chronaris_fusion_kind=chronaris_fusion_kind,
        chronaris_lag_aware_weight=chronaris_lag_aware_weight,
        chronaris_mechanism_enabled=chronaris_mechanism_enabled,
        chronaris_explicit_shift_enabled=chronaris_explicit_shift_enabled,
        chronaris_explicit_shift_weight=chronaris_explicit_shift_weight,
        chronaris_event_pair_weight=chronaris_event_pair_weight,
        include_candidate_subdirectory=False,
        resume=resume,
    )
    payload = _load_checkpoint_payload(candidate_result.best_checkpoint_path)
    return _result_from_payload(
        payload,
        Path(candidate_result.best_checkpoint_path),
        Path(candidate_result.last_checkpoint_path),
        status=candidate_result.status,
    )


def load_common_pretraining_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    allow_legacy_implementation: bool = False,
):
    payload = _load_checkpoint_payload(path, device=device)
    if payload.get("training_status") != "completed":
        raise RepresentationContractError("common pretraining checkpoint is incomplete")
    if bool(payload.get("label_used_for_encoder_training")):
        raise RepresentationContractError("pretraining checkpoint used downstream labels")
    method_name = str(payload["method_name"])
    if method_name == "chronaris" and payload.get("implementation_revision") != "causal_fusion_v4" and not allow_legacy_implementation:
        raise RepresentationContractError(
            "legacy Chronaris weights require their frozen source tag; "
            "set allow_legacy_implementation=True only for an explicit v4 weight-migration diagnostic"
        )
    physiology_names = tuple(payload["physiology_feature_names"])
    vehicle_names = tuple(payload["vehicle_feature_names"])
    field_labels = tuple(tuple(value) for value in payload["vehicle_field_labels"])
    candidate = EncoderCandidateConfig(**payload.get("candidate_config", {}))
    backbone_config = payload.get("encoder_manifest", {}).get("backbone_config", {})
    chronaris_variant = str(backbone_config.get("variant", "full"))
    chronaris_fusion_kind = str(backbone_config.get("fusion_kind", "multiscale"))
    chronaris_max_ode_step_s = backbone_config.get("max_ode_step_s")
    chronaris_ode_method = str(backbone_config.get("ode_method", "euler"))
    chronaris_semantic_event_enabled = bool(
        backbone_config.get("semantic_event_enabled", False)
    )
    chronaris_learnable_semantic_queries = bool(
        backbone_config.get("learnable_semantic_queries", False)
    )
    encoder = build_trainable_fusion_encoder(
        method_name,
        physiology_feature_names=physiology_names,
        vehicle_feature_names=vehicle_names,
        vehicle_field_labels=field_labels,
        candidate_config=candidate,
        chronaris_variant=chronaris_variant,
        chronaris_fusion_kind=chronaris_fusion_kind,
        chronaris_max_ode_step_s=chronaris_max_ode_step_s,
        chronaris_ode_method=chronaris_ode_method,
        chronaris_semantic_event_enabled=chronaris_semantic_event_enabled,
        chronaris_learnable_semantic_queries=chronaris_learnable_semantic_queries,
        chronaris_physics_calibration=backbone_config.get("physics_calibration"),
        chronaris_physics_weight=float(backbone_config.get("physics_weight", 0.1)),
    ).to(device)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    heads = CommonPretextHeadBundle(
        representation_dim=FUSION_OUTPUT_DIM,
        target_feature_count=len(physiology_names) + len(vehicle_names),
        **payload.get("pretext_head_config", {}),
    ).to(device)
    heads.load_state_dict(payload["head_state_dict"], strict=True)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(payload["normalizer"])
    return encoder, heads, normalizer, payload


def _load_checkpoint_payload(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") not in {
        "chronaris.common_pretraining_checkpoint.v1",
        "chronaris.common_pretraining_checkpoint.v2",
    }:
        raise RepresentationContractError("unsupported common pretraining checkpoint")
    return payload


def _result_from_payload(payload, best_path, last_path, *, status):
    return CommonPretrainingResult(
        method_name=str(payload["method_name"]),
        status=status,
        best_checkpoint_path=str(best_path),
        last_checkpoint_path=str(last_path),
        protocol_sha256=str(payload["protocol_sha256"]),
        training_elapsed_s=float(payload["training_elapsed_s"]),
        parameter_count=int(payload["parameter_count"]),
        head_parameter_count=int(payload["head_parameter_count"]),
        step_count=int(payload["step_count"]),
        training_rows=tuple(payload["training_rows"]),
        augmentation_rows=tuple(payload["augmentation_rows"]),
    )
