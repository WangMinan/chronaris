"""Locked Chronaris compatibility wrapper over the canonical candidate trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from chronaris.modeling.training.candidate_screen import (
    CandidateScreenConfig,
    train_pretext_candidate,
)
from chronaris.modeling.training.pretraining_encoders import (
    ENCODER_SCREEN_CANDIDATES,
    EncoderCandidateConfig,
)
from chronaris.representation import (
    AugmentationPolicy,
    DualStreamObservationBatch,
    FoldLineage,
    TrainOnlyRobustNormalizer,
)


@dataclass(frozen=True, slots=True)
class LockedChronarisTrainingConfig:
    max_epochs: int = 50
    batch_size: int = 128
    patience: int = 8
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    seed: int = 17
    device: str = "cpu"
    deterministic: bool = True
    max_ode_step_s: float | None = None
    ode_method: str = "euler"
    semantic_event_enabled: bool = False
    learnable_semantic_queries: bool = False
    lag_aware_weight: float = 0.0
    explicit_shift_weight: float = 0.0
    event_pair_weight: float = 0.0
    heartbeat_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if min(self.max_epochs, self.batch_size, self.patience) <= 0:
            raise ValueError("locked Chronaris epoch/batch/patience must be positive")
        if self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("locked Chronaris optimizer configuration is invalid")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("locked Chronaris device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("locked Chronaris requested unavailable CUDA device")
        if self.max_ode_step_s is not None and (
            not math.isfinite(self.max_ode_step_s) or self.max_ode_step_s <= 0
        ):
            raise ValueError("max_ode_step_s must be finite and positive when set")
        if self.ode_method not in {"euler", "midpoint", "rk4", "dopri5"}:
            raise ValueError("unsupported locked Chronaris ODE method")
        if min(
            self.lag_aware_weight,
            self.explicit_shift_weight,
            self.event_pair_weight,
        ) < 0:
            raise ValueError("locked Chronaris objective weights must be non-negative")
        if self.learnable_semantic_queries and not self.semantic_event_enabled:
            raise ValueError("learnable semantic queries require semantic_event_enabled")
        if self.event_pair_weight > 0 and not self.semantic_event_enabled:
            raise ValueError("event-pair objective requires semantic_event_enabled")
        if not 0 < self.heartbeat_interval_s <= 60:
            raise ValueError("heartbeat_interval_s must be in (0,60]")


@dataclass(frozen=True, slots=True)
class LockedChronarisTrainingResult:
    status: str
    best_checkpoint_path: str
    last_checkpoint_path: str
    protocol_sha256: str
    best_epoch: int
    completed_epochs: int
    stopped_early: bool
    best_validation_losses: Mapping[str, float]
    best_public_selection_loss: float
    training_elapsed_s: float
    parameter_count: int
    epoch_rows: tuple[Mapping[str, object], ...]
    auxiliary_rows: tuple[Mapping[str, object], ...]


def train_locked_chronaris(
    *,
    batch: DualStreamObservationBatch | None,
    fold: FoldLineage,
    physiology_feature_names: tuple[str, ...],
    vehicle_feature_names: tuple[str, ...],
    vehicle_field_labels: tuple[tuple[str, str], ...],
    normalizer: TrainOnlyRobustNormalizer,
    output_root: str | Path,
    config: LockedChronarisTrainingConfig | None = None,
    augmentation_policy: AugmentationPolicy | None = None,
    batch_provider: Callable[[Sequence[str]], DualStreamObservationBatch] | None = None,
    candidate_config: EncoderCandidateConfig | None = None,
    variant: str = "full",
    fusion_kind: str = "multiscale",
    initialization_checkpoint: str | Path | None = None,
    resume: bool = True,
) -> LockedChronarisTrainingResult:
    resolved = config or LockedChronarisTrainingConfig()
    candidate = candidate_config or ENCODER_SCREEN_CANDIDATES[0]
    result = train_pretext_candidate(
        "chronaris",
        candidate=candidate,
        batch=batch,
        fold=fold,
        physiology_feature_names=physiology_feature_names,
        vehicle_feature_names=vehicle_feature_names,
        vehicle_field_labels=vehicle_field_labels,
        normalizer=normalizer,
        output_root=output_root,
        config=CandidateScreenConfig(
            max_epochs=resolved.max_epochs,
            batch_size=resolved.batch_size,
            patience=resolved.patience,
            weight_decay=resolved.weight_decay,
            gradient_clip_norm=resolved.gradient_clip_norm,
            seed=resolved.seed,
            device=resolved.device,
            deterministic=resolved.deterministic,
            max_ode_step_s=resolved.max_ode_step_s,
            ode_method=resolved.ode_method,
            semantic_event_enabled=resolved.semantic_event_enabled,
            learnable_semantic_queries=resolved.learnable_semantic_queries,
            heartbeat_interval_s=resolved.heartbeat_interval_s,
        ),
        augmentation_policy=augmentation_policy,
        batch_provider=batch_provider,
        initialization_checkpoint=initialization_checkpoint,
        chronaris_variant=variant,
        chronaris_fusion_kind=fusion_kind,
        chronaris_lag_aware_weight=resolved.lag_aware_weight,
        chronaris_mechanism_enabled=True,
        chronaris_explicit_shift_weight=resolved.explicit_shift_weight,
        chronaris_event_pair_weight=resolved.event_pair_weight,
        include_candidate_subdirectory=False,
        resume=resume,
    )
    payload = torch.load(
        result.best_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    auxiliary_names = {
        "chronaris_continuous_alignment",
        "chronaris_physical_consistency",
        "chronaris_causal_direction",
        "lag_aware_alignment",
        "explicit_time_shift",
        "event_response_pairing",
    }
    return LockedChronarisTrainingResult(
        status=result.status,
        best_checkpoint_path=result.best_checkpoint_path,
        last_checkpoint_path=result.last_checkpoint_path,
        protocol_sha256=result.protocol_sha256,
        best_epoch=result.best_epoch,
        completed_epochs=result.completed_epochs,
        stopped_early=result.stopped_early,
        best_validation_losses=result.best_validation_losses,
        best_public_selection_loss=result.best_public_selection_loss,
        training_elapsed_s=result.training_elapsed_s,
        parameter_count=result.parameter_count,
        epoch_rows=result.epoch_rows,
        auxiliary_rows=tuple(
            row
            for row in payload.get("training_rows", ())
            if row.get("term_name") in auxiliary_names
        ),
    )
