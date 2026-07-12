"""Predeclared missingness and timing curriculum for Chronaris v2."""

from __future__ import annotations

from dataclasses import dataclass

from chronaris.representation.augmentation import AugmentationPolicy


@dataclass(frozen=True, slots=True)
class ChronarisV2CurriculumStage:
    stage_index: int
    epoch_start: int
    point_dropout_probability: float
    block_duration_s: float
    clock_offset_limit_s: float


CURRICULUM_STAGES = (
    ChronarisV2CurriculumStage(0, 1, 0.10, 5.0, 0.25),
    ChronarisV2CurriculumStage(1, 6, 0.20, 5.0, 0.50),
    ChronarisV2CurriculumStage(2, 11, 0.30, 10.0, 0.50),
    ChronarisV2CurriculumStage(3, 16, 0.40, 15.0, 1.00),
)


def chronaris_v2_curriculum_stage(epoch: int) -> ChronarisV2CurriculumStage:
    if epoch <= 0:
        raise ValueError("curriculum epoch must be one-based")
    return max(
        (stage for stage in CURRICULUM_STAGES if stage.epoch_start <= epoch),
        key=lambda stage: stage.epoch_start,
    )


def chronaris_v2_augmentation_policy(epoch: int) -> AugmentationPolicy:
    stage = chronaris_v2_curriculum_stage(epoch)
    return AugmentationPolicy(
        point_dropout_probability=stage.point_dropout_probability,
        block_duration_min_s=stage.block_duration_s,
        block_duration_max_s=stage.block_duration_s,
        modality_dropout_probability=0.05,
        timestamp_jitter_sigma_s=0.05,
        clock_offset_limit_s=stage.clock_offset_limit_s,
    )
