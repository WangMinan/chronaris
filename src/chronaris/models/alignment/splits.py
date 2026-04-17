"""Chronological split utilities for Stage E alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from chronaris.features.experiment_input import E0ExperimentSample


@dataclass(frozen=True, slots=True)
class ChronologicalSplitConfig:
    """Controls how E0 samples are split into train/validation/test partitions."""

    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2
    gap_windows: int = 0

    def __post_init__(self) -> None:
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("Split ratios must be non-negative.")
        ratio_sum = sum(ratios)
        if not math.isclose(ratio_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Split ratios must sum to 1.0.")
        if self.gap_windows < 0:
            raise ValueError("gap_windows must be non-negative.")


@dataclass(frozen=True, slots=True)
class ChronologicalSampleSplit:
    """A train/validation/test split for E0 experiment samples."""

    train: tuple[E0ExperimentSample, ...]
    validation: tuple[E0ExperimentSample, ...]
    test: tuple[E0ExperimentSample, ...]
    skipped_between_train_validation: tuple[E0ExperimentSample, ...] = field(default_factory=tuple)
    skipped_between_validation_test: tuple[E0ExperimentSample, ...] = field(default_factory=tuple)


def split_e0_samples_chronologically(
    samples: tuple[E0ExperimentSample, ...],
    *,
    config: ChronologicalSplitConfig | None = None,
) -> ChronologicalSampleSplit:
    """Split E0 samples by time order for Stage E preview training."""

    if not samples:
        raise ValueError("At least one E0ExperimentSample is required for chronological splitting.")

    active_config = config or ChronologicalSplitConfig()
    ordered_samples = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.start_offset_ms,
                sample.end_offset_ms,
                sample.sample_id,
            ),
        )
    )

    usable_count = len(ordered_samples) - (2 * active_config.gap_windows)
    if usable_count <= 0:
        raise ValueError("gap_windows leaves no usable samples for train/validation/test splits.")

    train_count = int(math.floor(usable_count * active_config.train_ratio))
    validation_count = int(math.floor(usable_count * active_config.validation_ratio))
    test_count = usable_count - train_count - validation_count

    train_end = train_count
    skipped_after_train_end = train_end + active_config.gap_windows
    validation_end = skipped_after_train_end + validation_count
    skipped_after_validation_end = validation_end + active_config.gap_windows
    test_end = skipped_after_validation_end + test_count

    return ChronologicalSampleSplit(
        train=ordered_samples[:train_end],
        skipped_between_train_validation=ordered_samples[train_end:skipped_after_train_end],
        validation=ordered_samples[skipped_after_train_end:validation_end],
        skipped_between_validation_test=ordered_samples[validation_end:skipped_after_validation_end],
        test=ordered_samples[skipped_after_validation_end:test_end],
    )
