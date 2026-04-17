"""Reference time-grid builders for Stage E alignment experiments."""

from __future__ import annotations

from dataclasses import dataclass

from chronaris.features.experiment_input import E0ExperimentSample


@dataclass(frozen=True, slots=True)
class ReferenceGridConfig:
    """Controls how window-level reference grids are generated."""

    point_count: int = 16
    include_end: bool = True

    def __post_init__(self) -> None:
        if self.point_count <= 0:
            raise ValueError("point_count must be positive.")


@dataclass(frozen=True, slots=True)
class ReferenceGrid:
    """A shared reference time grid for one Stage E sample window."""

    sample_id: str
    start_offset_ms: int
    end_offset_ms: int
    duration_ms: int
    relative_offsets_ms: tuple[float, ...]
    absolute_offsets_ms: tuple[float, ...]
    relative_offsets_s: tuple[float, ...]


def build_reference_grid(
    sample: E0ExperimentSample,
    *,
    config: ReferenceGridConfig | None = None,
) -> ReferenceGrid:
    """Build a shared reference grid for one E0 sample window."""

    active_config = config or ReferenceGridConfig()
    if sample.end_offset_ms < sample.start_offset_ms:
        raise ValueError("sample end_offset_ms must be greater than or equal to start_offset_ms.")

    duration_ms = sample.end_offset_ms - sample.start_offset_ms
    relative_offsets_ms = _build_relative_offsets_ms(duration_ms, active_config)
    absolute_offsets_ms = tuple(sample.start_offset_ms + offset for offset in relative_offsets_ms)
    relative_offsets_s = tuple(offset / 1000.0 for offset in relative_offsets_ms)

    return ReferenceGrid(
        sample_id=sample.sample_id,
        start_offset_ms=sample.start_offset_ms,
        end_offset_ms=sample.end_offset_ms,
        duration_ms=duration_ms,
        relative_offsets_ms=relative_offsets_ms,
        absolute_offsets_ms=absolute_offsets_ms,
        relative_offsets_s=relative_offsets_s,
    )


def build_reference_grids(
    samples: tuple[E0ExperimentSample, ...],
    *,
    config: ReferenceGridConfig | None = None,
) -> tuple[ReferenceGrid, ...]:
    """Build reference grids for a sequence of E0 sample windows."""

    active_config = config or ReferenceGridConfig()
    return tuple(build_reference_grid(sample, config=active_config) for sample in samples)


def _build_relative_offsets_ms(
    duration_ms: int,
    config: ReferenceGridConfig,
) -> tuple[float, ...]:
    if config.point_count == 1:
        return (0.0,)

    if config.include_end:
        denominator = config.point_count - 1
        return tuple((duration_ms * index) / denominator for index in range(config.point_count))

    return tuple((duration_ms * index) / config.point_count for index in range(config.point_count))
