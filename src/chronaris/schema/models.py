"""Core data contracts shared across Chronaris modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from typing import Mapping

ScalarValue = str | int | float | bool | None


class StreamKind(StrEnum):
    """Supported upstream time-series stream categories."""

    PHYSIOLOGY = "physiology"
    VEHICLE = "vehicle"


@dataclass(frozen=True, slots=True)
class SortieLocator:
    """Identifies a target sortie and optional query boundaries."""

    sortie_id: str
    pilot_id: str | None = None
    aircraft_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class RawPoint:
    """One raw upstream time-series point."""

    stream_kind: StreamKind
    measurement: str
    timestamp: datetime
    values: Mapping[str, ScalarValue]
    clock_time: time | None = None
    timestamp_precision_digits: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    source: str | None = None


@dataclass(frozen=True, slots=True)
class SortieMetadata:
    """Business metadata attached to a sortie."""

    sortie_id: str
    flight_task_id: int | None = None
    flight_batch_id: int | None = None
    flight_date: date | None = None
    mission_code: str | None = None
    aircraft_model: str | None = None
    aircraft_tail: str | None = None
    pilot_code: str | None = None
    batch_id: str | None = None
    sortie_number: str | None = None
    batch_number: str | None = None
    extra: Mapping[str, ScalarValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SortieBundle:
    """Raw data grouped by sortie before timebase normalization."""

    locator: SortieLocator
    metadata: SortieMetadata
    physiology_points: tuple[RawPoint, ...] = ()
    vehicle_points: tuple[RawPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class AlignedPoint:
    """A raw point projected into a relative-time coordinate system."""

    point: RawPoint
    offset_ms: int


@dataclass(frozen=True, slots=True)
class AlignedSortieBundle:
    """Sortie data after reference-time selection and offset projection."""

    locator: SortieLocator
    metadata: SortieMetadata
    reference_time: datetime
    physiology_points: tuple[AlignedPoint, ...] = ()
    vehicle_points: tuple[AlignedPoint, ...] = ()


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Controls how one aligned sortie is sliced into windows."""

    duration_ms: int
    stride_ms: int
    min_physiology_points: int = 1
    min_vehicle_points: int = 1
    allow_partial_last_window: bool = True

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive.")
        if self.stride_ms <= 0:
            raise ValueError("stride_ms must be positive.")
        if self.min_physiology_points < 0:
            raise ValueError("min_physiology_points cannot be negative.")
        if self.min_vehicle_points < 0:
            raise ValueError("min_vehicle_points cannot be negative.")


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """A windowed sample ready for later model or evaluation stages."""

    sample_id: str
    sortie_id: str
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    physiology_points: tuple[AlignedPoint, ...] = ()
    vehicle_points: tuple[AlignedPoint, ...] = ()
    labels: Mapping[str, ScalarValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    """Result of building one sortie into aligned windowed samples."""

    aligned_bundle: AlignedSortieBundle
    windows: tuple[SampleWindow, ...] = ()

    def summary(self) -> dict[str, str | int]:
        return {
            "sortie_id": self.aligned_bundle.locator.sortie_id,
            "reference_time": self.aligned_bundle.reference_time.isoformat(),
            "physiology_points": len(self.aligned_bundle.physiology_points),
            "vehicle_points": len(self.aligned_bundle.vehicle_points),
            "windows": len(self.windows),
        }
