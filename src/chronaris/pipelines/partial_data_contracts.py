"""Contracts and JSONL helpers for partial-data assets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from chronaris.schema.models import RawPoint, WindowConfig


VEHICLE_ONLY_FEATURE_BUNDLE_KEYS = (
    "feature_names",
    "values",
    "point_offsets_s",
    "valid_mask",
    "point_counts",
    "sample_ids",
    "sortie_ids",
    "window_indices",
    "window_start_offsets_s",
    "window_end_offsets_s",
)


@dataclass(frozen=True, slots=True)
class PartialDataEntry:
    """Standardized repo-consumable record for non-dual-stream assets."""

    sortie_id: str
    source_type: str
    stream_kind: str
    data_tier: str
    time_range: Mapping[str, str | None]
    measurement_family: tuple[str, ...]
    usable_for_pretraining: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    raw_manifest_path: str | None = None
    inventory_reference: str | None = None
    bucket: str | None = None
    tag_filters: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sortie_id": self.sortie_id,
            "source_type": self.source_type,
            "stream_kind": self.stream_kind,
            "data_tier": self.data_tier,
            "raw_manifest_path": self.raw_manifest_path,
            "inventory_reference": self.inventory_reference,
            "time_range": dict(self.time_range),
            "measurement_family": list(self.measurement_family),
            "usable_for_pretraining": self.usable_for_pretraining,
            "notes": list(self.notes),
            "bucket": self.bucket,
            "tag_filters": dict(self.tag_filters),
        }


@dataclass(frozen=True, slots=True)
class PartialStreamSample:
    """One windowed single-stream sample produced from a partial-data entry."""

    sample_id: str
    sortie_id: str
    stream_kind: str
    window_index: int
    start_offset_ms: int
    end_offset_ms: int
    point_count: int
    feature_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartialDataManifest:
    """Output summary for one partial-data builder run."""

    export_version: str
    generated_at_utc: str
    entry_count: int
    built_entry_count: int
    skipped_entry_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "export_version": self.export_version,
            "generated_at_utc": self.generated_at_utc,
            "entry_count": self.entry_count,
            "built_entry_count": self.built_entry_count,
            "skipped_entry_count": self.skipped_entry_count,
        }


@dataclass(frozen=True, slots=True)
class PartialDataConfig:
    """Controls partial-data manifesting and vehicle-only sample export."""

    export_version: str = "partial-data-v1"
    point_limit_per_measurement: int | None = None
    max_points_per_field_per_window: int = 32
    window_config: WindowConfig = field(
        default_factory=lambda: WindowConfig(
            duration_ms=5_000,
            stride_ms=5_000,
            min_physiology_points=0,
            min_vehicle_points=1,
        )
    )

    def __post_init__(self) -> None:
        if self.point_limit_per_measurement is not None and self.point_limit_per_measurement <= 0:
            raise ValueError("point_limit_per_measurement must be positive when provided.")
        if self.max_points_per_field_per_window <= 0:
            raise ValueError("max_points_per_field_per_window must be positive.")


@dataclass(frozen=True, slots=True)
class PartialDataBuildResult:
    """Concrete artifact paths from one partial-data builder run."""

    manifest: PartialDataManifest
    manifest_path: str
    window_manifest_path: str
    feature_bundle_path: str | None
    built_samples: tuple[PartialStreamSample, ...]


@dataclass(frozen=True, slots=True)
class PartialPointChunk:
    """A bounded chunk of raw points for streaming partial-data builds."""

    start_utc: datetime
    stop_utc: datetime
    points: tuple[RawPoint, ...]


@dataclass(frozen=True, slots=True)
class PartialMeasurementMetadata:
    """RealBus field metadata resolved for one vehicle measurement."""

    measurement: str
    status: str
    field_names: tuple[str, ...] = ()
    reason: str | None = None
    allow_all_fields: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "measurement": self.measurement,
            "status": self.status,
            "field_count": None if self.allow_all_fields else len(self.field_names),
            "allow_all_fields": self.allow_all_fields,
            "reason": self.reason,
        }


PartialPointProvider = Callable[[PartialDataEntry], Sequence[RawPoint]]
PartialPointChunkProvider = Callable[[PartialDataEntry], Iterable[PartialPointChunk]]
PartialVehicleMetadataProvider = Callable[[PartialDataEntry], Mapping[str, PartialMeasurementMetadata]]


def load_partial_data_entries(path: str | Path) -> tuple[PartialDataEntry, ...]:
    """Load standardized partial-data entries from JSONL."""

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(
        PartialDataEntry(
            sortie_id=row["sortie_id"],
            source_type=row["source_type"],
            stream_kind=row["stream_kind"],
            data_tier=row["data_tier"],
            raw_manifest_path=row.get("raw_manifest_path"),
            inventory_reference=row.get("inventory_reference"),
            time_range=dict(row.get("time_range", {})),
            measurement_family=tuple(row.get("measurement_family", ())),
            usable_for_pretraining=bool(row.get("usable_for_pretraining", False)),
            notes=tuple(row.get("notes", ())),
            bucket=row.get("bucket"),
            tag_filters=dict(row.get("tag_filters", {})),
        )
        for row in rows
    )


def dump_partial_data_entries(
    entries: Iterable[PartialDataEntry],
    *,
    path: str | Path,
) -> None:
    """Write standardized partial-data entries to JSONL."""

    write_jsonl(Path(path), [entry.to_dict() for entry in entries])


def concrete_measurements(entry: PartialDataEntry) -> tuple[str, ...]:
    """Return concrete measurement names, excluding placeholder markers."""

    return tuple(
        measurement
        for measurement in entry.measurement_family
        if measurement and not measurement.endswith("_PENDING")
    )


def parse_required_utc(value: str | None, field_name: str) -> datetime:
    """Parse a required UTC timestamp from a manifest field."""

    if not value:
        raise ValueError(f"vehicle-only partial entry requires {field_name}.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write mappings as UTF-8 JSON Lines."""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
