"""Contracts for the ignored local snapshot of fixed Dingxin raw points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


FIXED_DINGXIN_SORTIE_IDS = (
    "20251005_四01_ACT-4_云_J20_22#01",
    "20251002_单01_ACT-8_翼云_J16_12#01",
)
RAW_POINT_FORMAT = "chronaris.raw_point_jsonl_gzip.v1"


@dataclass(frozen=True, slots=True)
class SnapshotViewPlan:
    """One fixed physiology view and its shared sortie vehicle scope."""

    sortie_id: str
    view_id: str
    pilot_id: int
    start_utc: str
    stop_utc: str
    physiology_measurements: tuple[str, ...]
    vehicle_measurements: tuple[str, ...]
    expected_physiology_point_count: int
    expected_vehicle_point_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["physiology_measurements"] = list(self.physiology_measurements)
        payload["vehicle_measurements"] = list(self.vehicle_measurements)
        return payload


@dataclass(frozen=True, slots=True)
class SnapshotFileRecord:
    """Compact lineage for one local raw-point file."""

    stream_kind: str
    sortie_id: str
    view_id: str | None
    pilot_id: int | None
    relative_path: str
    sha256: str
    byte_count: int
    point_count: int
    measurement_counts: Mapping[str, int]
    field_count: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None
    format: str = RAW_POINT_FORMAT

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "measurement_counts": dict(self.measurement_counts),
        }


@dataclass(frozen=True, slots=True)
class SnapshotRunResult:
    """Result returned by the G2a snapshot orchestrator."""

    run_id: str
    status: str
    snapshot_root: str
    compact_run_root: str
    snapshot_manifest_path: str
    compact_manifest_path: str
    report_path: str
    evidence_manifest_path: str
    file_count: int
    physiology_point_count: int
    vehicle_point_count: int
    resumed: bool = False
