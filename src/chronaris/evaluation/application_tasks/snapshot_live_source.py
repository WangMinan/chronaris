"""Read-only Influx point source for G2a fixed-data snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from chronaris.access.influx_cli import InfluxQueryRunner
from chronaris.access.overlap_preview import DirectInfluxScopeConfig, ScopedInfluxPointReader
from chronaris.feature_export.profile import StageHSortieProfile, StageHViewProfile
from chronaris.schema.models import RawPoint, SortieLocator, StreamKind


class SnapshotPointSource(Protocol):
    """Minimal injectable source used by the snapshot orchestrator."""

    def fetch_physiology(
        self,
        profile: StageHSortieProfile,
        view: StageHViewProfile,
        *,
        start_utc: datetime,
        stop_utc: datetime,
    ) -> Sequence[RawPoint]: ...

    def fetch_vehicle(
        self,
        profile: StageHSortieProfile,
        *,
        start_utc: datetime,
        stop_utc: datetime,
    ) -> Sequence[RawPoint]: ...


@dataclass(frozen=True, slots=True)
class InfluxSnapshotPointSource:
    """Execute only fixed-scope read queries against the current Influx copy."""

    runner: InfluxQueryRunner

    def fetch_physiology(
        self,
        profile: StageHSortieProfile,
        view: StageHViewProfile,
        *,
        start_utc: datetime,
        stop_utc: datetime,
    ) -> Sequence[RawPoint]:
        reader = ScopedInfluxPointReader(
            runner=self.runner,
            stream_kind=StreamKind.PHYSIOLOGY,
            scope=DirectInfluxScopeConfig(
                bucket=profile.physiology_bucket,
                measurements=profile.model_physiology_measurements,
                start_time_utc=start_utc,
                stop_time_utc=stop_utc,
                tag_filters={
                    "collect_task_id": str(profile.collect_task_id),
                    "pilot_id": str(view.pilot_id),
                },
            ),
        )
        return reader.fetch_points(
            SortieLocator(sortie_id=profile.sortie_id, pilot_id=str(view.pilot_id))
        )

    def fetch_vehicle(
        self,
        profile: StageHSortieProfile,
        *,
        start_utc: datetime,
        stop_utc: datetime,
    ) -> Sequence[RawPoint]:
        reader = ScopedInfluxPointReader(
            runner=self.runner,
            stream_kind=StreamKind.VEHICLE,
            scope=DirectInfluxScopeConfig(
                bucket=profile.vehicle_bucket,
                measurements=profile.vehicle_measurements,
                start_time_utc=start_utc,
                stop_time_utc=stop_utc,
                tag_filters={"sortie_number": profile.sortie_id},
            ),
        )
        return reader.fetch_points(SortieLocator(sortie_id=profile.sortie_id))
