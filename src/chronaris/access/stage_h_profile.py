"""Stage H multi-sortie export profile resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from chronaris.access.influx_cli import InfluxDistinctMeasurementReader
from chronaris.access.mysql_metadata import (
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLStorageAnalysisReader,
)
from chronaris.access.physiology_context import derive_physiology_query_context
from chronaris.schema.models import SortieLocator, SortieMetadata
from chronaris.schema.real_bus import CollectTaskMetadata, FlightTaskMetadata, StorageAnalysis


@dataclass(frozen=True, slots=True)
class StageHViewProfile:
    """Resolved one-pilot export view for a Stage H sortie."""

    view_id: str
    pilot_id: int


@dataclass(frozen=True, slots=True)
class StageHSortieProfile:
    """Resolved metadata and stream scope for one Stage H sortie."""

    sortie_id: str
    flight_task: FlightTaskMetadata
    collect_task: CollectTaskMetadata
    pilot_ids: tuple[int, ...]
    views: tuple[StageHViewProfile, ...]
    physiology_bucket: str
    vehicle_bucket: str
    available_physiology_measurements: tuple[str, ...]
    model_physiology_measurements: tuple[str, ...]
    vehicle_measurements: tuple[str, ...]
    vehicle_analysis_ids: Mapping[str, int] = field(default_factory=dict)
    clip_start_utc: datetime | None = None
    clip_stop_utc: datetime | None = None
    pilot_resolution_source: str = "unknown"

    @property
    def collect_task_id(self) -> int:
        return self.collect_task.collect_task_id

    def to_sortie_metadata(self) -> SortieMetadata:
        """Convert the resolved profile into repo-wide sortie metadata."""

        return SortieMetadata(
            sortie_id=self.sortie_id,
            flight_task_id=self.flight_task.flight_task_id,
            flight_batch_id=self.flight_task.flight_batch_id,
            flight_date=self.flight_task.flight_date,
            mission_code=self.flight_task.mission_code,
            aircraft_model=self.flight_task.aircraft_model,
            aircraft_tail=self.flight_task.aircraft_number,
            pilot_code=self.flight_task.pilot_code,
            batch_id=(
                str(self.flight_task.flight_batch_id)
                if self.flight_task.flight_batch_id is not None
                else None
            ),
            sortie_number=self.flight_task.sortie_number,
            batch_number=self.flight_task.batch_number,
            extra={
                "collect_task_id": self.collect_task.collect_task_id,
                "pilot_ids": ",".join(str(pilot_id) for pilot_id in self.pilot_ids),
                "source_sortie_id": self.flight_task.source_sortie_id,
                "pilot_resolution_source": self.pilot_resolution_source,
                "clip_start_utc": (
                    None if self.clip_start_utc is None else self.clip_start_utc.isoformat()
                ),
                "clip_stop_utc": (
                    None if self.clip_stop_utc is None else self.clip_stop_utc.isoformat()
                ),
            },
        )


@dataclass(slots=True)
class StageHProfileResolver:
    """Resolve Stage H export profiles from MySQL and Influx facts."""

    flight_task_reader: MySQLFlightTaskReader
    collect_task_reader: MySQLCollectTaskReader
    storage_analysis_reader: MySQLStorageAnalysisReader
    distinct_measurement_reader: InfluxDistinctMeasurementReader
    physiology_bucket: str = "physiological_input"
    model_physiology_measurements: tuple[str, ...] = ("eeg", "spo2")
    vehicle_category: str = "BUS"
    require_model_physiology_measurements: bool = True

    def resolve(self, locator: SortieLocator) -> StageHSortieProfile:
        """Resolve one sortie into the exact Stage H export contract."""

        flight_task = self.flight_task_reader.fetch_by_locator(locator)
        collect_task = self.collect_task_reader.fetch_for_flight_task(flight_task)
        physiology_context = derive_physiology_query_context(
            flight_task,
            collect_task=collect_task,
        )
        pilot_ids = physiology_context.pilot_ids
        views = tuple(
            StageHViewProfile(
                view_id=f"{flight_task.sortie_number}__pilot_{pilot_id}",
                pilot_id=pilot_id,
            )
            for pilot_id in pilot_ids
        )
        physiology_measurements = self.distinct_measurement_reader.fetch_measurements(
            bucket=self.physiology_bucket,
            start=physiology_context.start_time_utc,
            stop=physiology_context.stop_time_utc,
            tag_filters={"collect_task_id": str(physiology_context.collect_task_id)},
            tag_filters_any={"pilot_id": tuple(str(pilot_id) for pilot_id in pilot_ids)},
        )
        model_physiology_measurements = tuple(
            measurement
            for measurement in self.model_physiology_measurements
            if measurement in physiology_measurements
        )
        if self.require_model_physiology_measurements and (
            model_physiology_measurements != self.model_physiology_measurements
        ):
            missing = sorted(
                set(self.model_physiology_measurements) - set(model_physiology_measurements)
            )
            raise ValueError(
                f"Stage H sortie {flight_task.sortie_number} is missing required physiology "
                f"measurements: {', '.join(missing)}"
            )

        analyses = self.storage_analysis_reader.list_for_sortie(
            locator,
            category=self.vehicle_category,
        )
        vehicle_measurements = tuple(
            dict.fromkeys(
                analysis.measurement
                for analysis in analyses
                if analysis.measurement
            )
        )
        if not vehicle_measurements:
            raise ValueError(
                f"Stage H sortie {flight_task.sortie_number} has no "
                f"{self.vehicle_category} storage_data_analysis measurements."
            )
        return StageHSortieProfile(
            sortie_id=flight_task.sortie_number,
            flight_task=flight_task,
            collect_task=collect_task,
            pilot_ids=pilot_ids,
            views=views,
            physiology_bucket=self.physiology_bucket,
            vehicle_bucket=_resolve_vehicle_bucket(analyses),
            available_physiology_measurements=physiology_measurements,
            model_physiology_measurements=model_physiology_measurements,
            vehicle_measurements=vehicle_measurements,
            vehicle_analysis_ids={
                analysis.measurement: analysis.analysis_id
                for analysis in analyses
                if analysis.measurement
            },
            clip_start_utc=physiology_context.start_time_utc,
            clip_stop_utc=physiology_context.stop_time_utc,
            pilot_resolution_source=_resolve_pilot_resolution_source(flight_task),
        )

    def resolve_many(self, sortie_ids: Sequence[str]) -> tuple[StageHSortieProfile, ...]:
        """Resolve a stable sequence of Stage H profiles."""

        return tuple(self.resolve(SortieLocator(sortie_id=sortie_id)) for sortie_id in sortie_ids)


def _resolve_vehicle_bucket(analyses: Sequence[StorageAnalysis]) -> str:
    for analysis in analyses:
        if analysis.bucket:
            return analysis.bucket
    return "bus"


def _resolve_pilot_resolution_source(flight_task: FlightTaskMetadata) -> str:
    if flight_task.source_sortie_id:
        return "source_sortie_id"
    if flight_task.up_pilot_id is not None or flight_task.down_pilot_id is not None:
        return "up_down_pilot_id"
    return "unknown"
