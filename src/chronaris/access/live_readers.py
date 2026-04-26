"""Real Influx-backed point readers for Chronaris Stage B."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from chronaris.access.influx_cli import (
    InfluxDistinctMeasurementReader,
    InfluxMeasurementPointReader,
    InfluxQueryRunner,
    InfluxQuerySpec,
)
from chronaris.access.mysql_metadata import (
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
)
from chronaris.access.physiology_context import derive_physiology_query_context
from chronaris.access.real_bus_context import derive_real_bus_context
from chronaris.schema.models import RawPoint, SortieLocator, StreamKind


@dataclass(frozen=True, slots=True)
class RealBusInfluxPointReader:
    """Reads vehicle-side bus points for one sortie using real metadata context."""

    context_reader: MySQLRealBusContextReader
    runner: InfluxQueryRunner
    access_rule_id: int
    analysis_id: int
    point_limit: int | None = None
    start_time_override_utc: datetime | None = None
    stop_time_override_utc: datetime | None = None

    def fetch_points(self, locator: SortieLocator) -> tuple[RawPoint, ...]:
        context = self.context_reader.fetch_context(
            locator=locator,
            access_rule_id=self.access_rule_id,
            analysis_id=self.analysis_id,
        )
        derived = derive_real_bus_context(
            flight_task=context.flight_task,
            category=context.analysis.category,
            access_rule_details=context.access_rule_details,
            detail_list=context.detail_list,
        )

        scoped_reader = InfluxMeasurementPointReader(
            runner=self.runner,
            stream_kind=StreamKind.VEHICLE,
            query_builder=lambda _: InfluxQuerySpec(
                bucket=context.analysis.bucket or "bus",
                measurement=context.analysis.measurement,
                start=self.start_time_override_utc or _day_start_utc(derived.flight_date),
                stop=self.stop_time_override_utc or _day_stop_utc(derived.flight_date),
                tag_filters={"sortie_number": context.flight_task.sortie_number},
                limit=self.point_limit,
            ),
        )
        return scoped_reader.fetch_points(locator)


@dataclass(frozen=True, slots=True)
class PhysiologyInfluxPointReader:
    """Reads physiology-side points for one sortie using source_sortie_id-derived context."""

    flight_task_reader: MySQLFlightTaskReader
    runner: InfluxQueryRunner
    collect_task_reader: MySQLCollectTaskReader | None = None
    measurement_names: tuple[str, ...] | None = None
    point_limit_per_measurement: int | None = None
    start_time_override_utc: datetime | None = None
    stop_time_override_utc: datetime | None = None

    def fetch_points(self, locator: SortieLocator) -> tuple[RawPoint, ...]:
        flight_task = self.flight_task_reader.fetch_by_locator(locator)
        collect_task = None
        if self.collect_task_reader is not None:
            collect_task = self.collect_task_reader.fetch_for_flight_task(flight_task)
        context = derive_physiology_query_context(flight_task, collect_task=collect_task)
        measurement_names = self.measurement_names or InfluxDistinctMeasurementReader(self.runner).fetch_measurements(
            bucket="physiological_input",
            start=context.start_time_utc,
            stop=context.stop_time_utc,
            tag_filters={"collect_task_id": str(context.collect_task_id)},
            tag_filters_any={"pilot_id": tuple(str(pilot_id) for pilot_id in context.pilot_ids)},
        )

        points: list[RawPoint] = []
        for measurement_name in measurement_names:
            scoped_reader = InfluxMeasurementPointReader(
                runner=self.runner,
                stream_kind=StreamKind.PHYSIOLOGY,
                query_builder=lambda _, measurement_name=measurement_name: InfluxQuerySpec(
                    bucket="physiological_input",
                    measurement=measurement_name,
                    start=self.start_time_override_utc or context.start_time_utc,
                    stop=self.stop_time_override_utc or context.stop_time_utc,
                    tag_filters={"collect_task_id": str(context.collect_task_id)},
                    tag_filters_any={"pilot_id": tuple(str(pilot_id) for pilot_id in context.pilot_ids)},
                    limit=self.point_limit_per_measurement,
                ),
            )
            points.extend(scoped_reader.fetch_points(locator))

        points.sort(key=lambda point: (point.timestamp, point.measurement))
        return tuple(points)


def _day_start_utc(day) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def _day_stop_utc(day) -> datetime:
    return _day_start_utc(day) + timedelta(days=1)
