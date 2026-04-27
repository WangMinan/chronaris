"""Live providers for partial-data vehicle-only assets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from chronaris.access.influx_cli import (
    InfluxMeasurementPointReader,
    InfluxQueryRunner,
    InfluxQuerySpec,
)
from chronaris.access.mysql_metadata import (
    MySQLRealBusContextReader,
    MySQLStorageAnalysisReader,
)
from chronaris.pipelines.partial_data_contracts import (
    PartialDataEntry,
    PartialMeasurementMetadata,
    PartialPointChunk,
    concrete_measurements,
    parse_required_utc,
)
from chronaris.schema.models import RawPoint, SortieLocator, StreamKind
from chronaris.schema.real_bus import RealBusContext


@dataclass(frozen=True, slots=True)
class MySQLPartialVehicleMetadataProvider:
    """Resolve RealBus metadata for vehicle-only partial-data entries."""

    storage_analysis_reader: MySQLStorageAnalysisReader
    real_bus_context_reader: MySQLRealBusContextReader
    access_rule_id: int = 6000019510066
    vehicle_category: str = "BUS"

    def __call__(self, entry: PartialDataEntry) -> Mapping[str, PartialMeasurementMetadata]:
        analyses = self.storage_analysis_reader.list_for_sortie(
            SortieLocator(sortie_id=entry.sortie_id),
            category=self.vehicle_category,
        )
        analysis_by_measurement = {
            analysis.measurement: analysis
            for analysis in analyses
            if analysis.measurement
        }
        result: dict[str, PartialMeasurementMetadata] = {}
        for measurement in concrete_measurements(entry):
            analysis = analysis_by_measurement.get(measurement)
            if analysis is None:
                result[measurement] = PartialMeasurementMetadata(
                    measurement=measurement,
                    status="skipped",
                    reason="storage_analysis_missing",
                )
                continue
            try:
                context = self.real_bus_context_reader.fetch_context(
                    locator=SortieLocator(sortie_id=entry.sortie_id),
                    access_rule_id=self.access_rule_id,
                    analysis_id=analysis.analysis_id,
                )
            except Exception as exc:  # pragma: no cover - live metadata fallback.
                result[measurement] = PartialMeasurementMetadata(
                    measurement=measurement,
                    status="skipped",
                    reason=f"context_error: {exc}",
                )
                continue

            field_names = _resolve_real_bus_field_names(context)
            if not field_names:
                result[measurement] = PartialMeasurementMetadata(
                    measurement=measurement,
                    status="skipped",
                    reason="no_realbus_field_mapping",
                )
                continue
            result[measurement] = PartialMeasurementMetadata(
                measurement=measurement,
                status="available",
                field_names=field_names,
            )
        return result


@dataclass(frozen=True, slots=True)
class InfluxPartialVehiclePointProvider:
    """Read vehicle-only partial-data entries from Influx using bounded chunks."""

    runner: InfluxQueryRunner
    point_limit_per_measurement: int | None = None
    chunk_duration_seconds: int = 300
    window_duration_ms: int = 5_000
    window_limit_per_field: int = 32

    def __post_init__(self) -> None:
        if self.point_limit_per_measurement is not None and self.point_limit_per_measurement <= 0:
            raise ValueError("point_limit_per_measurement must be positive when provided.")
        if self.chunk_duration_seconds <= 0:
            raise ValueError("chunk_duration_seconds must be positive.")
        if self.window_duration_ms <= 0:
            raise ValueError("window_duration_ms must be positive.")
        if self.window_limit_per_field <= 0:
            raise ValueError("window_limit_per_field must be positive.")

    def __call__(self, entry: PartialDataEntry) -> tuple[RawPoint, ...]:
        points: list[RawPoint] = []
        for chunk in self.iter_chunks(entry):
            points.extend(chunk.points)
        points.sort(key=lambda point: (point.timestamp, point.measurement))
        return tuple(points)

    def iter_chunks(self, entry: PartialDataEntry) -> Iterable[PartialPointChunk]:
        if entry.stream_kind != StreamKind.VEHICLE.value:
            return ()
        bucket = entry.bucket
        if not bucket:
            raise ValueError("vehicle-only partial entry requires bucket.")
        measurements = concrete_measurements(entry)
        if not measurements:
            raise ValueError("vehicle-only partial entry requires concrete measurement_family.")
        start = parse_required_utc(entry.time_range.get("start_utc"), "time_range.start_utc")
        stop = parse_required_utc(entry.time_range.get("stop_utc"), "time_range.stop_utc")
        if stop <= start:
            raise ValueError("time_range.stop_utc must be greater than start_utc.")

        return self._iter_influx_chunks(
            entry=entry,
            bucket=bucket,
            measurements=measurements,
            start=start,
            stop=stop,
        )

    def _iter_influx_chunks(
        self,
        *,
        entry: PartialDataEntry,
        bucket: str,
        measurements: tuple[str, ...],
        start: datetime,
        stop: datetime,
    ) -> Iterable[PartialPointChunk]:
        chunk_delta = timedelta(seconds=self.chunk_duration_seconds)
        chunk_start = start
        while chunk_start < stop:
            chunk_stop = min(chunk_start + chunk_delta, stop)
            reader = InfluxMeasurementPointReader(
                runner=self.runner,
                stream_kind=StreamKind.VEHICLE,
                query_builder=lambda _, chunk_start=chunk_start, chunk_stop=chunk_stop: InfluxQuerySpec(
                    bucket=bucket,
                    measurement=measurements[0] if len(measurements) == 1 else None,
                    measurement_any=() if len(measurements) == 1 else measurements,
                    start=chunk_start,
                    stop=chunk_stop,
                    tag_filters=entry.tag_filters,
                    limit=self.point_limit_per_measurement,
                    sort_by_time=True,
                    window_every=_format_flux_duration_ms(self.window_duration_ms),
                    window_limit=self.window_limit_per_field,
                ),
            )
            yield PartialPointChunk(
                start_utc=chunk_start,
                stop_utc=chunk_stop,
                points=reader.fetch_points(SortieLocator(sortie_id=entry.sortie_id)),
            )
            chunk_start = chunk_stop


def _resolve_real_bus_field_names(context: RealBusContext) -> tuple[str, ...]:
    seen: set[str] = set()
    fields: list[str] = []
    for detail in context.detail_list:
        if detail.col_field and detail.col_field not in seen:
            seen.add(detail.col_field)
            fields.append(detail.col_field)
    for structure in context.structure_list:
        if structure.col_field and structure.col_field not in seen:
            seen.add(structure.col_field)
            fields.append(structure.col_field)
    for access_detail in context.access_rule_details:
        if access_detail.col_field and access_detail.col_field not in seen:
            seen.add(access_detail.col_field)
            fields.append(access_detail.col_field)
    return tuple(fields)


def _format_flux_duration_ms(duration_ms: int) -> str:
    if duration_ms % 1000 == 0:
        return f"{duration_ms // 1000}s"
    return f"{duration_ms}ms"
