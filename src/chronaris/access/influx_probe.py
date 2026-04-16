"""Lightweight coverage probes for Influx measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chronaris.access.influx_cli import (
    InfluxDistinctMeasurementReader,
    InfluxMeasurementPointReader,
    InfluxQueryRunner,
    InfluxQuerySpec,
)
from chronaris.schema.models import SortieLocator, StreamKind


@dataclass(frozen=True, slots=True)
class MeasurementTimeBounds:
    """Earliest and latest timestamps observed for one measurement."""

    measurement: str
    first_time: datetime
    last_time: datetime


def fetch_measurement_time_bounds(
    *,
    runner: InfluxQueryRunner,
    bucket: str,
    start: datetime,
    stop: datetime,
    tag_filters: dict[str, str],
    stream_kind: StreamKind,
    measurement_names: tuple[str, ...] | None = None,
) -> tuple[MeasurementTimeBounds, ...]:
    """Fetch first/last timestamps for scoped measurements without full data scans."""

    names = measurement_names or InfluxDistinctMeasurementReader(runner).fetch_measurements(
        bucket=bucket,
        start=start,
        stop=stop,
        tag_filters=tag_filters,
    )

    bounds: list[MeasurementTimeBounds] = []
    for measurement_name in names:
        first_time = _fetch_edge_time(
            runner=runner,
            bucket=bucket,
            measurement=measurement_name,
            start=start,
            stop=stop,
            tag_filters=tag_filters,
            stream_kind=stream_kind,
            descending=False,
        )
        last_time = _fetch_edge_time(
            runner=runner,
            bucket=bucket,
            measurement=measurement_name,
            start=start,
            stop=stop,
            tag_filters=tag_filters,
            stream_kind=stream_kind,
            descending=True,
        )
        if first_time is not None and last_time is not None:
            bounds.append(
                MeasurementTimeBounds(
                    measurement=measurement_name,
                    first_time=first_time,
                    last_time=last_time,
                )
            )

    bounds.sort(key=lambda item: (item.first_time, item.measurement))
    return tuple(bounds)


def _fetch_edge_time(
    *,
    runner: InfluxQueryRunner,
    bucket: str,
    measurement: str,
    start: datetime,
    stop: datetime,
    tag_filters: dict[str, str],
    stream_kind: StreamKind,
    descending: bool,
) -> datetime | None:
    point_reader = InfluxMeasurementPointReader(
        runner=runner,
        stream_kind=stream_kind,
        query_builder=lambda _: InfluxQuerySpec(
            bucket=bucket,
            measurement=measurement,
            start=start,
            stop=stop,
            tag_filters=tag_filters,
            limit=1,
            sort_by_time=True,
            time_desc=descending,
        ),
    )
    points = point_reader.fetch_points(SortieLocator(sortie_id=f"probe:{measurement}"))
    if not points:
        return None
    return points[0].timestamp
