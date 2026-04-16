"""Helpers for deriving physiology-query context from flight-task metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from chronaris.schema.real_bus import CollectTaskMetadata, FlightTaskMetadata


@dataclass(frozen=True, slots=True)
class PhysiologyQueryContext:
    """Minimal query context for physiology-side Influx access."""

    collect_task_id: int
    pilot_ids: tuple[int, ...]
    start_time_utc: datetime
    stop_time_utc: datetime


SOURCE_TIMEZONE = timezone(timedelta(hours=8))


def derive_physiology_query_context(
    flight_task: FlightTaskMetadata,
    *,
    collect_task: CollectTaskMetadata | None = None,
) -> PhysiologyQueryContext:
    """
    Derive physiology query keys from flight-task metadata.

    Current minimum-data convention:
    `source_sortie_id` is shaped like `collect_task_id-pilot_id[-pilot_id...]`.
    """

    collect_task_id, pilot_ids = _derive_ids(flight_task, collect_task=collect_task)
    start_time_utc, stop_time_utc = _derive_time_bounds(
        flight_task,
        collect_task=collect_task,
    )

    return PhysiologyQueryContext(
        collect_task_id=collect_task_id,
        pilot_ids=pilot_ids,
        start_time_utc=start_time_utc,
        stop_time_utc=stop_time_utc,
    )


def _derive_ids(
    flight_task: FlightTaskMetadata,
    *,
    collect_task: CollectTaskMetadata | None,
) -> tuple[int, tuple[int, ...]]:
    collect_task_id: int | None = None
    pilot_ids: tuple[int, ...] = ()

    if flight_task.source_sortie_id:
        segments = [segment for segment in flight_task.source_sortie_id.split("-") if segment]
        if segments:
            collect_task_id = int(segments[0])
            pilot_ids = tuple(int(segment) for segment in segments[1:] if segment.isdigit())

    if collect_task_id is None and collect_task is not None:
        collect_task_id = collect_task.collect_task_id

    if collect_task_id is None:
        raise ValueError(
            f"Neither source_sortie_id nor collect_task metadata could provide collect_task_id "
            f"for sortie {flight_task.sortie_number}."
        )

    if not pilot_ids:
        fallback_ids = tuple(
            pilot_id
            for pilot_id in (flight_task.up_pilot_id, flight_task.down_pilot_id)
            if pilot_id is not None
        )
        pilot_ids = fallback_ids

    if not pilot_ids:
        raise ValueError(
            f"No pilot ids could be derived for sortie {flight_task.sortie_number}."
        )

    return collect_task_id, pilot_ids


def _derive_flight_day(
    flight_task: FlightTaskMetadata,
    *,
    collect_task: CollectTaskMetadata | None,
):
    if flight_task.flight_date is not None:
        return flight_task.flight_date
    if collect_task is not None and collect_task.collect_date is not None:
        return collect_task.collect_date
    raise ValueError(
        f"flight_date is required to derive physiology query context for sortie "
        f"{flight_task.sortie_number}."
    )


def _derive_time_bounds(
    flight_task: FlightTaskMetadata,
    *,
    collect_task: CollectTaskMetadata | None,
) -> tuple[datetime, datetime]:
    if collect_task is not None and collect_task.collect_start_time and collect_task.collect_end_time:
        return (
            _to_utc(collect_task.collect_start_time),
            _to_utc(collect_task.collect_end_time),
        )

    if flight_task.car_start_time and flight_task.car_end_time:
        return (
            _to_utc(flight_task.car_start_time),
            _to_utc(flight_task.car_end_time),
        )

    day = _derive_flight_day(flight_task, collect_task=collect_task)
    start_time_utc = datetime.combine(day, time.min, tzinfo=timezone.utc)
    stop_time_utc = start_time_utc + timedelta(days=1)
    return start_time_utc, stop_time_utc


def _to_utc(naive_local: datetime) -> datetime:
    if naive_local.tzinfo is not None:
        return naive_local.astimezone(timezone.utc)
    return naive_local.replace(tzinfo=SOURCE_TIMEZONE).astimezone(timezone.utc)
