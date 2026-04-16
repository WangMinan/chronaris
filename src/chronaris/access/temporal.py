"""Temporal parsing helpers for physiology and vehicle-side raw data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Sequence


@dataclass(frozen=True, slots=True)
class AttachedClockTime:
    """A time-of-day value lifted into a full datetime with day offset."""

    clock_time: time
    timestamp: datetime
    day_offset: int


def parse_physiology_timestamp(raw_value: str) -> datetime:
    """Parse a physiology timestamp and preserve up to microsecond precision."""

    value = raw_value.strip()
    _validate_fractional_precision(value, max_digits=6, label="Physiology timestamp")
    return _parse_datetime(value, ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"))


def parse_bus_clock_time(raw_value: str) -> time:
    """Parse a vehicle time-of-day value and preserve up to millisecond precision."""

    value = raw_value.strip()
    _validate_fractional_precision(value, max_digits=3, label="Bus clock time")
    return _parse_datetime(value, ("%H:%M:%S.%f", "%H:%M:%S")).time()


def attach_bus_timestamps(base_date: date, raw_clock_values: Sequence[str | time]) -> tuple[AttachedClockTime, ...]:
    """Attach a flight date to vehicle clock values and reserve cross-day handling."""

    parsed_times = tuple(
        parse_bus_clock_time(value) if isinstance(value, str) else value for value in raw_clock_values
    )
    return attach_cross_day_times(base_date, parsed_times)


def attach_cross_day_times(base_date: date, clock_times: Sequence[time]) -> tuple[AttachedClockTime, ...]:
    """
    Lift time-of-day values into full datetimes.

    This mirrors the semantics of the original TimeSequenceProcessor:
    when the current reference time is earlier than the previous one,
    the date offset increases by one day.
    """

    day_offset = 0
    prev_time: time | None = None
    attached_times: list[AttachedClockTime] = []

    for clock_time in clock_times:
        if prev_time is not None and clock_time < prev_time:
            day_offset += 1

        attached_times.append(
            AttachedClockTime(
                clock_time=clock_time,
                timestamp=datetime.combine(base_date + timedelta(days=day_offset), clock_time),
                day_offset=day_offset,
            )
        )
        prev_time = clock_time

    return tuple(attached_times)


def _parse_datetime(value: str, formats: Sequence[str]) -> datetime:
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Failed to parse datetime value: {value}")


def _validate_fractional_precision(value: str, *, max_digits: int, label: str) -> None:
    if "." not in value:
        return

    fractional = value.rsplit(".", maxsplit=1)[1]
    if len(fractional) > max_digits:
        raise ValueError(f"{label} cannot exceed {max_digits} fractional digits: {value}")
