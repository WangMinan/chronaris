"""Influx query helpers backed by the local influx CLI."""

from __future__ import annotations

import csv
import io
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Protocol

from chronaris.access.settings import InfluxSettings
from chronaris.access.temporal import parse_bus_clock_time
from chronaris.schema.models import RawPoint, SortieLocator, StreamKind


RowMapping = Mapping[str, str | None]


class InfluxQueryRunner(Protocol):
    """Minimal query runner protocol for Influx readers."""

    def query(self, flux: str) -> tuple[RowMapping, ...]:
        """Run a Flux query and return raw row mappings."""


@dataclass(frozen=True, slots=True)
class InfluxQuerySpec:
    """A minimal Flux query specification for one measurement."""

    bucket: str
    measurement: str | None
    start: datetime
    stop: datetime
    tag_filters: Mapping[str, str] = field(default_factory=dict)
    tag_filters_any: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    limit: int | None = None
    sort_by_time: bool = False
    time_desc: bool = False


@dataclass(frozen=True, slots=True)
class InfluxCliRunner:
    """Runs Flux queries via the local influx executable."""

    settings: InfluxSettings
    influx_binary: str = "influx"

    def query(self, flux: str) -> tuple[RowMapping, ...]:
        completed = subprocess.run(
            [
                self.influx_binary,
                "query",
                "--raw",
                flux,
            ],
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=self._build_env(),
        )
        return parse_influx_annotated_csv(completed.stdout)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["INFLUX_HOST"] = self.settings.url
        env["INFLUX_ORG"] = self.settings.org
        env["INFLUX_TOKEN"] = self.settings.token
        return env


@dataclass(frozen=True, slots=True)
class InfluxMeasurementPointReader:
    """Reads grouped points from one Influx measurement."""

    runner: InfluxQueryRunner
    stream_kind: StreamKind
    query_builder: Callable[[SortieLocator], InfluxQuerySpec]

    def fetch_points(self, locator: SortieLocator) -> tuple[RawPoint, ...]:
        spec = self.query_builder(locator)
        rows = self.runner.query(build_flux_query(spec))
        return rows_to_raw_points(rows, self.stream_kind)


@dataclass(frozen=True, slots=True)
class InfluxDistinctMeasurementReader:
    """Lists distinct measurements in one bucket for a scoped tag/time range."""

    runner: InfluxQueryRunner

    def fetch_measurements(
        self,
        *,
        bucket: str,
        start: datetime,
        stop: datetime,
        tag_filters: Mapping[str, str] | None = None,
        tag_filters_any: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        rows = self.runner.query(
            build_distinct_measurements_query(
                bucket=bucket,
                start=start,
                stop=stop,
                tag_filters=tag_filters or {},
                tag_filters_any=tag_filters_any or {},
            )
        )
        names = [row["_value"] for row in rows if row.get("_value")]
        return tuple(dict.fromkeys(names))


def build_flux_query(spec: InfluxQuerySpec) -> str:
    """Build a simple Flux query for one measurement and tag filter set."""

    filters: list[str] = []
    if spec.measurement is not None:
        filters.append(f'r._measurement == "{_escape_flux_string(spec.measurement)}"')
    for key, value in spec.tag_filters.items():
        filters.append(f'r.{key} == "{_escape_flux_string(value)}"')
    for key, values in spec.tag_filters_any.items():
        normalized_values = tuple(dict.fromkeys(value for value in values if value))
        if not normalized_values:
            continue
        if len(normalized_values) == 1:
            filters.append(f'r.{key} == "{_escape_flux_string(normalized_values[0])}"')
            continue
        filters.append(
            "("
            + " or ".join(
                f'r.{key} == "{_escape_flux_string(value)}"'
                for value in normalized_values
            )
            + ")"
        )

    flux = (
        f'from(bucket:"{_escape_flux_string(spec.bucket)}")'
        f' |> range(start: {spec.start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")},'
        f' stop: {spec.stop.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})'
        f' |> filter(fn: (r) => {" and ".join(filters)})'
    )
    if spec.sort_by_time:
        flux += f' |> sort(columns: ["_time"], desc: {"true" if spec.time_desc else "false"})'
    if spec.limit is not None:
        flux += f" |> limit(n: {spec.limit})"
    return flux


def build_distinct_measurements_query(
    *,
    bucket: str,
    start: datetime,
    stop: datetime,
    tag_filters: Mapping[str, str] | None = None,
    tag_filters_any: Mapping[str, tuple[str, ...]] | None = None,
) -> str:
    """Build a Flux query that lists distinct measurements for a scoped tag range."""

    filters = [
        f'r.{key} == "{_escape_flux_string(value)}"'
        for key, value in (tag_filters or {}).items()
    ]
    for key, values in (tag_filters_any or {}).items():
        normalized_values = tuple(dict.fromkeys(value for value in values if value))
        if not normalized_values:
            continue
        if len(normalized_values) == 1:
            filters.append(f'r.{key} == "{_escape_flux_string(normalized_values[0])}"')
            continue
        filters.append(
            "("
            + " or ".join(
                f'r.{key} == "{_escape_flux_string(value)}"'
                for value in normalized_values
            )
            + ")"
        )
    filter_clause = ""
    if filters:
        filter_clause = f' |> filter(fn: (r) => {" and ".join(filters)})'
    return (
        f'from(bucket:"{_escape_flux_string(bucket)}")'
        f' |> range(start: {start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")},'
        f' stop: {stop.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")})'
        f"{filter_clause}"
        ' |> keep(columns: ["_measurement"])'
        " |> group()"
        ' |> distinct(column: "_measurement")'
    )


def parse_influx_annotated_csv(content: str) -> tuple[RowMapping, ...]:
    """Parse annotated CSV returned by `influx query --raw`."""

    if not content.strip():
        return ()

    result: list[RowMapping] = []
    current_header: list[str] | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        row_values = next(csv.reader(io.StringIO(raw_line)))

        if _is_influx_header_row(row_values):
            current_header = row_values
            continue

        if current_header is None:
            continue

        padded_values = row_values + [""] * max(0, len(current_header) - len(row_values))
        row = {
            key: _normalize_influx_value(value)
            for key, value in zip(current_header, padded_values)
        }
        result.append(row)
    return tuple(result)


def rows_to_raw_points(rows: tuple[RowMapping, ...], stream_kind: StreamKind) -> tuple[RawPoint, ...]:
    """Group long-form Influx rows back into point-wise RawPoint objects."""

    grouped: dict[tuple[str, str, tuple[tuple[str, str], ...]], dict[str, object]] = {}

    for row in rows:
        measurement = _require_value(row, "_measurement")
        timestamp = _parse_influx_time(_require_value(row, "_time"))
        field_name = _require_value(row, "_field")
        field_value = row.get("_value")
        tags = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "result",
                "table",
                "_start",
                "_stop",
                "_time",
                "_value",
                "_field",
                "_measurement",
            }
            and value is not None
        }
        group_key = (
            measurement,
            timestamp.isoformat(),
            tuple(sorted(tags.items())),
        )
        payload = grouped.setdefault(
            group_key,
            {
                "measurement": measurement,
                "timestamp": timestamp,
                "values": {},
                "tags": tags,
            },
        )
        payload["values"][field_name] = field_value

    points: list[RawPoint] = []
    for payload in grouped.values():
        values = dict(payload["values"])
        timestamp = payload["timestamp"]
        clock_time = None
        precision_digits = 6 if stream_kind == StreamKind.PHYSIOLOGY else 3
        if stream_kind == StreamKind.VEHICLE and isinstance(values.get("code1001"), str):
            clock_time = parse_bus_clock_time(values["code1001"])

        points.append(
            RawPoint(
                stream_kind=stream_kind,
                measurement=payload["measurement"],
                timestamp=timestamp,
                values=values,
                clock_time=clock_time,
                timestamp_precision_digits=precision_digits,
                tags=payload["tags"],
                source="influx",
            )
        )

    points.sort(key=lambda point: (point.timestamp, point.measurement))
    return tuple(points)


def _parse_influx_time(raw_value: str) -> datetime:
    return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))


def _escape_flux_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _normalize_influx_value(value: str) -> str | None:
    if value == "":
        return None
    return value


def _require_value(row: RowMapping, key: str) -> str:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing required Influx column '{key}'.")
    return value


def _is_influx_header_row(values: list[str]) -> bool:
    return "result" in values and "table" in values
