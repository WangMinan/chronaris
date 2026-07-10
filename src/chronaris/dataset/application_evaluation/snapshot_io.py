"""Deterministic raw-point snapshot serialization."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from datetime import datetime, time
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from chronaris.dataset.application_evaluation.snapshot_contracts import SnapshotFileRecord
from chronaris.schema.models import RawPoint, StreamKind


def write_raw_point_snapshot(
    path: str | Path,
    points: Iterable[RawPoint],
    *,
    snapshot_root: str | Path,
    sortie_id: str,
    view_id: str | None,
    pilot_id: int | None,
    expected_stream_kind: StreamKind,
) -> SnapshotFileRecord:
    """Write one deterministic gzip JSONL file and return its compact lineage."""

    resolved_path = Path(path)
    resolved_root = Path(snapshot_root)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_name(resolved_path.name + ".tmp")
    measurement_counts: dict[str, int] = {}
    field_names: set[str] = set()
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    point_count = 0
    previous_sort_key: tuple[datetime, str] | None = None

    try:
        with temporary_path.open("wb") as raw_stream:
            with gzip.GzipFile(filename="", fileobj=raw_stream, mode="wb", mtime=0) as gzip_stream:
                with io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="\n") as text_stream:
                    for point in points:
                        if point.stream_kind != expected_stream_kind:
                            raise ValueError(
                                f"snapshot stream mismatch: expected {expected_stream_kind}, "
                                f"got {point.stream_kind}"
                            )
                        sort_key = (point.timestamp, point.measurement)
                        if previous_sort_key is not None and sort_key < previous_sort_key:
                            raise ValueError("raw points must be sorted by timestamp and measurement")
                        previous_sort_key = sort_key
                        payload = raw_point_to_dict(point)
                        text_stream.write(
                            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                            + "\n"
                        )
                        point_count += 1
                        measurement_counts[point.measurement] = (
                            measurement_counts.get(point.measurement, 0) + 1
                        )
                        field_names.update(f"{point.measurement}.{name}" for name in point.values)
                        first_timestamp = first_timestamp or point.timestamp
                        last_timestamp = point.timestamp
        temporary_path.replace(resolved_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return SnapshotFileRecord(
        stream_kind=expected_stream_kind.value,
        sortie_id=sortie_id,
        view_id=view_id,
        pilot_id=pilot_id,
        relative_path=resolved_path.relative_to(resolved_root).as_posix(),
        sha256=sha256_file(resolved_path),
        byte_count=resolved_path.stat().st_size,
        point_count=point_count,
        measurement_counts=dict(sorted(measurement_counts.items())),
        field_count=len(field_names),
        first_timestamp_utc=None if first_timestamp is None else first_timestamp.isoformat(),
        last_timestamp_utc=None if last_timestamp is None else last_timestamp.isoformat(),
    )


def iter_raw_point_snapshot(path: str | Path) -> Iterator[RawPoint]:
    """Read a previously written snapshot without loading the full file."""

    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield raw_point_from_dict(json.loads(line))


def raw_point_to_dict(point: RawPoint) -> dict[str, object]:
    return {
        "stream_kind": point.stream_kind.value,
        "measurement": point.measurement,
        "timestamp_utc": point.timestamp.isoformat(),
        "clock_time": None if point.clock_time is None else point.clock_time.isoformat(),
        "timestamp_precision_digits": point.timestamp_precision_digits,
        "values": dict(point.values),
        "tags": dict(point.tags),
        "source": point.source,
    }


def raw_point_from_dict(payload: Mapping[str, object]) -> RawPoint:
    raw_clock_time = payload.get("clock_time")
    return RawPoint(
        stream_kind=StreamKind(str(payload["stream_kind"])),
        measurement=str(payload["measurement"]),
        timestamp=datetime.fromisoformat(str(payload["timestamp_utc"]).replace("Z", "+00:00")),
        values=dict(payload.get("values") or {}),
        clock_time=None if raw_clock_time is None else time.fromisoformat(str(raw_clock_time)),
        timestamp_precision_digits=(
            None
            if payload.get("timestamp_precision_digits") is None
            else int(payload["timestamp_precision_digits"])
        ),
        tags={str(key): str(value) for key, value in dict(payload.get("tags") or {}).items()},
        source=None if payload.get("source") is None else str(payload["source"]),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
