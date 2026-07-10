"""Deterministic local raw-point snapshot tests."""

from __future__ import annotations

from datetime import datetime, time, timezone

from chronaris.dataset.application_evaluation import (
    iter_raw_point_snapshot,
    write_raw_point_snapshot,
)
from chronaris.schema.models import RawPoint, StreamKind


def test_raw_point_snapshot_is_deterministic_and_round_trips(tmp_path) -> None:
    points = tuple(
        RawPoint(
            stream_kind=StreamKind.PHYSIOLOGY,
            measurement="eeg",
            timestamp=datetime(2025, 10, 2, 8, 35, second, tzinfo=timezone.utc),
            values={"af3": str(second), "valid": True},
            clock_time=time(8, 35, second),
            timestamp_precision_digits=6,
            tags={"pilot_id": "10033"},
            source="influx",
        )
        for second in (0, 1)
    )
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"
    first = write_raw_point_snapshot(
        first_path,
        points,
        snapshot_root=tmp_path,
        sortie_id="sortie-a",
        view_id="view-a",
        pilot_id=10033,
        expected_stream_kind=StreamKind.PHYSIOLOGY,
    )
    second = write_raw_point_snapshot(
        second_path,
        points,
        snapshot_root=tmp_path,
        sortie_id="sortie-a",
        view_id="view-a",
        pilot_id=10033,
        expected_stream_kind=StreamKind.PHYSIOLOGY,
    )

    assert first.sha256 == second.sha256
    assert first.point_count == 2
    restored = tuple(iter_raw_point_snapshot(first_path))
    assert restored == points
