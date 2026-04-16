"""Tests for lightweight Influx coverage probes."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.influx_probe import fetch_measurement_time_bounds
from chronaris.schema.models import StreamKind


class FakeProbeRunner:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, flux: str):
        self.queries.append(flux)
        if 'distinct(column: "_measurement")' in flux:
            return (
                {"_value": "eeg"},
                {"_value": "spo2"},
            )
        if 'r._measurement == "eeg"' in flux and 'desc: false' in flux:
            return (
                {"result": None, "table": "0", "_time": "2025-10-05T01:10:00Z", "_field": "af3", "_measurement": "eeg", "_value": "0"},
            )
        if 'r._measurement == "eeg"' in flux and 'desc: true' in flux:
            return (
                {"result": None, "table": "0", "_time": "2025-10-05T01:20:00Z", "_field": "af3", "_measurement": "eeg", "_value": "1"},
            )
        if 'r._measurement == "spo2"' in flux and 'desc: false' in flux:
            return (
                {"result": None, "table": "0", "_time": "2025-10-05T01:11:00Z", "_field": "spo2", "_measurement": "spo2", "_value": "95"},
            )
        if 'r._measurement == "spo2"' in flux and 'desc: true' in flux:
            return (
                {"result": None, "table": "0", "_time": "2025-10-05T01:21:00Z", "_field": "spo2", "_measurement": "spo2", "_value": "96"},
            )
        return ()


class InfluxProbeTest(unittest.TestCase):
    def test_fetch_measurement_time_bounds_reads_distinct_and_edges(self) -> None:
        runner = FakeProbeRunner()
        bounds = fetch_measurement_time_bounds(
            runner=runner,
            bucket="physiological_input",
            start=datetime(2025, 10, 5, 1, 0, tzinfo=timezone.utc),
            stop=datetime(2025, 10, 5, 7, 0, tzinfo=timezone.utc),
            tag_filters={"collect_task_id": "2100448"},
            stream_kind=StreamKind.PHYSIOLOGY,
        )

        self.assertEqual(len(bounds), 2)
        self.assertEqual(bounds[0].measurement, "eeg")
        self.assertEqual(bounds[0].first_time.isoformat(), "2025-10-05T01:10:00+00:00")
        self.assertEqual(bounds[0].last_time.isoformat(), "2025-10-05T01:20:00+00:00")
        self.assertEqual(bounds[1].measurement, "spo2")


if __name__ == "__main__":
    unittest.main()
