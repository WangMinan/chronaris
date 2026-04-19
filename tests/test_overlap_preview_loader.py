"""Tests for direct overlap-focused Influx preview readers."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.overlap_preview import DirectInfluxScopeConfig, ScopedInfluxPointReader
from chronaris.schema.models import SortieLocator, StreamKind


class FakeInfluxRunner:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, flux: str):
        self.queries.append(flux)
        return ()


def _utc(hour: int, minute: int, second: int) -> datetime:
    return datetime(2025, 10, 5, hour, minute, second, tzinfo=timezone.utc)


class OverlapPreviewLoaderTest(unittest.TestCase):
    def test_scoped_influx_point_reader_builds_fixed_measurement_queries(self) -> None:
        runner = FakeInfluxRunner()
        reader = ScopedInfluxPointReader(
            runner=runner,
            stream_kind=StreamKind.PHYSIOLOGY,
            scope=DirectInfluxScopeConfig(
                bucket="physiological_input",
                measurements=("eeg", "spo2"),
                start_time_utc=_utc(1, 35, 0),
                stop_time_utc=_utc(1, 38, 1),
                tag_filters={"collect_task_id": "2100448", "pilot_id": "10033"},
                point_limit_per_measurement=500,
            ),
        )

        reader.fetch_points(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(len(runner.queries), 2)
        self.assertIn('from(bucket:"physiological_input")', runner.queries[0])
        self.assertIn('r._measurement == "eeg"', runner.queries[0])
        self.assertIn('r.collect_task_id == "2100448"', runner.queries[0])
        self.assertIn('r.pilot_id == "10033"', runner.queries[0])
        self.assertIn("|> limit(n: 500)", runner.queries[0])
        self.assertIn('r._measurement == "spo2"', runner.queries[1])


if __name__ == "__main__":
    unittest.main()
