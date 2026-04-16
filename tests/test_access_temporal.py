"""Tests for physiology and vehicle-side temporal parsing helpers."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.temporal import (
    attach_bus_timestamps,
    parse_bus_clock_time,
    parse_physiology_timestamp,
)


class TemporalParsingTest(unittest.TestCase):
    def test_parse_physiology_timestamp_preserves_microseconds(self) -> None:
        value = parse_physiology_timestamp("2025-10-05 10:00:00.123456")
        self.assertEqual(value.microsecond, 123456)

    def test_parse_bus_clock_time_preserves_milliseconds(self) -> None:
        value = parse_bus_clock_time("10:00:00.123")
        self.assertEqual(value.microsecond, 123000)

    def test_parse_bus_clock_time_rejects_more_than_milliseconds(self) -> None:
        with self.assertRaises(ValueError):
            parse_bus_clock_time("10:00:00.1234")

    def test_attach_bus_timestamps_reserves_cross_day_logic(self) -> None:
        attached = attach_bus_timestamps(
            date(2025, 10, 5),
            ["23:59:59.900", "00:00:00.100", "00:00:01.200"],
        )

        self.assertEqual(attached[0].timestamp.isoformat(), "2025-10-05T23:59:59.900000")
        self.assertEqual(attached[1].timestamp.isoformat(), "2025-10-06T00:00:00.100000")
        self.assertEqual(attached[1].day_offset, 1)
        self.assertEqual(attached[2].day_offset, 1)


if __name__ == "__main__":
    unittest.main()
