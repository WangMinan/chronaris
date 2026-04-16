"""Tests for deriving physiology query context from flight-task metadata."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.physiology_context import derive_physiology_query_context
from chronaris.schema.real_bus import CollectTaskMetadata, FlightTaskMetadata


class PhysiologyContextTest(unittest.TestCase):
    def test_derive_context_from_source_sortie_id(self) -> None:
        context = derive_physiology_query_context(
            FlightTaskMetadata(
                flight_task_id=6000001,
                sortie_number="20251005_四01_ACT-4_云_J20_22#01",
                source_sortie_id="2100448-10033",
                flight_date=date(2025, 10, 5),
                up_pilot_id=10033,
            )
        )

        self.assertEqual(context.collect_task_id, 2100448)
        self.assertEqual(context.pilot_ids, (10033,))
        self.assertEqual(context.start_time_utc.isoformat(), "2025-10-05T00:00:00+00:00")
        self.assertEqual(context.stop_time_utc.isoformat(), "2025-10-06T00:00:00+00:00")

    def test_derive_context_can_fallback_to_up_and_down_pilot_ids(self) -> None:
        context = derive_physiology_query_context(
            FlightTaskMetadata(
                flight_task_id=6000002,
                sortie_number="20251002_单01_ACT-8_翼云_J16_12#01",
                source_sortie_id="2100450",
                flight_date=date(2025, 10, 2),
                up_pilot_id=10035,
                down_pilot_id=10033,
            )
        )

        self.assertEqual(context.collect_task_id, 2100450)
        self.assertEqual(context.pilot_ids, (10035, 10033))

    def test_derive_context_can_fallback_to_collect_task_when_source_sortie_id_missing(self) -> None:
        context = derive_physiology_query_context(
            FlightTaskMetadata(
                flight_task_id=6000001,
                sortie_number="20251005_四01_ACT-4_云_J20_22#01",
                flight_date=date(2025, 10, 5),
                up_pilot_id=10033,
            ),
            collect_task=CollectTaskMetadata(
                collect_task_id=2100448,
                collect_date=date(2025, 10, 5),
            ),
        )

        self.assertEqual(context.collect_task_id, 2100448)
        self.assertEqual(context.pilot_ids, (10033,))

    def test_derive_context_prefers_collect_task_time_bounds(self) -> None:
        context = derive_physiology_query_context(
            FlightTaskMetadata(
                flight_task_id=6000001,
                sortie_number="20251005_四01_ACT-4_云_J20_22#01",
                flight_date=date(2025, 10, 5),
                up_pilot_id=10033,
            ),
            collect_task=CollectTaskMetadata(
                collect_task_id=2100448,
                collect_date=date(2025, 10, 5),
                collect_start_time=datetime(2025, 10, 5, 9, 0, 0),
                collect_end_time=datetime(2025, 10, 5, 15, 0, 0),
            ),
        )

        self.assertEqual(context.start_time_utc.isoformat(), "2025-10-05T01:00:00+00:00")
        self.assertEqual(context.stop_time_utc.isoformat(), "2025-10-05T07:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
