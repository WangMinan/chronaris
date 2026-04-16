"""Tests for Influx CLI parsing and point grouping."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.influx_cli import (
    build_distinct_measurements_query,
    InfluxQuerySpec,
    build_flux_query,
    parse_influx_annotated_csv,
    rows_to_raw_points,
)
from chronaris.schema.models import StreamKind


RAW_INFLUX_CSV = """#group,false,false,true,true,false,false,true,true,true,true
#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,dateTime:RFC3339,string,string,string,string,string
#default,_result,,,,,,,,,
,result,table,_start,_stop,_time,_value,_field,_measurement,import_id,sortie_number
,,0,2025-10-05T00:00:00Z,2025-10-06T00:00:00Z,2025-10-05T01:35:00Z,09:35:00.000,code1001,BUS6000019110020,1,20251005_四01_ACT-4_云_J20_22#01
,,1,2025-10-05T00:00:00Z,2025-10-06T00:00:00Z,2025-10-05T01:35:00Z,3422130991200,code1002,BUS6000019110020,1,20251005_四01_ACT-4_云_J20_22#01
,,2,2025-10-05T00:00:00Z,2025-10-06T00:00:00Z,2025-10-05T01:35:00.255Z,09:35:00.255,code1001,BUS6000019110020,1,20251005_四01_ACT-4_云_J20_22#01
,,3,2025-10-05T00:00:00Z,2025-10-06T00:00:00Z,2025-10-05T01:35:00.255Z,3422381216000,code1002,BUS6000019110020,1,20251005_四01_ACT-4_云_J20_22#01
"""

RAW_DISTINCT_MEASUREMENTS_CSV = """#group,false,false,false
#datatype,string,long,string
#default,_result,,
,result,table,_value
,,0,eeg
,,0,spo2
"""


class InfluxCliTest(unittest.TestCase):
    def test_build_flux_query_includes_measurement_and_tags(self) -> None:
        query = build_flux_query(
            InfluxQuerySpec(
                bucket="bus",
                measurement="BUS6000019110020",
                start=datetime(2025, 10, 5, 0, 0, tzinfo=timezone.utc),
                stop=datetime(2025, 10, 6, 0, 0, tzinfo=timezone.utc),
                tag_filters={"sortie_number": "20251005_四01_ACT-4_云_J20_22#01"},
                limit=10,
            )
        )

        self.assertIn('from(bucket:"bus")', query)
        self.assertIn('r._measurement == "BUS6000019110020"', query)
        self.assertIn('r.sortie_number == "20251005_四01_ACT-4_云_J20_22#01"', query)
        self.assertIn("|> limit(n: 10)", query)

    def test_parse_influx_annotated_csv_ignores_annotation_lines(self) -> None:
        rows = parse_influx_annotated_csv(RAW_INFLUX_CSV)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["_field"], "code1001")
        self.assertEqual(rows[1]["_value"], "3422130991200")

    def test_rows_to_raw_points_groups_fields_by_timestamp(self) -> None:
        rows = parse_influx_annotated_csv(RAW_INFLUX_CSV)
        points = rows_to_raw_points(rows, StreamKind.VEHICLE)

        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].measurement, "BUS6000019110020")
        self.assertEqual(points[0].timestamp.isoformat(), "2025-10-05T01:35:00+00:00")
        self.assertEqual(points[0].values["code1001"], "09:35:00.000")
        self.assertEqual(points[0].values["code1002"], "3422130991200")
        self.assertEqual(points[0].clock_time.isoformat(), "09:35:00")
        self.assertEqual(points[0].timestamp_precision_digits, 3)

    def test_parse_influx_annotated_csv_supports_distinct_measurement_queries(self) -> None:
        rows = parse_influx_annotated_csv(RAW_DISTINCT_MEASUREMENTS_CSV)
        self.assertEqual(rows[0]["_value"], "eeg")
        self.assertEqual(rows[1]["_value"], "spo2")

    def test_build_distinct_measurements_query_includes_tag_filters(self) -> None:
        query = build_distinct_measurements_query(
            bucket="physiological_input",
            start=datetime(2025, 10, 5, 1, 0, tzinfo=timezone.utc),
            stop=datetime(2025, 10, 5, 7, 0, tzinfo=timezone.utc),
            tag_filters={"collect_task_id": "2100448"},
        )

        self.assertIn('from(bucket:"physiological_input")', query)
        self.assertIn('r.collect_task_id == "2100448"', query)
        self.assertIn('distinct(column: "_measurement")', query)


if __name__ == "__main__":
    unittest.main()
