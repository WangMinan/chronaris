"""Merged unit tests."""

from __future__ import annotations

# ---- merged from test_access_temporal.py ----
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


# ---- merged from test_influx_cli.py ----
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


# ---- merged from test_influx_probe.py ----
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


# ---- merged from test_overlap_preview_loader.py ----
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
