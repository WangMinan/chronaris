"""Tests for MySQL-backed metadata readers."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.mysql_cli import parse_tsv_rowset
from chronaris.access.mysql_metadata import (
    MySQLCollectTaskReader,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLSortieMetadataReader,
)
from chronaris.schema.models import SortieLocator


class FakeRunner:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, sql: str):
        self.queries.append(sql)
        if "FROM access_rule_detail" in sql:
            return (
                {
                    "access_rule_id": "6000019510001",
                    "storage_data_analysis_id": "6000019110032",
                    "col_field": "code1001",
                    "col_name": "消息发生时间",
                    "col_type": "datetime",
                    "measurement": "bus_main",
                    "bucket": "real",
                },
                {
                    "access_rule_id": "6000019510001",
                    "storage_data_analysis_id": "6000019110032",
                    "col_field": "code1002",
                    "col_name": "速度",
                    "col_type": "double",
                    "measurement": "bus_main",
                    "bucket": "real",
                },
            )
        if "FROM storage_data_analysis_detail" in sql:
            return (
                {
                    "storage_data_analysis_id": "6000019110032",
                    "order_num": "1",
                    "col_name": "消息发生时间",
                    "col_field": "code1001",
                },
                {
                    "storage_data_analysis_id": "6000019110032",
                    "order_num": "2",
                    "col_name": "速度",
                    "col_field": "code1002",
                },
            )
        if "FROM storage_data_structure" in sql:
            return (
                {
                    "storage_data_analysis_id": "6000019110032",
                    "col_field": "code1002",
                    "col_name": "速度",
                    "parent_id": "0",
                    "is_leaf": "1",
                },
            )
        return ()

    def query_one(self, sql: str):
        self.queries.append(sql)
        if "FROM flight_task" in sql:
            return {
                "flight_task_id": "6000001",
                "flight_batch_id": "6000001",
                "sortie_number": "20251005_四01_ACT-4_云_J20_22#01",
                "batch_number": "20251005_四01",
                "subject": "ACT-4",
                "airplane_model": "J20",
                "airplane_number": "22",
                "fly_num": "1",
                "up_pilot_id": "10033",
                "down_pilot_id": None,
                "source_sortie_id": "2100448-10033",
                "car_star_time": "2025-10-05 09:00:00",
                "car_end_time": "2025-10-05 15:00:00",
                "fly_date": "2025-10-05",
            }
        if "FROM storage_data_analysis" in sql:
            return {
                "id": "6000019110032",
                "category": "BUS",
                "bucket": "real",
                "measurement": "bus_main",
                "sortie_number": "20251005_四01_ACT-4_云_J20_22#01",
                "md5_val": "abc123",
            }
        if "FROM collect_task" in sql:
            return {
                "id": "2100448",
                "coding": "TASK_20251005_01",
                "collect_date": "2025-10-05",
                "subject": "ACT-4",
                "collect_start_date": "2025-10-05 09:00:00",
                "collect_end_date": "2025-10-05 15:00:00",
            }
        return None


class MySQLMetadataTest(unittest.TestCase):
    def test_parse_tsv_rowset_handles_headers_and_nulls(self) -> None:
        rows = parse_tsv_rowset("id\tsortie_number\tfly_date\n6000001\tS-1\t2025-10-05\n6000002\tS-2\tNULL\n")
        self.assertEqual(rows[0]["id"], "6000001")
        self.assertEqual(rows[0]["fly_date"], "2025-10-05")
        self.assertIsNone(rows[1]["fly_date"])

    def test_sortie_metadata_reader_maps_joined_flight_task_row(self) -> None:
        reader = MySQLSortieMetadataReader(flight_task_reader=MySQLFlightTaskReader(runner=FakeRunner()))
        metadata = reader.fetch_metadata(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(metadata.flight_task_id, 6000001)
        self.assertEqual(metadata.flight_batch_id, 6000001)
        self.assertEqual(metadata.flight_date.isoformat(), "2025-10-05")
        self.assertEqual(metadata.pilot_code, "云")
        self.assertEqual(metadata.aircraft_tail, "22")
        self.assertEqual(metadata.extra["source_sortie_id"], "2100448-10033")

    def test_real_bus_context_reader_aggregates_business_metadata(self) -> None:
        runner = FakeRunner()
        context_reader = MySQLRealBusContextReader(
            runner=runner,
            flight_task_reader=MySQLFlightTaskReader(runner=runner),
        )

        context = context_reader.fetch_context(
            locator=SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"),
            access_rule_id=6000019510001,
            analysis_id=6000019110032,
        )

        self.assertEqual(context.flight_task.flight_date.isoformat(), "2025-10-05")
        self.assertEqual(context.analysis.category, "BUS")
        self.assertEqual(len(context.access_rule_details), 2)
        self.assertEqual(len(context.detail_list), 2)
        self.assertEqual(len(context.structure_list), 1)

    def test_collect_task_reader_can_resolve_from_flight_task(self) -> None:
        runner = FakeRunner()
        flight_task = MySQLFlightTaskReader(runner).fetch_by_locator(
            SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01")
        )
        collect_task = MySQLCollectTaskReader(runner).fetch_for_flight_task(flight_task)

        self.assertEqual(collect_task.collect_task_id, 2100448)
        self.assertEqual(collect_task.collect_date.isoformat(), "2025-10-05")
        self.assertEqual(collect_task.subject, "ACT-4")


if __name__ == "__main__":
    unittest.main()
