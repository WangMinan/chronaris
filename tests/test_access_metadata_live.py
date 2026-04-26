"""Merged unit tests."""

from __future__ import annotations

# ---- merged from test_live_factory.py ----
import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.live_factory import StageBLiveLoaderConfig


class LiveFactoryConfigTest(unittest.TestCase):
    def test_stage_b_live_loader_config_defaults_are_sane(self) -> None:
        config = StageBLiveLoaderConfig(
            java_properties_path="human-machine.properties",
            access_rule_id=6000019510066,
            analysis_id=6000019110020,
        )

        self.assertEqual(config.mysql_database, "rjgx_backend")
        self.assertIsNone(config.physiology_measurements)
        self.assertIsNone(config.physiology_point_limit_per_measurement)
        self.assertIsNone(config.bus_point_limit)
        self.assertIsNone(config.start_time_override_utc)
        self.assertIsNone(config.stop_time_override_utc)

    def test_stage_b_live_loader_config_accepts_time_overrides(self) -> None:
        config = StageBLiveLoaderConfig(
            java_properties_path="human-machine.properties",
            access_rule_id=6000019510066,
            analysis_id=6000019110020,
            start_time_override_utc=datetime(2025, 10, 5, 1, 35, 0, tzinfo=timezone.utc),
            stop_time_override_utc=datetime(2025, 10, 5, 1, 38, 1, tzinfo=timezone.utc),
        )

        self.assertEqual(config.start_time_override_utc.isoformat(), "2025-10-05T01:35:00+00:00")
        self.assertEqual(config.stop_time_override_utc.isoformat(), "2025-10-05T01:38:01+00:00")



from datetime import datetime, timezone


# ---- merged from test_live_readers.py ----
import sys
from datetime import date
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.live_readers import PhysiologyInfluxPointReader, RealBusInfluxPointReader
from chronaris.schema.models import SortieLocator
from chronaris.schema.real_bus import (
    AccessRuleDetail,
    CollectTaskMetadata,
    FlightTaskMetadata,
    RealBusContext,
    StorageAnalysis,
    StorageAnalysisDetail,
    StorageStructure,
)


class FakeInfluxRunner:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, flux: str):
        self.queries.append(flux)
        if 'distinct(column: "_measurement")' in flux:
            return (
                {"_value": "eeg"},
                {"_value": "spo2"},
            )
        return ()


class FakeFlightTaskReader:
    def fetch_by_locator(self, locator: SortieLocator) -> FlightTaskMetadata:
        return FlightTaskMetadata(
            flight_task_id=6000001,
            sortie_number=locator.sortie_id,
            flight_date=date(2025, 10, 5),
            up_pilot_id=10033,
        )


class FakeCollectTaskReader:
    def fetch_for_flight_task(self, flight_task: FlightTaskMetadata) -> CollectTaskMetadata:
        return CollectTaskMetadata(
            collect_task_id=2100448,
            collect_date=date(2025, 10, 5),
            subject="ACT-4",
        )


class FakeDualPilotFlightTaskReader:
    def fetch_by_locator(self, locator: SortieLocator) -> FlightTaskMetadata:
        return FlightTaskMetadata(
            flight_task_id=6000002,
            sortie_number=locator.sortie_id,
            flight_date=date(2025, 10, 2),
            up_pilot_id=10035,
            down_pilot_id=10033,
        )


class FakeRealBusContextReader:
    def fetch_context(self, *, locator, flight_task_id=None, access_rule_id, analysis_id):
        return RealBusContext(
            flight_task=FlightTaskMetadata(
                flight_task_id=6000001,
                sortie_number=locator.sortie_id,
                flight_date=date(2025, 10, 5),
            ),
            analysis=StorageAnalysis(
                analysis_id=analysis_id,
                category="BUS",
                bucket="bus",
                measurement="BUS6000019110020",
                sortie_number=locator.sortie_id,
            ),
            access_rule_details=(
                AccessRuleDetail(
                    access_rule_id=access_rule_id,
                    storage_data_analysis_id=analysis_id,
                    col_field="code1001",
                ),
            ),
            detail_list=(
                StorageAnalysisDetail(
                    analysis_id=analysis_id,
                    order_num=1,
                    col_name="消息发生时间",
                    col_field="code1001",
                ),
            ),
            structure_list=(
                StorageStructure(
                    analysis_id=analysis_id,
                    col_field="code1001",
                    col_name="消息发生时间",
                ),
            ),
        )


class LiveReadersTest(unittest.TestCase):
    def test_physiology_reader_builds_collect_task_filter(self) -> None:
        runner = FakeInfluxRunner()
        reader = PhysiologyInfluxPointReader(
            flight_task_reader=FakeFlightTaskReader(),
            collect_task_reader=FakeCollectTaskReader(),
            runner=runner,
        )

        reader.fetch_points(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(len(runner.queries), 3)
        self.assertIn('distinct(column: "_measurement")', runner.queries[0])
        self.assertIn('r.collect_task_id == "2100448"', runner.queries[0])
        self.assertIn('r.pilot_id == "10033"', runner.queries[0])
        self.assertIn('r._measurement == "eeg"', runner.queries[1])
        self.assertIn('r._measurement == "spo2"', runner.queries[2])

    def test_physiology_reader_can_filter_multiple_pilot_ids(self) -> None:
        runner = FakeInfluxRunner()
        reader = PhysiologyInfluxPointReader(
            flight_task_reader=FakeDualPilotFlightTaskReader(),
            collect_task_reader=FakeCollectTaskReader(),
            runner=runner,
        )

        reader.fetch_points(SortieLocator(sortie_id="20251002_单01_ACT-8_翼云_J16_12#01"))

        self.assertIn('(r.pilot_id == "10035" or r.pilot_id == "10033")', runner.queries[0])
        self.assertIn('(r.pilot_id == "10035" or r.pilot_id == "10033")', runner.queries[1])

    def test_physiology_reader_can_query_specific_measurements(self) -> None:
        runner = FakeInfluxRunner()
        reader = PhysiologyInfluxPointReader(
            flight_task_reader=FakeFlightTaskReader(),
            collect_task_reader=FakeCollectTaskReader(),
            runner=runner,
            measurement_names=("eeg", "spo2"),
            point_limit_per_measurement=10,
        )

        reader.fetch_points(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(len(runner.queries), 2)
        self.assertIn('r._measurement == "eeg"', runner.queries[0])
        self.assertIn("|> limit(n: 10)", runner.queries[0])
        self.assertIn('r._measurement == "spo2"', runner.queries[1])

    def test_real_bus_reader_builds_measurement_and_sortie_filter(self) -> None:
        runner = FakeInfluxRunner()
        reader = RealBusInfluxPointReader(
            context_reader=FakeRealBusContextReader(),
            runner=runner,
            access_rule_id=6000019510066,
            analysis_id=6000019110020,
        )

        reader.fetch_points(SortieLocator(sortie_id="20251005_四01_ACT-4_云_J20_22#01"))

        self.assertEqual(len(runner.queries), 1)
        self.assertIn('from(bucket:"bus")', runner.queries[0])
        self.assertIn('r._measurement == "BUS6000019110020"', runner.queries[0])
        self.assertIn('r.sortie_number == "20251005_四01_ACT-4_云_J20_22#01"', runner.queries[0])


# ---- merged from test_mysql_metadata.py ----
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
    MySQLStorageAnalysisReader,
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
        if "FROM storage_data_analysis" in sql:
            return (
                {
                    "id": "6000019110021",
                    "category": "BUS",
                    "bucket": "bus",
                    "measurement": "BUS6000019110021",
                    "sortie_number": "20251002_单01_ACT-8_翼云_J16_12#01",
                    "md5_val": "md5-21",
                },
                {
                    "id": "6000019110022",
                    "category": "BUS",
                    "bucket": "bus",
                    "measurement": "BUS6000019110022",
                    "sortie_number": "20251002_单01_ACT-8_翼云_J16_12#01",
                    "md5_val": "md5-22",
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

    def test_storage_analysis_reader_lists_all_measurements_for_sortie(self) -> None:
        analyses = MySQLStorageAnalysisReader(FakeRunner()).list_for_sortie(
            SortieLocator(sortie_id="20251002_单01_ACT-8_翼云_J16_12#01"),
            category="BUS",
        )

        self.assertEqual(len(analyses), 2)
        self.assertEqual(analyses[0].measurement, "BUS6000019110021")
        self.assertEqual(analyses[1].measurement, "BUS6000019110022")


# ---- merged from test_physiology_context.py ----
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


# ---- merged from test_real_bus_context.py ----
import sys
from datetime import date
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.real_bus_context import derive_real_bus_context, resolve_time_column_index
from chronaris.schema.real_bus import AccessRuleDetail, FlightTaskMetadata, StorageAnalysisDetail


class RealBusContextTest(unittest.TestCase):
    def test_resolve_time_column_index_for_bus_prefers_exact_message_time(self) -> None:
        detail_list = (
            StorageAnalysisDetail(analysis_id=1, order_num=1, col_name="消息发生时间", col_field="code1001"),
            StorageAnalysisDetail(analysis_id=1, order_num=2, col_name="速度", col_field="code1002"),
        )

        self.assertEqual(resolve_time_column_index("BUS", detail_list), 0)

    def test_resolve_time_column_index_for_acmi_uses_fallback_time_column(self) -> None:
        detail_list = (
            StorageAnalysisDetail(analysis_id=1, order_num=1, col_name="生成时间", col_field="code1001"),
            StorageAnalysisDetail(analysis_id=1, order_num=2, col_name="速度", col_field="code1002"),
        )

        self.assertEqual(resolve_time_column_index("ACMI2", detail_list), 0)

    def test_derive_real_bus_context_requires_flight_date(self) -> None:
        with self.assertRaises(ValueError):
            derive_real_bus_context(
                flight_task=FlightTaskMetadata(flight_task_id=1, sortie_number="20251005_xxx"),
                category="BUS",
                access_rule_details=(),
                detail_list=(),
            )

    def test_derive_real_bus_context_collects_selected_fields_and_flight_date(self) -> None:
        context = derive_real_bus_context(
            flight_task=FlightTaskMetadata(
                flight_task_id=1,
                sortie_number="20251005_xxx",
                flight_date=date(2025, 10, 5),
            ),
            category="BUS",
            access_rule_details=(
                AccessRuleDetail(
                    access_rule_id=1,
                    storage_data_analysis_id=11,
                    col_field="code1001",
                    col_name="消息发生时间",
                ),
                AccessRuleDetail(
                    access_rule_id=1,
                    storage_data_analysis_id=11,
                    col_field="code1002",
                    col_name="速度",
                ),
            ),
            detail_list=(
                StorageAnalysisDetail(analysis_id=11, order_num=1, col_name="消息发生时间", col_field="code1001"),
                StorageAnalysisDetail(analysis_id=11, order_num=2, col_name="速度", col_field="code1002"),
            ),
        )

        self.assertEqual(context.flight_date.isoformat(), "2025-10-05")
        self.assertEqual(context.time_column_index, 0)
        self.assertEqual(context.selected_fields, frozenset({"code1001", "code1002"}))
