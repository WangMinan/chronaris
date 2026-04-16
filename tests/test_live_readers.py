"""Tests for Stage B live reader query construction."""

from __future__ import annotations

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
        self.assertIn('r._measurement == "eeg"', runner.queries[1])
        self.assertIn('r._measurement == "spo2"', runner.queries[2])

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


if __name__ == "__main__":
    unittest.main()
