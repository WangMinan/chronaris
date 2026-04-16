"""Tests for real-bus metadata derivation helpers."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
