"""Tests for Stage C single-sortie validation summaries."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.sortie_validation import (
    render_validation_markdown,
    summarize_cross_stream_timing,
    summarize_stream,
    validate_sortie_bundle,
)
from chronaris.schema.models import RawPoint, SortieBundle, SortieLocator, SortieMetadata, StreamKind, WindowConfig


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class SortieValidationTest(unittest.TestCase):
    def test_summarize_stream_groups_measurements(self) -> None:
        points = (
            RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 0), {"x": 1.0}),
            RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 5), {"x": 2.0}),
            RawPoint(StreamKind.PHYSIOLOGY, "spo2", _utc(2025, 10, 5, 1, 11, 0), {"y": 3.0}),
        )

        summary = summarize_stream(StreamKind.PHYSIOLOGY, points)

        self.assertEqual(summary.point_count, 3)
        self.assertEqual(summary.measurement_count, 2)
        self.assertEqual(summary.measurements[0].measurement, "eeg")
        self.assertEqual(summary.span_ms, 60000)

    def test_cross_stream_summary_reports_gap(self) -> None:
        physiology = summarize_stream(
            StreamKind.PHYSIOLOGY,
            (
                RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 0), {"x": 1.0}),
                RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 10), {"x": 2.0}),
            ),
        )
        vehicle = summarize_stream(
            StreamKind.VEHICLE,
            (
                RawPoint(StreamKind.VEHICLE, "bus", _utc(2025, 10, 5, 1, 35, 0), {"a": "09:35:00.000"}),
                RawPoint(StreamKind.VEHICLE, "bus", _utc(2025, 10, 5, 1, 35, 10), {"a": "09:35:10.000"}),
            ),
        )

        summary = summarize_cross_stream_timing(physiology, vehicle)

        self.assertEqual(summary.relation, "physiology_before_vehicle")
        self.assertEqual(summary.gap_duration_ms, 1490000)

    def test_validate_sortie_bundle_reports_window_trials(self) -> None:
        bundle = SortieBundle(
            locator=SortieLocator(sortie_id="sortie-001"),
            metadata=SortieMetadata(sortie_id="sortie-001"),
            physiology_points=(
                RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 0), {"x": 1.0}),
                RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 3), {"x": 2.0}),
            ),
            vehicle_points=(
                RawPoint(StreamKind.VEHICLE, "bus", _utc(2025, 10, 5, 1, 10, 1), {"a": "09:10:01.000"}),
                RawPoint(StreamKind.VEHICLE, "bus", _utc(2025, 10, 5, 1, 10, 4), {"a": "09:10:04.000"}),
            ),
        )

        summary = validate_sortie_bundle(
            bundle,
            window_configs=(
                WindowConfig(duration_ms=5000, stride_ms=5000),
                WindowConfig(duration_ms=4000, stride_ms=1000),
            ),
        )

        self.assertEqual(summary.cross_stream_timing.relation, "overlap")
        self.assertEqual(summary.window_trials[0].window_count, 1)
        self.assertEqual(summary.window_trials[1].window_count, 4)

    def test_render_validation_markdown_contains_core_sections(self) -> None:
        bundle = SortieBundle(
            locator=SortieLocator(sortie_id="sortie-002"),
            metadata=SortieMetadata(sortie_id="sortie-002"),
            physiology_points=(
                RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 10, 0), {"x": 1.0}),
            ),
            vehicle_points=(
                RawPoint(StreamKind.VEHICLE, "bus", _utc(2025, 10, 5, 1, 10, 0), {"a": "09:10:00.000"}),
            ),
        )

        summary = validate_sortie_bundle(
            bundle,
            window_configs=(WindowConfig(duration_ms=1000, stride_ms=1000),),
        )
        markdown = render_validation_markdown(summary, title="Validation Preview")

        self.assertIn("# Validation Preview", markdown)
        self.assertIn("## Stream Coverage", markdown)
        self.assertIn("## Cross-Stream Timing", markdown)
        self.assertIn("## Window Trials", markdown)


if __name__ == "__main__":
    unittest.main()
