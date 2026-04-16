"""Tests for Chronaris dataset timebase and window construction."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.dataset.timebase import ReferenceStrategy, TimebaseError, TimebasePolicy, align_sortie_bundle
from chronaris.schema.models import RawPoint, SortieBundle, SortieLocator, SortieMetadata, StreamKind, WindowConfig


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int, millisecond: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=timezone.utc)


class DatasetBuilderTest(unittest.TestCase):
    def test_align_sortie_bundle_uses_earliest_observation_by_default(self) -> None:
        locator = SortieLocator(sortie_id="sortie-001")
        bundle = SortieBundle(
            locator=locator,
            metadata=SortieMetadata(sortie_id=locator.sortie_id),
            physiology_points=(
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 100),
                    values={"hr": 81.0},
                ),
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 800),
                    values={"hr": 84.0},
                ),
            ),
            vehicle_points=(
                RawPoint(
                    stream_kind=StreamKind.VEHICLE,
                    measurement="bus",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 0),
                    values={"speed": 230.0},
                ),
            ),
        )

        aligned = align_sortie_bundle(bundle)

        self.assertEqual(aligned.reference_time, _utc(2025, 10, 5, 10, 0, 0, 0))
        self.assertEqual([point.offset_ms for point in aligned.physiology_points], [100, 800])
        self.assertEqual([point.offset_ms for point in aligned.vehicle_points], [0])

    def test_builder_creates_windows_with_point_thresholds(self) -> None:
        locator = SortieLocator(sortie_id="sortie-002")
        bundle = SortieBundle(
            locator=locator,
            metadata=SortieMetadata(sortie_id=locator.sortie_id),
            physiology_points=(
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 100),
                    values={"hr": 80.0},
                ),
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 800),
                    values={"hr": 82.0},
                ),
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=_utc(2025, 10, 5, 10, 0, 1, 600),
                    values={"hr": 85.0},
                ),
            ),
            vehicle_points=(
                RawPoint(
                    stream_kind=StreamKind.VEHICLE,
                    measurement="bus",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 0),
                    values={"speed": 230.0},
                ),
                RawPoint(
                    stream_kind=StreamKind.VEHICLE,
                    measurement="bus",
                    timestamp=_utc(2025, 10, 5, 10, 0, 1, 0),
                    values={"speed": 235.0},
                ),
                RawPoint(
                    stream_kind=StreamKind.VEHICLE,
                    measurement="bus",
                    timestamp=_utc(2025, 10, 5, 10, 0, 2, 0),
                    values={"speed": 240.0},
                ),
            ),
        )

        builder = SortieDatasetBuilder(
            timebase_policy=TimebasePolicy(reference_strategy=ReferenceStrategy.EARLIEST_OBSERVATION),
            window_config=WindowConfig(
                duration_ms=1000,
                stride_ms=1000,
                min_physiology_points=1,
                min_vehicle_points=1,
                allow_partial_last_window=True,
            ),
        )

        result = builder.build(bundle)

        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.windows[0].start_offset_ms, 0)
        self.assertEqual(result.windows[0].end_offset_ms, 1000)
        self.assertEqual(len(result.windows[0].physiology_points), 2)
        self.assertEqual(len(result.windows[0].vehicle_points), 1)
        self.assertEqual(result.windows[1].start_offset_ms, 1000)
        self.assertEqual(len(result.windows[1].physiology_points), 1)
        self.assertEqual(len(result.windows[1].vehicle_points), 1)

    def test_mixed_datetime_awareness_raises(self) -> None:
        locator = SortieLocator(sortie_id="sortie-003")
        bundle = SortieBundle(
            locator=locator,
            metadata=SortieMetadata(sortie_id=locator.sortie_id),
            physiology_points=(
                RawPoint(
                    stream_kind=StreamKind.PHYSIOLOGY,
                    measurement="ecg",
                    timestamp=datetime(2025, 10, 5, 10, 0, 0),
                    values={"hr": 80.0},
                ),
            ),
            vehicle_points=(
                RawPoint(
                    stream_kind=StreamKind.VEHICLE,
                    measurement="bus",
                    timestamp=_utc(2025, 10, 5, 10, 0, 0, 0),
                    values={"speed": 230.0},
                ),
            ),
        )

        with self.assertRaises(TimebaseError):
            align_sortie_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
