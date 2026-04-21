"""Merged unit tests."""

from __future__ import annotations

# ---- merged from test_dataset_builder.py ----
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


# ---- merged from test_dataset_pipeline_v1.py ----
import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.access.loader import SortieLoader
from chronaris.access.memory import InMemoryMetadataReader, InMemoryPointReader
from chronaris.dataset.builder import SortieDatasetBuilder
from chronaris.pipelines.dataset_v1 import DatasetPipelineV1
from chronaris.schema.models import RawPoint, SortieLocator, SortieMetadata, StreamKind, WindowConfig


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int, millisecond: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=timezone.utc)


class DatasetPipelineV1Test(unittest.TestCase):
    def test_pipeline_runs_end_to_end_with_in_memory_sources(self) -> None:
        locator = SortieLocator(sortie_id="sortie-101")

        physiology_reader = InMemoryPointReader(
            points_by_sortie={
                locator.sortie_id: (
                    RawPoint(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        measurement="ecg",
                        timestamp=_utc(2025, 10, 5, 10, 0, 0, 200),
                        values={"hr": 78.0},
                    ),
                    RawPoint(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        measurement="ecg",
                        timestamp=_utc(2025, 10, 5, 10, 0, 1, 200),
                        values={"hr": 83.0},
                    ),
                )
            },
            expected_kind=StreamKind.PHYSIOLOGY,
        )
        vehicle_reader = InMemoryPointReader(
            points_by_sortie={
                locator.sortie_id: (
                    RawPoint(
                        stream_kind=StreamKind.VEHICLE,
                        measurement="bus",
                        timestamp=_utc(2025, 10, 5, 10, 0, 0, 0),
                        values={"speed": 220.0},
                    ),
                    RawPoint(
                        stream_kind=StreamKind.VEHICLE,
                        measurement="bus",
                        timestamp=_utc(2025, 10, 5, 10, 0, 1, 0),
                        values={"speed": 225.0},
                    ),
                )
            },
            expected_kind=StreamKind.VEHICLE,
        )
        metadata_reader = InMemoryMetadataReader(
            metadata_by_sortie={
                locator.sortie_id: SortieMetadata(
                    sortie_id=locator.sortie_id,
                    mission_code="ACT-4",
                    aircraft_model="J20",
                    pilot_code="YUN",
                )
            }
        )

        pipeline = DatasetPipelineV1(
            loader=SortieLoader(
                physiology_reader=physiology_reader,
                vehicle_reader=vehicle_reader,
                metadata_reader=metadata_reader,
            ),
            builder=SortieDatasetBuilder(
                window_config=WindowConfig(duration_ms=1000, stride_ms=1000),
            ),
        )

        result = pipeline.run(locator)

        self.assertEqual(result.aligned_bundle.metadata.mission_code, "ACT-4")
        self.assertEqual(len(result.aligned_bundle.physiology_points), 2)
        self.assertEqual(len(result.aligned_bundle.vehicle_points), 2)
        self.assertEqual(len(result.windows), 2)
        self.assertEqual(result.summary()["sortie_id"], locator.sortie_id)
