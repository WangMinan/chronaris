"""Tests for the first Chronaris dataset pipeline."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
