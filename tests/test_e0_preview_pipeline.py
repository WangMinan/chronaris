"""Tests for the E0 preview pipeline."""

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
from chronaris.features.experiment_input import E0InputConfig
from chronaris.pipelines.e0_preview import E0PreviewPipeline
from chronaris.schema.models import RawPoint, SortieLocator, SortieMetadata, StreamKind, WindowConfig
from chronaris.dataset.builder import SortieDatasetBuilder


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class E0PreviewPipelineTest(unittest.TestCase):
    def test_pipeline_runs_end_to_end(self) -> None:
        locator = SortieLocator(sortie_id="sortie-001")
        loader = SortieLoader(
            physiology_reader=InMemoryPointReader(
                points_by_sortie={
                    locator.sortie_id: (
                        RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 35, 0), {"af3": "1.0"}),
                        RawPoint(StreamKind.PHYSIOLOGY, "eeg", _utc(2025, 10, 5, 1, 35, 1), {"af3": "2.0"}),
                    )
                },
                expected_kind=StreamKind.PHYSIOLOGY,
            ),
            vehicle_reader=InMemoryPointReader(
                points_by_sortie={
                    locator.sortie_id: (
                        RawPoint(StreamKind.VEHICLE, "BUSX", _utc(2025, 10, 5, 1, 35, 0), {"code1002": "10.0"}),
                        RawPoint(StreamKind.VEHICLE, "BUSX", _utc(2025, 10, 5, 1, 35, 1), {"code1002": "11.0"}),
                    )
                },
                expected_kind=StreamKind.VEHICLE,
            ),
            metadata_reader=InMemoryMetadataReader(
                metadata_by_sortie={locator.sortie_id: SortieMetadata(sortie_id=locator.sortie_id)}
            ),
        )
        pipeline = E0PreviewPipeline(
            loader=loader,
            dataset_builder=SortieDatasetBuilder(
                window_config=WindowConfig(duration_ms=5000, stride_ms=5000),
            ),
            input_config=E0InputConfig(
                physiology_measurements=("eeg",),
                vehicle_measurements=("BUSX",),
            ),
        )

        samples = pipeline.run(locator)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].physiology.feature_names, ("eeg.af3",))
        self.assertEqual(samples[0].vehicle.feature_names, ("BUSX.code1002",))


if __name__ == "__main__":
    unittest.main()
