"""Tests for E0 minimal experiment-input adaptation."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import (
    E0InputConfig,
    build_e0_experiment_samples,
    build_numeric_stream_matrix,
)
from chronaris.schema.models import (
    AlignedPoint,
    AlignedSortieBundle,
    DatasetBuildResult,
    RawPoint,
    SampleWindow,
    SortieLocator,
    SortieMetadata,
    StreamKind,
)


def _utc(year: int, month: int, day: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class ExperimentInputTest(unittest.TestCase):
    def test_build_numeric_stream_matrix_keeps_only_numeric_features(self) -> None:
        matrix = build_numeric_stream_matrix(
            StreamKind.PHYSIOLOGY,
            (
                AlignedPoint(
                    point=RawPoint(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        measurement="eeg",
                        timestamp=_utc(2025, 10, 5, 1, 35, 0),
                        values={"af3": "1.5", "date_time": "2025-10-05 09:35:00.000000"},
                    ),
                    offset_ms=0,
                ),
                AlignedPoint(
                    point=RawPoint(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        measurement="spo2",
                        timestamp=_utc(2025, 10, 5, 1, 35, 1),
                        values={"spo2": 95, "status": "OK"},
                    ),
                    offset_ms=1000,
                ),
            ),
        )

        self.assertEqual(matrix.feature_names, ("eeg.af3", "spo2.spo2"))
        self.assertIn("eeg.date_time", matrix.dropped_fields)
        self.assertIn("spo2.status", matrix.dropped_fields)
        self.assertEqual(matrix.values[0][0], 1.5)
        self.assertTrue(math.isnan(matrix.values[0][1]))
        self.assertTrue(math.isnan(matrix.values[1][0]))
        self.assertEqual(matrix.values[1][1], 95.0)

    def test_build_e0_experiment_samples_adapts_windows(self) -> None:
        aligned_bundle = AlignedSortieBundle(
            locator=SortieLocator(sortie_id="sortie-001"),
            metadata=SortieMetadata(sortie_id="sortie-001"),
            reference_time=_utc(2025, 10, 5, 1, 35, 0),
        )
        result = DatasetBuildResult(
            aligned_bundle=aligned_bundle,
            windows=(
                SampleWindow(
                    sample_id="sortie-001:0000",
                    sortie_id="sortie-001",
                    window_index=0,
                    start_offset_ms=0,
                    end_offset_ms=5000,
                    physiology_points=(
                        AlignedPoint(
                            point=RawPoint(
                                stream_kind=StreamKind.PHYSIOLOGY,
                                measurement="eeg",
                                timestamp=_utc(2025, 10, 5, 1, 35, 0),
                                values={"af3": "1.0"},
                            ),
                            offset_ms=0,
                        ),
                    ),
                    vehicle_points=(
                        AlignedPoint(
                            point=RawPoint(
                                stream_kind=StreamKind.VEHICLE,
                                measurement="BUS6000019110020",
                                timestamp=_utc(2025, 10, 5, 1, 35, 0),
                                values={"code1001": "09:35:00.000", "code1002": "123.4"},
                            ),
                            offset_ms=0,
                        ),
                    ),
                ),
            ),
        )

        samples = build_e0_experiment_samples(
            result,
            config=E0InputConfig(
                physiology_measurements=("eeg",),
                vehicle_measurements=("BUS6000019110020",),
            ),
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].physiology.feature_names, ("eeg.af3",))
        self.assertEqual(samples[0].vehicle.feature_names, ("BUS6000019110020.code1002",))
        self.assertIn("BUS6000019110020.code1001", samples[0].vehicle.dropped_fields)


if __name__ == "__main__":
    unittest.main()
