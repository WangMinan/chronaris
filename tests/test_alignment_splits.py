"""Tests for Stage E chronological split utilities."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.models.alignment import ChronologicalSplitConfig, split_e0_samples_chronologically
from chronaris.schema.models import StreamKind


def _stream(kind: StreamKind) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=kind,
        point_count=1,
        feature_names=("feature",),
        point_offsets_ms=(0,),
        point_measurements=("measurement",),
        values=((1.0,),),
        dropped_fields=(),
    )


def _sample(index: int) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=f"sample-{index:03d}",
        sortie_id="sortie-001",
        start_offset_ms=index * 5000,
        end_offset_ms=(index + 1) * 5000,
        physiology=_stream(StreamKind.PHYSIOLOGY),
        vehicle=_stream(StreamKind.VEHICLE),
    )


class AlignmentSplitTest(unittest.TestCase):
    def test_split_e0_samples_chronologically_uses_default_15_5_5_shape_for_25_samples(self) -> None:
        samples = tuple(_sample(index) for index in range(25))

        split = split_e0_samples_chronologically(samples)

        self.assertEqual(len(split.train), 15)
        self.assertEqual(len(split.validation), 5)
        self.assertEqual(len(split.test), 5)
        self.assertEqual(split.train[0].sample_id, "sample-000")
        self.assertEqual(split.validation[0].sample_id, "sample-015")
        self.assertEqual(split.test[0].sample_id, "sample-020")
        self.assertEqual(split.skipped_between_train_validation, ())
        self.assertEqual(split.skipped_between_validation_test, ())

    def test_split_e0_samples_chronologically_reserves_gap_windows_between_partitions(self) -> None:
        samples = tuple(_sample(index) for index in range(12))

        split = split_e0_samples_chronologically(
            samples,
            config=ChronologicalSplitConfig(gap_windows=1),
        )

        self.assertEqual(tuple(sample.sample_id for sample in split.train), tuple(f"sample-{i:03d}" for i in range(6)))
        self.assertEqual(tuple(sample.sample_id for sample in split.skipped_between_train_validation), ("sample-006",))
        self.assertEqual(tuple(sample.sample_id for sample in split.validation), ("sample-007", "sample-008"))
        self.assertEqual(tuple(sample.sample_id for sample in split.skipped_between_validation_test), ("sample-009",))
        self.assertEqual(tuple(sample.sample_id for sample in split.test), ("sample-010", "sample-011"))

    def test_split_config_rejects_invalid_ratio_sum(self) -> None:
        with self.assertRaises(ValueError):
            ChronologicalSplitConfig(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)

    def test_split_rejects_gap_that_removes_all_usable_samples(self) -> None:
        samples = tuple(_sample(index) for index in range(2))

        with self.assertRaises(ValueError):
            split_e0_samples_chronologically(
                samples,
                config=ChronologicalSplitConfig(gap_windows=1),
            )


if __name__ == "__main__":
    unittest.main()
