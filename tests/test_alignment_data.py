"""Merged unit tests."""

from __future__ import annotations

# ---- merged from test_alignment_batching.py ----
import math
import os
import sys
from pathlib import Path
import unittest

ENABLE_NUMPY_RUNTIME_TESTS = os.environ.get("CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS") == "1"

if ENABLE_NUMPY_RUNTIME_TESTS:
    SRC = Path(__file__).resolve().parents[1] / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
    from chronaris.models.alignment.batching import build_alignment_batch
    from chronaris.schema.models import StreamKind


    class AlignmentBatchingTest(unittest.TestCase):
        def test_build_alignment_batch_pads_points_and_unifies_features(self) -> None:
            samples = (
                E0ExperimentSample(
                    sample_id="sample-001",
                    sortie_id="sortie-001",
                    start_offset_ms=0,
                    end_offset_ms=5000,
                    physiology=NumericStreamMatrix(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        point_count=2,
                        feature_names=("eeg.af3",),
                        point_offsets_ms=(0, 1000),
                        point_measurements=("eeg", "eeg"),
                        values=((1.0,), (2.0,)),
                        dropped_fields=(),
                    ),
                    vehicle=NumericStreamMatrix(
                        stream_kind=StreamKind.VEHICLE,
                        point_count=1,
                        feature_names=("BUS.code1002",),
                        point_offsets_ms=(0,),
                        point_measurements=("BUS",),
                        values=((10.0,),),
                        dropped_fields=(),
                    ),
                ),
                E0ExperimentSample(
                    sample_id="sample-002",
                    sortie_id="sortie-001",
                    start_offset_ms=5000,
                    end_offset_ms=10000,
                    physiology=NumericStreamMatrix(
                        stream_kind=StreamKind.PHYSIOLOGY,
                        point_count=1,
                        feature_names=("eeg.af4",),
                        point_offsets_ms=(0,),
                        point_measurements=("eeg",),
                        values=((3.0,),),
                        dropped_fields=(),
                    ),
                    vehicle=NumericStreamMatrix(
                        stream_kind=StreamKind.VEHICLE,
                        point_count=2,
                        feature_names=("BUS.code1002", "BUS.code1003"),
                        point_offsets_ms=(0, 1000),
                        point_measurements=("BUS", "BUS"),
                        values=((11.0, 21.0), (12.0, 22.0)),
                        dropped_fields=(),
                    ),
                ),
            )

            batch = build_alignment_batch(samples)

            self.assertEqual(batch.sample_ids, ("sample-001", "sample-002"))
            self.assertEqual(batch.physiology.values.shape, (2, 2, 2))
            self.assertEqual(batch.vehicle.values.shape, (2, 2, 2))
            self.assertEqual(batch.physiology.feature_names, ("eeg.af3", "eeg.af4"))
            self.assertEqual(batch.vehicle.feature_names, ("BUS.code1002", "BUS.code1003"))
            self.assertTrue(batch.physiology.mask[0, 1])
            self.assertFalse(batch.physiology.mask[1, 1])
            self.assertTrue(math.isnan(batch.physiology.values[0, 0, 1]))
            self.assertEqual(batch.vehicle.values[1, 1, 1], 22.0)
else:
    class AlignmentBatchingRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("numpy runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_numpy_runtime_disabled(self) -> None:
            pass


# ---- merged from test_alignment_reference_grid.py ----
import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.models.alignment import ReferenceGridConfig, build_reference_grid, build_reference_grids
from chronaris.schema.models import StreamKind


def _empty_stream(kind: StreamKind) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=kind,
        point_count=0,
        feature_names=(),
        point_offsets_ms=(),
        point_measurements=(),
        values=(),
        dropped_fields=(),
    )


def _reference_sample(sample_id: str, start_offset_ms: int, end_offset_ms: int) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=sample_id,
        sortie_id="sortie-001",
        start_offset_ms=start_offset_ms,
        end_offset_ms=end_offset_ms,
        physiology=_empty_stream(StreamKind.PHYSIOLOGY),
        vehicle=_empty_stream(StreamKind.VEHICLE),
    )


class AlignmentReferenceGridTest(unittest.TestCase):
    def test_build_reference_grid_spans_full_window_when_include_end_is_true(self) -> None:
        grid = build_reference_grid(
            _reference_sample("sample-001", 0, 5000),
            config=ReferenceGridConfig(point_count=5, include_end=True),
        )

        self.assertEqual(grid.duration_ms, 5000)
        self.assertEqual(grid.relative_offsets_ms, (0.0, 1250.0, 2500.0, 3750.0, 5000.0))
        self.assertEqual(grid.absolute_offsets_ms, (0.0, 1250.0, 2500.0, 3750.0, 5000.0))
        self.assertEqual(grid.relative_offsets_s, (0.0, 1.25, 2.5, 3.75, 5.0))

    def test_build_reference_grid_uses_left_closed_spacing_when_include_end_is_false(self) -> None:
        grid = build_reference_grid(
            _reference_sample("sample-002", 1000, 5000),
            config=ReferenceGridConfig(point_count=4, include_end=False),
        )

        self.assertEqual(grid.duration_ms, 4000)
        self.assertEqual(grid.relative_offsets_ms, (0.0, 1000.0, 2000.0, 3000.0))
        self.assertEqual(grid.absolute_offsets_ms, (1000.0, 2000.0, 3000.0, 4000.0))

    def test_build_reference_grid_supports_single_reference_point(self) -> None:
        grid = build_reference_grid(
            _reference_sample("sample-003", 2000, 7000),
            config=ReferenceGridConfig(point_count=1),
        )

        self.assertEqual(grid.relative_offsets_ms, (0.0,))
        self.assertEqual(grid.absolute_offsets_ms, (2000.0,))

    def test_build_reference_grids_preserves_sample_order(self) -> None:
        grids = build_reference_grids(
            (
                _reference_sample("sample-010", 0, 5000),
                _reference_sample("sample-011", 5000, 10000),
            ),
            config=ReferenceGridConfig(point_count=3),
        )

        self.assertEqual(tuple(grid.sample_id for grid in grids), ("sample-010", "sample-011"))
        self.assertEqual(grids[1].absolute_offsets_ms, (5000.0, 7500.0, 10000.0))


# ---- merged from test_alignment_splits.py ----
import sys
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.models.alignment import ChronologicalSplitConfig, split_e0_samples_chronologically
from chronaris.schema.models import StreamKind


def _split_stream(kind: StreamKind) -> NumericStreamMatrix:
    return NumericStreamMatrix(
        stream_kind=kind,
        point_count=1,
        feature_names=("feature",),
        point_offsets_ms=(0,),
        point_measurements=("measurement",),
        values=((1.0,),),
        dropped_fields=(),
    )


def _split_sample(index: int) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=f"sample-{index:03d}",
        sortie_id="sortie-001",
        start_offset_ms=index * 5000,
        end_offset_ms=(index + 1) * 5000,
        physiology=_split_stream(StreamKind.PHYSIOLOGY),
        vehicle=_split_stream(StreamKind.VEHICLE),
    )


class AlignmentSplitTest(unittest.TestCase):
    def test_split_e0_samples_chronologically_uses_default_15_5_5_shape_for_25_samples(self) -> None:
        samples = tuple(_split_sample(index) for index in range(25))

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
        samples = tuple(_split_sample(index) for index in range(12))

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
        samples = tuple(_split_sample(index) for index in range(2))

        with self.assertRaises(ValueError):
            split_e0_samples_chronologically(
                samples,
                config=ChronologicalSplitConfig(gap_windows=1),
            )


# ---- merged from test_alignment_torch_batch.py ----
import os
import sys
from pathlib import Path
import unittest

ENABLE_TORCH_RUNTIME_TESTS = os.environ.get("CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS") == "1"

if ENABLE_TORCH_RUNTIME_TESTS:
    SRC = Path(__file__).resolve().parents[1] / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import torch

    from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
    from chronaris.models.alignment.batching import build_alignment_batch
    from chronaris.models.alignment.torch_batch import build_torch_alignment_batch
    from chronaris.schema.models import StreamKind

    def _stream(
        kind: StreamKind,
        *,
        feature_names: tuple[str, ...],
        offsets_ms: tuple[int, ...],
        values: tuple[tuple[float, ...], ...],
    ) -> NumericStreamMatrix:
        return NumericStreamMatrix(
            stream_kind=kind,
            point_count=len(offsets_ms),
            feature_names=feature_names,
            point_offsets_ms=offsets_ms,
            point_measurements=tuple("measurement" for _ in offsets_ms),
            values=values,
            dropped_fields=(),
        )


    class AlignmentTorchBatchTest(unittest.TestCase):
        def test_build_torch_alignment_batch_converts_masks_offsets_and_nan_values(self) -> None:
            samples = (
                E0ExperimentSample(
                    sample_id="sample-001",
                    sortie_id="sortie-001",
                    start_offset_ms=0,
                    end_offset_ms=5000,
                    physiology=_stream(
                        StreamKind.PHYSIOLOGY,
                        feature_names=("eeg.af3",),
                        offsets_ms=(0, 1000),
                        values=((1.0,), (2.0,)),
                    ),
                    vehicle=_stream(
                        StreamKind.VEHICLE,
                        feature_names=("BUS.code1002",),
                        offsets_ms=(0,),
                        values=((10.0,),),
                    ),
                ),
                E0ExperimentSample(
                    sample_id="sample-002",
                    sortie_id="sortie-001",
                    start_offset_ms=5000,
                    end_offset_ms=10000,
                    physiology=_stream(
                        StreamKind.PHYSIOLOGY,
                        feature_names=("eeg.af4",),
                        offsets_ms=(0,),
                        values=((3.0,),),
                    ),
                    vehicle=_stream(
                        StreamKind.VEHICLE,
                        feature_names=("BUS.code1002", "BUS.code1003"),
                        offsets_ms=(0, 1000),
                        values=((11.0, 21.0), (12.0, 22.0)),
                    ),
                ),
            )

            numpy_batch = build_alignment_batch(samples)
            torch_batch = build_torch_alignment_batch(numpy_batch)

            self.assertEqual(torch_batch.sample_ids, ("sample-001", "sample-002"))
            self.assertEqual(tuple(torch_batch.physiology.values.shape), (2, 2, 2))
            self.assertEqual(torch_batch.physiology.values.dtype, torch.float32)
            self.assertTrue(bool(torch_batch.physiology.mask[0, 1]))
            self.assertFalse(bool(torch_batch.physiology.mask[1, 1]))
            self.assertEqual(float(torch_batch.physiology.values[0, 0, 0]), 1.0)
            self.assertEqual(float(torch_batch.physiology.values[0, 0, 1]), 0.0)
            self.assertTrue(bool(torch_batch.physiology.feature_valid_mask[0, 0, 0]))
            self.assertFalse(bool(torch_batch.physiology.feature_valid_mask[0, 0, 1]))
            self.assertEqual(float(torch_batch.physiology.offsets_s[0, 1]), 1.0)
            self.assertEqual(float(torch_batch.physiology.offsets_s[1, 1]), 0.0)
            self.assertEqual(float(torch_batch.physiology.delta_t_s[0, 0]), 0.0)
            self.assertEqual(float(torch_batch.physiology.delta_t_s[0, 1]), 1.0)
            self.assertEqual(float(torch_batch.physiology.delta_t_s[1, 1]), 0.0)

            self.assertEqual(tuple(torch_batch.vehicle.values.shape), (2, 2, 2))
            self.assertEqual(int(torch_batch.vehicle.point_counts[0]), 1)
            self.assertEqual(int(torch_batch.vehicle.point_counts[1]), 2)
            self.assertEqual(float(torch_batch.vehicle.delta_t_s[1, 1]), 1.0)
else:
    class AlignmentTorchRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass
