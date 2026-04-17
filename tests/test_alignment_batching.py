"""Tests for Stage E alignment batching."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
