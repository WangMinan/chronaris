"""Tests for torch-backed Stage E alignment batch adapters."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
