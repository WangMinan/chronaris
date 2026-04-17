"""Tests for the minimal Stage E ODE-RNN forward prototype."""

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
    from chronaris.models.alignment.config import AlignmentPrototypeConfig
    from chronaris.models.alignment.prototype import DualStreamODERNNPrototype
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


    class AlignmentPrototypeForwardTest(unittest.TestCase):
        def test_dual_stream_forward_returns_expected_shapes_and_masks(self) -> None:
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
            model = DualStreamODERNNPrototype.from_torch_alignment_batch(
                torch_batch,
                config=AlignmentPrototypeConfig(
                    hidden_dim=8,
                    embedding_dim=6,
                    encoder_hidden_dim=10,
                    decoder_hidden_dim=10,
                    dynamics_hidden_dim=12,
                    projection_dim=4,
                    ode_method="euler",
                ),
            )

            output = model(torch_batch)

            self.assertEqual(output.sample_ids, ("sample-001", "sample-002"))
            self.assertEqual(tuple(output.physiology.updated_hidden_states.shape), (2, 2, 8))
            self.assertEqual(tuple(output.vehicle.updated_hidden_states.shape), (2, 2, 8))
            self.assertEqual(tuple(output.physiology.reconstructions.shape), (2, 2, 2))
            self.assertEqual(tuple(output.vehicle.reconstructions.shape), (2, 2, 2))
            self.assertEqual(tuple(output.physiology.projected_states.shape), (2, 2, 4))
            self.assertEqual(tuple(output.vehicle.projected_states.shape), (2, 2, 4))
            self.assertEqual(tuple(output.physiology.final_hidden_state.shape), (2, 8))
            self.assertEqual(tuple(output.vehicle.final_hidden_state.shape), (2, 8))
            self.assertEqual(float(output.physiology.updated_hidden_states[1, 1].abs().sum().detach()), 0.0)
            self.assertEqual(float(output.physiology.reconstructions[1, 1].abs().sum().detach()), 0.0)
            self.assertTrue(bool(torch.isfinite(output.physiology.final_hidden_state).all()))
            self.assertTrue(bool(torch.isfinite(output.vehicle.final_hidden_state).all()))
            self.assertTrue(bool(torch.isfinite(output.physiology.reconstructions).all()))
            self.assertTrue(bool(torch.isfinite(output.vehicle.reconstructions).all()))
else:
    class AlignmentPrototypeForwardRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
