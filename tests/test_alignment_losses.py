"""Tests for minimal Stage E reconstruction losses."""

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
    from chronaris.models.alignment.losses import dual_stream_reconstruction_loss, masked_mean_squared_error
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


    class AlignmentLossesTest(unittest.TestCase):
        def test_masked_mean_squared_error_ignores_invalid_positions(self) -> None:
            predictions = torch.tensor([[[1.0, 9.0], [3.0, 7.0]]])
            targets = torch.tensor([[[1.0, 0.0], [2.0, 1.0]]])
            valid_mask = torch.tensor([[[True, False], [True, False]]])

            loss = masked_mean_squared_error(predictions, targets, valid_mask)

            self.assertAlmostEqual(float(loss), 0.5)

        def test_dual_stream_reconstruction_loss_returns_finite_breakdown(self) -> None:
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
            )

            numpy_batch = build_alignment_batch(samples)
            torch_batch = build_torch_alignment_batch(numpy_batch)
            model = DualStreamODERNNPrototype.from_torch_alignment_batch(
                torch_batch,
                config=AlignmentPrototypeConfig(
                    hidden_dim=6,
                    embedding_dim=4,
                    encoder_hidden_dim=8,
                    decoder_hidden_dim=8,
                    dynamics_hidden_dim=8,
                    projection_dim=3,
                    ode_method="euler",
                ),
            )

            output = model(torch_batch)
            loss = dual_stream_reconstruction_loss(output, torch_batch)

            self.assertTrue(bool(torch.isfinite(loss.physiology)))
            self.assertTrue(bool(torch.isfinite(loss.vehicle)))
            self.assertTrue(bool(torch.isfinite(loss.total)))
            self.assertGreaterEqual(float(loss.total.detach()), 0.0)
else:
    class AlignmentLossesRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
