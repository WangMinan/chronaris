"""Unit tests for Stage E reconstruction loss scaling behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is expected in Stage E envs.
    torch = None

if torch is not None:
    from chronaris.models.alignment.losses import dual_stream_reconstruction_loss
    from chronaris.models.alignment.torch_batch import TorchAlignmentBatch, TorchAlignmentStreamBatch

    @dataclass(frozen=True, slots=True)
    class _FakeStreamOutput:
        reconstructions: torch.Tensor


    @dataclass(frozen=True, slots=True)
    class _FakeDualOutput:
        physiology: _FakeStreamOutput
        vehicle: _FakeStreamOutput


    def _build_stream_batch(
        *,
        feature_names: tuple[str, ...],
        values: tuple[tuple[float, ...], ...],
    ) -> TorchAlignmentStreamBatch:
        value_tensor = torch.tensor([values], dtype=torch.float32)
        point_count = value_tensor.shape[1]
        feature_count = value_tensor.shape[2]
        offsets_ms = tuple(point_index * 1000 for point_index in range(point_count))
        return TorchAlignmentStreamBatch(
            values=value_tensor,
            mask=torch.ones((1, point_count), dtype=torch.bool),
            feature_valid_mask=torch.ones((1, point_count, feature_count), dtype=torch.bool),
            offsets_ms=torch.tensor([offsets_ms], dtype=torch.int64),
            offsets_s=torch.tensor([[offset / 1000.0 for offset in offsets_ms]], dtype=torch.float32),
            delta_t_s=torch.tensor(
                [[0.0] + [1.0 for _ in range(max(point_count - 1, 0))]],
                dtype=torch.float32,
            ),
            point_counts=torch.tensor([point_count], dtype=torch.int64),
            feature_names=feature_names,
        )


    class AlignmentLossScalingTest(unittest.TestCase):
        def test_relative_mse_reduces_cross_stream_scale_gap(self) -> None:
            physiology_batch = _build_stream_batch(
                feature_names=("eeg.af3", "eeg.af4"),
                values=((1.0, 2.0),),
            )
            vehicle_batch = _build_stream_batch(
                feature_names=("BUS.code1002", "BUS.code1003"),
                values=((1_000_000.0, 2_000_000.0),),
            )
            batch = TorchAlignmentBatch(
                sample_ids=("sample-001",),
                physiology=physiology_batch,
                vehicle=vehicle_batch,
            )
            output = _FakeDualOutput(
                physiology=_FakeStreamOutput(reconstructions=torch.tensor([[[2.0, 4.0]]], dtype=torch.float32)),
                vehicle=_FakeStreamOutput(
                    reconstructions=torch.tensor([[[2_000_000.0, 4_000_000.0]]], dtype=torch.float32)
                ),
            )

            mse = dual_stream_reconstruction_loss(output, batch, mode="mse")
            relative = dual_stream_reconstruction_loss(output, batch, mode="relative_mse")

            self.assertGreater(float(mse.vehicle), float(mse.physiology) * 1e10)
            self.assertAlmostEqual(float(relative.physiology), 1.0, places=6)
            self.assertAlmostEqual(float(relative.vehicle), 1.0, places=6)
            self.assertAlmostEqual(float(relative.total), 2.0, places=6)
else:
    class AlignmentLossScalingTorchMissingTest(unittest.TestCase):
        @unittest.skip("torch is not available in the current environment.")
        def test_torch_missing(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
