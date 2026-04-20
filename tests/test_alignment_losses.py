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
    from chronaris.models.alignment.losses import (
        build_stage_e_objective,
        dual_stream_alignment_loss,
        dual_stream_reconstruction_loss,
        masked_mean_squared_error,
        projection_alignment_loss,
    )
    from chronaris.models.alignment.prototype import (
        DualStreamODERNNPrototype,
        DualStreamPrototypeOutput,
        StreamPrototypeOutput,
    )
    from chronaris.models.alignment.reference_grid import ReferenceGridConfig, build_reference_grids
    from chronaris.models.alignment.torch_batch import (
        TorchAlignmentBatch,
        TorchAlignmentStreamBatch,
        build_torch_alignment_batch,
    )
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


    def _single_sample_stream_batch(
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
            offsets_s=torch.tensor([[offset_ms / 1000.0 for offset_ms in offsets_ms]], dtype=torch.float32),
            delta_t_s=torch.tensor(
                [[0.0] + [1.0 for _ in range(max(point_count - 1, 0))]],
                dtype=torch.float32,
            ),
            point_counts=torch.tensor([point_count], dtype=torch.int64),
            feature_names=feature_names,
        )


    def _stream_output_with_reconstructions(
        stream_batch: TorchAlignmentStreamBatch,
        *,
        reconstructions: tuple[tuple[float, ...], ...],
    ) -> StreamPrototypeOutput:
        reconstruction_tensor = torch.tensor([reconstructions], dtype=torch.float32)
        batch_size, point_count, _feature_count = reconstruction_tensor.shape
        hidden_dim = 3
        projection_dim = 2
        zeros_hidden = torch.zeros((batch_size, point_count, hidden_dim), dtype=torch.float32)
        zeros_projection = torch.zeros((batch_size, point_count, projection_dim), dtype=torch.float32)
        return StreamPrototypeOutput(
            feature_names=stream_batch.feature_names,
            observation_embeddings=zeros_hidden,
            evolved_hidden_states=zeros_hidden,
            updated_hidden_states=zeros_hidden,
            reconstructions=reconstruction_tensor,
            projected_states=zeros_projection,
            mask=stream_batch.mask,
            feature_valid_mask=stream_batch.feature_valid_mask,
            offsets_s=stream_batch.offsets_s,
            delta_t_s=stream_batch.delta_t_s,
            point_counts=stream_batch.point_counts,
            final_hidden_state=torch.zeros((batch_size, hidden_dim), dtype=torch.float32),
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

        def test_relative_mse_reconstruction_mode_reduces_stream_scale_imbalance(self) -> None:
            physiology_batch = _single_sample_stream_batch(
                feature_names=("eeg.af3", "eeg.af4"),
                values=((1.0, 2.0),),
            )
            vehicle_batch = _single_sample_stream_batch(
                feature_names=("BUS.code1002", "BUS.code1003"),
                values=((1_000_000.0, 2_000_000.0),),
            )
            torch_batch = TorchAlignmentBatch(
                sample_ids=("sample-001",),
                physiology=physiology_batch,
                vehicle=vehicle_batch,
            )
            output = DualStreamPrototypeOutput(
                sample_ids=("sample-001",),
                physiology=_stream_output_with_reconstructions(
                    physiology_batch,
                    reconstructions=((2.0, 4.0),),
                ),
                vehicle=_stream_output_with_reconstructions(
                    vehicle_batch,
                    reconstructions=((2_000_000.0, 4_000_000.0),),
                ),
            )

            mse = dual_stream_reconstruction_loss(output, torch_batch, mode="mse")
            relative = dual_stream_reconstruction_loss(output, torch_batch, mode="relative_mse")

            self.assertGreater(float(mse.vehicle), float(mse.physiology) * 1e10)
            self.assertAlmostEqual(float(relative.physiology), 1.0, places=6)
            self.assertAlmostEqual(float(relative.vehicle), 1.0, places=6)
            self.assertAlmostEqual(float(relative.total), 2.0, places=6)

        def test_projection_alignment_loss_is_zero_for_identical_reference_projections(self) -> None:
            projections = torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0]]],
                dtype=torch.float32,
            )

            loss = projection_alignment_loss(projections, projections, mode="mse")

            self.assertEqual(float(loss), 0.0)

        def test_dual_stream_alignment_and_objective_are_finite_when_reference_grid_is_provided(self) -> None:
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
                        offsets_ms=(0, 2000),
                        values=((10.0,), (11.0,)),
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
            reference_grids = build_reference_grids(samples, config=ReferenceGridConfig(point_count=4))
            reference_offsets_s = torch.tensor(
                [grid.relative_offsets_s for grid in reference_grids],
                dtype=torch.float32,
            )

            output = model(torch_batch, reference_offsets_s=reference_offsets_s)
            alignment = dual_stream_alignment_loss(output)
            objective = build_stage_e_objective(output, torch_batch)

            self.assertTrue(bool(torch.isfinite(alignment.alignment)))
            self.assertGreaterEqual(float(alignment.alignment.detach()), 0.0)
            self.assertTrue(bool(torch.isfinite(objective.total)))
            self.assertGreaterEqual(float(objective.total.detach()), 0.0)
else:
    class AlignmentLossesRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
