"""Merged unit tests."""

from __future__ import annotations

# ---- merged from test_alignment_losses.py ----
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
        projected_states: torch.Tensor | None = None,
        reference_projected_states: torch.Tensor | None = None,
        reference_offsets_s: torch.Tensor | None = None,
    ) -> StreamPrototypeOutput:
        reconstruction_tensor = torch.tensor([reconstructions], dtype=torch.float32)
        batch_size, point_count, _feature_count = reconstruction_tensor.shape
        hidden_dim = 3
        projection_dim = 2
        zeros_hidden = torch.zeros((batch_size, point_count, hidden_dim), dtype=torch.float32)
        resolved_projection = (
            projected_states
            if projected_states is not None
            else torch.zeros((batch_size, point_count, projection_dim), dtype=torch.float32)
        )
        resolved_reference_projection = (
            reference_projected_states
            if reference_projected_states is not None
            else resolved_projection.clone()
        )
        resolved_reference_offsets = reference_offsets_s
        if resolved_reference_offsets is not None and resolved_reference_offsets.ndim == 1:
            resolved_reference_offsets = resolved_reference_offsets.unsqueeze(0)
        if resolved_reference_offsets is None:
            resolved_reference_offsets = stream_batch.offsets_s.clone()
        return StreamPrototypeOutput(
            feature_names=stream_batch.feature_names,
            observation_embeddings=zeros_hidden,
            evolved_hidden_states=zeros_hidden,
            updated_hidden_states=zeros_hidden,
            reconstructions=reconstruction_tensor,
            projected_states=resolved_projection,
            mask=stream_batch.mask,
            feature_valid_mask=stream_batch.feature_valid_mask,
            offsets_s=stream_batch.offsets_s,
            delta_t_s=stream_batch.delta_t_s,
            point_counts=stream_batch.point_counts,
            final_hidden_state=torch.zeros((batch_size, hidden_dim), dtype=torch.float32),
            reference_projected_states=resolved_reference_projection,
            reference_offsets_s=resolved_reference_offsets,
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

        def test_stage_f_feature_vehicle_residual_is_zero_for_consistent_speed_and_acceleration(self) -> None:
            physiology_batch = _single_sample_stream_batch(
                feature_names=("eeg.af3",),
                values=((0.0,), (0.0,), (0.0,)),
            )
            vehicle_batch = _single_sample_stream_batch(
                feature_names=("speed", "acc"),
                values=((0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),
            )
            batch = TorchAlignmentBatch(
                sample_ids=("sample-001",),
                physiology=physiology_batch,
                vehicle=vehicle_batch,
            )
            output = DualStreamPrototypeOutput(
                sample_ids=("sample-001",),
                physiology=_stream_output_with_reconstructions(
                    physiology_batch,
                    reconstructions=((0.0,), (0.0,), (0.0,)),
                ),
                vehicle=_stream_output_with_reconstructions(
                    vehicle_batch,
                    reconstructions=((0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),
                ),
            )

            objective = build_stage_e_objective(
                output,
                batch,
                enable_physics_constraints=True,
                physics_constraint_mode="feature_first_with_latent_fallback",
                vehicle_physics_weight=1.0,
                physiology_physics_weight=1.0,
                physiology_envelope_lower=torch.tensor([-1.0], dtype=torch.float32),
                physiology_envelope_upper=torch.tensor([1.0], dtype=torch.float32),
            )

            self.assertAlmostEqual(float(objective.vehicle_physics), 0.0, places=6)
            self.assertAlmostEqual(float(objective.physics_total), float(objective.physiology_physics), places=6)

        def test_stage_f_physiology_envelope_penalizes_out_of_range_values(self) -> None:
            physiology_batch = _single_sample_stream_batch(
                feature_names=("eeg.af3",),
                values=((0.0,), (0.0,), (0.0,)),
            )
            vehicle_batch = _single_sample_stream_batch(
                feature_names=("speed", "acc"),
                values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            )
            batch = TorchAlignmentBatch(
                sample_ids=("sample-001",),
                physiology=physiology_batch,
                vehicle=vehicle_batch,
            )
            output = DualStreamPrototypeOutput(
                sample_ids=("sample-001",),
                physiology=_stream_output_with_reconstructions(
                    physiology_batch,
                    reconstructions=((2.0,), (3.0,), (4.0,)),
                ),
                vehicle=_stream_output_with_reconstructions(
                    vehicle_batch,
                    reconstructions=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
                ),
            )

            objective = build_stage_e_objective(
                output,
                batch,
                enable_physics_constraints=True,
                physics_constraint_mode="feature_only",
                vehicle_physics_weight=1.0,
                physiology_physics_weight=1.0,
                physiology_envelope_lower=torch.tensor([-1.0], dtype=torch.float32),
                physiology_envelope_upper=torch.tensor([1.0], dtype=torch.float32),
            )

            self.assertGreater(float(objective.physiology_physics), 0.0)
            self.assertGreater(float(objective.physics_total), 0.0)

        def test_stage_f_vehicle_fallback_to_latent_curvature_when_feature_tokens_absent(self) -> None:
            physiology_batch = _single_sample_stream_batch(
                feature_names=("eeg.af3",),
                values=((0.0,), (0.0,), (0.0,)),
            )
            vehicle_batch = _single_sample_stream_batch(
                feature_names=("BUS.code1002",),
                values=((0.0,), (0.0,), (0.0,)),
            )
            batch = TorchAlignmentBatch(
                sample_ids=("sample-001",),
                physiology=physiology_batch,
                vehicle=vehicle_batch,
            )
            curved_projection = torch.tensor(
                [[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]],
                dtype=torch.float32,
            )
            output = DualStreamPrototypeOutput(
                sample_ids=("sample-001",),
                physiology=_stream_output_with_reconstructions(
                    physiology_batch,
                    reconstructions=((0.0,), (0.0,), (0.0,)),
                ),
                vehicle=_stream_output_with_reconstructions(
                    vehicle_batch,
                    reconstructions=((0.0,), (0.0,), (0.0,)),
                    projected_states=curved_projection,
                ),
            )

            objective = build_stage_e_objective(
                output,
                batch,
                enable_physics_constraints=True,
                physics_constraint_mode="feature_first_with_latent_fallback",
                vehicle_physics_weight=1.0,
                physiology_physics_weight=0.0,
                physiology_envelope_lower=torch.tensor([-1.0], dtype=torch.float32),
                physiology_envelope_upper=torch.tensor([1.0], dtype=torch.float32),
            )

            self.assertGreater(float(objective.vehicle_physics), 0.0)
else:
    class AlignmentLossesRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


# ---- merged from test_alignment_loss_scaling.py ----
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


# ---- merged from test_alignment_prototype_forward.py ----
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
