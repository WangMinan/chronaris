"""Tests for the minimal Stage E preview training pipeline."""

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

    from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
    from chronaris.models.alignment import AlignmentPrototypeConfig, ChronologicalSplitConfig, ReferenceGridConfig
    from chronaris.pipelines.alignment_preview import AlignmentPreviewConfig, AlignmentPreviewPipeline
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


    def _sample(index: int) -> E0ExperimentSample:
        base = float(index + 1)
        return E0ExperimentSample(
            sample_id=f"sample-{index:03d}",
            sortie_id="sortie-001",
            start_offset_ms=index * 5000,
            end_offset_ms=(index + 1) * 5000,
            physiology=_stream(
                StreamKind.PHYSIOLOGY,
                feature_names=("eeg.af3", "eeg.af4"),
                offsets_ms=(0, 1000, 3000),
                values=(
                    (base, base + 0.5),
                    (base + 1.0, base + 1.5),
                    (base + 2.0, base + 2.5),
                ),
            ),
            vehicle=_stream(
                StreamKind.VEHICLE,
                feature_names=("BUS.code1002", "BUS.code1003"),
                offsets_ms=(0, 2000, 4000),
                values=(
                    (base + 10.0, base + 20.0),
                    (base + 11.0, base + 21.0),
                    (base + 12.0, base + 22.0),
                ),
            ),
        )


    class AlignmentPreviewPipelineTest(unittest.TestCase):
        def test_alignment_preview_pipeline_runs_train_validation_test_loop(self) -> None:
            samples = tuple(_sample(index) for index in range(25))
            pipeline = AlignmentPreviewPipeline(
                config=AlignmentPreviewConfig(
                    prototype_config=AlignmentPrototypeConfig(
                        hidden_dim=8,
                        embedding_dim=6,
                        encoder_hidden_dim=10,
                        decoder_hidden_dim=10,
                        dynamics_hidden_dim=12,
                        projection_dim=4,
                        ode_method="euler",
                    ),
                    split_config=ChronologicalSplitConfig(),
                    reference_grid_config=ReferenceGridConfig(point_count=4),
                    epoch_count=2,
                    batch_size=4,
                    learning_rate=1e-3,
                    device="cpu",
                )
            )

            result = pipeline.run(samples)

            self.assertEqual(len(result.split.train), 15)
            self.assertEqual(len(result.split.validation), 5)
            self.assertEqual(len(result.split.test), 5)
            self.assertEqual(len(result.train_history), 2)
            self.assertEqual(len(result.validation_history), 2)
            self.assertEqual(result.train_history[0].sample_count, 15)
            self.assertEqual(result.validation_history[0].sample_count, 5)
            self.assertEqual(result.test_metrics.sample_count, 5)
            self.assertGreater(result.train_history[0].batch_count, 0)
            self.assertGreater(result.validation_history[0].batch_count, 0)
            self.assertGreater(result.test_metrics.batch_count, 0)

            for metrics in (*result.train_history, *result.validation_history, result.test_metrics):
                self.assertGreaterEqual(metrics.physiology_reconstruction, 0.0)
                self.assertGreaterEqual(metrics.vehicle_reconstruction, 0.0)
                self.assertGreaterEqual(metrics.reconstruction_total, 0.0)
                self.assertGreaterEqual(metrics.alignment, 0.0)
                self.assertGreaterEqual(metrics.total, 0.0)

            self.assertIsNotNone(result.intermediate_export)
            assert result.intermediate_export is not None
            self.assertEqual(result.intermediate_export.partition, "test")
            self.assertEqual(result.intermediate_export.sample_count, 3)
            self.assertEqual(result.intermediate_export.reference_point_count, 4)
            self.assertEqual(len(result.intermediate_export.samples), 3)

            first_export = result.intermediate_export.samples[0]
            self.assertEqual(len(first_export.physiology.reference_offsets_s), 4)
            self.assertEqual(len(first_export.vehicle.reference_offsets_s), 4)
            self.assertGreaterEqual(first_export.physiology.mean_reference_projection_l2, 0.0)
            self.assertGreaterEqual(first_export.vehicle.mean_reference_projection_l2, 0.0)
            self.assertGreaterEqual(first_export.mean_reference_projection_cosine, -1.0)
            self.assertLessEqual(first_export.mean_reference_projection_cosine, 1.0)
else:
    class AlignmentPreviewPipelineRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
