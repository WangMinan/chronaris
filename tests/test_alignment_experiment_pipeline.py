"""Tests for the composed Stage E experiment pipeline."""

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
    from chronaris.pipelines.alignment_experiment import AlignmentExperimentPipeline
    from chronaris.pipelines.alignment_preview import AlignmentPreviewConfig, AlignmentPreviewPipeline
    from chronaris.schema.models import SortieLocator, StreamKind

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


    class StubE0Pipeline:
        def __init__(self, samples: tuple[E0ExperimentSample, ...]) -> None:
            self.samples = samples

        def run(self, locator: SortieLocator) -> tuple[E0ExperimentSample, ...]:
            assert locator.sortie_id == "sortie-001"
            return self.samples


    class AlignmentExperimentPipelineTest(unittest.TestCase):
        def test_alignment_experiment_pipeline_renders_markdown_summary(self) -> None:
            samples = tuple(_sample(index) for index in range(25))
            pipeline = AlignmentExperimentPipeline(
                e0_pipeline=StubE0Pipeline(samples),
                alignment_preview_pipeline=AlignmentPreviewPipeline(
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
                        epoch_count=1,
                        batch_size=4,
                        learning_rate=1e-3,
                        device="cpu",
                    )
                ),
            )

            result = pipeline.run(SortieLocator(sortie_id="sortie-001"))

            self.assertEqual(result.sample_summary.sample_count, 25)
            self.assertEqual(len(result.preview_result.split.train), 15)
            self.assertIsNotNone(result.preview_result.intermediate_export)
            assert result.preview_result.intermediate_export is not None
            self.assertEqual(result.preview_result.intermediate_export.partition, "test")
            self.assertIn("# Alignment Preview - sortie-001", result.report_markdown)
            self.assertIn("- sample count: `25`", result.report_markdown)
            self.assertIn("- train: `15`", result.report_markdown)
            self.assertIn("## Final Train Metrics", result.report_markdown)
            self.assertIn("## Reference Intermediate Export", result.report_markdown)
            self.assertIn("- partition: `test`", result.report_markdown)
            self.assertIn("- exported sample count: `3`", result.report_markdown)
            self.assertIn("## Test Metrics", result.report_markdown)
else:
    class AlignmentExperimentPipelineRuntimeDisabledTest(unittest.TestCase):
        @unittest.skip("torch runtime tests are disabled on this machine; enable in a suitable environment.")
        def test_torch_runtime_disabled(self) -> None:
            pass


if __name__ == "__main__":
    unittest.main()
