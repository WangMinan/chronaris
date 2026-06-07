"""Tests for streaming windows and checkpoint-backed runtime inference."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import StageIPrivateTaskEntry, StreamingPointEvent, StreamingWindowBuffer  # noqa: E402
from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix  # noqa: E402
from chronaris.pipelines import AlignmentPreviewConfig, StageIMultitaskTrainConfig, run_stage_i_multitask_train  # noqa: E402
from chronaris.models.alignment import AlignmentPrototypeConfig, ChronologicalSplitConfig  # noqa: E402
from chronaris.schema.models import AlignedPoint, RawPoint, StreamKind, WindowConfig  # noqa: E402
from chronaris.serving import (  # noqa: E402
    StageIRuntimeInferenceConfig,
    dump_runtime_samples_jsonl,
    load_runtime_samples_jsonl,
    run_stage_i_runtime_inference_batch,
    run_stage_i_runtime_inference_incremental,
    run_stage_i_runtime_inference,
)


class StreamingWindowBufferTest(unittest.TestCase):
    def test_streaming_window_buffer_emits_incremental_windows(self) -> None:
        buffer = StreamingWindowBuffer(
            sortie_id="sortie-001",
            window_config=WindowConfig(duration_ms=2000, stride_ms=1000),
        )
        events = [
            _event(StreamKind.PHYSIOLOGY, offset_ms=0, measurement="eeg", values={"alpha": 0.1}),
            _event(StreamKind.VEHICLE, offset_ms=0, measurement="bus", values={"speed": 100.0}),
            _event(StreamKind.PHYSIOLOGY, offset_ms=1000, measurement="eeg", values={"alpha": 0.2}),
            _event(StreamKind.VEHICLE, offset_ms=1000, measurement="bus", values={"speed": 101.0}),
            _event(StreamKind.PHYSIOLOGY, offset_ms=2000, measurement="eeg", values={"alpha": 0.3}),
            _event(StreamKind.VEHICLE, offset_ms=2000, measurement="bus", values={"speed": 102.0}),
        ]

        emitted = []
        for event in events:
            emitted.extend(buffer.push(event))
        emitted.extend(buffer.flush())

        self.assertEqual(len(emitted), 2)
        self.assertEqual(emitted[0].start_offset_ms, 0)
        self.assertEqual(emitted[0].end_offset_ms, 2000)
        self.assertEqual(emitted[1].start_offset_ms, 1000)
        self.assertEqual(emitted[1].end_offset_ms, 3000)

    def test_streaming_window_buffer_tracks_out_of_order_points(self) -> None:
        buffer = StreamingWindowBuffer(
            sortie_id="sortie-001",
            window_config=WindowConfig(duration_ms=2000, stride_ms=1000),
            max_cached_points_per_stream=4,
        )
        buffer.push(_event(StreamKind.PHYSIOLOGY, offset_ms=1000, measurement="eeg", values={"alpha": 0.1}))
        buffer.push(_event(StreamKind.PHYSIOLOGY, offset_ms=0, measurement="eeg", values={"alpha": 0.2}))

        diagnostics = buffer.diagnostics
        self.assertEqual(diagnostics.out_of_order_event_count, 1)
        self.assertEqual(diagnostics.max_cached_points_per_stream, 4)
        self.assertEqual(diagnostics.physiology_cached_points, 2)


class StageIRuntimeInferenceTest(unittest.TestCase):
    def test_runtime_inference_loads_checkpoint_and_writes_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _build_samples()
            task_entries = _build_task_entries(samples)
            train_result = run_stage_i_multitask_train(
                StageIMultitaskTrainConfig(
                    run_id="runtime-train",
                    output_root=str(root / "train"),
                    preview_config=AlignmentPreviewConfig(
                        prototype_config=AlignmentPrototypeConfig(
                            hidden_dim=8,
                            embedding_dim=8,
                            encoder_hidden_dim=12,
                            decoder_hidden_dim=12,
                            dynamics_hidden_dim=12,
                            projection_dim=4,
                            activation="relu",
                            ode_method="euler",
                        ),
                        split_config=ChronologicalSplitConfig(
                            train_ratio=1.0,
                            validation_ratio=0.0,
                            test_ratio=0.0,
                        ),
                        epoch_count=1,
                        batch_size=len(samples),
                        learning_rate=1e-3,
                        device="cpu",
                        reconstruction_loss_mode="relative_mse",
                        input_normalization_mode="zscore_train",
                        alignment_loss_mode="mse",
                        enable_physics_constraints=True,
                        physics_constraint_family="rigid_body",
                        vehicle_physics_weight=0.1,
                        physiology_physics_weight=0.1,
                        export_intermediate_states=False,
                        intermediate_partition="all",
                    ),
                ),
                samples=samples,
                task_entries=task_entries,
                source_summary={"source": "runtime_test"},
            )

            sample_jsonl_path = root / "runtime_samples.jsonl"
            dump_runtime_samples_jsonl(samples, path=sample_jsonl_path)
            reloaded_samples = load_runtime_samples_jsonl(sample_jsonl_path)
            self.assertEqual(tuple(sample.sample_id for sample in reloaded_samples), tuple(sample.sample_id for sample in samples))

            runtime_result = run_stage_i_runtime_inference(
                StageIRuntimeInferenceConfig(
                    run_id="runtime-infer",
                    checkpoint_path=train_result.checkpoint_path,
                    artifact_root=str(root / "runtime"),
                    report_root=str(root / "reports"),
                    device="cpu",
                ),
                samples=reloaded_samples,
            )

            self.assertTrue(Path(runtime_result.summary_path).exists())
            self.assertTrue(Path(runtime_result.report_path).exists())
            self.assertTrue(Path(runtime_result.predictions_csv_path).exists())

            summary = json.loads(Path(runtime_result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_count"], len(samples))
            self.assertEqual(summary["semantic_event"]["query_count"], 3)
            self.assertGreater(summary["semantic_event"]["mean_event_token_count"], 0.0)
            self.assertGreater(summary["mean_top_event_score"], 0.0)
            first_row = summary["samples"][0]
            self.assertIn("risk_proxy_prediction", first_row)
            self.assertIn("workload_proxy_prediction", first_row)
            self.assertIn("event_replay_tag_prediction", first_row)
            self.assertIn("semantic_top_query_name", first_row)

    def test_runtime_inference_strict_schema_rejects_feature_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _build_samples()
            task_entries = _build_task_entries(samples)
            train_result = run_stage_i_multitask_train(
                _build_train_config(root, sample_count=len(samples)),
                samples=samples,
                task_entries=task_entries,
                source_summary={"source": "runtime_test"},
            )
            mismatched = list(samples)
            mismatched[0] = E0ExperimentSample(
                sample_id=samples[0].sample_id,
                sortie_id=samples[0].sortie_id,
                start_offset_ms=samples[0].start_offset_ms,
                end_offset_ms=samples[0].end_offset_ms,
                physiology=samples[0].physiology,
                vehicle=NumericStreamMatrix(
                    stream_kind=StreamKind.VEHICLE,
                    point_count=samples[0].vehicle.point_count,
                    feature_names=samples[0].vehicle.feature_names + ("extra.vehicle",),
                    point_offsets_ms=samples[0].vehicle.point_offsets_ms,
                    point_measurements=samples[0].vehicle.point_measurements,
                    values=tuple(row + (0.0,) for row in samples[0].vehicle.values),
                    dropped_fields=(),
                ),
            )

            with self.assertRaisesRegex(ValueError, "runtime feature schema mismatch"):
                run_stage_i_runtime_inference(
                    StageIRuntimeInferenceConfig(
                        run_id="runtime-strict",
                        checkpoint_path=train_result.checkpoint_path,
                        artifact_root=str(root / "runtime"),
                        report_root=str(root / "reports"),
                        device="cpu",
                        strict_feature_schema=True,
                    ),
                    samples=tuple(mismatched),
                )

    def test_runtime_inference_batch_and_incremental_match_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _build_samples()
            task_entries = _build_task_entries(samples)
            train_result = run_stage_i_multitask_train(
                _build_train_config(root, sample_count=len(samples)),
                samples=samples,
                task_entries=task_entries,
                source_summary={"source": "runtime_test"},
            )

            batch_result = run_stage_i_runtime_inference_batch(
                StageIRuntimeInferenceConfig(
                    run_id="runtime-batch",
                    checkpoint_path=train_result.checkpoint_path,
                    artifact_root=str(root / "runtime_batch"),
                    report_root=str(root / "reports"),
                    device="cpu",
                    batch_size=2,
                    emit_predictions_jsonl=True,
                    max_windows=3,
                ),
                samples=samples,
            )
            incremental_result = run_stage_i_runtime_inference_incremental(
                StageIRuntimeInferenceConfig(
                    run_id="runtime-incremental",
                    checkpoint_path=train_result.checkpoint_path,
                    artifact_root=str(root / "runtime_incremental"),
                    report_root=str(root / "reports"),
                    device="cpu",
                    max_windows=3,
                ),
                samples=samples,
            )

            self.assertEqual(batch_result.summary["sample_count"], 3)
            self.assertEqual(incremental_result.summary["sample_count"], 3)
            self.assertEqual(batch_result.summary["sample_count"], incremental_result.summary["sample_count"])
            self.assertTrue(Path(batch_result.predictions_jsonl_path).exists())
            self.assertEqual(batch_result.summary["diagnostics"]["feature_schema_status"], "exact")
            self.assertEqual(incremental_result.summary["diagnostics"]["batch_size"], 1)

            jsonl_rows = [
                json.loads(line)
                for line in Path(batch_result.predictions_jsonl_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(jsonl_rows), 3)


def _event(
    stream_kind: StreamKind,
    *,
    offset_ms: int,
    measurement: str,
    values: dict[str, float],
) -> StreamingPointEvent:
    return StreamingPointEvent(
        stream_kind=stream_kind,
        point=AlignedPoint(
            point=RawPoint(
                stream_kind=stream_kind,
                measurement=measurement,
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=offset_ms),
                values=values,
            ),
            offset_ms=offset_ms,
        ),
    )


def _build_samples() -> tuple[E0ExperimentSample, ...]:
    samples = []
    for index in range(4):
        physiology = NumericStreamMatrix(
            stream_kind=StreamKind.PHYSIOLOGY,
            point_count=4,
            feature_names=("eeg.alpha", "spo2.spo2"),
            point_offsets_ms=(0, 1000, 2000, 3000),
            point_measurements=("eeg", "eeg", "spo2", "spo2"),
            values=(
                (0.1 + 0.1 * index, 97.5 - 0.1 * index),
                (0.2 + 0.1 * index, 97.2 - 0.1 * index),
                (0.3 + 0.1 * index, 96.9 - 0.1 * index),
                (0.4 + 0.1 * index, 96.6 - 0.1 * index),
            ),
            dropped_fields=(),
        )
        vehicle = NumericStreamMatrix(
            stream_kind=StreamKind.VEHICLE,
            point_count=4,
            feature_names=("speed", "acc", "altitude", "vertical_speed", "pitch", "pitch_rate"),
            point_offsets_ms=(0, 1000, 2000, 3000),
            point_measurements=("bus", "bus", "bus", "bus"),
            values=(
                (100.0 + index, 1.0, 1000.0, 0.0, 0.0, 0.0),
                (101.0 + index, 1.0, 1001.0, 0.5, 1.0, 0.5),
                (102.0 + index, 1.0, 1003.0, 0.5, 3.0, 0.5),
                (103.0 + index, 1.0, 1006.0, 0.5, 6.0, 0.5),
            ),
            dropped_fields=(),
        )
        samples.append(
            E0ExperimentSample(
                sample_id=f"sample-{index}",
                sortie_id="sortie-001",
                start_offset_ms=index * 3000,
                end_offset_ms=(index + 1) * 3000,
                physiology=physiology,
                vehicle=vehicle,
            )
        )
    return tuple(samples)


def _build_train_config(root: Path, *, sample_count: int) -> StageIMultitaskTrainConfig:
    return StageIMultitaskTrainConfig(
        run_id="runtime-train",
        output_root=str(root / "train"),
        preview_config=AlignmentPreviewConfig(
            prototype_config=AlignmentPrototypeConfig(
                hidden_dim=8,
                embedding_dim=8,
                encoder_hidden_dim=12,
                decoder_hidden_dim=12,
                dynamics_hidden_dim=12,
                projection_dim=4,
                activation="relu",
                ode_method="euler",
            ),
            split_config=ChronologicalSplitConfig(
                train_ratio=1.0,
                validation_ratio=0.0,
                test_ratio=0.0,
            ),
            epoch_count=1,
            batch_size=sample_count,
            learning_rate=1e-3,
            device="cpu",
            reconstruction_loss_mode="relative_mse",
            input_normalization_mode="zscore_train",
            alignment_loss_mode="mse",
            enable_physics_constraints=True,
            physics_constraint_family="rigid_body",
            vehicle_physics_weight=0.1,
            physiology_physics_weight=0.1,
            export_intermediate_states=False,
            intermediate_partition="all",
        ),
    )


def _build_task_entries(samples: Sequence[E0ExperimentSample]) -> tuple[StageIPrivateTaskEntry, ...]:
    entries = []
    labels = ("low", "medium", "high", "medium")
    workloads = (0.2, 0.4, 0.8, 0.5)
    pairs = ("sample-1", "sample-0", "sample-3", "sample-2")
    for index, sample in enumerate(samples):
        entries.append(
            StageIPrivateTaskEntry(
                sample_id=sample.sample_id,
                sortie_id=sample.sortie_id,
                pilot_id=10030 + index,
                view_id=f"view-{index % 2}",
                window_index=index,
                sample_partition="train",
                task_name="risk_proxy",
                task_type="classification",
                label_name="risk_proxy",
                label_value=labels[index],
                label_source="synthetic",
                source_refs={"source": "test"},
                benchmark_role="thesis_weak_label_benchmark",
                task_role="thesis_weak_label_task",
            )
        )
        entries.append(
            StageIPrivateTaskEntry(
                sample_id=sample.sample_id,
                sortie_id=sample.sortie_id,
                pilot_id=10030 + index,
                view_id=f"view-{index % 2}",
                window_index=index,
                sample_partition="train",
                task_name="workload_proxy",
                task_type="regression",
                label_name="workload_proxy",
                label_value=workloads[index],
                label_source="synthetic",
                source_refs={"source": "test"},
                benchmark_role="thesis_weak_label_benchmark",
                task_role="thesis_weak_label_task",
            )
        )
        entries.append(
            StageIPrivateTaskEntry(
                sample_id=sample.sample_id,
                sortie_id=sample.sortie_id,
                pilot_id=10030 + index,
                view_id=f"view-{index % 2}",
                window_index=index,
                sample_partition="train",
                task_name="event_replay_tag",
                task_type="retrieval",
                label_name="event_replay_tag",
                label_value="paired_window",
                label_source="synthetic",
                source_refs={"source": "test"},
                benchmark_role="thesis_weak_label_benchmark",
                task_role="thesis_weak_label_task",
                paired_sample_id=pairs[index],
            )
        )
    return tuple(entries)


if __name__ == "__main__":
    unittest.main()
