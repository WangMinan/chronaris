"""Tests for Stage I multitask training and thesis-task builders."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    TASK_EVENT_REPLAY_TAG,
    TASK_RISK_PROXY,
    TASK_WORKLOAD_PROXY,
    THESIS_WEAK_LABEL_BENCHMARK_ROLE,
    THESIS_WEAK_LABEL_BOUNDARY,
    THESIS_WEAK_LABEL_ROLE,
    build_stage_i_real_task_payload,
)
from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix  # noqa: E402
from chronaris.models.alignment.config import AlignmentPrototypeConfig  # noqa: E402
from chronaris.models.alignment.splits import ChronologicalSplitConfig  # noqa: E402
from chronaris.pipelines.alignment_preview import AlignmentPreviewConfig  # noqa: E402
from chronaris.pipelines.stage_i.private.benchmark_data import (  # noqa: E402
    load_aligned_private_records,
)
from chronaris.pipelines.stage_i.training.multitask_train import (  # noqa: E402
    StageIMultitaskTrainConfig,
    run_stage_i_multitask_train,
)
from chronaris.schema.models import StreamKind  # noqa: E402

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_deep_pipeline_helpers",
    Path(__file__).resolve().with_name("test_stage_i_deep_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load synthetic Stage H helper")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_private_stage_h_run = _HELPER_MODULE._write_private_stage_h_run


class StageIMultitaskTrainTest(unittest.TestCase):
    def test_real_task_builder_separates_thesis_weak_label_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(root / "stage-h-e" / "run_manifest.json"),
                f_run_manifest_path=str(f_manifest),
            )

            payload = build_stage_i_real_task_payload(records)

            self.assertEqual(payload["summary"]["benchmark_role"], THESIS_WEAK_LABEL_BENCHMARK_ROLE)
            self.assertEqual(payload["summary"]["task_role"], THESIS_WEAK_LABEL_ROLE)
            self.assertEqual(payload["summary"]["thesis_task_boundary"], THESIS_WEAK_LABEL_BOUNDARY)
            self.assertEqual(
                payload["summary"]["task_counts"],
                {
                    TASK_RISK_PROXY: len(records),
                    TASK_WORKLOAD_PROXY: len(records),
                    TASK_EVENT_REPLAY_TAG: len(records),
                },
            )
            self.assertGreater(
                payload["summary"]["coverage"][TASK_EVENT_REPLAY_TAG]["valid_label_count"],
                0,
            )
            workload_entry = payload["by_task"][TASK_WORKLOAD_PROXY][0]
            self.assertGreaterEqual(float(workload_entry.label_value), 0.0)
            self.assertLessEqual(float(workload_entry.label_value), 1.0)
            retrieval_entry = payload["by_task"][TASK_EVENT_REPLAY_TAG][0]
            self.assertEqual(retrieval_entry.task_type, "retrieval")
            self.assertIn("event_replay_tag", retrieval_entry.context_payload)

    def test_multitask_train_runs_end_to_end_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(e_manifest),
                f_run_manifest_path=str(f_manifest),
            )
            payload = build_stage_i_real_task_payload(records)
            samples = _build_synthetic_multitask_samples(records)

            result = run_stage_i_multitask_train(
                StageIMultitaskTrainConfig(
                    run_id="stage-i-multitask-smoke",
                    output_root=str(root / "artifacts"),
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
                        physics_constraint_family="full",
                        vehicle_physics_weight=0.1,
                        physiology_physics_weight=0.1,
                        export_intermediate_states=False,
                        intermediate_partition="all",
                    ),
                    task_head_hidden_dim=8,
                    retrieval_embedding_dim=6,
                    task_loss_weight=1.0,
                    causal_weight=0.1,
                ),
                samples=samples,
                task_entries=payload["entries"],
                source_summary={"source": "synthetic_stage_h_records"},
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue(Path(result.task_manifest_path).exists())
            self.assertEqual(
                set(summary["task_heads"]),
                {TASK_RISK_PROXY, TASK_WORKLOAD_PROXY, TASK_EVENT_REPLAY_TAG},
            )
            self.assertGreater(summary["test_metrics"]["task_total"], 0.0)
            self.assertGreater(summary["test_metrics"]["causal_total"], 0.0)
            self.assertIn(TASK_RISK_PROXY, summary["test_metrics"]["task_components"])
            self.assertIn(TASK_WORKLOAD_PROXY, summary["test_metrics"]["task_components"])
            self.assertIn(TASK_EVENT_REPLAY_TAG, summary["test_metrics"]["task_components"])
            self.assertEqual(summary["source_summary"]["source"], "synthetic_stage_h_records")


def _write_pair(root: Path) -> tuple[Path, Path]:
    e_manifest = _write_private_stage_h_run(
        root,
        run_name="stage-h-e",
        amplitude_scale=0.8,
        physics_enabled=False,
    )
    f_manifest = _write_private_stage_h_run(
        root,
        run_name="stage-h-f",
        amplitude_scale=1.4,
        physics_enabled=True,
    )
    return e_manifest, f_manifest


def _build_synthetic_multitask_samples(records) -> tuple[E0ExperimentSample, ...]:
    samples: list[E0ExperimentSample] = []
    for row in records.sort_values(["start_offset_ms", "sample_id"]).itertuples(index=False):
        level = float(row.window_index + 1)
        pilot_scale = 0.05 * (int(row.pilot_id) % 3)
        physiology = NumericStreamMatrix(
            stream_kind=StreamKind.PHYSIOLOGY,
            point_count=4,
            feature_names=("eeg.alpha", "spo2.spo2"),
            point_offsets_ms=(0, 1000, 2000, 3000),
            point_measurements=("eeg", "eeg", "spo2", "spo2"),
            values=(
                (0.5 * level + pilot_scale, 97.5 - 0.2 * level),
                (0.6 * level + pilot_scale, 97.2 - 0.2 * level),
                (0.7 * level + pilot_scale, 96.9 - 0.2 * level),
                (0.8 * level + pilot_scale, 96.6 - 0.2 * level),
            ),
            dropped_fields=(),
        )
        vehicle = NumericStreamMatrix(
            stream_kind=StreamKind.VEHICLE,
            point_count=4,
            feature_names=("BUS001.speed", "BUS001.acc"),
            point_offsets_ms=(0, 1000, 2000, 3000),
            point_measurements=("BUS001", "BUS001", "BUS001", "BUS001"),
            values=(
                (100.0 + 10.0 * level + pilot_scale, 0.2 * level),
                (101.0 + 10.0 * level + pilot_scale, 0.3 * level),
                (102.0 + 10.0 * level + pilot_scale, 0.4 * level),
                (103.0 + 10.0 * level + pilot_scale, 0.5 * level),
            ),
            dropped_fields=(),
        )
        samples.append(
            E0ExperimentSample(
                sample_id=str(row.sample_id),
                sortie_id=str(row.sortie_id),
                start_offset_ms=int(row.start_offset_ms),
                end_offset_ms=int(row.end_offset_ms),
                physiology=physiology,
                vehicle=vehicle,
            )
        )
    return tuple(samples)


if __name__ == "__main__":
    unittest.main()
