"""Tests for Stage I thesis weak-label multitask sweep."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import build_stage_i_real_task_payload  # noqa: E402
from chronaris.pipelines.stage_i.stage_i_multitask_sweep import (  # noqa: E402
    StageIMultitaskSweepConfig,
    discover_existing_child_summary_paths,
    run_stage_i_multitask_sweep,
)
from chronaris.pipelines.stage_i.stage_i_private_benchmark_data import (  # noqa: E402
    load_aligned_private_records,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_multitask_train_helpers",
    Path(__file__).resolve().with_name("test_stage_i_multitask_train.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load multitask sweep helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_pair = _HELPER_MODULE._write_pair
_build_synthetic_multitask_samples = _HELPER_MODULE._build_synthetic_multitask_samples
_ORIGINAL_SWEEP_RUNNER = run_stage_i_multitask_sweep.__globals__["run_stage_i_multitask_train"]


class StageIMultitaskSweepTest(unittest.TestCase):
    def test_multitask_sweep_writes_summary_table_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(e_manifest),
                f_run_manifest_path=str(f_manifest),
            )
            payload = build_stage_i_real_task_payload(records)
            samples = _build_synthetic_multitask_samples(records)

            result = run_stage_i_multitask_sweep(
                StageIMultitaskSweepConfig(
                    run_id="multitask-sweep-smoke",
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    physics_constraint_families=("minimal", "rigid_body"),
                    causal_weights=(0.0, 0.1),
                    task_loss_weights=(0.5,),
                    causal_lag_window_points=(None,),
                    max_runs=2,
                    epoch_count=1,
                    batch_size=len(samples),
                    device="cpu",
                    source_manifests={
                        "e_run_manifest_path": str(e_manifest),
                        "f_run_manifest_path": str(f_manifest),
                    },
                ),
                samples=samples,
                task_entries=payload["entries"],
                source_summary={"source": "synthetic_stage_h_records"},
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["evidence_layer"], "thesis_weak_label")
            self.assertEqual(summary["combination_count"], 2)
            self.assertEqual(len(summary["rows"]), 2)
            self.assertTrue(Path(result.table_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertEqual(
                summary["rows"][0]["source_manifests"]["e_run_manifest_path"],
                str(e_manifest),
            )
            self.assertIn("checkpoint_path", summary["best_run"])

    def test_multitask_sweep_writes_partial_summary_and_resume_avoids_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(e_manifest),
                f_run_manifest_path=str(f_manifest),
            )
            payload = build_stage_i_real_task_payload(records)
            samples = _build_synthetic_multitask_samples(records)
            config = StageIMultitaskSweepConfig(
                run_id="multitask-sweep-partial",
                output_root=str(root / "artifacts"),
                report_root=str(root / "reports"),
                physics_constraint_families=("minimal",),
                causal_weights=(0.0,),
                task_loss_weights=(0.5, 1.0),
                causal_lag_window_points=(None,),
                max_runs=2,
                epoch_count=1,
                batch_size=len(samples),
                device="cpu",
                source_manifests={
                    "e_run_manifest_path": str(e_manifest),
                    "f_run_manifest_path": str(f_manifest),
                },
            )

            call_state = {"count": 0}

            def flaky_runner(*args, **kwargs):
                call_state["count"] += 1
                if call_state["count"] == 2:
                    raise RuntimeError("synthetic blocker")
                return _ORIGINAL_SWEEP_RUNNER(*args, **kwargs)

            with self.assertRaisesRegex(RuntimeError, "synthetic blocker"), mock.patch(
                "chronaris.pipelines.stage_i.stage_i_multitask_sweep.run_stage_i_multitask_train",
                side_effect=flaky_runner,
            ):
                run_stage_i_multitask_sweep(
                    config,
                    samples=samples,
                    task_entries=payload["entries"],
                    source_summary={"source": "synthetic_stage_h_records"},
                )

            partial_summary_path = root / "artifacts" / "multitask-sweep-partial" / "partial_summary.json"
            partial_table_path = root / "artifacts" / "multitask-sweep-partial" / "thesis_weak_label_multitask_ablation.partial.csv"
            partial_summary = json.loads(partial_summary_path.read_text(encoding="utf-8"))
            self.assertEqual(partial_summary["status"], "partial_blocked")
            self.assertEqual(len(partial_summary["completed_child_runs"]), 1)
            self.assertEqual(partial_summary["blocked_at_run_index"], 2)
            self.assertTrue(partial_table_path.exists())
            self.assertEqual(call_state["count"], 2)

            existing_child_paths = discover_existing_child_summary_paths(
                StageIMultitaskSweepConfig(
                    run_id="multitask-sweep-partial",
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    physics_constraint_families=("minimal",),
                    causal_weights=(0.0,),
                    task_loss_weights=(0.5,),
                    causal_lag_window_points=(None,),
                    max_runs=1,
                    resume_existing=True,
                )
            )
            self.assertEqual(len(existing_child_paths), 1)

            with mock.patch(
                "chronaris.pipelines.stage_i.stage_i_multitask_sweep.run_stage_i_multitask_train",
                side_effect=AssertionError("resume should not rerun existing child summaries"),
            ):
                resumed_result = run_stage_i_multitask_sweep(
                    StageIMultitaskSweepConfig(
                        run_id="multitask-sweep-resumed",
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        physics_constraint_families=("minimal",),
                        causal_weights=(0.0,),
                        task_loss_weights=(0.5,),
                        causal_lag_window_points=(None,),
                        max_runs=1,
                        epoch_count=1,
                        batch_size=len(samples),
                        device="cpu",
                        resume_existing=True,
                        resume_run_root=str(root / "artifacts" / "multitask-sweep-partial"),
                        source_manifests={
                            "e_run_manifest_path": str(e_manifest),
                            "f_run_manifest_path": str(f_manifest),
                        },
                    ),
                    samples=(),
                    task_entries=payload["entries"],
                    source_summary={"source": "synthetic_stage_h_records"},
                )

            resumed_summary = json.loads(Path(resumed_result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(resumed_summary["status"], "completed")
            self.assertEqual(resumed_summary["derived_from_run_id"], "multitask-sweep-partial")
            self.assertEqual(resumed_summary["blocked_at_run_index"], 2)
            self.assertEqual(len(resumed_summary["completed_child_run_paths"]), 1)
            self.assertTrue(resumed_summary["blocked_attempt_log_paths"])

    def test_multitask_sweep_runtime_budget_guard_blocks_before_first_child(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            records = load_aligned_private_records(
                e_run_manifest_path=str(e_manifest),
                f_run_manifest_path=str(f_manifest),
            )
            payload = build_stage_i_real_task_payload(records)
            samples = _build_synthetic_multitask_samples(records)

            with self.assertRaisesRegex(Exception, "runtime budget"), mock.patch(
                "chronaris.pipelines.stage_i.stage_i_multitask_sweep.run_stage_i_multitask_train",
                side_effect=AssertionError("runtime budget guard should prevent child execution"),
            ):
                run_stage_i_multitask_sweep(
                    StageIMultitaskSweepConfig(
                        run_id="multitask-sweep-budget",
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        physics_constraint_families=("minimal",),
                        causal_weights=(0.0,),
                        task_loss_weights=(0.5,),
                        causal_lag_window_points=(None,),
                        max_runs=1,
                        max_runtime_seconds=0.0,
                        source_manifests={
                            "e_run_manifest_path": str(e_manifest),
                            "f_run_manifest_path": str(f_manifest),
                        },
                    ),
                    samples=samples,
                    task_entries=payload["entries"],
                    source_summary={"source": "synthetic_stage_h_records"},
                )

            partial_summary = json.loads(
                (root / "artifacts" / "multitask-sweep-budget" / "partial_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(partial_summary["status"], "partial_blocked")
            self.assertEqual(partial_summary["blocked_at_run_index"], 1)
            self.assertEqual(partial_summary["combination_count_completed"], 0)


if __name__ == "__main__":
    unittest.main()
