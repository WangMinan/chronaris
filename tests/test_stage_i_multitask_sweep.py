"""Tests for Stage I thesis weak-label multitask sweep."""

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

from chronaris.dataset import build_stage_i_real_task_payload  # noqa: E402
from chronaris.pipelines.stage_i.stage_i_multitask_sweep import (  # noqa: E402
    StageIMultitaskSweepConfig,
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


if __name__ == "__main__":
    unittest.main()
