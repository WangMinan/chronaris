"""Tests for the task evaluation optimized model summary package."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.optimized_model_summary import (  # noqa: E402
    StageIOptimizedModelSummaryConfig,
    run_task_eval_optimized_model_summary,
)


class StageIOptimizedModelSummaryTest(unittest.TestCase):
    def test_summary_collects_metrics_gates_gpu_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            p30 = root / "p30"
            p31 = root / "p31"
            p32 = root / "p32"
            p34 = root / "p34"
            p35 = root / "p35"
            p36 = root / "p36"
            for path in (p30, p31, p32, p34, p35, p36):
                path.mkdir()

            _write_json(
                p34 / "task_head_optimization_summary.json",
                {
                    "status": "partial",
                    "seeds": [42, 17, 29],
                    "split_strategies": ["leave_one_view_out", "leave_one_sortie_out"],
                },
            )
            pd.DataFrame([{"epoch": epoch} for epoch in range(1, 6)]).to_csv(
                p34 / "training_curves.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "task_name": "T2_next_window_physiology_response",
                        "split_strategy": "leave_one_view_out",
                        "metric": "rmse",
                        "p30_chronaris_v1": 10.0,
                        "p34_chronaris_v2": 6.0,
                        "delta_abs_positive_is_better": 4.0,
                    }
                ]
            ).to_csv(p34 / "improvement_vs_p30.csv", index=False)
            _write_json(
                p34 / "gpu_perf_summary.json",
                {
                    "runtime_device": "cuda",
                    "gpu_name": "unit-test-gpu",
                    "tensor_cache_mode": "auto",
                    "best_batch_size": 2048,
                    "amp_mode": "bf16",
                    "torch_compile_mode": "off",
                    "max_memory_allocated_gb": 0.5,
                },
            )
            (p34 / "resume_command.txt").write_text("python p34 --device cuda --require-cuda\n", encoding="utf-8")

            _write_json(p35 / "stream_role_fusion_summary.json", {"status": "partial"})
            pd.DataFrame(
                [
                    {
                        "dataset_id": "private_feature_export",
                        "second_stream_role": "real_vehicle",
                        "fusion_route": "causal_lagged_vehicle_to_physio",
                        "lag_gate_mean": 0.8,
                        "context_gate_mean": 0.1,
                        "vehicle_gate_mean": 0.9,
                        "causal_gate_mean": 0.85,
                    }
                ]
            ).to_csv(p35 / "gate_statistics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "task_group": "combined",
                        "metric": "combined_macro_f1",
                        "variant_id": "full",
                        "value_mean": 0.3,
                        "full_value_mean": 0.3,
                        "delta_abs_mean": 0.0,
                        "p35_status": "reference_only_until_v3_confirm",
                    }
                ]
            ).to_csv(p35 / "public_metrics.csv", index=False)
            _write_json(
                p35 / "gpu_perf_summary.json",
                {
                    "runtime_device": "cuda",
                    "gpu_name": "unit-test-gpu",
                    "tensor_cache_mode": "auto",
                    "amp_mode": "bf16",
                    "torch_compile_mode": "off",
                    "max_gpu_memory_gb": 0.25,
                },
            )
            (p35 / "resume_command.txt").write_text("python p35 --device cuda --require-cuda\n", encoding="utf-8")

            _write_json(p36 / "optimized_reevaluation_summary.json", {"status": "partial"})
            _write_json(
                p36 / "model_selection_summary.json",
                {
                    "selection_status": "partial_until_full_p35_v3_confirm",
                    "positive_delta_convention": "regression baseline-optimized",
                },
            )
            (p36 / "resume_command.txt").write_text("python p36\n", encoding="utf-8")

            result = run_task_eval_optimized_model_summary(
                StageIOptimizedModelSummaryConfig(
                    run_id="test-summary",
                    p30_root=str(p30),
                    p31_root=str(p31),
                    p32_root=str(p32),
                    p34_root=str(p34),
                    p35_root=str(p35),
                    p36_root=str(p36),
                    artifact_root=str(root / "out"),
                    report_root=str(root / "reports"),
                )
            )

            self.assertEqual(result.summary["status"], "partial")
            self.assertEqual(result.summary["runtime_device"], "cuda")
            self.assertTrue(Path(result.report_path).exists())
            metrics = pd.read_csv(result.summary["key_metric_summary_csv"])
            self.assertEqual(float(metrics.loc[0, "delta_positive_is_better"]), 4.0)
            self.assertIn("3 seeds", str(metrics.loc[0, "boundary"]))
            self.assertIn("5 epochs", str(metrics.loc[0, "boundary"]))
            gpu = pd.read_csv(result.summary["gpu_runtime_summary_csv"])
            self.assertEqual(set(gpu["runtime_device"]), {"cuda"})
            resume = Path(result.summary["resume_commands_txt"]).read_text(encoding="utf-8")
            self.assertIn("--require-cuda", resume)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
