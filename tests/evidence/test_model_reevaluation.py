"""Tests for P36 optimized Chronaris re-evaluation aggregation."""

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

from chronaris.evidence.optimized_reevaluation import (  # noqa: E402
    StageIOptimizedReevaluationConfig,
    run_task_eval_optimized_reevaluation,
)


class StageIOptimizedReevaluationTest(unittest.TestCase):
    def test_delta_direction_positive_means_optimized_better(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            p30 = root / "p30"
            p31 = root / "p31"
            p32 = root / "p32"
            p34 = root / "p34"
            p35 = root / "p35"
            for path in (p30, p31, p32, p34, p35):
                path.mkdir()
            _write_json(p34 / "task_head_optimization_summary.json", {"status": "completed", "runtime_device": "cuda"})
            _write_json(p35 / "stream_role_fusion_summary.json", {"status": "completed", "runtime_device": "cuda"})
            pd.DataFrame(
                [
                    {
                        "task_name": "T2_next_window_physiology_response",
                        "split_strategy": "leave_one_view_out",
                        "metric": "rmse",
                        "chronaris_full": 10.0,
                        "mult": 8.0,
                        "contiformer": 7.0,
                    }
                ]
            ).to_csv(p30 / "model_comparison_wide.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_name": "T2_next_window_physiology_response",
                        "split_strategy": "leave_one_view_out",
                        "metric": "rmse",
                        "chronaris_v2_task_heads": 6.0,
                    }
                ]
            ).to_csv(p34 / "task_head_metrics_wide.csv", index=False)
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
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "task_group": "combined",
                        "metric": "combined_macro_f1",
                        "variant_id": "full",
                        "value_mean": 0.3,
                    }
                ]
            ).to_csv(p31 / "ablation_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "task_group": "combined",
                        "metric": "combined_macro_f1",
                        "variant_id": "full",
                        "value_mean": 0.3,
                    }
                ]
            ).to_csv(p35 / "public_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_name": "T2_next_window_physiology_response",
                        "model_name": "chronaris_v3_stream_role_fusion",
                        "split_strategy": "leave_one_view_out",
                        "metric": "rmse",
                        "value_mean": 5.0,
                        "source_stage": "P35_v3_confirm",
                    }
                ]
            ).to_csv(p35 / "private_metrics.csv", index=False)
            pd.DataFrame().to_csv(p35 / "comparison_vs_p31.csv", index=False)
            pd.DataFrame(
                [{"evidence_quadrant": "base", "midterm_use": "reference"}]
            ).to_csv(p32 / "cross_evidence_matrix.csv", index=False)

            result = run_task_eval_optimized_reevaluation(
                StageIOptimizedReevaluationConfig(
                    run_id="test-p36",
                    p30_root=str(p30),
                    p31_root=str(p31),
                    p32_root=str(p32),
                    p34_root=str(p34),
                    p35_root=str(p35),
                    artifact_root=str(root / "out"),
                    report_root=str(root / "reports"),
                )
            )
            self.assertEqual(result.summary["status"], "completed")
            self.assertFalse((Path(result.artifact_root) / "partial_summary.json").exists())
            delta = pd.read_csv(result.summary["optimized_delta_vs_p30_csv"])
            self.assertEqual(float(delta["delta_abs_positive_is_better"].iloc[0]), 4.0)
            self.assertEqual(result.summary["runtime_device"], "cuda")
            private = pd.read_csv(result.summary["optimized_private_comparison_csv"])
            self.assertIn("P35_v3_confirm", set(private["source"]))
            model_summary = Path(result.summary["model_selection_summary_json"]).read_text(encoding="utf-8")
            self.assertIn('"runtime_device": "cuda"', model_summary)
            self.assertIn('"selection_status": "completed_requested_p35_v3_confirm"', model_summary)
            self.assertTrue(Path(result.report_path).exists())


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
