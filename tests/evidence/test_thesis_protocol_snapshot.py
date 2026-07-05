"""Tests for the P38 thesis protocol freeze builder."""

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

from chronaris.evidence.thesis_protocol import (  # noqa: E402
    MATRIX_COLUMNS,
    StageIThesisProtocolConfig,
    build_task_eval_thesis_protocol,
)


class StageIThesisProtocolTest(unittest.TestCase):
    def test_protocol_freeze_writes_registry_matrix_summary_and_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            p30 = root / "p30"
            p31 = root / "p31"
            p32 = root / "p32"
            p34 = root / "p34"
            p35 = root / "p35"
            p36 = root / "p36"
            p36_summary = root / "p36_summary"
            p37 = root / "p37"
            for path in (p30, p31, p32, p34, p35, p36, p36_summary, p37):
                path.mkdir()

            _write_json(p30 / "evidence_manifest.json", {"status": "completed"})
            _write_json(p31 / "evidence_manifest.json", {"status": "completed"})
            _write_json(p32 / "evidence_manifest.json", {"status": "completed"})
            _write_json(p34 / "task_head_optimization_summary.json", {"status": "completed"})
            _write_json(p35 / "stream_role_fusion_summary.json", {"status": "completed"})
            _write_json(p36 / "optimized_reevaluation_summary.json", {"status": "completed"})
            _write_json(p36_summary / "optimized_model_summary.json", {"status": "completed"})
            _write_json(p37 / "metric_calibration_summary.json", {"status": "completed"})

            pd.DataFrame(
                [
                    {
                        "evidence_quadrant": "private_thirdparty_comparison",
                        "dataset_role": "private_real_dual_stream",
                        "dataset_id": "private_feature_export",
                        "task_group": "T1_maneuver_intensity_class",
                        "model_or_component": "chronaris_full",
                        "metric": "macro_f1",
                        "value": 0.2,
                        "baseline": "",
                        "delta_abs": "",
                        "delta_rel_pct": "",
                        "protocol": "leakage_safe_v1/leave_one_view_out",
                        "artifact_path": str(p30 / "model_comparison_long.csv"),
                        "figure_path": "",
                        "midterm_use": "private model comparison",
                        "wording_boundary": "private proxy",
                    }
                ]
            ).to_csv(p32 / "cross_evidence_matrix.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "task_name": "T1_maneuver_intensity_class",
                        "model_name": "chronaris_v2_task_heads",
                        "split_strategy": "leave_one_view_out",
                        "metric": "macro_f1",
                        "value_mean": 0.3,
                        "value_std": 0.01,
                        "seed_count": 3,
                        "completed_fold_count": 9,
                        "sample_count": 333,
                        "higher_is_better": True,
                    }
                ]
            ).to_csv(p34 / "task_head_metrics_long.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_name": "T1_maneuver_intensity_class",
                        "split_strategy": "leave_one_view_out",
                        "metric": "macro_f1",
                        "delta_abs_positive_is_better": 0.1,
                    }
                ]
            ).to_csv(p34 / "improvement_vs_p30.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "task_name": "T1_maneuver_intensity_class",
                        "model_name": "v3_stream_role",
                        "split_strategy": "leave_one_view_out",
                        "metric": "macro_f1",
                        "value_mean": 0.31,
                        "seed_count": 3,
                        "source_stage": "P35_v3_confirm",
                        "p35_status": "v3_confirm",
                    }
                ]
            ).to_csv(p35 / "private_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "task_group": "combined",
                        "metric": "combined_macro_f1",
                        "variant_id": "v3_stream_role",
                        "value_mean": 0.4,
                        "full_value_mean": 0.35,
                        "delta_abs_mean": 0.05,
                        "value_count": 3,
                        "p35_status": "v3_confirm",
                    }
                ]
            ).to_csv(p35 / "public_metrics.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "stage": "P36",
                        "scope": "public_context_proxy",
                        "dataset_or_task": "nasa_csm",
                        "split_or_group": "combined",
                        "metric": "combined_macro_f1",
                        "reference": "P31 full",
                        "optimized": "P35 aggregate",
                        "optimized_value": 0.4,
                        "delta_positive_is_better": 0.05,
                        "status": "improved",
                        "boundary": "public context proxy",
                    }
                ]
            ).to_csv(p36_summary / "key_metric_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "claim": "unit",
                        "allowed_strength": "bounded",
                        "required_boundary": "do not overclaim",
                    }
                ]
            ).to_csv(p36_summary / "claim_boundary_summary.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "task_name": "T1_maneuver_intensity_class",
                        "model_name": "p37_t1",
                        "split_strategy": "leave_one_view_out",
                        "metric": "macro_f1",
                        "value_mean": 0.32,
                        "seed_count": 3,
                    }
                ]
            ).to_csv(p37 / "t1_calibration_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_name": "T3_paired_pilot_window_retrieval",
                        "model_name": "p37_t3",
                        "split_strategy": "leave_one_view_out",
                        "metric": "mrr",
                        "value_mean": 0.11,
                        "seed_count": 3,
                    }
                ]
            ).to_csv(p37 / "t3_metric_calibration_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_name": "T1_maneuver_intensity_class",
                        "split_strategy": "leave_one_view_out",
                        "metric": "macro_f1",
                        "delta_positive_is_better": 0.02,
                    },
                    {
                        "task_name": "T3_paired_pilot_window_retrieval",
                        "split_strategy": "leave_one_view_out",
                        "metric": "mrr",
                        "delta_positive_is_better": 0.0,
                    },
                ]
            ).to_csv(p37 / "p37_delta_vs_p34.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "variant_id": "p37_public",
                        "primary_metric": "combined_macro_f1",
                        "selection_score": 0.42,
                    }
                ]
            ).to_csv(p37 / "public_route_calibration_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "dataset_id": "nasa_csm",
                        "metric": "combined_macro_f1",
                        "p37_variant": "p37_public",
                        "reference": "P35 v3_stream_role",
                        "delta_positive_is_better": 0.02,
                    }
                ]
            ).to_csv(p37 / "p37_delta_vs_p35.csv", index=False)

            result = build_task_eval_thesis_protocol(
                StageIThesisProtocolConfig(
                    run_id="test-thesis-protocol",
                    p30_root=str(p30),
                    p31_root=str(p31),
                    p32_root=str(p32),
                    p34_root=str(p34),
                    p35_root=str(p35),
                    p36_root=str(p36),
                    p36_summary_root=str(p36_summary),
                    p37_root=str(p37),
                    artifact_root=str(root / "out"),
                    report_root=str(root / "reports"),
                )
            )

            summary = result.summary
            self.assertEqual(summary["status"], "completed")
            self.assertGreaterEqual(summary["matrix_rows"], 7)
            self.assertTrue(Path(summary["evidence_manifest_path"]).exists())
            matrix = pd.read_csv(summary["result_matrix_long_csv"])
            for column in MATRIX_COLUMNS:
                self.assertIn(column, matrix.columns)
            self.assertIn("P37", set(matrix["source_stage"]))
            self.assertIn("public_model_comparison", set(matrix["evidence_quadrant"]))
            claims = pd.read_csv(summary["claim_boundary_table_csv"])
            self.assertIn("synthetic_future_scope", set(claims["claim_id"]))
            registry = pd.read_csv(summary["experiment_registry_csv"])
            self.assertIn("P36_summary", set(registry["stage"]))
            report = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("不重跑训练", report)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
