"""Tests for Stage I leakage-safe private proxy ablation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.pipelines.stage_i.private.leakage_audit import (  # noqa: E402
    LabelFeatureLeakageError,
    audit_label_feature_overlap,
)
from chronaris.pipelines.stage_i.private.leakage_safe_ablation import (  # noqa: E402
    MAX_T3_DISTRIBUTION_ROWS,
    StageILeakageSafeAblationConfig,
    _compact_t3_similarity_distribution,
    run_stage_i_leakage_safe_ablation,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_leakage_safe_helpers",
    Path(__file__).resolve().with_name("test_stage_i_deep_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load private Stage H synthetic helper")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_private_stage_h_run = _HELPER_MODULE._write_private_stage_h_run


class StageILeakageAuditTest(unittest.TestCase):
    def test_direct_label_source_overlap_fails(self) -> None:
        with self.assertRaisesRegex(LabelFeatureLeakageError, "BUS001_speed"):
            audit_label_feature_overlap(
                task_name="T1_maneuver_intensity_class",
                label_source_fields=("BUS001.speed",),
                input_feature_fields=("feat__veh__BUS001_speed__mean",),
                leakage_safe=True,
            )

    def test_deterministic_label_feature_fails(self) -> None:
        with self.assertRaisesRegex(LabelFeatureLeakageError, "vehicle_proxy_score"):
            audit_label_feature_overlap(
                task_name="T1_maneuver_intensity_class",
                label_source_fields=("BUS001.speed",),
                input_feature_fields=("feat__residual__vehicle_proxy_score",),
                leakage_safe=True,
            )

    def test_t3_temporal_identity_feature_fails(self) -> None:
        with self.assertRaisesRegex(LabelFeatureLeakageError, "window_index"):
            audit_label_feature_overlap(
                task_name="T3_paired_pilot_window_retrieval",
                label_source_fields=("sample_id", "sortie_id", "pilot_id", "window_index"),
                input_feature_fields=("feat__ctx__window_index", "feat__opt_fused__dim_000__mean"),
                leakage_safe=True,
            )

    def test_t2_target_window_feature_fails(self) -> None:
        with self.assertRaisesRegex(LabelFeatureLeakageError, "target_window"):
            audit_label_feature_overlap(
                task_name="T2_next_window_physiology_response",
                label_source_fields=("next_window::eeg.alpha",),
                input_feature_fields=("feat__target_window_physiology_eeg_alpha",),
                leakage_safe=True,
            )


class StageILeakageSafeAblationTest(unittest.TestCase):
    def test_t3_distribution_compaction_retains_positive_pairs(self) -> None:
        rows = [
            {
                "seed": 17,
                "variant_name": "full_model",
                "query_sample_id": "q0",
                "candidate_sample_id": f"positive-{index}",
                "is_positive": 1,
                "similarity": 0.9,
            }
            for index in range(4)
        ]
        rows.extend(
            {
                "seed": 17,
                "variant_name": "full_model",
                "query_sample_id": f"q{index}",
                "candidate_sample_id": f"negative-{index}",
                "is_positive": 0,
                "similarity": 0.1,
            }
            for index in range(MAX_T3_DISTRIBUTION_ROWS + 100)
        )
        compact = _compact_t3_similarity_distribution(rows)
        self.assertEqual(len(compact), MAX_T3_DISTRIBUTION_ROWS)
        self.assertEqual(sum(int(row["is_positive"]) for row in compact), 4)

    def test_leakage_safe_ablation_writes_smoke_outputs_and_aggregates_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)

            result = run_stage_i_leakage_safe_ablation(
                StageILeakageSafeAblationConfig(
                    run_id="leakage-safe-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    seeds=(17, 29, 43),
                )
            )

            for path in (
                result.summary_path,
                result.report_path,
                result.label_feature_audit_json_path,
                result.label_feature_audit_csv_path,
                result.seed_metrics_path,
                result.split_manifest_path,
                result.model_backbone_csv_path,
                result.task_adapter_csv_path,
                result.model_backbone_figure_path,
                result.task_adapter_figure_path,
            ):
                self.assertTrue(Path(path).exists(), path)
                self.assertGreater(Path(path).stat().st_size, 0, path)

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["protocol"], "leakage_safe_v1")
            self.assertEqual(summary["audit_status"], "pass")
            self.assertTrue(summary["leakage_safe"])
            self.assertTrue(all(row["protocol"] == "leakage_safe_v1" for row in summary["rows"]))
            self.assertTrue(any(row["seed_count"] == 3 for row in summary["rows"]))
            self.assertIn("model_backbone_ablation_png", summary["paths"])
            self.assertIn("task_adapter_ablation_png", summary["paths"])

            seed_metrics = pd.read_csv(result.seed_metrics_path)
            self.assertIn("persistence_improvement_rate", seed_metrics.columns)
            t3_rows = seed_metrics.loc[seed_metrics["task_name"] == "T3_paired_pilot_window_retrieval"]
            self.assertTrue((t3_rows["valid_query_count"].dropna() > 0).any())
            self.assertTrue((t3_rows["skipped_fold_count"].dropna() > 0).any())
            self.assertTrue((t3_rows["candidate_pool_policy"].dropna() == "same_sortie_cross_pilot").any())
            self.assertTrue(any("top5_accuracy" in row for row in summary["rows"] if row["task_name"] == "T3_paired_pilot_window_retrieval"))

    def test_same_seed_reproduces_summary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest, f_manifest = _write_pair(root)
            first = run_stage_i_leakage_safe_ablation(
                StageILeakageSafeAblationConfig(
                    run_id="leakage-safe-repeat-a",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    seeds=(17,),
                )
            )
            second = run_stage_i_leakage_safe_ablation(
                StageILeakageSafeAblationConfig(
                    run_id="leakage-safe-repeat-b",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    seeds=(17,),
                )
            )
            first_summary = json.loads(Path(first.summary_path).read_text(encoding="utf-8"))
            second_summary = json.loads(Path(second.summary_path).read_text(encoding="utf-8"))
            first_rows = [
                {key: row[key] for key in ("task_name", "variant_name", "primary_metric_value", "delta_vs_full")}
                for row in first_summary["rows"]
            ]
            second_rows = [
                {key: row[key] for key in ("task_name", "variant_name", "primary_metric_value", "delta_vs_full")}
                for row in second_summary["rows"]
            ]
            self.assertEqual(first_rows, second_rows)


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


if __name__ == "__main__":
    unittest.main()
