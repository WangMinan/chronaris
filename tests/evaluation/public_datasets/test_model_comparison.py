"""Tests for task evaluation public model-comparison artifact builder."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.model_comparison import (  # noqa: E402
    StageIPublicModelComparisonConfig,
    build_task_eval_public_model_comparison,
)


class StageIPublicModelComparisonTest(unittest.TestCase):
    def test_builder_writes_midterm_comparison_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = build_task_eval_public_model_comparison(
                StageIPublicModelComparisonConfig(
                    run_id="test-public-model-comparison",
                    artifact_root=str(root / "assets"),
                    report_root=str(root / "reports"),
                )
            )
            self.assertTrue(Path(result.long_csv_path).exists())
            self.assertTrue(Path(result.wide_csv_path).exists())
            self.assertTrue(Path(result.improvement_summary_csv_path).exists())
            self.assertTrue(Path(result.evidence_manifest_path).exists())
            self.assertTrue(Path(result.report_path).exists())

            wide = pd.read_csv(result.wide_csv_path)
            nasa = wide[
                (wide["dataset_id"] == "nasa_csm")
                & (wide["task_group"] == "combined")
                & (wide["metric_name"] == "macro_f1")
            ].iloc[0]
            self.assertAlmostEqual(
                float(nasa["public_baseline"]),
                0.4549915569394342,
                places=6,
            )
            self.assertAlmostEqual(
                float(nasa["chronaris_public_fusion_current"]),
                0.5615950767425805,
                places=6,
            )
            self.assertGreater(float(nasa["chronaris_vs_public_abs_delta"]), 0.10)

            manifest = json.loads(Path(result.evidence_manifest_path).read_text(encoding="utf-8"))
            self.assertIn("figure_paths", manifest)
            self.assertIn("p28_refresh_included", manifest)
            self.assertIn("missing_metrics", manifest)
            for figure_path in manifest["figure_paths"].values():
                self.assertTrue(Path(figure_path).exists())

            improvement = pd.read_csv(result.improvement_summary_csv_path)
            self.assertTrue(
                (
                    (improvement["dataset_id"] == "nasa_csm")
                    & (improvement["task_group"] == "combined")
                    & (improvement["metric_name"] == "balanced_accuracy")
                ).any()
            )
            self.assertTrue(
                (
                    (improvement["dataset_id"] == "uab_workload_dataset")
                    & (improvement["task_group"] == "subjective_mean")
                    & (improvement["metric_name"] == "mean_rmse")
                ).any()
            )


if __name__ == "__main__":
    unittest.main()
