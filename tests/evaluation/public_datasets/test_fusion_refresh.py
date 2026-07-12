"""Tests for P28 public fusion refresh workflow."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

import pandas as pd

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.fusion_refresh import (  # noqa: E402
    StageIPublicFusionRefreshConfig,
    _locked_public_bridge_candidate,
    build_public_fusion_refresh_candidates,
    run_task_eval_public_fusion_refresh,
)
from chronaris.evaluation.public_datasets.pipelines.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_task_eval_sequence_preparation,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "task_eval_pipeline_helpers",
    SRC.parent / "tests" / "evaluation" / "public_datasets" / "test_pipeline.py",
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("failed to load task evaluation synthetic dataset helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_mini_uab_dataset = _HELPER_MODULE._write_mini_uab_dataset
_write_mini_nasa_csm_dataset = _HELPER_MODULE._write_mini_nasa_csm_dataset


class StageIPublicFusionRefreshTest(unittest.TestCase):
    def test_external_confirmation_uses_one_candidate_bound_to_v2_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lock = root / "lock.json"
            promotion = root / "promotion.json"
            lock.write_text(json.dumps({
                "format": "chronaris.v2_locked_configuration.v1",
                "configuration_locked": True,
                "selection_uses_downstream_labels": False,
                "candidate": {
                    "candidate_id": "v2_locked",
                    "internal_hidden_dim": 96,
                    "dropout": 0.1,
                    "learning_rate": 1e-3,
                },
            }), encoding="utf-8")
            promotion.write_text(json.dumps({
                "status": "completed",
                "locked_results_returned_to_development": False,
            }), encoding="utf-8")
            config = StageIPublicFusionRefreshConfig(
                run_id="external",
                dataset_prepared_roots={},
                locked_configuration_path=str(lock),
                promotion_evidence_path=str(promotion),
                external_confirmation_only=True,
            )

            candidate, bridge = _locked_public_bridge_candidate(config)

            self.assertEqual(candidate.hidden_dim, 96)
            self.assertEqual(candidate.dropout, 0.1)
            self.assertEqual(bridge["screen_candidate_limit"], 1)
            self.assertFalse(bridge["public_label_candidate_selection"])

    def test_candidate_grid_records_required_refresh_dimensions(self) -> None:
        candidates, grid = build_public_fusion_refresh_candidates(limit=2)
        self.assertEqual(len(candidates), 2)
        self.assertIn(128, grid["grid_spec"]["hidden_dim"])
        self.assertIn("huber", grid["grid_spec"]["regression_loss"])
        self.assertIn("robust_train", grid["grid_spec"]["target_transform"])
        self.assertGreater(grid["full_candidate_count"], len(candidates))

    def test_refresh_runs_screen_and_confirm_on_synthetic_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "datasets"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)
            uab_root = root / "uab_sequences"
            nasa_root = root / "nasa_sequences"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=16,
                )
            )
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=16,
                )
            )

            result = run_task_eval_public_fusion_refresh(
                StageIPublicFusionRefreshConfig(
                    run_id="test-public-fusion-refresh",
                    dataset_prepared_roots={
                        "nasa_csm": str(nasa_root),
                        "uab_workload_dataset": str(uab_root),
                    },
                    artifact_root=str(root / "refresh"),
                    report_root=str(root / "reports"),
                    screen_epochs=1,
                    confirm_epochs=1,
                    screen_max_folds=1,
                    confirm_max_folds=1,
                    screen_candidate_limit=1,
                    confirm_top_k=1,
                    device="cpu",
                    require_cuda=False,
                )
            )
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertIn("nasa_csm", summary["best_by_dataset_task"])
            self.assertIn("uab_workload_dataset", summary["best_by_dataset_task"])
            self.assertTrue(Path(summary["screen_leaderboard_csv"]).exists())
            self.assertTrue(Path(summary["confirm_leaderboard_csv"]).exists())
            self.assertTrue(Path(summary["fold_metrics_csv"]).exists())
            self.assertTrue(Path(summary["training_curves_csv"]).exists())
            self.assertTrue(Path(summary["progress_path"]).exists())
            self.assertTrue(Path(summary["run_log_path"]).exists())
            self.assertIn(summary["status"], {"completed", "partial"})
            self.assertIn("fig_public_fusion_win_summary", summary["figure_paths"])
            self.assertTrue(Path(summary["figure_paths"]["fig_public_fusion_win_summary"]).exists())
            self.assertTrue(Path(result.evidence_manifest_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            confirm = pd.read_csv(summary["confirm_leaderboard_csv"])
            self.assertEqual(set(confirm["dataset_id"]), {"nasa_csm", "uab_workload_dataset"})
            for candidate_root in confirm["artifact_root"]:
                self.assertTrue((Path(candidate_root) / "config.json").exists())
                self.assertTrue((Path(candidate_root) / "fold_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
