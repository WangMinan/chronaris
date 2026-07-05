"""task evaluation public-opt regression aggregation tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evaluation.public_datasets.pipelines.opt import (  # noqa: E402
    StageIPublicOptConfig,
    run_task_eval_public_opt,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch import (  # noqa: E402
    StageIPublicOptTorchUABConfig,
    run_task_eval_public_opt_torch_uab,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "task_eval_public_opt_helpers",
    SRC.parent / "tests" / "evaluation" / "public_datasets" / "test_opt.py",
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load task evaluation public-opt helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_build_prepared_uab_root = _HELPER_MODULE._build_prepared_uab_root
_write_reference_public_opt_summary = _HELPER_MODULE._write_reference_public_opt_summary
_write_reference_summaries = _HELPER_MODULE._write_reference_summaries


class StageIPublicOptAggregationTest(unittest.TestCase):
    def test_session_mean_broadcast_makes_each_head_constant_within_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-session-mean",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    prediction_aggregation_policy="session_mean_broadcast",
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["prediction_aggregation_policy"],
                "session_mean_broadcast",
            )
            predictions = pd.read_csv(result.predictions_path)
            grouped = predictions.groupby(
                ["evaluation_group", "head_name", "session_id"],
                sort=False,
            )["y_pred"].nunique()
            self.assertTrue((grouped == 1).all())

    def test_session_median_broadcast_keeps_metrics_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-session-median",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    ensemble_policy="mean_top2",
                    prediction_aggregation_policy="session_median_broadcast",
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["prediction_aggregation_policy"],
                "session_median_broadcast",
            )
            self.assertEqual(summary["ensemble_policy"], "mean_top2")
            for subset_id in ("n_back", "heat_the_chair"):
                metrics_by_head = summary["subset_results"][subset_id]["heads"]
                self.assertIn("mean_top2_ensemble", metrics_by_head)
                for metrics in metrics_by_head.values():
                    self.assertTrue(np.isfinite(float(metrics["mae"])))
                    self.assertTrue(np.isfinite(float(metrics["rmse"])))
            report_text = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("prediction_aggregation_policy", report_text)

    def test_torch_session_mean_broadcast_makes_selected_predictions_constant_within_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared_root = _build_prepared_uab_root(root)
            reference_public_opt = root / "reference_public_opt_summary.json"
            reference_deep = root / "reference_deep_summary.json"
            _write_reference_public_opt_summary(reference_public_opt)
            _, deep_summary_path = _write_reference_summaries(root)
            reference_deep.write_text(
                Path(deep_summary_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = run_task_eval_public_opt_torch_uab(
                StageIPublicOptTorchUABConfig(
                    run_id="public-opt-uab-torch-session-mean",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    device="cpu",
                    screen_max_folds=1,
                    full_max_folds=1,
                    batch_size=32,
                    epochs=1,
                    patience=1,
                    full_candidate_limit=2,
                    ensemble_policy="mean_top2",
                    prediction_aggregation_policy="session_mean_broadcast",
                    feature_profiles=("full",),
                    learning_rates=(1e-3,),
                    weight_decays=(1e-4,),
                    reference_public_opt_summary_path=str(reference_public_opt),
                    reference_deep_comparison_summary_path=str(reference_deep),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["screen_config"]["prediction_aggregation_policy"],
                "session_mean_broadcast",
            )
            predictions = pd.read_csv(result.predictions_path)
            grouped = predictions.groupby(
                ["subset_id", "session_id"],
                sort=False,
            )["y_pred"].nunique()
            self.assertTrue((grouped == 1).all())

    def test_torch_session_pooled_supervision_runs_with_physiology_only_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared_root = _build_prepared_uab_root(root)
            reference_public_opt = root / "reference_public_opt_summary.json"
            reference_deep = root / "reference_deep_summary.json"
            _write_reference_public_opt_summary(reference_public_opt)
            _, deep_summary_path = _write_reference_summaries(root)
            reference_deep.write_text(
                Path(deep_summary_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = run_task_eval_public_opt_torch_uab(
                StageIPublicOptTorchUABConfig(
                    run_id="public-opt-uab-torch-session-pooled",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    device="cpu",
                    screen_max_folds=1,
                    full_max_folds=1,
                    batch_size=32,
                    epochs=1,
                    patience=1,
                    full_candidate_limit=1,
                    full_group_winner_limit=1,
                    ensemble_policy="none",
                    prediction_aggregation_policy="session_mean_broadcast",
                    supervision_granularity="session_pooled_broadcast",
                    feature_profiles=("physiology_only",),
                    learning_rates=(1e-3,),
                    weight_decays=(1e-4,),
                    reference_public_opt_summary_path=str(reference_public_opt),
                    reference_deep_comparison_summary_path=str(reference_deep),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(
                summary["screen_config"]["supervision_granularity"],
                "session_pooled_broadcast",
            )
            self.assertEqual(
                summary["screen_config"]["feature_profiles"],
                ["physiology_only"],
            )
            predictions = pd.read_csv(result.predictions_path)
            grouped = predictions.groupby(
                ["subset_id", "session_id"],
                sort=False,
            )["y_pred"].nunique()
            self.assertTrue((grouped == 1).all())
