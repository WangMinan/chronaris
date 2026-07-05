"""Tests for P37 metric calibration aggregation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.evidence.metric_calibration import (  # noqa: E402
    StageIMetricCalibrationConfig,
    _resume_command,
    _status_from_outputs,
    _write_private_delta,
)
from chronaris.evaluation.dingxin.pipelines.thirdparty_comparison import _class_balanced_weights  # noqa: E402


class StageIMetricCalibrationTest(unittest.TestCase):
    def test_delta_direction_positive_means_p37_better(self) -> None:
        p37 = pd.DataFrame(
            [
                {
                    "task_name": "T1_maneuver_intensity_class",
                    "model_name": "p37_t1_focal",
                    "split_strategy": "leave_one_view_out",
                    "metric": "macro_f1",
                    "value_mean": 0.30,
                },
                {
                    "task_name": "T2_next_window_physiology_response",
                    "model_name": "p37_t2",
                    "split_strategy": "leave_one_view_out",
                    "metric": "rmse",
                    "value_mean": 2.0,
                },
            ]
        )
        baseline = pd.DataFrame(
            [
                {
                    "task_name": "T1_maneuver_intensity_class",
                    "model_name": "chronaris_v2_task_heads",
                    "split_strategy": "leave_one_view_out",
                    "metric": "macro_f1",
                    "value_mean": 0.25,
                },
                {
                    "task_name": "T2_next_window_physiology_response",
                    "model_name": "chronaris_v2_task_heads",
                    "split_strategy": "leave_one_view_out",
                    "metric": "rmse",
                    "value_mean": 3.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            delta = _write_private_delta(p37, baseline, Path(tmp) / "delta.csv")
        self.assertAlmostEqual(delta.loc[delta["metric"] == "macro_f1", "delta_positive_is_better"].iloc[0], 0.05)
        self.assertAlmostEqual(delta.loc[delta["metric"] == "rmse", "delta_positive_is_better"].iloc[0], 1.0)

    def test_t1_class_weights_use_train_fold_labels(self) -> None:
        weights = _class_balanced_weights(pd.Series([0, 0, 1]).to_numpy(), device="cpu")
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertEqual(float(weights[2]), 0.0)

    def test_resume_command_and_status(self) -> None:
        config = StageIMetricCalibrationConfig(run_id="20260702T-task-eval-metric-calibration-r1")
        command = _resume_command(config)
        self.assertIn("--require-cuda", command)
        self.assertIn("--skip-completed", command)
        self.assertEqual(
            _status_from_outputs(config, {"private_confirm_root": "/x", "public_confirm_root": "/y"}),
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
