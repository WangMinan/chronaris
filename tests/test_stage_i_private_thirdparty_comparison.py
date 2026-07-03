"""Tests for P30 private Stage H third-party comparison."""

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

from chronaris.pipelines.stage_i.private.thirdparty_comparison import (  # noqa: E402
    StageIPrivateThirdPartyComparisonConfig,
    _build_improvement_summary,
    run_stage_i_private_thirdparty_comparison,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_deep_helpers",
    Path(__file__).resolve().with_name("test_stage_i_deep_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("failed to load Stage H synthetic helper")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_private_stage_h_run = _HELPER_MODULE._write_private_stage_h_run


class StageIPrivateThirdPartyComparisonTest(unittest.TestCase):
    def test_private_thirdparty_smoke_writes_real_vehicle_schema_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
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
            result = run_stage_i_private_thirdparty_comparison(
                StageIPrivateThirdPartyComparisonConfig(
                    run_id="p30-test",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    models=("chronaris_full", "mult", "contiformer"),
                    seeds=(17,),
                    split_strategy=("leave_one_view_out",),
                    epochs=1,
                    batch_size=4,
                    hidden_dim=16,
                    num_heads=2,
                    layers=1,
                    device="cpu",
                    require_cuda=False,
                )
            )
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            schema = json.loads(Path(summary["private_sequence_schema_json"]).read_text(encoding="utf-8"))
            self.assertEqual(schema["second_stream_role"], "real_vehicle_timeseries")
            self.assertTrue(schema["second_stream_is_real_vehicle"])
            self.assertEqual(summary["evidence_role"], "private_real_dual_stream")
            for key in (
                "label_feature_overlap_audit_json",
                "split_manifest_json",
                "model_comparison_long_csv",
                "model_comparison_wide_csv",
                "private_third_party_config_json",
                "private_third_party_long_csv",
                "private_third_party_wide_csv",
                "private_third_party_leaderboard_csv",
                "improvement_summary_csv",
                "evidence_manifest_path",
                "resume_command_txt",
            ):
                self.assertTrue(Path(summary[key]).exists(), key)
            long_frame = pd.read_csv(summary["model_comparison_long_csv"])
            self.assertIn("mult", set(long_frame["model_name"]))
            self.assertIn("contiformer", set(long_frame["model_name"]))
            self.assertTrue(summary["gpuopt_enabled"])
            self.assertIn("tensor_cache", summary["gpuopt_config"])
            for figure_key in (
                "fig_private_third_party_task_leaderboard",
                "fig_private_third_party_delta_heatmap",
                "fig_private_third_party_fold_variance",
                "fig_private_third_party_training_curves",
                "fig_private_third_party_gpu_throughput",
            ):
                self.assertTrue(Path(summary["figure_paths"][figure_key]).exists(), figure_key)

    def test_improvement_delta_direction_uses_metric_direction(self) -> None:
        long_frame = pd.DataFrame(
            [
                {
                    "task_name": "T1_maneuver_intensity_class",
                    "model_name": "chronaris_full",
                    "split_strategy": "leave_one_view_out",
                    "metric": "macro_f1",
                    "value_mean": 0.7,
                },
                {
                    "task_name": "T1_maneuver_intensity_class",
                    "model_name": "mult",
                    "split_strategy": "leave_one_view_out",
                    "metric": "macro_f1",
                    "value_mean": 0.5,
                },
                {
                    "task_name": "T2_next_window_physiology_response",
                    "model_name": "chronaris_full",
                    "split_strategy": "leave_one_view_out",
                    "metric": "rmse",
                    "value_mean": 2.0,
                },
                {
                    "task_name": "T2_next_window_physiology_response",
                    "model_name": "mult",
                    "split_strategy": "leave_one_view_out",
                    "metric": "rmse",
                    "value_mean": 3.0,
                },
            ]
        )
        improvement = _build_improvement_summary(long_frame)
        macro = improvement[improvement["metric"] == "macro_f1"].iloc[0]
        rmse = improvement[improvement["metric"] == "rmse"].iloc[0]
        self.assertAlmostEqual(float(macro["delta_abs"]), 0.2)
        self.assertAlmostEqual(float(rmse["delta_abs"]), 1.0)


if __name__ == "__main__":
    unittest.main()
