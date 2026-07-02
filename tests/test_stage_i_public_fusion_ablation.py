"""Tests for P31 public fusion ablation workflow."""

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

from chronaris.pipelines.stage_i.public.fusion_ablation import (  # noqa: E402
    StageIPublicFusionAblationConfig,
    _build_ablation_summary,
    default_public_fusion_ablation_variants,
    run_stage_i_public_fusion_ablation,
)
from chronaris.pipelines.stage_i.public.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_stage_i_sequence_preparation,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_pipeline_helpers",
    Path(__file__).resolve().with_name("test_stage_i_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("failed to load Stage I synthetic dataset helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_mini_uab_dataset = _HELPER_MODULE._write_mini_uab_dataset
_write_mini_nasa_csm_dataset = _HELPER_MODULE._write_mini_nasa_csm_dataset


class StageIPublicFusionAblationTest(unittest.TestCase):
    def test_default_variant_manifest_contains_required_ablation_ids(self) -> None:
        ids = {variant.variant_id for variant in default_public_fusion_ablation_variants()}
        for expected in (
            "full",
            "no_lag_window",
            "no_event_bias",
            "no_causal_fusion",
            "physiology_only",
            "single_stream_physio_only",
            "no_context_proxy_stream",
            "context_only",
            "simple_dual_stream_concat",
            "no_normalize_states",
            "no_target_transform",
            "mse_loss",
            "no_balanced_class",
            "smaller_capacity_h64_l1",
            "best_full_reproduce",
            "regression_loss_huber",
        ):
            self.assertIn(expected, ids)

    def test_public_ablation_smoke_writes_context_proxy_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "datasets"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)
            uab_root = root / "uab_sequences"
            nasa_root = root / "nasa_sequences"
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=16,
                )
            )
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=16,
                )
            )
            result = run_stage_i_public_fusion_ablation(
                StageIPublicFusionAblationConfig(
                    run_id="p31-test",
                    dataset_prepared_roots={
                        "nasa_csm": str(nasa_root),
                        "uab_workload_dataset": str(uab_root),
                    },
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    variants=("full", "no_lag_window", "physiology_only", "context_only"),
                    screen_epochs=1,
                    confirm_epochs=1,
                    screen_max_folds=1,
                    confirm_max_folds=1,
                    device="cpu",
                    require_cuda=False,
                )
            )
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["evidence_role"], "public_adapter_context_proxy_evidence")
            manifest = json.loads(Path(summary["ablation_variant_manifest_json"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["second_stream_is_real_vehicle"])
            self.assertEqual(manifest["second_stream_role"], "context_proxy")
            self.assertTrue(Path(summary["ablation_summary_csv"]).exists())
            self.assertTrue(Path(summary["ablation_long_csv"]).exists())
            self.assertTrue(Path(summary["ablation_wide_csv"]).exists())
            self.assertTrue(Path(summary["component_contribution_csv"]).exists())
            self.assertTrue(Path(summary["public_fusion_ablation_config_json"]).exists())
            self.assertTrue(Path(summary["public_fusion_ablation_variants_json"]).exists())
            self.assertTrue(Path(summary["gpu_perf_batches_csv"]).exists())
            self.assertTrue(Path(summary["gpu_perf_fold_summary_csv"]).exists())
            self.assertTrue(Path(summary["resume_command_txt"]).exists())
            gpu_perf = json.loads(Path(summary["gpu_perf_summary_json"]).read_text(encoding="utf-8"))
            self.assertIn("tensor_cache_mode", gpu_perf)
            self.assertIn("auto_batch_size", gpu_perf)
            self.assertTrue(Path(summary["figure_paths"]["fig_public_ablation_delta_heatmap"]).exists())
            self.assertTrue(Path(summary["figure_paths"]["fig_public_ablation_component_contribution"]).exists())
            self.assertTrue(Path(summary["figure_paths"]["fig_public_ablation_gpu_throughput"]).exists())

    def test_ablation_delta_direction_handles_rmse_and_macro_f1(self) -> None:
        long = pd.DataFrame(
            [
                {"dataset_id": "nasa_csm", "task_group": "combined", "metric": "combined_macro_f1", "variant_id": "full", "ablation_family": "full", "seed": 42, "value": 0.7},
                {"dataset_id": "nasa_csm", "task_group": "combined", "metric": "combined_macro_f1", "variant_id": "no_lag_window", "ablation_family": "lag", "seed": 42, "value": 0.5},
                {"dataset_id": "uab_workload_dataset", "task_group": "n_back", "metric": "n_back_rmse", "variant_id": "full", "ablation_family": "full", "seed": 42, "value": 2.0},
                {"dataset_id": "uab_workload_dataset", "task_group": "n_back", "metric": "n_back_rmse", "variant_id": "no_lag_window", "ablation_family": "lag", "seed": 42, "value": 3.0},
            ]
        )
        summary = _build_ablation_summary(long)
        macro = summary[(summary["metric"] == "combined_macro_f1") & (summary["variant_id"] == "no_lag_window")].iloc[0]
        rmse = summary[(summary["metric"] == "n_back_rmse") & (summary["variant_id"] == "no_lag_window")].iloc[0]
        self.assertAlmostEqual(float(macro["delta_abs_mean"]), 0.2)
        self.assertAlmostEqual(float(rmse["delta_abs_mean"]), 1.0)


if __name__ == "__main__":
    unittest.main()
