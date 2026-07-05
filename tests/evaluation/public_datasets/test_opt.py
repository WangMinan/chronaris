"""task evaluation public-opt regression and classification tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    dump_task_eval_sequence_entries,
    load_task_eval_sequence_bundle,
    load_task_eval_sequence_entries,
    save_task_eval_sequence_bundle,
)
from chronaris.evaluation import save_bar_plot, save_grouped_bar_plot  # noqa: E402
from chronaris.evaluation.public_datasets.pipelines.mainline_report import (  # noqa: E402
    StageIPublicMainlineReportConfig,
    run_task_eval_public_mainline_report,
)
from chronaris.evaluation.public_datasets.pipelines.opt import (  # noqa: E402
    StageIPublicOptConfig,
    run_task_eval_public_opt,
)
from chronaris.evaluation.public_datasets.pipelines.opt_torch import (  # noqa: E402
    StageIPublicOptTorchUABConfig,
    run_task_eval_public_opt_torch_uab,
)
from chronaris.evaluation.public_datasets.pipelines.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_task_eval_sequence_preparation,
)
from chronaris.modeling.common.baseline_models import build_loso_splits  # noqa: E402
from chronaris.evaluation.public_datasets.pipelines.opt_data import (  # noqa: E402
    build_task_eval_public_opt_feature_frame,
)
from chronaris.evaluation.public_datasets.pipelines import opt_sklearn as task_eval_public_opt_sklearn_module  # noqa: E402
from chronaris.evaluation.public_datasets.pipelines import opt_torch as task_eval_public_opt_torch_module  # noqa: E402

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "task_eval_pipeline_helpers",
    SRC.parent / "tests" / "evaluation" / "public_datasets" / "test_pipeline.py",
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load task evaluation synthetic dataset helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_mini_uab_dataset = _HELPER_MODULE._write_mini_uab_dataset
_write_mini_nasa_csm_dataset = _HELPER_MODULE._write_mini_nasa_csm_dataset

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_task_eval_public_opt_script",
    SRC.parent / "scripts" / "evaluation" / "public_datasets" / "run_opt.py",
)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load run_task_eval_public_opt script")
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)


class StageIPublicOptTest(unittest.TestCase):
    def test_public_opt_feature_frame_filters_to_primary_subjective_uab_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            entries = load_task_eval_sequence_entries(prepared_root / "task_manifest.jsonl")
            bundle = load_task_eval_sequence_bundle(prepared_root / "sequence_bundle.npz")

            feature_result = build_task_eval_public_opt_feature_frame(
                entries,
                bundle,
                dataset_id="uab_workload_dataset",
                profile="window_v2",
            )

            feature_frame = feature_result.feature_frame
            self.assertEqual(feature_result.track, "subjective")
            self.assertEqual(feature_result.task_type, "regression")
            self.assertEqual(
                set(feature_frame["subset_id"].unique()),
                {"n_back", "heat_the_chair"},
            )
            self.assertNotIn("flight_simulator", set(feature_frame["subset_id"].unique()))
            self.assertTrue((feature_frame["dataset_id"] == "uab_workload_dataset").all())
            self.assertFalse(
                any(
                    name in {"objective_label_value", "subjective_target_value"}
                    for name in feature_result.feature_columns
                )
            )
            self.assertEqual(
                set(feature_result.feature_groups),
                {
                    "full",
                    "physiology_only",
                    "physiology_lowdim",
                    "physiology_scalar_only",
                    "context_only",
                    "residual_only",
                },
            )
            for head_name in (
                "target_prior_median",
                "target_prior_trimmed_mean",
                "physiology_persistence",
                "ridge_residual_cv",
                "elasticnet_residual",
                "huber_residual",
                "heat_prior_residual_guarded",
            ):
                self.assertIn(head_name, feature_result.head_feature_columns)
            self.assertEqual(feature_result.head_feature_columns["target_prior_median"], ())
            self.assertEqual(feature_result.head_feature_columns["target_prior_trimmed_mean"], ())
            for head_name in (
                "physiology_persistence",
                "ridge_residual_cv",
                "elasticnet_residual",
                "huber_residual",
                "heat_prior_residual_guarded",
            ):
                self.assertTrue(feature_result.head_feature_columns[head_name])

    def test_public_opt_feature_frame_filters_to_primary_nasa_attention_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            entries = load_task_eval_sequence_entries(prepared_root / "task_manifest.jsonl")
            bundle = load_task_eval_sequence_bundle(prepared_root / "sequence_bundle.npz")

            feature_result = build_task_eval_public_opt_feature_frame(
                entries,
                bundle,
                dataset_id="nasa_csm",
                profile="window_v2",
            )

            feature_frame = feature_result.feature_frame
            self.assertEqual(feature_result.track, "objective")
            self.assertEqual(feature_result.task_type, "classification")
            self.assertEqual(set(feature_frame["subset_id"].unique()), {"benchmark", "loft"})
            self.assertEqual(
                feature_result.evaluation_groups["combined"],
                ("benchmark", "loft"),
            )
            self.assertEqual(feature_result.label_order, (1, 2, 5))
            self.assertIn("physiology_margin_balanced_logistic", feature_result.head_feature_columns)
            self.assertIn("balanced_logistic_context", feature_result.head_feature_columns)
            self.assertIn("balanced_linear_svc_context", feature_result.head_feature_columns)

    def test_run_task_eval_public_opt_writes_expected_uab_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            phase3_summary_path, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-uab",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    reference_phase3_closure_summary_path=str(phase3_summary_path),
                    reference_deep_comparison_summary_path=str(deep_summary_path),
                )
            )

            self.assertTrue(Path(result.feature_frame_path).exists())
            self.assertTrue(Path(result.predictions_path).exists())
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue((Path(result.artifact_root) / "run.log").exists())
            self.assertTrue((Path(result.artifact_root) / "progress.json").exists())

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["track"], "subjective")
            self.assertEqual(summary["feature_profile"], "full")
            self.assertEqual(summary["subset_order"], ["n_back", "heat_the_chair"])
            self.assertEqual(
                set(summary["heads"]),
                {
                    "physiology_persistence",
                    "ridge_residual_cv",
                    "elasticnet_residual",
                    "huber_residual",
                },
            )
            self.assertIn("reference_comparison", summary)
            self.assertIn("n_back", summary["reference_comparison"]["groups"])
            self.assertIn("winning_margin_vs_deep", summary)
            self.assertIn("run_log_path", summary)
            progress = json.loads(
                (Path(result.artifact_root) / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["dataset_id"], "uab_workload_dataset")
            self.assertEqual(progress["last_event"], "finished")
            for subset_id in ("n_back", "heat_the_chair"):
                subset_payload = summary["subset_results"][subset_id]
                self.assertEqual(
                    set(subset_payload["heads"]),
                    {
                        "physiology_persistence",
                        "ridge_residual_cv",
                        "elasticnet_residual",
                        "huber_residual",
                    },
                )
                self.assertIn(
                    subset_payload["best_head"],
                    {
                        "physiology_persistence",
                        "ridge_residual_cv",
                        "elasticnet_residual",
                        "huber_residual",
                    },
                )

    def test_run_task_eval_public_opt_uab_hybrid_splits_heads_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            phase3_summary_path, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-uab-hybrid",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    head_catalog="uab_hybrid",
                    reference_phase3_closure_summary_path=str(phase3_summary_path),
                    reference_deep_comparison_summary_path=str(deep_summary_path),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["head_catalog"], "uab_hybrid")
            self.assertEqual(
                set(summary["subset_results"]["n_back"]["heads"]),
                {
                    "physiology_persistence",
                    "ridge_residual_cv",
                    "elasticnet_residual",
                    "huber_residual",
                },
            )
            self.assertEqual(
                set(summary["subset_results"]["heat_the_chair"]["heads"]),
                {
                    "target_prior_median",
                    "target_prior_trimmed_mean",
                    "heat_prior_residual_guarded",
                    "physiology_persistence",
                    "ridge_heat_physiology_lowdim",
                    "huber_heat_physiology_lowdim",
                },
            )
            self.assertIn(
                summary["subset_results"]["heat_the_chair"]["best_head"],
                {
                    "target_prior_median",
                    "target_prior_trimmed_mean",
                    "heat_prior_residual_guarded",
                    "physiology_persistence",
                    "ridge_heat_physiology_lowdim",
                    "huber_heat_physiology_lowdim",
                },
            )
            report_text = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("target_prior_median", report_text)
            self.assertIn("heat_prior_residual_guarded", report_text)
            self.assertIn("ridge_heat_physiology_lowdim", report_text)
            self.assertNotIn("| huber_residual |", report_text.split("### heat_the_chair", 1)[1])
            self.assertIn("context proxy / adapter stream", report_text)
            self.assertIn("public adapter evidence", report_text)

    def test_public_opt_prior_heads_use_only_outer_train_labels(self) -> None:
        subset_frame = _build_public_opt_prior_probe_frame(
            y_by_group={"s1": 0.0, "s2": 10.0, "s3": 20.0, "s4": 999.0},
            feature_by_group={"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0},
        )
        loso_splits = build_loso_splits(
            subset_frame["split_group"].astype(str).to_numpy()
        )

        for head_name, reducer in (
            ("target_prior_median", _expected_train_median),
            ("target_prior_trimmed_mean", _expected_train_trimmed_mean),
        ):
            predictions = task_eval_public_opt_sklearn_module._run_one_regression_head(
                subset_frame=subset_frame,
                evaluation_group="heat_the_chair",
                head_name=head_name,
                feature_columns=(),
                loso_splits=loso_splits,
                progress=None,
            )
            for split_group, group_predictions in predictions.groupby("split_group"):
                train_values = subset_frame.loc[
                    subset_frame["split_group"] != split_group,
                    "y_true",
                ].to_numpy(dtype=float)
                self.assertTrue(
                    np.allclose(
                        group_predictions["y_pred"].to_numpy(dtype=float),
                        reducer(train_values),
                    ),
                    msg=f"{head_name} leaked labels for {split_group}",
                )

    def test_heat_prior_residual_guarded_falls_back_when_inner_cv_does_not_improve(self) -> None:
        subset_frame = _build_public_opt_prior_probe_frame(
            y_by_group={"s1": 0.0, "s2": 10.0, "s3": 20.0, "s4": 30.0},
            feature_by_group={"s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0},
        )
        loso_splits = build_loso_splits(
            subset_frame["split_group"].astype(str).to_numpy()
        )

        predictions = task_eval_public_opt_sklearn_module._run_one_regression_head(
            subset_frame=subset_frame,
            evaluation_group="heat_the_chair",
            head_name="heat_prior_residual_guarded",
            feature_columns=("residual__physiology_intensity_mean",),
            loso_splits=loso_splits,
            progress=None,
        )

        for split_group, group_predictions in predictions.groupby("split_group"):
            train_values = subset_frame.loc[
                subset_frame["split_group"] != split_group,
                "y_true",
            ].to_numpy(dtype=float)
            self.assertTrue(
                np.allclose(
                    group_predictions["y_pred"].to_numpy(dtype=float),
                    np.median(train_values),
                )
            )

    def test_run_task_eval_public_opt_writes_expected_nasa_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            phase3_summary_path, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-nasa",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    dataset_id="nasa_csm",
                    profile="window_v2",
                    reference_phase3_closure_summary_path=str(phase3_summary_path),
                    reference_deep_comparison_summary_path=str(deep_summary_path),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["track"], "objective")
            self.assertEqual(summary["task_type"], "classification")
            self.assertEqual(
                summary["subset_order"],
                ["benchmark_only", "loft_only", "combined"],
            )
            self.assertEqual(
                set(summary["heads"]),
                {
                    "physiology_margin_balanced_logistic",
                    "balanced_logistic_context",
                    "balanced_linear_svc_context",
                },
            )
            self.assertIn("combined", summary["reference_comparison"]["groups"])
            self.assertIn("winning_margin_vs_deep", summary)
            report_text = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("scenario_context", report_text)
            self.assertIn("public adapter evidence", report_text)

    def test_nasa_public_opt_rejects_prepared_assets_without_leakage_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            schema_path = prepared_root / "sequence_schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema.pop("label_leakage_guard", None)
            schema_path.write_text(
                json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "label_leakage_guard"):
                run_task_eval_public_opt(
                    StageIPublicOptConfig(
                        run_id="public-opt-nasa-stale",
                        prepared_artifact_root=str(prepared_root),
                        artifact_root=str(Path(temp_dir) / "artifacts"),
                        report_root=str(Path(temp_dir) / "reports"),
                        dataset_id="nasa_csm",
                        profile="window_v2",
                    )
                )

    def test_public_opt_predictions_preserve_subject_loso_split_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-loso",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                )
            )

            predictions = pd.read_csv(result.predictions_path)
            feature_frame = pd.read_parquet(result.feature_frame_path)
            for subset_id in ("n_back", "heat_the_chair"):
                expected_groups = set(
                    feature_frame.loc[feature_frame["subset_id"] == subset_id, "split_group"]
                    .astype(str)
                    .unique()
                )
                observed_groups = set(
                    predictions.loc[predictions["evaluation_group"] == subset_id, "split_group"]
                    .astype(str)
                    .unique()
                )
                self.assertEqual(observed_groups, expected_groups)

    def test_nasa_public_opt_predictions_preserve_combined_subject_loso_split_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-nasa-loso",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    dataset_id="nasa_csm",
                    profile="window_v2",
                )
            )

            predictions = pd.read_csv(result.predictions_path)
            combined = predictions.loc[predictions["evaluation_group"] == "combined"]
            self.assertTrue((combined["split_group"] == combined["subject_id"]).all())

    def test_public_opt_regression_fallback_keeps_predictions_and_metrics_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            _force_constant_subjective_targets(prepared_root, value=0.75)

            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-fallback",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                )
            )

            predictions = pd.read_csv(result.predictions_path)
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            for subset_id in ("n_back", "heat_the_chair"):
                for head_name in (
                    "physiology_persistence",
                    "ridge_residual_cv",
                    "elasticnet_residual",
                    "huber_residual",
                ):
                    metrics = summary["subset_results"][subset_id]["heads"][head_name]
                    self.assertTrue(np.isfinite(float(metrics["mae"])))
                    self.assertTrue(np.isfinite(float(metrics["rmse"])))

    def test_public_opt_classification_fallback_keeps_predictions_and_metrics_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            _force_constant_objective_targets(prepared_root, value=2)

            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-classification-fallback",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    dataset_id="nasa_csm",
                    profile="window_v2",
                )
            )

            predictions = pd.read_csv(result.predictions_path)
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            for subset_id in ("benchmark_only", "loft_only", "combined"):
                for head_name in (
                    "physiology_margin_balanced_logistic",
                    "balanced_logistic_context",
                    "balanced_linear_svc_context",
                ):
                    metrics = summary["subset_results"][subset_id]["heads"][head_name]
                    self.assertTrue(np.isfinite(float(metrics["macro_f1"])))
                    self.assertTrue(np.isfinite(float(metrics["balanced_accuracy"])))

    def test_public_opt_supports_feature_profile_and_ensemble_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            result = run_task_eval_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-nasa-ensemble",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    dataset_id="nasa_csm",
                    profile="window_v2",
                    feature_profile="residual_only",
                    ensemble_policy="vote_top2",
                )
            )
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["feature_profile"], "residual_only")
            self.assertEqual(summary["ensemble_policy"], "vote_top2")
            self.assertIn(
                "vote_top2_ensemble",
                summary["subset_results"]["combined"]["heads"],
            )

    def test_task_eval_metric_plots_fallback_to_ascii_safe_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                save_bar_plot(
                    {"20251005_四01_ACT-4_云_J20_22#01": 1.0},
                    path=root / "bar.png",
                    title="中文标题",
                    ylabel="指标值",
                )
                save_grouped_bar_plot(
                    {
                        "20251005_四01_ACT-4_云_J20_22#01": {"宏平均": 1.0},
                        "20251002_单01_ACT-8_翼云_J16_12#01": {"宏平均": 0.8},
                    },
                    path=root / "grouped.png",
                    title="中文分组标题",
                    ylabel="指标值",
                )
            self.assertTrue((root / "bar.png").exists())
            self.assertTrue((root / "grouped.png").exists())
            self.assertFalse(
                any("Glyph" in str(item.message) for item in caught),
                msg=[str(item.message) for item in caught],
            )

    def test_public_mainline_prefers_best_uab_source_and_tie_breaks_by_mae(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, deep_summary_path = _write_reference_summaries(root)
            nasa_summary_path = _write_reference_nasa_public_opt_summary(root)
            torch_summary_path = _write_reference_torch_uab_summary(
                root / "torch_uab_summary.json",
                n_back_rmse=5.0,
                n_back_mae=4.0,
                heat_rmse=1.7,
                heat_mae=1.3,
            )
            legacy_summary_path = _write_reference_legacy_uab_summary(
                root / "legacy_uab_summary.json",
                n_back_rmse=4.59,
                n_back_mae=3.79,
                heat_rmse=1.45,
                heat_mae=1.15,
            )
            tie_summary_path = _write_reference_legacy_uab_summary(
                root / "tie_uab_summary.json",
                n_back_rmse=4.6000005,
                n_back_mae=3.7000,
                heat_rmse=1.4500005,
                heat_mae=1.1500,
            )

            result = run_task_eval_public_mainline_report(
                StageIPublicMainlineReportConfig(
                    run_id="public-mainline-best-of-test",
                    uab_summary_path=str(torch_summary_path),
                    extra_uab_summary_paths=(
                        str(legacy_summary_path),
                        str(tie_summary_path),
                    ),
                    nasa_summary_path=str(nasa_summary_path),
                    deep_comparison_summary_path=str(deep_summary_path),
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["public_mainline_status"], "public opt closed")
            self.assertEqual(summary["thesis_facing_status"], "public_adapter_evidence")
            self.assertFalse(summary["thesis_dual_stream_mainline_closed"])
            self.assertEqual(
                summary["public_branch_semantics"]["uab"]["second_stream_name"],
                "task_context",
            )
            self.assertEqual(summary["uab"]["source_type"], "multi_source_best_of")
            self.assertEqual(
                summary["uab"]["groups"]["n_back"]["best_public_source_type"],
                "legacy_public_opt",
            )
            self.assertTrue(summary["uab"]["groups"]["heat_the_chair"]["clean_win"])
            self.assertTrue(summary["uab"]["groups"]["heat_the_chair"]["tie_break_used"])
            report_text = Path(result.report_path).read_text(encoding="utf-8")
            self.assertIn("current torch-native branch", report_text)
            self.assertIn("public adapter evidence", report_text)
            self.assertIn("context proxy", report_text)
            self.assertIn("不是论文里的真实 vehicle stream", report_text)

    def test_public_mainline_can_promote_new_prior_summary_from_extra_uab_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, deep_summary_path = _write_reference_summaries(root)
            nasa_summary_path = _write_reference_nasa_public_opt_summary(root)
            current_uab_path = _write_reference_torch_uab_summary(
                root / "current_torch_uab_summary.json",
                n_back_rmse=5.0,
                n_back_mae=4.0,
                heat_rmse=1.7,
                heat_mae=1.3,
            )
            legacy_uab_path = _write_reference_legacy_uab_summary(
                root / "legacy_uab_summary.json",
                n_back_rmse=4.59,
                n_back_mae=3.79,
                heat_rmse=1.4567586,
                heat_mae=1.1637,
            )
            prior_uab_path = _write_reference_prior_uab_summary(
                root / "prior_uab_summary.json",
                n_back_rmse=6.0,
                n_back_mae=5.0,
                heat_rmse=1.44,
                heat_mae=1.14,
            )

            result = run_task_eval_public_mainline_report(
                StageIPublicMainlineReportConfig(
                    run_id="public-mainline-prior-extra-test",
                    uab_summary_path=str(current_uab_path),
                    extra_uab_summary_paths=(
                        str(legacy_uab_path),
                        str(prior_uab_path),
                    ),
                    nasa_summary_path=str(nasa_summary_path),
                    deep_comparison_summary_path=str(deep_summary_path),
                    artifact_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["public_mainline_status"], "public opt closed")
            self.assertEqual(
                summary["public_branch_semantics"]["nasa"]["second_stream_name"],
                "scenario_context",
            )
            self.assertEqual(
                summary["uab"]["groups"]["n_back"]["best_public_head"],
                "ridge_residual",
            )
            self.assertEqual(
                summary["uab"]["groups"]["heat_the_chair"]["best_public_head"],
                "target_prior_median",
            )
            self.assertEqual(
                summary["uab"]["groups"]["heat_the_chair"]["best_public_source_type"],
                "uab_public_adapter",
            )

    def test_public_opt_script_auto_routes_uab_to_torch_backend(self) -> None:
        args = _build_public_opt_script_args()
        fake_result = SimpleNamespace(
            feature_frame_path="/tmp/public_opt_torch_feature_frame.parquet",
            predictions_path="/tmp/public_opt_torch_predictions.csv",
            summary_path="/tmp/public_opt_torch_summary.json",
            report_path="/tmp/public_opt_torch_report.md",
            summary={"runtime_device": "cuda"},
        )
        with patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt_torch_uab",
            return_value=fake_result,
        ) as torch_runner, patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt",
        ) as sklearn_runner:
            payload = _SCRIPT_MODULE._run_from_args(args)

        torch_runner.assert_called_once()
        sklearn_runner.assert_not_called()
        config = torch_runner.call_args.args[0]
        self.assertEqual(config.dataset_id, "uab_workload_dataset")
        self.assertEqual(config.device, "auto")
        self.assertTrue(config.require_cuda)
        self.assertEqual(config.candidate_catalog, "heat_specialist")
        self.assertEqual(config.selected_subsets, ("heat_the_chair",))
        self.assertEqual(
            config.feature_profiles,
            ("physiology_lowdim",),
        )
        self.assertTrue(config.artifact_root.endswith("docs/artifacts/runs"))
        self.assertEqual(payload["backend"], "torch")
        self.assertEqual(payload["runtime_device"], "cuda")

    def test_public_opt_script_auto_routes_nasa_to_sklearn_backend(self) -> None:
        args = _build_public_opt_script_args(dataset_id="nasa_csm")
        fake_result = SimpleNamespace(
            feature_frame_path="/tmp/public_opt_feature_frame.parquet",
            predictions_path="/tmp/public_opt_predictions.csv",
            summary_path="/tmp/public_opt_summary.json",
            report_path="/tmp/public_opt_report.md",
        )
        with patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt",
            return_value=fake_result,
        ) as sklearn_runner, patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt_torch_uab",
        ) as torch_runner:
            payload = _SCRIPT_MODULE._run_from_args(args)

        sklearn_runner.assert_called_once()
        torch_runner.assert_not_called()
        config = sklearn_runner.call_args.args[0]
        self.assertEqual(config.dataset_id, "nasa_csm")
        self.assertTrue(config.artifact_root.endswith("docs/artifacts/runs"))
        self.assertEqual(payload["backend"], "sklearn")

    def test_public_opt_script_supports_sklearn_uab_hybrid_catalog(self) -> None:
        args = _build_public_opt_script_args(
            backend="sklearn",
            head_catalog="uab_hybrid",
            allow_cpu_heavy_sklearn=True,
        )
        fake_result = SimpleNamespace(
            feature_frame_path="/tmp/public_opt_feature_frame.parquet",
            predictions_path="/tmp/public_opt_predictions.csv",
            summary_path="/tmp/public_opt_summary.json",
            report_path="/tmp/public_opt_report.md",
        )
        with patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt",
            return_value=fake_result,
        ) as sklearn_runner, patch.object(
            _SCRIPT_MODULE,
            "run_task_eval_public_opt_torch_uab",
        ) as torch_runner:
            payload = _SCRIPT_MODULE._run_from_args(args)

        sklearn_runner.assert_called_once()
        torch_runner.assert_not_called()
        config = sklearn_runner.call_args.args[0]
        self.assertEqual(config.dataset_id, "uab_workload_dataset")
        self.assertEqual(config.head_catalog, "uab_hybrid")
        self.assertTrue(config.artifact_root.endswith("docs/artifacts/runs"))
        self.assertEqual(payload["backend"], "sklearn")

    def test_public_opt_script_rejects_uab_hybrid_without_cpu_heavy_override(self) -> None:
        args = _build_public_opt_script_args(
            backend="sklearn",
            head_catalog="uab_hybrid",
        )
        with self.assertRaisesRegex(ValueError, "CPU-heavy historical reproduction"):
            _SCRIPT_MODULE._run_from_args(args)

    def test_public_opt_torch_uab_auto_device_falls_back_to_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            reference_public_opt = Path(temp_dir) / "reference_public_opt_summary.json"
            reference_deep = Path(temp_dir) / "reference_deep_summary.json"
            _write_reference_public_opt_summary(reference_public_opt)
            _, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            reference_deep.write_text(
                Path(deep_summary_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with patch.object(
                task_eval_public_opt_torch_module,
                "resolve_torch_device_name",
                return_value="cpu",
            ):
                result = run_task_eval_public_opt_torch_uab(
                    StageIPublicOptTorchUABConfig(
                        run_id="public-opt-uab-torch-auto-cpu",
                        prepared_artifact_root=str(prepared_root),
                        artifact_root=str(Path(temp_dir) / "artifacts"),
                        report_root=str(Path(temp_dir) / "reports"),
                        device="auto",
                        screen_max_folds=1,
                        full_max_folds=1,
                        batch_size=32,
                        epochs=1,
                        patience=1,
                        full_candidate_limit=2,
                        ensemble_policy="mean_top2",
                        feature_profiles=("full",),
                        learning_rates=(1e-3,),
                        weight_decays=(1e-4,),
                        reference_public_opt_summary_path=str(reference_public_opt),
                        reference_deep_comparison_summary_path=str(reference_deep),
                    )
                )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_device"], "cpu")
            self.assertEqual(summary["screen_config"]["device"], "auto")
            self.assertEqual(summary["screen_config"]["ensemble_policy"], "mean_top2")
            self.assertEqual(
                summary["selected_result"]["selection_policy"]["ensemble_policy"],
                "mean_top2",
            )
            predictions = pd.read_csv(result.predictions_path)
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())

    def test_public_opt_torch_uab_require_cuda_fails_before_training_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            with patch.object(
                task_eval_public_opt_torch_module,
                "resolve_torch_device_name",
                return_value="cpu",
            ):
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    run_task_eval_public_opt_torch_uab(
                        StageIPublicOptTorchUABConfig(
                            run_id="public-opt-uab-torch-require-cuda",
                            prepared_artifact_root=str(prepared_root),
                            artifact_root=str(Path(temp_dir) / "artifacts"),
                            report_root=str(Path(temp_dir) / "reports"),
                            device="auto",
                            require_cuda=True,
                            screen_max_folds=1,
                            run_full_loso=False,
                            feature_profiles=("physiology_lowdim",),
                            learning_rates=(1e-3,),
                            weight_decays=(1e-4,),
                        )
                    )
            run_root = Path(temp_dir) / "artifacts" / "public-opt-uab-torch-require-cuda"
            self.assertTrue((run_root / "run.log").exists())
            progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["last_event"], "failed")

    def test_public_opt_torch_heat_specialist_runs_heat_only_with_finite_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_task_eval_public_opt_torch_uab(
                StageIPublicOptTorchUABConfig(
                    run_id="public-opt-uab-torch-heat-only",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    device="cpu",
                    candidate_catalog="heat_specialist",
                    selected_subsets=("heat_the_chair",),
                    feature_profiles=("physiology_lowdim",),
                    screen_max_folds=1,
                    full_max_folds=1,
                    batch_size=32,
                    epochs=1,
                    patience=1,
                    full_candidate_limit=1,
                    full_group_winner_limit=1,
                    learning_rates=(1e-3,),
                    weight_decays=(1e-4,),
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["screen_config"]["candidate_catalog"], "heat_specialist")
            self.assertEqual(summary["screen_config"]["selected_subsets"], ["heat_the_chair"])
            self.assertEqual(set(summary["selected_result"]["groups"]), {"heat_the_chair"})
            predictions = pd.read_csv(result.predictions_path)
            self.assertEqual(set(predictions["subset_id"]), {"heat_the_chair"})
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())
            self.assertTrue((Path(result.artifact_root) / "run.log").exists())

    def test_torch_uab_heat_specialist_shortlist_preserves_mae_and_blend_winners(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "candidate_id": "heat_linear_huber_lowdim__lr0p001__wd0p0001",
                    "screen_mean_rmse": 1.00,
                    "screen_mean_mae": 1.20,
                    "heat_the_chair_rmse": 1.00,
                },
                {
                    "candidate_id": "heat_mlp_lowdim__lr0p001__wd0p0001",
                    "screen_mean_rmse": 1.05,
                    "screen_mean_mae": 0.95,
                    "heat_the_chair_rmse": 1.07,
                },
                {
                    "candidate_id": "heat_affine_calibrated_blend__lr0p001__wd0p0001",
                    "screen_mean_rmse": 1.08,
                    "screen_mean_mae": 1.05,
                    "heat_the_chair_rmse": 1.09,
                },
                {
                    "candidate_id": "heat_residual_correction__lr0p001__wd0p0001",
                    "screen_mean_rmse": 1.20,
                    "screen_mean_mae": 1.10,
                    "heat_the_chair_rmse": 0.98,
                },
            ]
        )

        shortlist_rows = task_eval_public_opt_torch_module._build_torch_uab_full_shortlist_rows(
            leaderboard=leaderboard,
            full_candidate_limit=1,
            group_winner_limit=1,
            ensemble_policy="none",
            selected_subsets=("heat_the_chair",),
            candidate_catalog="heat_specialist",
        )

        shortlisted_candidate_ids = [str(row["candidate_id"]) for row in shortlist_rows]
        self.assertEqual(
            shortlisted_candidate_ids[0],
            "heat_linear_huber_lowdim__lr0p001__wd0p0001",
        )
        self.assertIn(
            "heat_mlp_lowdim__lr0p001__wd0p0001",
            shortlisted_candidate_ids,
        )
        self.assertIn(
            "heat_affine_calibrated_blend__lr0p001__wd0p0001",
            shortlisted_candidate_ids,
        )
        self.assertIn(
            "heat_residual_correction__lr0p001__wd0p0001",
            shortlisted_candidate_ids,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_public_opt_torch_uab_runs_on_cuda_with_finite_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            reference_public_opt = Path(temp_dir) / "reference_public_opt_summary.json"
            reference_deep = Path(temp_dir) / "reference_deep_summary.json"
            _write_reference_public_opt_summary(reference_public_opt)
            _, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            reference_deep.write_text(
                Path(deep_summary_path).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = run_task_eval_public_opt_torch_uab(
                StageIPublicOptTorchUABConfig(
                    run_id="public-opt-uab-torch-smoke",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    device="cuda",
                    screen_max_folds=1,
                    full_max_folds=1,
                    batch_size=32,
                    epochs=1,
                    patience=1,
                    full_candidate_limit=1,
                    ensemble_policy="none",
                    feature_profiles=("full",),
                    learning_rates=(1e-3,),
                    weight_decays=(1e-4,),
                    reference_public_opt_summary_path=str(reference_public_opt),
                    reference_deep_comparison_summary_path=str(reference_deep),
                )
            )

            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.predictions_path).exists())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_device"], "cuda")
            self.assertTrue(summary["screen_leaderboard"])
            self.assertIn("selected_result", summary)
            predictions = pd.read_csv(result.predictions_path)
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())
            for subset_id in ("n_back", "heat_the_chair"):
                metrics = summary["selected_result"]["groups"][subset_id]
                self.assertTrue(np.isfinite(float(metrics["rmse"])))
                self.assertTrue(np.isfinite(float(metrics["mae"])))


def _build_prepared_uab_root(temp_root: Path) -> Path:
    dataset_root = temp_root / "dataset"
    prepared_root = temp_root / "prepared_uab"
    _write_mini_uab_dataset(dataset_root)
    run_task_eval_sequence_preparation(
        StageISequencePreparationConfig(
            dataset_id="uab_workload_dataset",
            artifact_root=str(prepared_root),
            dataset_root=str(dataset_root),
            profile="window_v2",
            target_steps=64,
        )
    )
    return prepared_root


def _build_public_opt_prior_probe_frame(
    *,
    y_by_group: dict[str, float],
    feature_by_group: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for index, (split_group, y_true) in enumerate(y_by_group.items()):
        rows.append(
            {
                "track": "subjective",
                "dataset_id": "uab_workload_dataset",
                "profile": "window_v2",
                "evaluation_group": "heat_the_chair",
                "subset_id": "heat_the_chair",
                "split_group": split_group,
                "sample_id": f"sample-{index}",
                "subject_id": split_group,
                "session_id": f"session-{split_group}",
                "window_index": index,
                "y_true": float(y_true),
                "residual__physiology_intensity_mean": float(
                    feature_by_group[split_group]
                ),
            }
        )
    return pd.DataFrame(rows)


def _expected_train_median(values: np.ndarray) -> float:
    return float(np.median(values.astype(float, copy=False)))


def _expected_train_trimmed_mean(values: np.ndarray) -> float:
    finite_values = values.astype(float, copy=False)
    lower, upper = np.quantile(finite_values, [0.1, 0.9])
    trimmed = finite_values[(finite_values >= lower) & (finite_values <= upper)]
    if trimmed.size == 0:
        trimmed = finite_values
    return float(np.mean(trimmed, dtype=np.float64))


def _build_public_opt_script_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "run_id": None,
        "prepared_artifact_root": "prepared_assets",
        "artifact_root": None,
        "report_root": "docs/artifacts",
        "dataset_id": "uab_workload_dataset",
        "profile": "window_v2",
        "seed": 42,
        "backend": "auto",
        "feature_profile": "full",
        "head_catalog": "expanded",
        "train_balance_policy": "class_weight_balanced",
        "ensemble_policy": "none",
        "prediction_aggregation_policy": "none",
        "winner_margin_policy": "paper_gate",
        "device": "auto",
        "batch_size": 256,
        "epochs": 20,
        "patience": 4,
        "screen_max_folds": 2,
        "full_max_folds": None,
        "full_candidate_limit": 2,
        "full_group_winner_limit": 1,
        "skip_full_loso": False,
        "supervision_granularity": "window",
        "torch_feature_profiles": [],
        "learning_rates": [],
        "weight_decays": [],
        "allow_cpu_debug": False,
        "allow_cpu_heavy_sklearn": False,
        "selected_subsets": [],
        "torch_candidate_catalog": "heat_specialist",
        "reference_public_opt_summary": None,
        "reference_phase3_closure_summary": None,
        "reference_deep_comparison_summary": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _build_prepared_nasa_root(temp_root: Path) -> Path:
    dataset_root = temp_root / "dataset"
    prepared_root = temp_root / "prepared_nasa"
    _write_mini_nasa_csm_dataset(dataset_root)
    run_task_eval_sequence_preparation(
        StageISequencePreparationConfig(
            dataset_id="nasa_csm",
            artifact_root=str(prepared_root),
            dataset_root=str(dataset_root),
            profile="window_v2",
            target_steps=64,
        )
    )
    return prepared_root


def _force_constant_subjective_targets(prepared_root: Path, *, value: float) -> None:
    entries = load_task_eval_sequence_entries(prepared_root / "task_manifest.jsonl")
    constant_entries = tuple(
        replace(entry, subjective_target_value=value)
        if entry.training_role == "primary" and entry.subset_id in {"n_back", "heat_the_chair"}
        else entry
        for entry in entries
    )
    dump_task_eval_sequence_entries(
        constant_entries,
        path=prepared_root / "task_manifest.jsonl",
    )

    bundle = load_task_eval_sequence_bundle(prepared_root / "sequence_bundle.npz")
    updated_bundle = replace(
        bundle,
        subjective_target_values=np.asarray(
            [
                value
                if entry.training_role == "primary" and entry.subset_id in {"n_back", "heat_the_chair"}
                else bundle.subjective_target_values[index]
                for index, entry in enumerate(constant_entries)
            ],
            dtype=np.float32,
        ),
    )
    save_task_eval_sequence_bundle(
        updated_bundle,
        path=prepared_root / "sequence_bundle.npz",
    )


def _force_constant_objective_targets(prepared_root: Path, *, value: int) -> None:
    entries = load_task_eval_sequence_entries(prepared_root / "task_manifest.jsonl")
    constant_entries = tuple(
        replace(entry, objective_label_value=value)
        if entry.training_role == "primary" and entry.subset_id in {"benchmark", "loft"}
        else entry
        for entry in entries
    )
    dump_task_eval_sequence_entries(
        constant_entries,
        path=prepared_root / "task_manifest.jsonl",
    )

    bundle = load_task_eval_sequence_bundle(prepared_root / "sequence_bundle.npz")
    updated_bundle = replace(
        bundle,
        objective_label_values=np.asarray(
            [
                float(value)
                if entry.training_role == "primary" and entry.subset_id in {"benchmark", "loft"}
                else bundle.objective_label_values[index]
                for index, entry in enumerate(constant_entries)
            ],
            dtype=np.float32,
        ),
    )
    save_task_eval_sequence_bundle(
        updated_bundle,
        path=prepared_root / "sequence_bundle.npz",
    )


def _write_reference_summaries(temp_root: Path) -> tuple[Path, Path]:
    phase3_summary_path = temp_root / "phase3_closure_summary.json"
    deep_summary_path = temp_root / "deep_comparison_summary.json"
    phase3_summary_path.write_text(
        json.dumps(
            {
                "uab_window": {
                    "subjective_primary": {
                        "n_back": {"rmse": 10.0, "mae": 4.8, "sample_count": 10},
                        "heat_the_chair": {"rmse": 1.8, "mae": 1.2, "sample_count": 10},
                    }
                },
                "nasa_attention": {
                    "objective_primary": {
                        "benchmark_only": {"macro_f1": 0.46, "balanced_accuracy": 0.49},
                        "loft_only": {"macro_f1": 0.37, "balanced_accuracy": 0.38},
                        "combined": {"macro_f1": 0.37, "balanced_accuracy": 0.37},
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    deep_summary_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "uab_workload_dataset": {
                        "status": "completed",
                        "models": {
                            "mult": {
                                "summary": {
                                    "subjective": {
                                        "groups": {
                                            "n_back": {"rmse": 5.8, "mae": 4.6},
                                            "heat_the_chair": {"rmse": 2.8, "mae": 2.3},
                                        }
                                    }
                                }
                            },
                            "contiformer": {
                                "summary": {
                                    "subjective": {
                                        "groups": {
                                            "n_back": {"rmse": 4.6, "mae": 3.8},
                                            "heat_the_chair": {"rmse": 1.45, "mae": 1.16},
                                        }
                                    }
                                }
                            },
                        },
                    },
                    "nasa_csm": {
                        "status": "completed",
                        "models": {
                            "mult": {
                                "summary": {
                                    "objective": {
                                        "groups": {
                                            "benchmark_only": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                            "loft_only": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                            "combined": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                        }
                                    }
                                }
                            },
                            "contiformer": {
                                "summary": {
                                    "objective": {
                                        "groups": {
                                            "benchmark_only": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                            "loft_only": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                            "combined": {"macro_f1": 0.30, "balanced_accuracy": 0.33},
                                        }
                                    }
                                }
                            },
                        },
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return phase3_summary_path, deep_summary_path


def _write_reference_public_opt_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "subset_results": {
                    "n_back": {
                        "best_head": "ridge_residual",
                        "heads": {
                            "ridge_residual": {
                                "rmse": 4.6103,
                                "mae": 3.8043,
                            }
                        },
                    },
                    "heat_the_chair": {
                        "best_head": "physiology_persistence",
                        "heads": {
                            "physiology_persistence": {
                                "rmse": 1.4568,
                                "mae": 1.1637,
                            }
                        },
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_reference_legacy_uab_summary(
    path: Path,
    *,
    n_back_rmse: float,
    n_back_mae: float,
    heat_rmse: float,
    heat_mae: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "subset_results": {
                    "n_back": {
                        "best_head": "ridge_residual",
                        "heads": {
                            "ridge_residual": {
                                "rmse": n_back_rmse,
                                "mae": n_back_mae,
                            }
                        },
                    },
                    "heat_the_chair": {
                        "best_head": "physiology_persistence",
                        "heads": {
                            "physiology_persistence": {
                                "rmse": heat_rmse,
                                "mae": heat_mae,
                            }
                        },
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_reference_prior_uab_summary(
    path: Path,
    *,
    n_back_rmse: float,
    n_back_mae: float,
    heat_rmse: float,
    heat_mae: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "head_catalog": "uab_hybrid",
                "prediction_aggregation_policy": "none",
                "subset_results": {
                    "n_back": {
                        "best_head": "ridge_residual",
                        "heads": {
                            "ridge_residual": {
                                "rmse": n_back_rmse,
                                "mae": n_back_mae,
                            }
                        },
                    },
                    "heat_the_chair": {
                        "best_head": "target_prior_median",
                        "heads": {
                            "target_prior_median": {
                                "rmse": heat_rmse,
                                "mae": heat_mae,
                            }
                        },
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_reference_torch_uab_summary(
    path: Path,
    *,
    n_back_rmse: float,
    n_back_mae: float,
    heat_rmse: float,
    heat_mae: float,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "selected_result": {
                    "groups": {
                        "n_back": {
                            "rmse": n_back_rmse,
                            "mae": n_back_mae,
                            "r2": 0.0,
                            "spearman": 0.0,
                        },
                        "heat_the_chair": {
                            "rmse": heat_rmse,
                            "mae": heat_mae,
                            "r2": 0.0,
                            "spearman": 0.0,
                        },
                    }
                },
                "acceptance": {
                    "groups": {
                        "n_back": {
                            "threshold_rmse": 4.6,
                            "observed_rmse": n_back_rmse,
                            "passed": n_back_rmse < 4.6,
                        },
                        "heat_the_chair": {
                            "threshold_rmse": 1.45,
                            "observed_rmse": heat_rmse,
                            "passed": heat_rmse < 1.45,
                        },
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_reference_nasa_public_opt_summary(path_root: Path) -> Path:
    path = path_root / "nasa_public_opt_summary.json"
    path.write_text(
        json.dumps(
            {
                "subset_results": {
                    "benchmark_only": {
                        "best_head": "balanced_linear_svc_context",
                        "heads": {
                            "balanced_linear_svc_context": {
                                "macro_f1": 0.74,
                                "balanced_accuracy": 0.75,
                            }
                        },
                    },
                    "loft_only": {
                        "best_head": "balanced_linear_svc_context",
                        "heads": {
                            "balanced_linear_svc_context": {
                                "macro_f1": 0.36,
                                "balanced_accuracy": 0.40,
                            }
                        },
                    },
                    "combined": {
                        "best_head": "balanced_logistic_context",
                        "heads": {
                            "balanced_logistic_context": {
                                "macro_f1": 0.46,
                                "balanced_accuracy": 0.56,
                            }
                        },
                    },
                },
                "winning_margin_vs_deep": {
                    "combined": {
                        "best_public_head": "balanced_logistic_context",
                        "best_public_value": 0.46,
                        "best_deep_model": "mult",
                        "best_deep_value": 0.30,
                        "margin_vs_best_deep": 0.16,
                        "gate_passed": True,
                        "rerun_threshold": 0.005,
                        "needs_deep_rerun": False,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
