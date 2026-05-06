"""Stage I public-opt regression and classification tests."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    dump_stage_i_sequence_entries,
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    save_stage_i_sequence_bundle,
)
from chronaris.pipelines import (  # noqa: E402
    StageIPublicOptConfig,
    StageIPublicOptTorchUABConfig,
    StageISequencePreparationConfig,
    run_stage_i_public_opt,
    run_stage_i_public_opt_torch_uab,
    run_stage_i_sequence_preparation,
)
from chronaris.pipelines.stage_i.stage_i_public_opt_data import (  # noqa: E402
    build_stage_i_public_opt_feature_frame,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "stage_i_pipeline_helpers",
    Path(__file__).resolve().with_name("test_stage_i_pipeline.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load Stage I synthetic dataset helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_write_mini_uab_dataset = _HELPER_MODULE._write_mini_uab_dataset
_write_mini_nasa_csm_dataset = _HELPER_MODULE._write_mini_nasa_csm_dataset


class StageIPublicOptTest(unittest.TestCase):
    def test_public_opt_feature_frame_filters_to_primary_subjective_uab_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            entries = load_stage_i_sequence_entries(prepared_root / "task_manifest.jsonl")
            bundle = load_stage_i_sequence_bundle(prepared_root / "sequence_bundle.npz")

            feature_result = build_stage_i_public_opt_feature_frame(
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
                {"full", "physiology_only", "context_only", "residual_only"},
            )
            for head_name in (
                "physiology_persistence",
                "ridge_residual_cv",
                "elasticnet_residual",
                "huber_residual",
            ):
                self.assertIn(head_name, feature_result.head_feature_columns)
                self.assertTrue(feature_result.head_feature_columns[head_name])

    def test_public_opt_feature_frame_filters_to_primary_nasa_attention_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            entries = load_stage_i_sequence_entries(prepared_root / "task_manifest.jsonl")
            bundle = load_stage_i_sequence_bundle(prepared_root / "sequence_bundle.npz")

            feature_result = build_stage_i_public_opt_feature_frame(
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

    def test_run_stage_i_public_opt_writes_expected_uab_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            phase3_summary_path, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            result = run_stage_i_public_opt(
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

    def test_run_stage_i_public_opt_writes_expected_nasa_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_nasa_root(Path(temp_dir))
            phase3_summary_path, deep_summary_path = _write_reference_summaries(Path(temp_dir))
            result = run_stage_i_public_opt(
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

    def test_public_opt_predictions_preserve_subject_loso_split_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_stage_i_public_opt(
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
            result = run_stage_i_public_opt(
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

            result = run_stage_i_public_opt(
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

            result = run_stage_i_public_opt(
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
            result = run_stage_i_public_opt(
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

            result = run_stage_i_public_opt_torch_uab(
                StageIPublicOptTorchUABConfig(
                    run_id="public-opt-uab-torch-smoke",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                    device="cuda",
                    screen_max_folds=1,
                    full_max_folds=1,
                    batch_size=32,
                    epochs=2,
                    patience=1,
                    reference_public_opt_summary_path=str(reference_public_opt),
                    reference_deep_comparison_summary_path=str(reference_deep),
                )
            )

            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.predictions_path).exists())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_device"], "cuda")
            self.assertTrue(summary["screen_leaderboard"])
            self.assertIn("final_result", summary)
            predictions = pd.read_csv(result.predictions_path)
            self.assertTrue(np.isfinite(predictions["y_pred"].to_numpy(dtype=float)).all())
            for subset_id in ("n_back", "heat_the_chair"):
                metrics = summary["final_result"]["groups"][subset_id]
                self.assertTrue(np.isfinite(float(metrics["rmse"])))
                self.assertTrue(np.isfinite(float(metrics["mae"])))


def _build_prepared_uab_root(temp_root: Path) -> Path:
    dataset_root = temp_root / "dataset"
    prepared_root = temp_root / "prepared_uab"
    _write_mini_uab_dataset(dataset_root)
    run_stage_i_sequence_preparation(
        StageISequencePreparationConfig(
            dataset_id="uab_workload_dataset",
            artifact_root=str(prepared_root),
            dataset_root=str(dataset_root),
            profile="window_v2",
            target_steps=64,
        )
    )
    return prepared_root


def _build_prepared_nasa_root(temp_root: Path) -> Path:
    dataset_root = temp_root / "dataset"
    prepared_root = temp_root / "prepared_nasa"
    _write_mini_nasa_csm_dataset(dataset_root)
    run_stage_i_sequence_preparation(
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
    entries = load_stage_i_sequence_entries(prepared_root / "task_manifest.jsonl")
    constant_entries = tuple(
        replace(entry, subjective_target_value=value)
        if entry.training_role == "primary" and entry.subset_id in {"n_back", "heat_the_chair"}
        else entry
        for entry in entries
    )
    dump_stage_i_sequence_entries(
        constant_entries,
        path=prepared_root / "task_manifest.jsonl",
    )

    bundle = load_stage_i_sequence_bundle(prepared_root / "sequence_bundle.npz")
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
    save_stage_i_sequence_bundle(
        updated_bundle,
        path=prepared_root / "sequence_bundle.npz",
    )


def _force_constant_objective_targets(prepared_root: Path, *, value: int) -> None:
    entries = load_stage_i_sequence_entries(prepared_root / "task_manifest.jsonl")
    constant_entries = tuple(
        replace(entry, objective_label_value=value)
        if entry.training_role == "primary" and entry.subset_id in {"benchmark", "loft"}
        else entry
        for entry in entries
    )
    dump_stage_i_sequence_entries(
        constant_entries,
        path=prepared_root / "task_manifest.jsonl",
    )

    bundle = load_stage_i_sequence_bundle(prepared_root / "sequence_bundle.npz")
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
    save_stage_i_sequence_bundle(
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
