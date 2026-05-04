"""Stage I public-opt regression tests."""

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
    StageISequencePreparationConfig,
    run_stage_i_public_opt,
    run_stage_i_sequence_preparation,
)
from chronaris.pipelines.stage_i_public_opt_data import (  # noqa: E402
    PUBLIC_OPT_HEAD_FEATURES,
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
            self.assertEqual(
                set(feature_frame["subset_id"].unique()),
                {"n_back", "heat_the_chair"},
            )
            self.assertNotIn("flight_simulator", set(feature_frame["subset_id"].unique()))
            self.assertTrue((feature_frame["dataset_id"] == "uab_workload_dataset").all())
            self.assertFalse(any(name in {"objective_label_value", "subjective_target_value"} for name in feature_result.feature_columns))
            for head_name, feature_columns in PUBLIC_OPT_HEAD_FEATURES.items():
                self.assertTrue(feature_columns)
                self.assertTrue(set(feature_columns).issubset(set(feature_result.feature_columns)))
                self.assertIn(head_name, feature_result.head_feature_columns)

    def test_run_stage_i_public_opt_writes_expected_artifacts_and_subset_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = _build_prepared_uab_root(Path(temp_dir))
            result = run_stage_i_public_opt(
                StageIPublicOptConfig(
                    run_id="public-opt-test",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "artifacts"),
                    report_root=str(Path(temp_dir) / "reports"),
                )
            )

            self.assertTrue(Path(result.feature_frame_path).exists())
            self.assertTrue(Path(result.predictions_path).exists())
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.report_path).exists())

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["subset_order"], ["n_back", "heat_the_chair"])
            self.assertEqual(set(summary["heads"]), {"physiology_persistence", "ridge_residual"})
            for subset_id in ("n_back", "heat_the_chair"):
                subset_payload = summary["subset_results"][subset_id]
                self.assertEqual(
                    set(subset_payload["heads"]),
                    {"physiology_persistence", "ridge_residual"},
                )
                self.assertIn(
                    subset_payload["best_head"],
                    {"physiology_persistence", "ridge_residual"},
                )

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
                    predictions.loc[predictions["subset_id"] == subset_id, "split_group"]
                    .astype(str)
                    .unique()
                )
                self.assertEqual(observed_groups, expected_groups)

    def test_public_opt_fallback_keeps_predictions_and_metrics_finite(self) -> None:
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
                for head_name in ("physiology_persistence", "ridge_residual"):
                    metrics = summary["subset_results"][subset_id]["heads"][head_name]
                    self.assertTrue(np.isfinite(float(metrics["mae"])))
                    self.assertTrue(np.isfinite(float(metrics["rmse"])))


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
