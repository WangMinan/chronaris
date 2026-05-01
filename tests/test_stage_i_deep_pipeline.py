"""Stage I deep-baseline sequence preparation and pipeline tests."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    StageISequenceBundle,
    StageISequenceEntry,
    dump_stage_i_sequence_entries,
    load_stage_i_sequence_bundle,
    load_stage_i_sequence_entries,
    save_stage_i_sequence_bundle,
)
from chronaris.features import (  # noqa: E402
    prepare_nasa_sequences,
    prepare_stage_h_case_sequences,
    prepare_uab_sequences,
)
from chronaris.pipelines import (  # noqa: E402
    StageIDeepBaselineConfig,
    StageIDeepComparisonConfig,
    StageISequencePreparationConfig,
    run_stage_i_deep_baseline,
    run_stage_i_deep_comparison,
    run_stage_i_sequence_preparation,
)
from chronaris.pipelines.stage_i_deep_baseline import (  # noqa: E402
    _sanitize_regression_outputs,
)

REAL_DATASET_ROOT = Path("/home/wangminan/dataset/chronaris")
REAL_STAGE_H_RUN_MANIFEST = Path(
    "/home/wangminan/projects/chronaris/docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json",
)
ENABLE_LIVE_SEQUENCE_TESTS = (
    os.environ.get("CHRONARIS_ENABLE_STAGE_I_LIVE_SEQUENCE_TESTS") == "1"
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


class StageISequenceContractTest(unittest.TestCase):
    def test_sequence_entry_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "task_manifest.jsonl"
            entry = StageISequenceEntry(
                sample_id="sample-001",
                dataset_id="uab_workload_dataset",
                subset_id="n_back",
                subject_id="subject_01",
                session_id="session-001",
                split_group="subject_01",
                training_role="primary",
                sequence_bundle_path="/tmp/sequence_bundle.npz",
                sequence_length=64,
                modality_schema={"eeg": {"feature_dim": 2}},
                source_origin="uab_window_v2",
                task_family="workload",
                label_namespace="workload_level",
                objective_label_name="workload_level",
                objective_label_value=1,
                subjective_target_name="tlx_mean",
                subjective_target_value=42.0,
            )
            dump_stage_i_sequence_entries((entry,), path=manifest_path)
            loaded = load_stage_i_sequence_entries(manifest_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, entry.sample_id)
            self.assertEqual(loaded[0].sequence_length, 64)
            self.assertEqual(loaded[0].modality_schema["eeg"]["feature_dim"], 2)

    def test_sequence_bundle_loader_validates_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "invalid_bundle.npz"
            np.savez(bundle_path, sample_ids=np.asarray(["sample-001"], dtype=str))
            with self.assertRaises(ValueError):
                load_stage_i_sequence_bundle(bundle_path)

    def test_sequence_bundle_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "sequence_bundle.npz"
            bundle = StageISequenceBundle(
                sample_ids=("sample-001", "sample-002"),
                time_axis=np.asarray([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32),
                modality_arrays={
                    "eeg": np.asarray(
                        [
                            [[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]],
                        ],
                        dtype=np.float32,
                    ),
                },
                modality_masks={
                    "eeg": np.asarray([[1, 1], [1, 0]], dtype=np.uint8),
                },
                objective_label_values=np.asarray([0.0, 1.0], dtype=np.float32),
                objective_label_mask=np.asarray([1, 1], dtype=np.uint8),
                subjective_target_values=np.asarray([10.0, np.nan], dtype=np.float32),
                subjective_target_mask=np.asarray([1, 0], dtype=np.uint8),
                metadata_json=(
                    json.dumps({"sample_id": "sample-001"}),
                    json.dumps({"sample_id": "sample-002"}),
                ),
                extras={"event_scores": np.asarray([[0.1, 0.2], [0.3, 0.4]])},
            )
            save_stage_i_sequence_bundle(bundle, path=bundle_path)
            loaded = load_stage_i_sequence_bundle(bundle_path)
            self.assertEqual(loaded.sample_ids, bundle.sample_ids)
            self.assertEqual(loaded.modality_arrays["eeg"].shape, (2, 2, 2))
            self.assertIn("event_scores", loaded.extras)

    def test_regression_prediction_sanitizer_replaces_nonfinite_values(self) -> None:
        sanitized, nonfinite_mask = _sanitize_regression_outputs(
            np.asarray([1.0, np.nan, np.inf, -np.inf], dtype=np.float32),
            fallback_value=3.5,
        )
        np.testing.assert_allclose(sanitized, np.asarray([1.0, 3.5, 3.5, 3.5], dtype=np.float32))
        np.testing.assert_array_equal(nonfinite_mask, np.asarray([False, True, True, True]))


class StageIRealSortieSequenceTest(unittest.TestCase):
    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "Stage H closure asset is not present")
    def test_prepare_real_sortie_sequences_keeps_three_views_and_37_windows(self) -> None:
        payload = prepare_stage_h_case_sequences(REAL_STAGE_H_RUN_MANIFEST)
        self.assertEqual(payload.summary.dataset_id, "stage_h_case")
        self.assertEqual(payload.summary.extra_summary["view_count"], 3)
        view_window_counts = payload.summary.extra_summary["view_window_counts"]
        self.assertEqual(set(view_window_counts.values()), {37})
        metadata = [json.loads(item) for item in payload.bundle.metadata_json]
        by_sortie = {}
        for item in metadata:
            by_sortie.setdefault(item["sortie_id"], set()).add(item["pilot_id"])
        self.assertEqual(by_sortie["20251002_单01_ACT-8_翼云_J16_12#01"], {10033, 10035})
        verdicts = payload.summary.extra_summary["projection_diagnostics_verdict_counts"]
        self.assertEqual(verdicts["PASS"], 2)
        self.assertEqual(verdicts["WARN"], 1)

    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "Stage H closure asset is not present")
    def test_real_sortie_baseline_smoke_runs_for_mult_and_contiformer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = Path(temp_dir) / "prepared"
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="stage_h_case",
                    artifact_root=str(prepared_root),
                    stage_h_run_manifest_path=str(REAL_STAGE_H_RUN_MANIFEST),
                    profile="real_sortie_v1",
                ),
            )
            for model_name in ("mult", "contiformer"):
                result = run_stage_i_deep_baseline(
                    StageIDeepBaselineConfig(
                        model_name=model_name,
                        dataset_id="stage_h_case",
                        profile="real_sortie_v1",
                        prepared_artifact_root=str(prepared_root),
                        artifact_root=str(Path(temp_dir) / model_name),
                        epochs=1,
                        batch_size=8,
                    ),
                )
                self.assertEqual(result.summary["view_count"], 3)
                self.assertTrue(Path(result.summary_path).exists())
                self.assertTrue(Path(result.report_path).exists())


class StageIPublicSequencePreparationTest(unittest.TestCase):
    def test_prepare_uab_and_nasa_sequences_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)

            uab_payload = prepare_uab_sequences(dataset_root)
            self.assertEqual(uab_payload.summary.dataset_id, "uab_workload_dataset")
            self.assertEqual(uab_payload.bundle.time_axis.shape[1], 64)
            self.assertIn("n_back", uab_payload.summary.subset_counts)
            ecg_zero_mask = uab_payload.summary.extra_summary["ecg_zero_mask_samples"]
            self.assertEqual(ecg_zero_mask["n_back"], 3)

            nasa_payload = prepare_nasa_sequences(dataset_root)
            self.assertEqual(nasa_payload.summary.dataset_id, "nasa_csm")
            self.assertEqual(nasa_payload.bundle.time_axis.shape[1], 64)
            self.assertIn("benchmark", nasa_payload.summary.subset_counts)
            self.assertGreater(
                nasa_payload.summary.extra_summary["inventory_only_background_count"],
                0,
            )

    @unittest.skipUnless(
        ENABLE_LIVE_SEQUENCE_TESTS and (REAL_DATASET_ROOT / "uab_workload_dataset").exists(),
        "live UAB sequence test disabled",
    )
    def test_live_uab_sequence_counts_preserve_window_profile(self) -> None:
        payload = prepare_uab_sequences(REAL_DATASET_ROOT)
        self.assertEqual(payload.summary.entry_count, payload.bundle.entry_count)
        self.assertGreater(payload.summary.subset_counts["n_back"], 0)
        self.assertIn("heat_the_chair", payload.summary.subset_counts)
        self.assertGreater(payload.summary.extra_summary["ecg_zero_mask_samples"]["n_back"], 0)

    @unittest.skipUnless(
        ENABLE_LIVE_SEQUENCE_TESTS and (REAL_DATASET_ROOT / "nasa_csm").exists(),
        "live NASA sequence test disabled",
    )
    def test_live_nasa_sequence_preserves_benchmark_loft_and_background(self) -> None:
        payload = prepare_nasa_sequences(REAL_DATASET_ROOT)
        self.assertEqual(set(payload.summary.subset_counts), {"benchmark", "loft"})
        self.assertGreater(payload.summary.extra_summary["inventory_only_background_count"], 0)


class StageIDeepComparisonPipelineTest(unittest.TestCase):
    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "Stage H closure asset is not present")
    def test_prepare_baseline_and_comparison_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "datasets"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)

            stage_h_root = Path(temp_dir) / "stage_h_case"
            uab_root = Path(temp_dir) / "uab_sequences"
            nasa_root = Path(temp_dir) / "nasa_sequences"
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="stage_h_case",
                    artifact_root=str(stage_h_root),
                    stage_h_run_manifest_path=str(REAL_STAGE_H_RUN_MANIFEST),
                    profile="real_sortie_v1",
                ),
            )
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )
            run_stage_i_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )

            comparison = run_stage_i_deep_comparison(
                StageIDeepComparisonConfig(
                    model_names=("mult", "contiformer"),
                    dataset_artifact_roots={
                        "stage_h_case": str(stage_h_root),
                        "uab_workload_dataset": str(uab_root),
                        "nasa_csm": str(nasa_root),
                    },
                    output_root=str(Path(temp_dir) / "comparison"),
                    epochs=1,
                    batch_size=8,
                    max_folds=1,
                ),
            )
            self.assertTrue(Path(comparison.summary_path).exists())
            self.assertTrue(Path(comparison.report_path).exists())
            self.assertEqual(
                comparison.summary["dataset_order"],
                ["stage_h_case", "uab_workload_dataset", "nasa_csm"],
            )
            self.assertEqual(
                comparison.summary["datasets"]["stage_h_case"]["status"],
                "completed",
            )
