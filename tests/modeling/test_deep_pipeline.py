"""task evaluation deep-baseline sequence preparation and pipeline tests."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.dataset import (  # noqa: E402
    StageISequenceBundle,
    StageISequenceEntry,
    dump_task_eval_sequence_entries,
    load_task_eval_private_task_entries,
    load_task_eval_sequence_bundle,
    load_task_eval_sequence_entries,
    save_task_eval_sequence_bundle,
)
from chronaris.features import (  # noqa: E402
    prepare_nasa_sequences,
    prepare_feature_export_case_sequences,
    prepare_uab_sequences,
)
from chronaris.evaluation.dingxin.pipelines.benchmark import (  # noqa: E402
    StageIPrivateBenchmarkConfig,
    run_task_eval_private_benchmark,
)
from chronaris.evaluation.public_datasets.pipelines.deep_baseline import (  # noqa: E402
    StageIDeepBaselineConfig,
    StageIDeepComparisonConfig,
    run_task_eval_deep_baseline,
    run_task_eval_deep_comparison,
)
from chronaris.evaluation.public_datasets.pipelines.fusion_screen import (  # noqa: E402
    StageIPublicFusionCandidate,
    StageIPublicFusionScreenConfig,
    run_task_eval_public_fusion_screen,
)
from chronaris.evaluation.public_datasets.pipelines.sequence_preparation import (  # noqa: E402
    StageISequencePreparationConfig,
    run_task_eval_sequence_preparation,
)
from chronaris.evaluation.public_datasets.pipelines.deep_baseline import (  # noqa: E402
    _sanitize_regression_outputs,
)
from chronaris.evaluation.public_datasets.pipelines.deep_baseline_runtime import (  # noqa: E402
    _apply_classification_logit_adjustment,
    _classification_loss,
)
from chronaris.modeling.common.deep_models import build_task_eval_deep_model  # noqa: E402
from chronaris.evaluation.public_datasets.pipelines import fusion_screen as public_fusion_module  # noqa: E402

REAL_DATASET_ROOT = Path("/home/wangminan/dataset/chronaris")
REAL_STAGE_H_RUN_MANIFEST = Path(
    "/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-04-27_feature-export-closure/run_manifest.json",
)
ENABLE_LIVE_SEQUENCE_TESTS = (
    os.environ.get("CHRONARIS_ENABLE_STAGE_I_LIVE_SEQUENCE_TESTS") == "1"
)

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
            dump_task_eval_sequence_entries((entry,), path=manifest_path)
            loaded = load_task_eval_sequence_entries(manifest_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].sample_id, entry.sample_id)
            self.assertEqual(loaded[0].sequence_length, 64)
            self.assertEqual(loaded[0].modality_schema["eeg"]["feature_dim"], 2)

    def test_sequence_bundle_loader_validates_required_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle_path = Path(temp_dir) / "invalid_bundle.npz"
            np.savez(bundle_path, sample_ids=np.asarray(["sample-001"], dtype=str))
            with self.assertRaises(ValueError):
                load_task_eval_sequence_bundle(bundle_path)

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
            save_task_eval_sequence_bundle(bundle, path=bundle_path)
            loaded = load_task_eval_sequence_bundle(bundle_path)
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

    def test_balanced_class_path_uses_prior_logit_adjustment_without_weighted_loss(self) -> None:
        logits = np.zeros((2, 3), dtype=np.float32)
        train_targets = np.asarray([0, 0, 0, 1], dtype=int)
        adjusted = _apply_classification_logit_adjustment(
            logits,
            train_targets=train_targets,
            output_dim=3,
            sampling_policy="balanced_class",
        )
        self.assertGreater(adjusted[0, 0], adjusted[0, 1])
        self.assertAlmostEqual(float(adjusted[0, 1]), float(adjusted[0, 2]), places=6)
        balanced_loss = _classification_loss(
            train_targets,
            device="cpu",
            output_dim=3,
            sampling_policy="balanced_class",
        )
        default_loss = _classification_loss(
            train_targets,
            device="cpu",
            output_dim=3,
            sampling_policy="none",
        )
        self.assertIsNone(balanced_loss.weight)
        self.assertIsNotNone(default_loss.weight)


class StageIRealSortieSequenceTest(unittest.TestCase):
    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "feature export closure asset is not present")
    def test_prepare_real_sortie_sequences_keeps_three_views_and_37_windows(self) -> None:
        payload = prepare_feature_export_case_sequences(REAL_STAGE_H_RUN_MANIFEST)
        self.assertEqual(payload.summary.dataset_id, "feature_export_case")
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

    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "feature export closure asset is not present")
    def test_real_sortie_baseline_smoke_runs_for_mult_and_contiformer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_root = Path(temp_dir) / "prepared"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="feature_export_case",
                    artifact_root=str(prepared_root),
                    feature_export_run_manifest_path=str(REAL_STAGE_H_RUN_MANIFEST),
                    profile="real_sortie_v1",
                ),
            )
            for model_name in ("mult", "contiformer"):
                result = run_task_eval_deep_baseline(
                    StageIDeepBaselineConfig(
                        model_name=model_name,
                        dataset_id="feature_export_case",
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
            self.assertEqual(uab_payload.sequence_schema["adapter_id"], "chronaris_public_uab_v1")
            self.assertEqual(uab_payload.sequence_schema["evidence_role"], "public_adapter_evidence")
            self.assertEqual(uab_payload.sequence_schema["second_stream_role"], "context_proxy")
            self.assertFalse(uab_payload.sequence_schema["second_stream_is_real_vehicle"])
            self.assertEqual(
                set(uab_payload.bundle.modality_arrays),
                {"physiology", "task_context"},
            )
            self.assertIn("n_back", uab_payload.summary.subset_counts)
            ecg_zero_mask = uab_payload.summary.extra_summary["ecg_zero_mask_samples"]
            self.assertEqual(ecg_zero_mask["n_back"], 3)
            self.assertNotIn("objective_label_text", uab_payload.entries[0].context_payload)
            self.assertEqual(uab_payload.entries[0].context_payload["second_stream_name"], "task_context")
            self.assertEqual(uab_payload.entries[0].context_payload["second_stream_role"], "context_proxy")
            self.assertFalse(uab_payload.entries[0].context_payload["second_stream_is_real_vehicle"])

            nasa_payload = prepare_nasa_sequences(dataset_root)
            self.assertEqual(nasa_payload.summary.dataset_id, "nasa_csm")
            self.assertEqual(nasa_payload.bundle.time_axis.shape[1], 64)
            self.assertEqual(nasa_payload.sequence_schema["adapter_id"], "chronaris_public_nasa_v1")
            self.assertEqual(nasa_payload.sequence_schema["evidence_role"], "public_adapter_evidence")
            self.assertEqual(nasa_payload.sequence_schema["second_stream_role"], "context_proxy")
            self.assertFalse(nasa_payload.sequence_schema["second_stream_is_real_vehicle"])
            self.assertEqual(
                set(nasa_payload.bundle.modality_arrays),
                {"physiology", "scenario_context"},
            )
            diagnostics = nasa_payload.summary.extra_summary["processing_diagnostics"]
            self.assertEqual(diagnostics["modalities"], ["physiology", "scenario_context"])
            self.assertGreater(diagnostics["csv_file_count"], 0)
            self.assertNotIn("event_code", diagnostics["context_feature_names"])
            self.assertIn("benchmark", nasa_payload.summary.subset_counts)
            self.assertGreater(
                nasa_payload.summary.extra_summary["inventory_only_background_count"],
                0,
            )
            self.assertNotIn("event_code", nasa_payload.entries[0].context_payload)
            self.assertEqual(
                nasa_payload.entries[0].context_payload["second_stream_name"],
                "scenario_context",
            )
            self.assertEqual(nasa_payload.entries[0].context_payload["second_stream_role"], "context_proxy")
            self.assertFalse(nasa_payload.entries[0].context_payload["second_stream_is_real_vehicle"])

    def test_public_fusion_wrapper_marks_second_stream_as_context_proxy(self) -> None:
        model = build_task_eval_deep_model(
            model_name="chronaris_public_fusion",
            ordered_modalities=("physiology", "task_context"),
            modality_input_dims={"physiology": 4, "task_context": 3},
            output_dim=1,
        )
        self.assertEqual(model.thesis_facing_contract["evidence_role"], "public_adapter_evidence")
        self.assertEqual(model.thesis_facing_contract["second_stream_name"], "task_context")
        self.assertEqual(model.thesis_facing_contract["second_stream_role"], "context_proxy")
        self.assertFalse(model.thesis_facing_contract["second_stream_is_real_vehicle"])

    def test_chronaris_public_fusion_runs_on_synthetic_uab_and_nasa(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "dataset"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)

            uab_root = Path(temp_dir) / "uab_sequences"
            nasa_root = Path(temp_dir) / "nasa_sequences"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                )
            )
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                )
            )
            self.assertTrue((nasa_root / "run.log").exists())
            self.assertTrue((nasa_root / "progress.json").exists())
            self.assertTrue((nasa_root / "processing_diagnostics.json").exists())
            nasa_progress = json.loads(
                (nasa_root / "progress.json").read_text(encoding="utf-8")
            )
            nasa_events = [event["event"] for event in nasa_progress["events"]]
            self.assertIn("source_processing_started", nasa_events)
            self.assertIn("source_processing_finished", nasa_events)
            self.assertEqual(nasa_progress["last_event"], "finished")
            for dataset_id, prepared_root in (
                ("uab_workload_dataset", uab_root),
                ("nasa_csm", nasa_root),
            ):
                result = run_task_eval_deep_baseline(
                    StageIDeepBaselineConfig(
                        model_name="chronaris_public_fusion",
                        dataset_id=dataset_id,
                        profile="window_v2",
                        prepared_artifact_root=str(prepared_root),
                        artifact_root=str(Path(temp_dir) / dataset_id / "fusion"),
                        epochs=1,
                        batch_size=8,
                        max_folds=1,
                        fusion_event_bias_weight=0.5,
                        fusion_lag_window_points=4,
                        fusion_normalize_states=False,
                    )
                )
                self.assertEqual(result.summary["model_name"], "chronaris_public_fusion")
                self.assertEqual(result.summary["model_config"]["fusion_event_bias_weight"], 0.5)
                self.assertEqual(result.summary["model_config"]["fusion_lag_window_points"], 4)
                self.assertFalse(result.summary["model_config"]["fusion_normalize_states"])
                self.assertTrue(Path(result.summary_path).exists())
                self.assertTrue(Path(result.predictions_path).exists())
                self.assertTrue((Path(result.artifact_root) / "run.log").exists())
                self.assertTrue((Path(result.artifact_root) / "progress.json").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_public_deep_baseline_supports_cuda_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "datasets"
            _write_mini_uab_dataset(dataset_root)

            prepared_root = Path(temp_dir) / "uab_sequences"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(prepared_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )
            result = run_task_eval_deep_baseline(
                StageIDeepBaselineConfig(
                    model_name="mult",
                    dataset_id="uab_workload_dataset",
                    profile="window_v2",
                    prepared_artifact_root=str(prepared_root),
                    artifact_root=str(Path(temp_dir) / "mult_cuda"),
                    epochs=1,
                    batch_size=8,
                    max_folds=1,
                    device="cuda",
                )
            )
            self.assertEqual(result.summary["runtime_device"], "cuda")

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
    @unittest.skipUnless(REAL_STAGE_H_RUN_MANIFEST.exists(), "feature export closure asset is not present")
    def test_prepare_baseline_and_comparison_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "datasets"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)

            feature_export_root = Path(temp_dir) / "feature_export_case"
            uab_root = Path(temp_dir) / "uab_sequences"
            nasa_root = Path(temp_dir) / "nasa_sequences"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="feature_export_case",
                    artifact_root=str(feature_export_root),
                    feature_export_run_manifest_path=str(REAL_STAGE_H_RUN_MANIFEST),
                    profile="real_sortie_v1",
                ),
            )
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )

            comparison = run_task_eval_deep_comparison(
                StageIDeepComparisonConfig(
                    model_names=("mult", "contiformer"),
                    dataset_artifact_roots={
                        "feature_export_case": str(feature_export_root),
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
                ["feature_export_case", "uab_workload_dataset", "nasa_csm"],
            )
            self.assertEqual(
                comparison.summary["datasets"]["feature_export_case"]["status"],
                "completed",
            )


def _write_private_feature_export_run(
    root: Path,
    *,
    run_name: str,
    amplitude_scale: float,
    physics_enabled: bool,
) -> Path:
    run_root = root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    view_specs = (
        ("sortie-A", "sortie-A__pilot_10001", 10001, (0, 1, 2)),
        ("sortie-B", "sortie-B__pilot_10002", 10002, (0, 1, 2)),
        ("sortie-B", "sortie-B__pilot_10003", 10003, (0, 1, 2)),
    )
    sortie_manifest_paths: dict[str, str] = {}
    by_sortie: dict[str, list[tuple[str, Path]]] = {}
    for sortie_id, view_id, pilot_id, class_codes in view_specs:
        view_dir = run_root / "sorties" / sortie_id / "views" / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        raw_sample_ids = tuple(f"{sortie_id}:{index:04d}" for index in range(len(class_codes)))
        partitions = ("train", "validation", "test")
        bundle_path = view_dir / "feature_bundle.npz"
        window_manifest_path = view_dir / "window_manifest.jsonl"
        raw_summary_path = view_dir / "raw_window_summary.jsonl"
        projection_path = view_dir / "projection_diagnostics_summary.json"
        intermediate_path = view_dir / "intermediate_summary.json"
        causal_path = view_dir / "causal_fusion_summary.json"

        time_axis = np.tile(np.linspace(0.0, 5.0, 4, dtype=np.float32), (len(class_codes), 1))
        physiology_projection = []
        vehicle_projection = []
        physiology_hidden = []
        vehicle_hidden = []
        raw_rows = []
        window_rows = []
        for index, class_code in enumerate(class_codes):
            level = float(class_code + 1)
            physiology_projection.append(
                np.asarray(
                    [
                        [level * amplitude_scale, 0.1 * level],
                        [level * amplitude_scale + 0.2, 0.2 * level],
                        [level * amplitude_scale + 0.4, 0.3 * level],
                        [level * amplitude_scale + 0.6, 0.4 * level],
                    ],
                    dtype=np.float32,
                )
            )
            vehicle_projection.append(
                np.asarray(
                    [
                        [0.3 * level, level * amplitude_scale],
                        [0.4 * level, level * amplitude_scale + 0.2],
                        [0.5 * level, level * amplitude_scale + 0.4],
                        [0.6 * level, level * amplitude_scale + 0.6],
                    ],
                    dtype=np.float32,
                )
            )
            physiology_hidden.append(
                np.asarray(
                    [
                        [level, 0.2 * level],
                        [level + 0.3, 0.3 * level],
                        [level + 0.6, 0.4 * level],
                        [level + 0.9, 0.5 * level],
                    ],
                    dtype=np.float32,
                )
            )
            vehicle_hidden.append(
                np.asarray(
                    [
                        [0.2 * level, level],
                        [0.3 * level, level + 0.3],
                        [0.4 * level, level + 0.6],
                        [0.5 * level, level + 0.9],
                    ],
                    dtype=np.float32,
                )
            )
            raw_rows.append(
                {
                    "sample_id": raw_sample_ids[index],
                    "sortie_id": sortie_id,
                    "sample_partition": partitions[index],
                    "start_offset_ms": index * 5000,
                    "end_offset_ms": (index + 1) * 5000,
                    "physiology_feature_stats": {
                        "feature_count": 2,
                        "features": {
                            "eeg.alpha": {
                                "count": 4,
                                "mean": 0.5 * level,
                                "std": 0.1 * level,
                                "min": 0.4 * level,
                                "max": 0.7 * level,
                                "start": 0.4 * level,
                                "end": 0.7 * level,
                                "delta": 0.3 * level,
                            },
                            "spo2.spo2": {
                                "count": 4,
                                "mean": 97.0 - level,
                                "std": 0.2 * level,
                                "min": 96.5 - level,
                                "max": 97.3 - level,
                                "start": 97.2 - level,
                                "end": 96.8 - level,
                                "delta": -0.4 * level,
                            },
                        },
                    },
                    "vehicle_feature_stats": {
                        "feature_count": 2,
                        "features": {
                            "BUS001.speed": {
                                "count": 4,
                                "mean": 100.0 + 20.0 * level,
                                "std": 1.0 * level,
                                "min": 99.0 + 20.0 * level,
                                "max": 101.0 + 20.0 * level,
                                "start": 99.0 + 20.0 * level,
                                "end": 101.0 + 20.0 * level,
                                "delta": 2.0 * level,
                            },
                            "BUS001.accel": {
                                "count": 4,
                                "mean": 0.5 * level,
                                "std": 0.3 * level,
                                "min": 0.1 * level,
                                "max": 0.9 * level,
                                "start": 0.1 * level,
                                "end": 0.9 * level,
                                "delta": 0.8 * level,
                            },
                        },
                    },
                }
            )
            window_rows.append(
                {
                    "sample_id": raw_sample_ids[index],
                    "sortie_id": sortie_id,
                    "window_index": index,
                    "start_offset_ms": index * 5000,
                    "end_offset_ms": (index + 1) * 5000,
                    "physiology_point_count": 4,
                    "vehicle_point_count": 4,
                    "sample_partition": partitions[index],
                    "selected_for_model": True,
                }
            )

        np.savez(
            bundle_path,
            sample_ids=np.asarray(raw_sample_ids, dtype=str),
            sample_partitions=np.asarray(partitions, dtype=str),
            physiology_reference_projection=np.asarray(physiology_projection, dtype=np.float32),
            vehicle_reference_projection=np.asarray(vehicle_projection, dtype=np.float32),
            physiology_reference_hidden=np.asarray(physiology_hidden, dtype=np.float32),
            vehicle_reference_hidden=np.asarray(vehicle_hidden, dtype=np.float32),
            fused_representation=np.zeros((0,), dtype=np.float32),
            reference_offsets_s=time_axis,
            attention_weights=np.zeros((0,), dtype=np.float32),
            vehicle_event_scores=np.zeros((0,), dtype=np.float32),
        )
        window_manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in window_rows),
            encoding="utf-8",
        )
        raw_summary_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_rows),
            encoding="utf-8",
        )
        projection_path.write_text(json.dumps({"threshold_evaluation": {"checks": []}}), encoding="utf-8")
        causal_path.write_text(json.dumps({}), encoding="utf-8")
        intermediate_path.write_text(
            json.dumps(
                {
                    "partition": "all",
                    "sample_count": len(raw_sample_ids),
                    "reference_point_count": 4,
                    "sample_ids": list(raw_sample_ids),
                }
            ),
            encoding="utf-8",
        )
        view_manifest = {
            "view_id": view_id,
            "sortie_id": sortie_id,
            "pilot_id": pilot_id,
            "projection_diagnostics_verdict": "PASS",
            "intermediate_summary": {
                "partition": "all",
                "sample_count": len(raw_sample_ids),
                "reference_point_count": 4,
                "sample_ids": list(raw_sample_ids),
            },
            "artifact_paths": {
                "feature_bundle_npz": str(bundle_path),
                "projection_diagnostics_summary_json": str(projection_path),
                "causal_fusion_summary_json": str(causal_path),
                "intermediate_summary_json": str(intermediate_path),
                "window_manifest_jsonl": str(window_manifest_path),
                "raw_window_summary_jsonl": str(raw_summary_path),
            },
        }
        view_manifest_path = view_dir / "view_manifest.json"
        view_manifest_path.write_text(
            json.dumps(view_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        by_sortie.setdefault(sortie_id, []).append((view_id, view_manifest_path))

    for sortie_id, items in by_sortie.items():
        sortie_manifest = {
            "sortie_id": sortie_id,
            "view_manifest_paths": {
                view_id: str(path)
                for view_id, path in items
            },
        }
        sortie_manifest_path = run_root / "sorties" / sortie_id / "sortie_manifest.json"
        sortie_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        sortie_manifest_path.write_text(
            json.dumps(sortie_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        sortie_manifest_paths[sortie_id] = str(sortie_manifest_path)

    run_manifest = {
        "run_id": run_name,
        "export_version": "feature-export-v1",
        "output_root": str(run_root),
        "generated_view_count": 3,
        "generated_view_ids": [view_id for items in by_sortie.values() for view_id, _ in items],
        "config": {
            "export_profile": "validation",
            "physics_constraints_enabled": physics_enabled,
            "physics_constraint_family": "full",
            "causal_fusion_enabled": False,
            "intermediate_partition": "all",
            "physiology_point_limit_per_measurement": None,
            "vehicle_point_limit_per_measurement": None,
            "point_limit_note": "no per-measurement point cap",
        },
        "sortie_manifest_paths": sortie_manifest_paths,
        "partial_data": None,
    }
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_manifest_path


class StageIPrivateBenchmarkPipelineTest(unittest.TestCase):
    def test_private_benchmark_runs_on_synthetic_feature_export_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-e",
                amplitude_scale=0.8,
                physics_enabled=False,
            )
            f_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-f",
                amplitude_scale=1.4,
                physics_enabled=True,
            )

            result = run_task_eval_private_benchmark(
                StageIPrivateBenchmarkConfig(
                    run_id="private-benchmark-smoke",
                    e_run_manifest_path=str(e_manifest),
                    f_run_manifest_path=str(f_manifest),
                    output_root=str(root / "artifacts"),
                    report_root=str(root / "reports"),
                    max_deep_folds=1,
                    deep_epochs=1,
                    deep_batch_size=4,
                )
            )

            self.assertTrue(Path(result.task_manifest_path).exists())
            self.assertTrue(Path(result.task_summary_path).exists())
            self.assertTrue(Path(result.benchmark_summary_path).exists())
            self.assertTrue(Path(result.alignment_report_path).exists())
            self.assertTrue(Path(result.causal_report_path).exists())
            self.assertTrue(Path(result.optimality_report_path).exists())

            task_summary = json.loads(Path(result.task_summary_path).read_text(encoding="utf-8"))
            self.assertEqual(task_summary["entry_count"], 27)
            self.assertEqual(task_summary["benchmark_role"], "private_proxy_benchmark")
            self.assertEqual(task_summary["task_role"], "proxy_task")
            self.assertEqual(task_summary["task_role_counts"]["proxy_task"], 27)
            self.assertEqual(task_summary["coverage"]["T1_maneuver_intensity_class"]["valid_label_count"], 9)
            self.assertEqual(task_summary["coverage"]["T2_next_window_physiology_response"]["valid_label_count"], 6)
            self.assertEqual(task_summary["coverage"]["T3_paired_pilot_window_retrieval"]["valid_label_count"], 6)
            self.assertIn("BUS001.speed", task_summary["selected_vehicle_fields"])
            self.assertIn("eeg.alpha", task_summary["selected_physiology_fields"])
            self.assertEqual(
                task_summary["proxy_task_definitions"]["T1_maneuver_intensity_class"]["proxy_task_id"],
                "maneuver_intensity_proxy",
            )
            task_entries = load_task_eval_private_task_entries(result.task_manifest_path)
            self.assertTrue(all(entry.benchmark_role == "private_proxy_benchmark" for entry in task_entries))
            self.assertTrue(all(entry.task_role == "proxy_task" for entry in task_entries))
            self.assertTrue(
                all(
                    entry.context_payload["thesis_task_boundary"]
                    == "proxy_task_not_direct_thesis_task"
                    for entry in task_entries
                )
            )

            summary = json.loads(Path(result.benchmark_summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_role"], "private_proxy_benchmark")
            self.assertEqual(summary["task_role"], "proxy_task")
            self.assertEqual(summary["records"]["sample_count"], 9)
            self.assertEqual(summary["records"]["view_count"], 3)
            self.assertIn("g_min", summary["tasks"]["T1_maneuver_intensity_class"]["variants"])
            self.assertEqual(
                summary["tasks"]["T1_maneuver_intensity_class"]["variants"]["naive_sync"]["status"],
                "completed",
            )
            self.assertEqual(
                summary["tasks"]["T2_next_window_physiology_response"]["variants"]["f_full"]["status"],
                "completed",
            )
            self.assertEqual(
                summary["tasks"]["T3_paired_pilot_window_retrieval"]["variants"]["g_min"]["status"],
                "completed",
            )
            self.assertIn("mult", summary["tasks"]["T1_maneuver_intensity_class"]["deep_models"])
            self.assertIn("contiformer", summary["tasks"]["T2_next_window_physiology_response"]["deep_models"])
            self.assertTrue(Path(summary["plots"]["t1_metrics"]).exists())
            self.assertTrue(Path(summary["plots"]["t2_metrics"]).exists())
            self.assertTrue(Path(summary["plots"]["t3_metrics"]).exists())

    def test_public_fusion_screen_require_cuda_fails_before_candidates_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                public_fusion_module,
                "resolve_torch_device_name",
                return_value="cpu",
            ):
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    run_task_eval_public_fusion_screen(
                        StageIPublicFusionScreenConfig(
                            run_id="fusion-screen-require-cuda",
                            dataset_prepared_roots={"nasa_csm": str(Path(temp_dir) / "missing")},
                            artifact_root=str(Path(temp_dir) / "fusion_screen"),
                            report_root=str(Path(temp_dir) / "reports"),
                            device="auto",
                            require_cuda=True,
                        )
                    )
            run_root = Path(temp_dir) / "fusion_screen" / "fusion-screen-require-cuda"
            self.assertTrue((run_root / "run.log").exists())
            progress = json.loads((run_root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["last_event"], "failed")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_public_fusion_screen_runs_on_cuda_with_synthetic_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir) / "datasets"
            _write_mini_uab_dataset(dataset_root)
            _write_mini_nasa_csm_dataset(dataset_root)

            uab_root = Path(temp_dir) / "uab_sequences"
            nasa_root = Path(temp_dir) / "nasa_sequences"
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="uab_workload_dataset",
                    artifact_root=str(uab_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )
            run_task_eval_sequence_preparation(
                StageISequencePreparationConfig(
                    dataset_id="nasa_csm",
                    artifact_root=str(nasa_root),
                    dataset_root=str(dataset_root),
                    profile="window_v2",
                    target_steps=64,
                ),
            )

            result = run_task_eval_public_fusion_screen(
                StageIPublicFusionScreenConfig(
                    run_id="fusion-screen-test",
                    dataset_prepared_roots={
                        "uab_workload_dataset": str(uab_root),
                        "nasa_csm": str(nasa_root),
                    },
                    artifact_root=str(Path(temp_dir) / "fusion_screen"),
                    report_root=str(Path(temp_dir) / "reports"),
                    epochs=1,
                    batch_size=32,
                    max_folds=1,
                    device="cuda",
                    candidates=(
                        StageIPublicFusionCandidate(
                            candidate_id="test_candidate",
                            hidden_dim=32,
                            num_heads=2,
                            layers=1,
                            dropout=0.1,
                            fusion_event_bias_weight=0.25,
                            fusion_lag_window_points=4,
                            fusion_normalize_states=True,
                        ),
                    ),
                )
            )
            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.leaderboard_csv_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime_device"], "cuda")
            self.assertIn("nasa_csm", summary["per_dataset_rankings"])
            self.assertIn("uab_workload_dataset", summary["per_dataset_rankings"])

    def test_private_benchmark_rejects_dirty_e_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-e-dirty",
                amplitude_scale=0.8,
                physics_enabled=True,
            )
            f_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-f",
                amplitude_scale=1.4,
                physics_enabled=True,
            )

            with self.assertRaisesRegex(ValueError, "physics_constraints_enabled=false"):
                run_task_eval_private_benchmark(
                    StageIPrivateBenchmarkConfig(
                        run_id="private-benchmark-dirty-e",
                        e_run_manifest_path=str(e_manifest),
                        f_run_manifest_path=str(f_manifest),
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        max_deep_folds=1,
                    )
                )

    def test_private_benchmark_rejects_view_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            e_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-e",
                amplitude_scale=0.8,
                physics_enabled=False,
            )
            f_manifest = _write_private_feature_export_run(
                root,
                run_name="feature-export-f-mismatch",
                amplitude_scale=1.4,
                physics_enabled=True,
            )
            f_payload = json.loads(Path(f_manifest).read_text(encoding="utf-8"))
            sortie_manifest_path = Path(next(iter(f_payload["sortie_manifest_paths"].values())))
            sortie_payload = json.loads(sortie_manifest_path.read_text(encoding="utf-8"))
            removed_view_id = next(iter(sortie_payload["view_manifest_paths"]))
            sortie_payload["view_manifest_paths"].pop(removed_view_id)
            sortie_manifest_path.write_text(
                json.dumps(sortie_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "same view ids"):
                run_task_eval_private_benchmark(
                    StageIPrivateBenchmarkConfig(
                        run_id="private-benchmark-view-mismatch",
                        e_run_manifest_path=str(e_manifest),
                        f_run_manifest_path=str(f_manifest),
                        output_root=str(root / "artifacts"),
                        report_root=str(root / "reports"),
                        max_deep_folds=1,
                    )
                )
