"""Tests for task evaluation runtime/service smoke outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest

SRC = next(parent / "src" for parent in Path(__file__).resolve().parents if (parent / "src" / "chronaris").exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix  # noqa: E402
from chronaris.serving.runtime_inference import dump_runtime_samples_jsonl  # noqa: E402
from chronaris.serving.runtime_service_smoke import (  # noqa: E402
    StageIRuntimeSmokeConfig,
    run_task_eval_runtime_smoke,
)

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "runtime_inference_helpers",
    Path(__file__).resolve().with_name("test_inference.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load runtime smoke helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_build_samples = _HELPER_MODULE._build_samples
_build_task_entries = _HELPER_MODULE._build_task_entries
_build_train_config = _HELPER_MODULE._build_train_config
run_task_eval_multitask_train = _HELPER_MODULE.run_task_eval_multitask_train


class StageIRuntimeSmokeTest(unittest.TestCase):
    def test_runtime_service_smoke_writes_outputs_and_error_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = _build_samples()
            task_entries = _build_task_entries(samples)
            train_result = run_task_eval_multitask_train(
                _build_train_config(root, sample_count=len(samples)),
                samples=samples,
                task_entries=task_entries,
                source_summary={"source": "runtime_service_smoke_test"},
            )

            sample_jsonl_path = root / "runtime_samples.jsonl"
            dump_runtime_samples_jsonl(samples, path=sample_jsonl_path)
            result = run_task_eval_runtime_smoke(
                StageIRuntimeSmokeConfig(
                    run_id="runtime-service-smoke",
                    checkpoint_path=train_result.checkpoint_path,
                    sample_jsonl_path=str(sample_jsonl_path),
                    artifact_root=str(root / "service"),
                    report_root=str(root / "reports"),
                    device="cpu",
                )
            )

            self.assertTrue(Path(result.summary_path).exists())
            self.assertTrue(Path(result.report_path).exists())
            self.assertTrue(Path(result.figure_manifest_path).exists())
            self.assertTrue(Path(result.predictions_jsonl_path).exists())
            self.assertTrue(Path(result.error_cases_path).exists())

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["input_sample_count"], len(samples))
            self.assertEqual(summary["view_count"], 1)
            self.assertEqual(summary["canonical_feature_schema_status"], "exact")
            self.assertTrue(Path(summary["schema_contract_path"]).exists())

            error_cases = json.loads(Path(result.error_cases_path).read_text(encoding="utf-8"))
            self.assertEqual(
                [row["case_id"] for row in error_cases],
                ["missing_checkpoint", "missing_fields", "empty_window", "schema_mismatch"],
            )
            self.assertTrue(all(row["status"] == "expected_failure" for row in error_cases))

            figure_manifest = json.loads(Path(result.figure_manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(
                [row["figure_id"] for row in figure_manifest["figures"]],
                ["runtime_service_flow", "runtime_payload_schema", "runtime_error_cases"],
            )

    def test_runtime_service_smoke_records_native_aligned_and_canonical_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = tuple(_drop_last_vehicle_feature(sample) for sample in _build_samples())
            task_entries = _build_task_entries(_build_samples())
            train_result = run_task_eval_multitask_train(
                _build_train_config(root, sample_count=len(samples)),
                samples=_build_samples(),
                task_entries=task_entries,
                source_summary={"source": "runtime_service_schema_contract_test"},
            )

            sample_jsonl_path = root / "runtime_samples_aligned.jsonl"
            dump_runtime_samples_jsonl(samples, path=sample_jsonl_path)
            result = run_task_eval_runtime_smoke(
                StageIRuntimeSmokeConfig(
                    run_id="runtime-service-smoke-aligned",
                    checkpoint_path=train_result.checkpoint_path,
                    sample_jsonl_path=str(sample_jsonl_path),
                    artifact_root=str(root / "service"),
                    report_root=str(root / "reports"),
                    device="cpu",
                    strict_feature_schema=True,
                )
            )

            summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
            self.assertEqual(summary["native_feature_schema_status"], "aligned")
            self.assertEqual(summary["canonical_feature_schema_status"], "exact")
            self.assertGreater(summary["missing_vehicle_feature_count"], 0)
            self.assertTrue(Path(summary["canonical_payload_path"]).exists())
            self.assertTrue(Path(summary["canonical_runtime_summary_path"]).exists())

            error_cases = json.loads(Path(result.error_cases_path).read_text(encoding="utf-8"))
            case_ids = [row["case_id"] for row in error_cases]
            self.assertIn("native_strict_feature_schema", case_ids)
            native_probe = next(row for row in error_cases if row["case_id"] == "native_strict_feature_schema")
            self.assertEqual(native_probe["status"], "expected_failure")


def _drop_last_vehicle_feature(sample: E0ExperimentSample) -> E0ExperimentSample:
    return E0ExperimentSample(
        sample_id=sample.sample_id,
        sortie_id=sample.sortie_id,
        start_offset_ms=sample.start_offset_ms,
        end_offset_ms=sample.end_offset_ms,
        physiology=sample.physiology,
        vehicle=NumericStreamMatrix(
            stream_kind=sample.vehicle.stream_kind,
            point_count=sample.vehicle.point_count,
            feature_names=sample.vehicle.feature_names[:-1],
            point_offsets_ms=sample.vehicle.point_offsets_ms,
            point_measurements=sample.vehicle.point_measurements,
            values=tuple(tuple(row[:-1]) for row in sample.vehicle.values),
            dropped_fields=sample.vehicle.dropped_fields,
        ),
        notes=sample.notes,
    )


if __name__ == "__main__":
    unittest.main()
