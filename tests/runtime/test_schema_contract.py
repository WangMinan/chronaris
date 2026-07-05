"""Tests for runtime schema contract export and canonical payload generation."""

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
from chronaris.serving.runtime_schema_contract import build_runtime_schema_contract  # noqa: E402

_HELPER_SPEC = importlib.util.spec_from_file_location(
    "runtime_inference_helpers",
    Path(__file__).resolve().with_name("test_inference.py"),
)
if _HELPER_SPEC is None or _HELPER_SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError("failed to load runtime schema contract helpers")
_HELPER_MODULE = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER_MODULE)
_build_samples = _HELPER_MODULE._build_samples
_build_task_entries = _HELPER_MODULE._build_task_entries
_build_train_config = _HELPER_MODULE._build_train_config
run_task_eval_multitask_train = _HELPER_MODULE.run_task_eval_multitask_train


class RuntimeSchemaContractTest(unittest.TestCase):
    def test_runtime_schema_contract_reports_native_gap_and_canonical_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_samples = _build_samples()
            task_entries = _build_task_entries(original_samples)
            train_result = run_task_eval_multitask_train(
                _build_train_config(root, sample_count=len(original_samples)),
                samples=original_samples,
                task_entries=task_entries,
                source_summary={"source": "runtime_schema_contract_test"},
            )

            trimmed_samples = tuple(_drop_last_vehicle_feature(sample) for sample in original_samples)
            result = build_runtime_schema_contract(
                checkpoint_path=train_result.checkpoint_path,
                samples=trimmed_samples,
                output_root=root / "schema_contract",
            )

            contract = json.loads(Path(result.schema_contract_path).read_text(encoding="utf-8"))
            self.assertEqual(contract["native_input"]["comparison"]["status"], "aligned")
            self.assertEqual(contract["canonical_payload"]["comparison"]["status"], "exact")
            self.assertEqual(contract["expected_schema"]["vehicle"]["feature_count"], len(original_samples[0].vehicle.feature_names))
            self.assertEqual(contract["native_input"]["vehicle"]["feature_count"], len(trimmed_samples[0].vehicle.feature_names))
            self.assertEqual(contract["native_input"]["comparison"]["vehicle"]["missing_feature_count"], 1)
            self.assertEqual(contract["canonical_payload"]["vehicle"]["feature_count"], len(original_samples[0].vehicle.feature_names))
            self.assertTrue(Path(result.canonical_payload_path).exists())


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
