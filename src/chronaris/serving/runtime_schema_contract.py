"""Runtime payload schema contract helpers for task evaluation service smoke."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from chronaris.features.experiment_input import E0ExperimentSample, NumericStreamMatrix
from chronaris.pipelines.alignment_preview import _deserialize_input_normalization_stats
from chronaris.schema.models import StreamKind
from chronaris.serving.runtime_inference import dump_runtime_samples_jsonl


@dataclass(frozen=True, slots=True)
class RuntimeSchemaContractRunResult:
    checkpoint_path: str
    schema_contract_path: str
    canonical_payload_path: str | None
    contract: Mapping[str, object]
    canonical_samples: tuple[E0ExperimentSample, ...]


def build_runtime_schema_contract(
    *,
    checkpoint_path: str | Path,
    samples: Sequence[E0ExperimentSample],
    output_root: str | Path,
    export_canonical_payload: bool = True,
    canonical_payload_filename: str = "canonical_runtime_samples.jsonl",
    schema_contract_filename: str = "runtime_schema_contract.json",
) -> RuntimeSchemaContractRunResult:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    expected_schema = _load_expected_schema(checkpoint)
    native_contract = _build_sample_schema_contract(samples=samples, expected_schema=expected_schema)
    canonical_samples = canonicalize_runtime_samples(
        samples=samples,
        expected_schema=expected_schema,
    )
    canonical_contract = _build_sample_schema_contract(
        samples=canonical_samples,
        expected_schema=expected_schema,
    )
    canonical_payload_path = None
    if export_canonical_payload:
        canonical_payload = output_root / canonical_payload_filename
        dump_runtime_samples_jsonl(canonical_samples, path=canonical_payload)
        canonical_payload_path = str(canonical_payload)

    contract = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "schema_source": expected_schema["schema_source"],
        "schema_hash": expected_schema["schema_hash"],
        "expected_schema": expected_schema["streams"],
        "native_input": native_contract,
        "canonical_payload": canonical_contract,
    }
    schema_contract_path = output_root / schema_contract_filename
    schema_contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return RuntimeSchemaContractRunResult(
        checkpoint_path=str(Path(checkpoint_path)),
        schema_contract_path=str(schema_contract_path),
        canonical_payload_path=canonical_payload_path,
        contract=contract,
        canonical_samples=canonical_samples,
    )


def canonicalize_runtime_samples(
    *,
    samples: Sequence[E0ExperimentSample],
    expected_schema: Mapping[str, object],
) -> tuple[E0ExperimentSample, ...]:
    physiology_feature_names = tuple(expected_schema["streams"]["physiology"]["feature_names"])
    vehicle_feature_names = tuple(expected_schema["streams"]["vehicle"]["feature_names"])
    return tuple(
        E0ExperimentSample(
            sample_id=sample.sample_id,
            sortie_id=sample.sortie_id,
            start_offset_ms=sample.start_offset_ms,
            end_offset_ms=sample.end_offset_ms,
            physiology=_align_stream_to_schema(sample.physiology, physiology_feature_names),
            vehicle=_align_stream_to_schema(sample.vehicle, vehicle_feature_names),
            notes=sample.notes,
        )
        for sample in samples
    )


def _load_expected_schema(checkpoint: Mapping[str, object]) -> dict[str, object]:
    feature_schema = dict(checkpoint.get("feature_schema") or {})
    physiology_feature_names = tuple(feature_schema.get("physiology_feature_names", ()))
    vehicle_feature_names = tuple(feature_schema.get("vehicle_feature_names", ()))
    schema_source = "feature_schema"
    if not physiology_feature_names or not vehicle_feature_names:
        normalization_stats = _deserialize_input_normalization_stats(
            checkpoint.get("input_normalization_stats")
        )
        if normalization_stats is not None:
            physiology_feature_names = physiology_feature_names or tuple(normalization_stats.physiology.feature_names)
            vehicle_feature_names = vehicle_feature_names or tuple(normalization_stats.vehicle.feature_names)
            schema_source = "input_normalization_stats"
    schema_hash = _build_schema_hash(physiology_feature_names, vehicle_feature_names)
    return {
        "schema_source": schema_source if (physiology_feature_names or vehicle_feature_names) else "input_samples",
        "schema_hash": schema_hash,
        "streams": {
            "physiology": _stream_schema_payload(physiology_feature_names),
            "vehicle": _stream_schema_payload(vehicle_feature_names),
        },
    }


def _build_sample_schema_contract(
    *,
    samples: Sequence[E0ExperimentSample],
    expected_schema: Mapping[str, object],
) -> dict[str, object]:
    physiology_summary = _summarize_sample_stream_schema(samples, "physiology")
    vehicle_summary = _summarize_sample_stream_schema(samples, "vehicle")
    physiology_expected = tuple(expected_schema["streams"]["physiology"]["feature_names"])
    vehicle_expected = tuple(expected_schema["streams"]["vehicle"]["feature_names"])
    physiology_comparison = _compare_feature_names(
        expected_feature_names=physiology_expected,
        observed_feature_names=tuple(physiology_summary["feature_names"]),
    )
    vehicle_comparison = _compare_feature_names(
        expected_feature_names=vehicle_expected,
        observed_feature_names=tuple(vehicle_summary["feature_names"]),
    )
    if physiology_comparison["status"] == "exact" and vehicle_comparison["status"] == "exact":
        status = "exact"
    else:
        status = "aligned"
    return {
        "status": status,
        "sample_count": len(samples),
        "physiology": physiology_summary,
        "vehicle": vehicle_summary,
        "comparison": {
            "status": status,
            "physiology": physiology_comparison,
            "vehicle": vehicle_comparison,
        },
    }


def _summarize_sample_stream_schema(
    samples: Sequence[E0ExperimentSample],
    stream_name: str,
) -> dict[str, object]:
    feature_sets = [
        tuple(getattr(sample, stream_name).feature_names)
        for sample in samples
    ]
    if not feature_sets:
        return {
            "feature_names": [],
            "feature_count": 0,
            "measurement_group_counts": {},
            "schema_variant_count": 0,
            "dominant_schema_coverage": 0,
        }
    grouped: dict[tuple[str, ...], int] = {}
    for feature_names in feature_sets:
        grouped[feature_names] = grouped.get(feature_names, 0) + 1
    dominant_feature_names = max(grouped.items(), key=lambda item: item[1])[0]
    return {
        "feature_names": list(dominant_feature_names),
        "feature_count": len(dominant_feature_names),
        "measurement_group_counts": _count_measurement_groups(dominant_feature_names),
        "schema_variant_count": len(grouped),
        "dominant_schema_coverage": grouped[dominant_feature_names],
    }


def _compare_feature_names(
    *,
    expected_feature_names: tuple[str, ...],
    observed_feature_names: tuple[str, ...],
) -> dict[str, object]:
    missing = [name for name in expected_feature_names if name not in set(observed_feature_names)]
    extra = [name for name in observed_feature_names if name not in set(expected_feature_names)]
    status = "exact" if not missing and not extra and observed_feature_names == expected_feature_names else "aligned"
    return {
        "status": status,
        "expected_feature_count": len(expected_feature_names),
        "observed_feature_count": len(observed_feature_names),
        "missing_feature_count": len(missing),
        "extra_feature_count": len(extra),
        "missing_feature_names_preview": missing[:20],
        "extra_feature_names_preview": extra[:20],
        "missing_measurement_group_counts": _count_measurement_groups(missing),
        "extra_measurement_group_counts": _count_measurement_groups(extra),
    }


def _build_schema_hash(
    physiology_feature_names: Sequence[str],
    vehicle_feature_names: Sequence[str],
) -> str:
    payload = json.dumps(
        {
            "physiology": list(physiology_feature_names),
            "vehicle": list(vehicle_feature_names),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stream_schema_payload(feature_names: Sequence[str]) -> dict[str, object]:
    return {
        "feature_names": list(feature_names),
        "feature_count": len(tuple(feature_names)),
        "measurement_group_counts": _count_measurement_groups(feature_names),
    }


def _count_measurement_groups(feature_names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in feature_names:
        group = _feature_group_name(name)
        counts[group] = counts.get(group, 0) + 1
    return counts


def _feature_group_name(feature_name: str) -> str:
    if ".code" in feature_name:
        return feature_name.split(".code", 1)[0]
    if "." in feature_name:
        return feature_name.split(".", 1)[0]
    return feature_name


def _align_stream_to_schema(
    stream: NumericStreamMatrix,
    expected_feature_names: Sequence[str],
) -> NumericStreamMatrix:
    expected_tuple = tuple(expected_feature_names)
    if not expected_tuple:
        return stream
    feature_index = {name: index for index, name in enumerate(stream.feature_names)}
    aligned_rows = []
    for row in stream.values:
        aligned_rows.append(
            tuple(float("nan") if name not in feature_index else row[feature_index[name]] for name in expected_tuple)
        )
    return NumericStreamMatrix(
        stream_kind=stream.stream_kind,
        point_count=stream.point_count,
        feature_names=expected_tuple,
        point_offsets_ms=stream.point_offsets_ms,
        point_measurements=stream.point_measurements,
        values=tuple(aligned_rows),
        dropped_fields=stream.dropped_fields,
    )
