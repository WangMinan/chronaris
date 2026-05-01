"""Sequence export for Stage I deep baselines."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset import (
    StageISequenceBundle,
    StageISequenceDatasetSummary,
    StageISequenceEntry,
    StageITaskEntry,
    build_nasa_csm_task_entries,
    build_uab_task_entries,
)
from chronaris.dataset.nasa_csm_stage_i import DATASET_ID as NASA_DATASET_ID
from chronaris.dataset.nasa_csm_stage_i import (
    WINDOW_DURATION_SECONDS as NASA_WINDOW_DURATION_SECONDS,
)
from chronaris.dataset.uab_stage_i import DATASET_ID as UAB_DATASET_ID
from chronaris.dataset.uab_stage_i import (
    WINDOW_DURATION_SECONDS as UAB_WINDOW_DURATION_SECONDS,
)
from chronaris.features.stage_i_case import load_stage_i_case_study_run
from chronaris.features.stage_i_feature_helpers import (
    build_window_specs,
    entry_iso_window_bounds,
    group_entries_by_csv_path,
)
from chronaris.features.stage_i_sequence_helpers import (
    build_common_time_axis,
    infer_csv_value_columns,
    infer_parquet_value_columns,
    stream_window_sequence_arrays_from_csv,
    stream_window_sequence_arrays_from_parquet,
)

STAGE_H_CASE_DATASET_ID = "stage_h_case"
REAL_SORTIE_V1 = "real_sortie_v1"
WINDOW_V2 = "window_v2"
DEFAULT_SEQUENCE_STEPS = 64


@dataclass(frozen=True, slots=True)
class StageISequencePreparationPayload:
    """In-memory payload produced by one sequence preparation pass."""

    entries: tuple[StageISequenceEntry, ...]
    bundle: StageISequenceBundle
    summary: StageISequenceDatasetSummary
    sequence_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ParquetSourceSpec:
    relative_path: str
    modality_name: str
    key_columns: tuple[str, ...]
    recording_id_builder: Callable[[tuple[object, ...]], str]
    selector: Callable[[tuple[str, ...]], tuple[str, ...]]
    invalid_nonpositive_columns: tuple[str, ...] = ()


def prepare_stage_h_case_sequences(
    run_manifest_path: str | Path,
    *,
    profile: str = REAL_SORTIE_V1,
) -> StageISequencePreparationPayload:
    run_input = load_stage_i_case_study_run(run_manifest_path)
    entries: list[StageISequenceEntry] = []
    sample_ids: list[str] = []
    time_axis: list[np.ndarray] = []
    physiology_values: list[np.ndarray] = []
    vehicle_values: list[np.ndarray] = []
    physiology_masks: list[np.ndarray] = []
    vehicle_masks: list[np.ndarray] = []
    objective_values: list[float] = []
    objective_mask: list[int] = []
    subjective_values: list[float] = []
    subjective_mask: list[int] = []
    metadata_json: list[str] = []
    event_scores: list[np.ndarray] = []
    attention_weights: list[np.ndarray] = []
    view_ids: list[str] = []
    sortie_ids: list[str] = []
    pilot_ids: list[int] = []
    verdict_codes: list[int] = []
    view_window_counts: dict[str, int] = {}

    modality_schema = _build_bimodal_schema(
        first_name="physiology",
        first_features=[f"projection_{index:02d}" for index in range(16)],
        second_name="vehicle",
        second_features=[f"projection_{index:02d}" for index in range(16)],
        notes={
            "physiology": "Stage H physiology_reference_projection",
            "vehicle": "Stage H vehicle_reference_projection",
        },
    )

    for view in run_input.views:
        view_window_counts[view.view_id] = view.window_count
        case_rows = {row.sample_id: row for row in view.case_window_rows}
        for sample_index, raw_sample_id in enumerate(view.sample_ids):
            sample_id = f"{view.view_id}::{raw_sample_id}"
            sample_ids.append(sample_id)
            time_values = view.stage_h_view.reference_offsets_s[sample_index].astype(
                np.float32,
            )
            physiology = view.stage_h_view.physiology_reference_projection[
                sample_index
            ].astype(np.float32)
            vehicle = view.stage_h_view.vehicle_reference_projection[
                sample_index
            ].astype(np.float32)
            time_axis.append(time_values)
            physiology_values.append(physiology)
            vehicle_values.append(vehicle)
            physiology_masks.append(
                np.isfinite(physiology).any(axis=1).astype(np.uint8),
            )
            vehicle_masks.append(np.isfinite(vehicle).any(axis=1).astype(np.uint8))
            verdict_value = (
                1.0 if view.projection_diagnostics_verdict == "PASS" else 0.0
            )
            objective_values.append(verdict_value)
            objective_mask.append(1)
            subjective_values.append(np.nan)
            subjective_mask.append(0)
            row = case_rows.get(raw_sample_id)
            payload = {
                "view_id": view.view_id,
                "sortie_id": view.sortie_id,
                "pilot_id": view.pilot_id,
                "projection_diagnostics_verdict": view.projection_diagnostics_verdict,
                "window_count": view.window_count,
                "selected_window_count": view.selected_window_count,
                "case_partition_sample_count": view.case_partition_sample_count,
                "sample_order": sample_index,
                "raw_sample_id": raw_sample_id,
                "run_manifest_path": run_input.run_manifest_path,
                "window_manifest": row.to_dict() if row is not None else None,
                "artifact_paths": dict(
                    view.stage_h_view.view_manifest.get("artifact_paths", {}),
                ),
            }
            metadata_json.append(json.dumps(payload, ensure_ascii=False))
            event_scores.append(
                view.stage_h_view.vehicle_event_scores[sample_index].astype(np.float32),
            )
            attention_weights.append(
                view.stage_h_view.attention_weights[sample_index].astype(np.float32),
            )
            view_ids.append(view.view_id)
            sortie_ids.append(view.sortie_id)
            pilot_ids.append(view.pilot_id)
            verdict_codes.append(int(verdict_value))
            entries.append(
                StageISequenceEntry(
                    sample_id=sample_id,
                    dataset_id=STAGE_H_CASE_DATASET_ID,
                    subset_id="case_partition",
                    subject_id=f"pilot_{view.pilot_id}",
                    session_id=view.view_id,
                    recording_id=view.view_id,
                    split_group=view.view_id,
                    training_role="case_study",
                    sequence_bundle_path="sequence_bundle.npz",
                    sequence_length=int(physiology.shape[0]),
                    modality_schema=modality_schema,
                    source_origin="stage_h_closure_validation",
                    window_index=row.window_index if row is not None else sample_index,
                    window_duration_s=5.0,
                    task_family="real_sortie_case_study",
                    label_namespace="projection_diagnostics_verdict",
                    objective_label_name="projection_diagnostics_verdict_code",
                    objective_label_value=verdict_value,
                    context_payload=payload,
                )
            )

    bundle = StageISequenceBundle(
        sample_ids=tuple(sample_ids),
        time_axis=np.stack(time_axis, axis=0),
        modality_arrays={
            "physiology": np.stack(physiology_values, axis=0),
            "vehicle": np.stack(vehicle_values, axis=0),
        },
        modality_masks={
            "physiology": np.stack(physiology_masks, axis=0),
            "vehicle": np.stack(vehicle_masks, axis=0),
        },
        objective_label_values=np.asarray(objective_values, dtype=np.float32),
        objective_label_mask=np.asarray(objective_mask, dtype=np.uint8),
        subjective_target_values=np.asarray(subjective_values, dtype=np.float32),
        subjective_target_mask=np.asarray(subjective_mask, dtype=np.uint8),
        metadata_json=tuple(metadata_json),
        extras={
            "vehicle_event_scores": np.stack(event_scores, axis=0),
            "attention_weights": np.stack(attention_weights, axis=0),
            "view_ids": np.asarray(view_ids, dtype=str),
            "sortie_ids": np.asarray(sortie_ids, dtype=str),
            "pilot_ids": np.asarray(pilot_ids, dtype=np.int32),
            "verdict_codes": np.asarray(verdict_codes, dtype=np.int32),
        },
    )
    summary = _build_summary(
        entries=entries,
        bundle=bundle,
        dataset_id=STAGE_H_CASE_DATASET_ID,
        profile=profile,
        modality_feature_counts={
            "physiology": 16,
            "vehicle": 16,
        },
        extra_summary={
            "view_count": len(run_input.views),
            "view_window_counts": view_window_counts,
            "projection_diagnostics_verdict_counts": dict(
                Counter(view.projection_diagnostics_verdict for view in run_input.views),
            ),
        },
    )
    return StageISequencePreparationPayload(
        entries=tuple(entries),
        bundle=bundle,
        summary=summary,
        sequence_schema={
            "dataset_id": STAGE_H_CASE_DATASET_ID,
            "profile": profile,
            "sequence_length": int(bundle.time_axis.shape[1]),
            "modalities": modality_schema,
            "extras": {
                "vehicle_event_scores": list(
                    bundle.extras["vehicle_event_scores"].shape,
                ),
                "attention_weights": list(bundle.extras["attention_weights"].shape),
            },
        },
    )


def prepare_uab_sequences(
    dataset_root: str | Path,
    *,
    profile: str = WINDOW_V2,
    target_steps: int = DEFAULT_SEQUENCE_STEPS,
) -> StageISequencePreparationPayload:
    prepared = build_uab_task_entries(dataset_root, profile=profile)
    entries = tuple(
        entry for entry in prepared.entries if entry.sample_granularity == "window"
    )
    if not entries:
        raise ValueError("UAB sequence export requires window_v2 entries.")
    root = Path(dataset_root) / UAB_DATASET_ID
    sample_id_to_index = {entry.sample_id: index for index, entry in enumerate(entries)}
    modality_specs = _uab_parquet_specs()
    eeg_feature_names = _union_parquet_feature_names(
        root=root,
        specs=tuple(spec for spec in modality_specs if spec.modality_name == "eeg"),
    )
    ecg_feature_names = _union_parquet_feature_names(
        root=root,
        specs=tuple(spec for spec in modality_specs if spec.modality_name == "ecg"),
    )
    eeg_values = np.zeros(
        (len(entries), target_steps, len(eeg_feature_names)),
        dtype=np.float32,
    )
    ecg_values = np.zeros(
        (len(entries), target_steps, len(ecg_feature_names)),
        dtype=np.float32,
    )
    eeg_masks = np.zeros((len(entries), target_steps), dtype=np.uint8)
    ecg_masks = np.zeros((len(entries), target_steps), dtype=np.uint8)
    feature_position_maps = {
        "eeg": {
            feature_name: feature_index
            for feature_index, feature_name in enumerate(eeg_feature_names)
        },
        "ecg": {
            feature_name: feature_index
            for feature_index, feature_name in enumerate(ecg_feature_names)
        },
    }
    window_specs = build_window_specs(entries, time_range_builder=entry_iso_window_bounds)
    for spec in modality_specs:
        parquet_path = root / spec.relative_path
        feature_names = infer_parquet_value_columns(
            parquet_path,
            selector=spec.selector,
        )
        if not feature_names:
            continue
        results = stream_window_sequence_arrays_from_parquet(
            parquet_path=parquet_path,
            key_columns=spec.key_columns,
            recording_id_builder=spec.recording_id_builder,
            window_specs=window_specs,
            value_columns=feature_names,
            target_steps=target_steps,
            invalid_nonpositive_columns=spec.invalid_nonpositive_columns,
        )
        _merge_sequence_results(
            results=results,
            feature_names=feature_names,
            target_arrays=eeg_values if spec.modality_name == "eeg" else ecg_values,
            target_masks=eeg_masks if spec.modality_name == "eeg" else ecg_masks,
            sample_id_to_index=sample_id_to_index,
            feature_positions=feature_position_maps[spec.modality_name],
        )

    modality_schema = _build_bimodal_schema(
        first_name="eeg",
        first_features=eeg_feature_names,
        second_name="ecg",
        second_features=ecg_feature_names,
    )
    sequence_entries = tuple(
        _sequence_entry_from_task_entry(
            entry,
            bundle_path="sequence_bundle.npz",
            sequence_length=target_steps,
            source_origin="uab_window_v2",
            modality_schema=modality_schema,
        )
        for entry in entries
    )
    metadata_json = tuple(
        json.dumps(
            {
                "sample_id": entry.sample_id,
                "recording_id": entry.recording_id,
                "source_refs": dict(entry.source_refs),
                "context_payload": dict(entry.context_payload),
            },
            ensure_ascii=False,
        )
        for entry in entries
    )
    bundle = StageISequenceBundle(
        sample_ids=tuple(entry.sample_id for entry in entries),
        time_axis=build_common_time_axis(
            entry_count=len(entries),
            target_steps=target_steps,
            window_duration_s=UAB_WINDOW_DURATION_SECONDS,
        ),
        modality_arrays={
            "eeg": eeg_values,
            "ecg": ecg_values,
        },
        modality_masks={
            "eeg": eeg_masks,
            "ecg": ecg_masks,
        },
        objective_label_values=np.asarray(
            [
                float(entry.objective_label_value)
                if entry.objective_label_value is not None
                else np.nan
                for entry in entries
            ],
            dtype=np.float32,
        ),
        objective_label_mask=np.asarray(
            [1 if entry.objective_label_value is not None else 0 for entry in entries],
            dtype=np.uint8,
        ),
        subjective_target_values=np.asarray(
            [
                float(entry.subjective_target_value)
                if entry.subjective_target_value is not None
                else np.nan
                for entry in entries
            ],
            dtype=np.float32,
        ),
        subjective_target_mask=np.asarray(
            [1 if entry.subjective_target_value is not None else 0 for entry in entries],
            dtype=np.uint8,
        ),
        metadata_json=metadata_json,
    )
    summary = _build_summary(
        entries=sequence_entries,
        bundle=bundle,
        dataset_id=UAB_DATASET_ID,
        profile=profile,
        modality_feature_counts={
            "eeg": len(eeg_feature_names),
            "ecg": len(ecg_feature_names),
        },
        extra_summary={
            "prepared_subset_counts": dict(prepared.subset_counts),
            "ecg_zero_mask_samples": _count_zero_mask_samples(
                entries=entries,
                mask=ecg_masks,
            ),
        },
    )
    return StageISequencePreparationPayload(
        entries=sequence_entries,
        bundle=bundle,
        summary=summary,
        sequence_schema={
            "dataset_id": UAB_DATASET_ID,
            "profile": profile,
            "sequence_length": target_steps,
            "modalities": modality_schema,
        },
    )


def prepare_nasa_sequences(
    dataset_root: str | Path,
    *,
    profile: str = WINDOW_V2,
    target_steps: int = DEFAULT_SEQUENCE_STEPS,
) -> StageISequencePreparationPayload:
    prepared = build_nasa_csm_task_entries(dataset_root)
    entries = tuple(prepared.entries)
    if not entries:
        raise ValueError("NASA CSM sequence export requires local extracted CSV files.")
    root = Path(dataset_root)
    sample_id_to_index = {entry.sample_id: index for index, entry in enumerate(entries)}
    grouped = group_entries_by_csv_path(entries)
    csv_headers = {
        relative_path: infer_csv_value_columns(
            root / relative_path,
            selector=lambda columns: columns,
        )
        for relative_path in grouped
    }
    eeg_feature_names = tuple(
        sorted(
            {
                column
                for columns in csv_headers.values()
                for column in columns
                if column.startswith("EEG_")
            },
        ),
    )
    peripheral_feature_names = tuple(
        column
        for column in ("ECG", "R", "GSR")
        if any(column in columns for columns in csv_headers.values())
    )
    eeg_values = np.zeros(
        (len(entries), target_steps, len(eeg_feature_names)),
        dtype=np.float32,
    )
    peripheral_values = np.zeros(
        (len(entries), target_steps, len(peripheral_feature_names)),
        dtype=np.float32,
    )
    eeg_masks = np.zeros((len(entries), target_steps), dtype=np.uint8)
    peripheral_masks = np.zeros((len(entries), target_steps), dtype=np.uint8)
    eeg_feature_positions = {
        feature_name: index
        for index, feature_name in enumerate(eeg_feature_names)
    }
    peripheral_feature_positions = {
        feature_name: index
        for index, feature_name in enumerate(peripheral_feature_names)
    }
    for relative_path, group_entries in sorted(grouped.items()):
        csv_path = root / relative_path
        recording_id = group_entries[0].recording_id or group_entries[0].session_id
        window_specs = {
            recording_id: build_window_specs(
                group_entries,
                time_range_builder=lambda entry: (
                    float(entry.context_payload["window_start_s"]),
                    float(entry.context_payload["window_end_s"]),
                ),
            )[recording_id]
        }
        eeg_columns = tuple(
            column
            for column in csv_headers[relative_path]
            if column.startswith("EEG_")
        )
        if eeg_columns:
            eeg_results = stream_window_sequence_arrays_from_csv(
                csv_path=csv_path,
                recording_id=recording_id,
                window_specs=window_specs,
                value_columns=eeg_columns,
                target_steps=target_steps,
                invalid_nonpositive_columns=(),
            )
            _merge_sequence_results(
                results=eeg_results,
                feature_names=eeg_columns,
                target_arrays=eeg_values,
                target_masks=eeg_masks,
                sample_id_to_index=sample_id_to_index,
                feature_positions=eeg_feature_positions,
            )
        peripheral_columns = tuple(
            column
            for column in peripheral_feature_names
            if column in csv_headers[relative_path]
        )
        if peripheral_columns:
            peripheral_results = stream_window_sequence_arrays_from_csv(
                csv_path=csv_path,
                recording_id=recording_id,
                window_specs=window_specs,
                value_columns=peripheral_columns,
                target_steps=target_steps,
                invalid_nonpositive_columns=peripheral_columns,
            )
            _merge_sequence_results(
                results=peripheral_results,
                feature_names=peripheral_columns,
                target_arrays=peripheral_values,
                target_masks=peripheral_masks,
                sample_id_to_index=sample_id_to_index,
                feature_positions=peripheral_feature_positions,
            )

    modality_schema = _build_bimodal_schema(
        first_name="eeg",
        first_features=eeg_feature_names,
        second_name="peripheral",
        second_features=peripheral_feature_names,
    )
    sequence_entries = tuple(
        _sequence_entry_from_task_entry(
            entry,
            bundle_path="sequence_bundle.npz",
            sequence_length=target_steps,
            source_origin="nasa_attention_state_window_v2",
            modality_schema=modality_schema,
        )
        for entry in entries
    )
    metadata_json = tuple(
        json.dumps(
            {
                "sample_id": entry.sample_id,
                "recording_id": entry.recording_id,
                "source_refs": dict(entry.source_refs),
                "context_payload": dict(entry.context_payload),
            },
            ensure_ascii=False,
        )
        for entry in entries
    )
    bundle = StageISequenceBundle(
        sample_ids=tuple(entry.sample_id for entry in entries),
        time_axis=build_common_time_axis(
            entry_count=len(entries),
            target_steps=target_steps,
            window_duration_s=NASA_WINDOW_DURATION_SECONDS,
        ),
        modality_arrays={
            "eeg": eeg_values,
            "peripheral": peripheral_values,
        },
        modality_masks={
            "eeg": eeg_masks,
            "peripheral": peripheral_masks,
        },
        objective_label_values=np.asarray(
            [float(entry.objective_label_value) for entry in entries],
            dtype=np.float32,
        ),
        objective_label_mask=np.asarray(
            [1 if entry.objective_label_value is not None else 0 for entry in entries],
            dtype=np.uint8,
        ),
        subjective_target_values=np.full((len(entries),), np.nan, dtype=np.float32),
        subjective_target_mask=np.zeros((len(entries),), dtype=np.uint8),
        metadata_json=metadata_json,
    )
    summary = _build_summary(
        entries=sequence_entries,
        bundle=bundle,
        dataset_id=NASA_DATASET_ID,
        profile=profile,
        modality_feature_counts={
            "eeg": len(eeg_feature_names),
            "peripheral": len(peripheral_feature_names),
        },
        extra_summary={
            "prepared_subset_counts": dict(prepared.subset_counts),
            "inventory_only_background_count": sum(
                1 for entry in entries if entry.training_role == "inventory_only"
            ),
        },
    )
    return StageISequencePreparationPayload(
        entries=sequence_entries,
        bundle=bundle,
        summary=summary,
        sequence_schema={
            "dataset_id": NASA_DATASET_ID,
            "profile": profile,
            "sequence_length": target_steps,
            "modalities": modality_schema,
        },
    )


def _sequence_entry_from_task_entry(
    entry: StageITaskEntry,
    *,
    bundle_path: str,
    sequence_length: int,
    source_origin: str,
    modality_schema: Mapping[str, object],
) -> StageISequenceEntry:
    return StageISequenceEntry(
        sample_id=entry.sample_id,
        dataset_id=entry.dataset_id,
        subset_id=entry.subset_id,
        subject_id=entry.subject_id,
        session_id=entry.session_id,
        recording_id=entry.recording_id,
        split_group=entry.split_group,
        training_role=entry.training_role,
        sequence_bundle_path=bundle_path,
        sequence_length=sequence_length,
        modality_schema=modality_schema,
        source_origin=source_origin,
        sample_granularity="sequence",
        window_index=entry.window_index,
        window_duration_s=entry.window_duration_s,
        task_family=entry.task_family,
        label_namespace=entry.label_namespace,
        objective_label_name=entry.objective_label_name,
        objective_label_value=entry.objective_label_value,
        subjective_target_name=entry.subjective_target_name,
        subjective_target_value=entry.subjective_target_value,
        window_start_utc=entry.window_start_utc,
        window_end_utc=entry.window_end_utc,
        context_payload=dict(entry.context_payload),
    )


def _build_summary(
    *,
    entries: Sequence[StageISequenceEntry],
    bundle: StageISequenceBundle,
    dataset_id: str,
    profile: str,
    modality_feature_counts: Mapping[str, int],
    extra_summary: Mapping[str, object] | None = None,
) -> StageISequenceDatasetSummary:
    subset_counts = Counter(entry.subset_id for entry in entries)
    training_role_counts = Counter(entry.training_role for entry in entries)
    task_family_counts = Counter(entry.task_family or "unknown" for entry in entries)
    label_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    source_origin_counts = Counter(entry.source_origin for entry in entries)
    for entry in entries:
        if entry.objective_label_name is not None and entry.objective_label_value is not None:
            label_distribution[entry.objective_label_name][
                str(entry.objective_label_value)
            ] += 1
    return StageISequenceDatasetSummary(
        dataset_id=dataset_id,
        profile=profile,
        generated_at_utc=pd.Timestamp.now("UTC").isoformat().replace(
            "+00:00",
            "Z",
        ),
        entry_count=len(entries),
        recording_count=len(
            {entry.recording_id or entry.session_id for entry in entries},
        ),
        split_group_count=len({entry.split_group for entry in entries}),
        sequence_length=int(bundle.time_axis.shape[1]),
        subset_counts=dict(subset_counts),
        training_role_counts=dict(training_role_counts),
        task_family_counts=dict(task_family_counts),
        label_distribution={
            label_name: dict(counter)
            for label_name, counter in label_distribution.items()
        },
        modality_feature_counts=dict(modality_feature_counts),
        source_origin_counts=dict(source_origin_counts),
        extra_summary=dict(extra_summary or {}),
    )


def _build_bimodal_schema(
    *,
    first_name: str,
    first_features: Sequence[str],
    second_name: str,
    second_features: Sequence[str],
    notes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    schema = {
        first_name: {
            "feature_dim": len(first_features),
            "feature_names": list(first_features),
            "time_axis_key": "time_axis",
        },
        second_name: {
            "feature_dim": len(second_features),
            "feature_names": list(second_features),
            "time_axis_key": "time_axis",
        },
    }
    if notes:
        for modality_name, note in notes.items():
            if modality_name in schema:
                schema[modality_name]["notes"] = note
    return schema


def _merge_sequence_results(
    *,
    results: Mapping[str, tuple[np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    target_arrays: np.ndarray,
    target_masks: np.ndarray,
    sample_id_to_index: Mapping[str, int],
    feature_positions: Mapping[str, int],
) -> None:
    for sample_id, (values, mask) in results.items():
        sample_index = sample_id_to_index.get(sample_id)
        if sample_index is None:
            continue
        for local_feature_index, feature_name in enumerate(feature_names):
            target_feature_index = feature_positions[feature_name]
            target_arrays[sample_index, :, target_feature_index] = values[
                :,
                local_feature_index,
            ]
        target_masks[sample_index] = np.maximum(target_masks[sample_index], mask)


def _union_parquet_feature_names(
    *,
    root: Path,
    specs: Sequence[_ParquetSourceSpec],
) -> tuple[str, ...]:
    feature_names: set[str] = set()
    for spec in specs:
        feature_names.update(
            infer_parquet_value_columns(root / spec.relative_path, selector=spec.selector),
        )
    return tuple(sorted(feature_names))


def _uab_parquet_specs() -> tuple[_ParquetSourceSpec, ...]:
    return (
        _ParquetSourceSpec(
            relative_path="data_n_back_test/eeg/eeg.parquet",
            modality_name="eeg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(
                column
                for column in columns
                if column.startswith("POW.") or column.startswith("PM.")
            ),
        ),
        _ParquetSourceSpec(
            relative_path="data_heat_the_chair/eeg/eeg.parquet",
            modality_name="eeg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(
                column
                for column in columns
                if column.startswith("POW.") or column.startswith("PM.")
            ),
        ),
        _ParquetSourceSpec(
            relative_path="data_flight_simulator/eeg/eeg.parquet",
            modality_name="eeg",
            key_columns=("subject", "flight"),
            recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
            selector=lambda columns: tuple(
                column
                for column in columns
                if column.startswith("POW.") or column.startswith("PM.")
            ),
        ),
        _ParquetSourceSpec(
            relative_path="data_n_back_test/ecg/ecg_hr.parquet",
            modality_name="ecg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(column for column in ("hr",) if column in columns),
            invalid_nonpositive_columns=("hr",),
        ),
        _ParquetSourceSpec(
            relative_path="data_n_back_test/ecg/ecg_ibi.parquet",
            modality_name="ecg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(
                column for column in ("rr_int",) if column in columns
            ),
            invalid_nonpositive_columns=("rr_int",),
        ),
        _ParquetSourceSpec(
            relative_path="data_n_back_test/ecg/ecg_br.parquet",
            modality_name="ecg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_n_back__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(column for column in ("br",) if column in columns),
            invalid_nonpositive_columns=("br",),
        ),
        _ParquetSourceSpec(
            relative_path="data_heat_the_chair/ecg/ecg.parquet",
            modality_name="ecg",
            key_columns=("subject", "test"),
            recording_id_builder=lambda key: f"uab_heat__{key[0]}__test_{int(key[1])}",
            selector=lambda columns: tuple(
                column for column in ("hr", "rr_int") if column in columns
            ),
            invalid_nonpositive_columns=("hr", "rr_int"),
        ),
        _ParquetSourceSpec(
            relative_path="data_flight_simulator/ecg/ecg_hr.parquet",
            modality_name="ecg",
            key_columns=("subject", "flight"),
            recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
            selector=lambda columns: tuple(column for column in ("hr",) if column in columns),
            invalid_nonpositive_columns=("hr",),
        ),
        _ParquetSourceSpec(
            relative_path="data_flight_simulator/ecg/ecg_ibi.parquet",
            modality_name="ecg",
            key_columns=("subject", "flight"),
            recording_id_builder=lambda key: f"uab_flight__subject_{key[0]}__flight_{int(key[1])}",
            selector=lambda columns: tuple(
                column for column in ("rr_int",) if column in columns
            ),
            invalid_nonpositive_columns=("rr_int",),
        ),
    )


def _count_zero_mask_samples(
    *,
    entries: Sequence[StageITaskEntry],
    mask: np.ndarray,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry, sample_mask in zip(entries, mask, strict=True):
        if not np.any(sample_mask):
            counts[entry.subset_id] += 1
    return dict(counts)
