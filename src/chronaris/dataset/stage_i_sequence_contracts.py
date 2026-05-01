"""Contracts and loaders for Stage I sequence-based deep baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class StageISequenceEntry:
    """One sequence sample prepared for Stage I deep baselines."""

    sample_id: str
    dataset_id: str
    subset_id: str
    subject_id: str
    session_id: str
    split_group: str
    training_role: str
    sequence_bundle_path: str
    sequence_length: int
    modality_schema: Mapping[str, object]
    source_origin: str
    sample_granularity: str = "sequence"
    recording_id: str | None = None
    window_index: int | None = None
    window_duration_s: float | None = None
    task_family: str | None = None
    label_namespace: str | None = None
    objective_label_name: str | None = None
    objective_label_value: float | int | None = None
    subjective_target_name: str | None = None
    subjective_target_value: float | None = None
    window_start_utc: str | None = None
    window_end_utc: str | None = None
    context_payload: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "subset_id": self.subset_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "recording_id": self.recording_id or self.session_id,
            "split_group": self.split_group,
            "training_role": self.training_role,
            "sample_granularity": self.sample_granularity,
            "window_index": self.window_index,
            "window_duration_s": self.window_duration_s,
            "task_family": self.task_family,
            "label_namespace": self.label_namespace,
            "objective_label_name": self.objective_label_name,
            "objective_label_value": self.objective_label_value,
            "subjective_target_name": self.subjective_target_name,
            "subjective_target_value": self.subjective_target_value,
            "window_start_utc": self.window_start_utc,
            "window_end_utc": self.window_end_utc,
            "sequence_bundle_path": self.sequence_bundle_path,
            "sequence_length": self.sequence_length,
            "modality_schema": dict(self.modality_schema),
            "source_origin": self.source_origin,
            "context_payload": dict(self.context_payload),
        }


@dataclass(frozen=True, slots=True)
class StageISequenceDatasetSummary:
    """Compact machine-readable summary for one prepared sequence dataset."""

    dataset_id: str
    profile: str
    generated_at_utc: str
    entry_count: int
    recording_count: int
    split_group_count: int
    sequence_length: int
    subset_counts: Mapping[str, int]
    training_role_counts: Mapping[str, int]
    task_family_counts: Mapping[str, int]
    label_distribution: Mapping[str, Mapping[str, int]]
    modality_feature_counts: Mapping[str, int]
    source_origin_counts: Mapping[str, int]
    extra_summary: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "profile": self.profile,
            "generated_at_utc": self.generated_at_utc,
            "entry_count": self.entry_count,
            "recording_count": self.recording_count,
            "split_group_count": self.split_group_count,
            "sequence_length": self.sequence_length,
            "subset_counts": dict(self.subset_counts),
            "training_role_counts": dict(self.training_role_counts),
            "task_family_counts": dict(self.task_family_counts),
            "label_distribution": {
                key: dict(value)
                for key, value in self.label_distribution.items()
            },
            "modality_feature_counts": dict(self.modality_feature_counts),
            "source_origin_counts": dict(self.source_origin_counts),
            "extra_summary": dict(self.extra_summary),
        }


@dataclass(frozen=True, slots=True)
class StageISequenceBundle:
    """Tensor bundle stored in `sequence_bundle.npz`."""

    sample_ids: tuple[str, ...]
    time_axis: np.ndarray
    modality_arrays: Mapping[str, np.ndarray]
    modality_masks: Mapping[str, np.ndarray]
    objective_label_values: np.ndarray
    objective_label_mask: np.ndarray
    subjective_target_values: np.ndarray
    subjective_target_mask: np.ndarray
    metadata_json: tuple[str, ...]
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def entry_count(self) -> int:
        return len(self.sample_ids)


def dump_stage_i_sequence_entries(
    entries: Sequence[StageISequenceEntry],
    *,
    path: str | Path,
) -> None:
    Path(path).write_text(
        "".join(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def load_stage_i_sequence_entries(path: str | Path) -> tuple[StageISequenceEntry, ...]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries: list[StageISequenceEntry] = []
    for row in rows:
        _validate_sequence_entry_row(row)
        entries.append(
            StageISequenceEntry(
                sample_id=str(row["sample_id"]),
                dataset_id=str(row["dataset_id"]),
                subset_id=str(row["subset_id"]),
                subject_id=str(row["subject_id"]),
                session_id=str(row["session_id"]),
                recording_id=str(row.get("recording_id") or row["session_id"]),
                split_group=str(row["split_group"]),
                training_role=str(row["training_role"]),
                sequence_bundle_path=str(row["sequence_bundle_path"]),
                sequence_length=int(row["sequence_length"]),
                modality_schema=dict(row["modality_schema"]),
                source_origin=str(row["source_origin"]),
                sample_granularity=str(row.get("sample_granularity") or "sequence"),
                window_index=_coerce_int(row.get("window_index")),
                window_duration_s=_coerce_float(row.get("window_duration_s")),
                task_family=_optional_str(row.get("task_family")),
                label_namespace=_optional_str(row.get("label_namespace")),
                objective_label_name=_optional_str(row.get("objective_label_name")),
                objective_label_value=row.get("objective_label_value"),
                subjective_target_name=_optional_str(row.get("subjective_target_name")),
                subjective_target_value=_coerce_float(row.get("subjective_target_value")),
                window_start_utc=_optional_str(row.get("window_start_utc")),
                window_end_utc=_optional_str(row.get("window_end_utc")),
                context_payload=dict(row.get("context_payload", {})),
            )
        )
    return tuple(entries)


def dump_stage_i_sequence_summary(
    summary: StageISequenceDatasetSummary,
    *,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_stage_i_sequence_summary(path: str | Path) -> StageISequenceDatasetSummary:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    return StageISequenceDatasetSummary(
        dataset_id=str(row["dataset_id"]),
        profile=str(row["profile"]),
        generated_at_utc=str(row["generated_at_utc"]),
        entry_count=int(row["entry_count"]),
        recording_count=int(row["recording_count"]),
        split_group_count=int(row["split_group_count"]),
        sequence_length=int(row["sequence_length"]),
        subset_counts=dict(row["subset_counts"]),
        training_role_counts=dict(row["training_role_counts"]),
        task_family_counts=dict(row["task_family_counts"]),
        label_distribution={
            str(key): dict(value)
            for key, value in dict(row["label_distribution"]).items()
        },
        modality_feature_counts=dict(row["modality_feature_counts"]),
        source_origin_counts=dict(row["source_origin_counts"]),
        extra_summary=dict(row.get("extra_summary", {})),
    )


def save_stage_i_sequence_bundle(
    bundle: StageISequenceBundle,
    *,
    path: str | Path,
) -> None:
    payload: dict[str, object] = {
        "sample_ids": np.asarray(bundle.sample_ids, dtype=str),
        "time_axis": np.asarray(bundle.time_axis, dtype=np.float32),
        "objective_label_values": np.asarray(
            bundle.objective_label_values,
            dtype=np.float32,
        ),
        "objective_label_mask": np.asarray(
            bundle.objective_label_mask,
            dtype=np.uint8,
        ),
        "subjective_target_values": np.asarray(
            bundle.subjective_target_values,
            dtype=np.float32,
        ),
        "subjective_target_mask": np.asarray(
            bundle.subjective_target_mask,
            dtype=np.uint8,
        ),
        "metadata_json": np.asarray(bundle.metadata_json, dtype=str),
    }
    modality_names = tuple(sorted(bundle.modality_arrays))
    payload["modality_names"] = np.asarray(modality_names, dtype=str)
    payload["extra_names"] = np.asarray(tuple(sorted(bundle.extras)), dtype=str)
    for modality_name in modality_names:
        payload[f"modality__{modality_name}__values"] = np.asarray(
            bundle.modality_arrays[modality_name],
            dtype=np.float32,
        )
        payload[f"modality__{modality_name}__mask"] = np.asarray(
            bundle.modality_masks[modality_name],
            dtype=np.uint8,
        )
    for extra_name, value in bundle.extras.items():
        payload[f"extra__{extra_name}"] = np.asarray(value)
    np.savez_compressed(Path(path), **payload)


def load_stage_i_sequence_bundle(path: str | Path) -> StageISequenceBundle:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "sample_ids",
            "time_axis",
            "objective_label_values",
            "objective_label_mask",
            "subjective_target_values",
            "subjective_target_mask",
            "metadata_json",
            "modality_names",
            "extra_names",
        }
        missing = tuple(sorted(required.difference(payload.files)))
        if missing:
            raise ValueError(
                "Stage I sequence bundle is missing keys: "
                + ", ".join(missing)
            )
        sample_ids = tuple(str(value) for value in payload["sample_ids"].tolist())
        metadata_json = tuple(str(value) for value in payload["metadata_json"].tolist())
        time_axis = np.asarray(payload["time_axis"], dtype=np.float32)
        if time_axis.ndim != 2:
            raise ValueError("sequence bundle time_axis must have shape [N, T].")
        _validate_sample_axis(
            sample_ids=sample_ids,
            metadata_json=metadata_json,
            arrays={
                "time_axis": time_axis,
                "objective_label_values": payload["objective_label_values"],
                "objective_label_mask": payload["objective_label_mask"],
                "subjective_target_values": payload["subjective_target_values"],
                "subjective_target_mask": payload["subjective_target_mask"],
            },
        )
        modality_arrays: dict[str, np.ndarray] = {}
        modality_masks: dict[str, np.ndarray] = {}
        modality_names = tuple(str(value) for value in payload["modality_names"].tolist())
        for modality_name in modality_names:
            values_key = f"modality__{modality_name}__values"
            mask_key = f"modality__{modality_name}__mask"
            if values_key not in payload.files or mask_key not in payload.files:
                raise ValueError(
                    f"sequence bundle is missing modality payload for '{modality_name}'."
                )
            values = np.asarray(payload[values_key], dtype=np.float32)
            mask = np.asarray(payload[mask_key], dtype=np.uint8)
            if values.ndim != 3:
                raise ValueError(
                    f"modality '{modality_name}' values must have shape [N, T, D]."
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"modality '{modality_name}' mask must have shape [N, T]."
                )
            if values.shape[:2] != mask.shape:
                raise ValueError(
                    f"modality '{modality_name}' values/mask shapes do not align."
                )
            if values.shape[:2] != time_axis.shape:
                raise ValueError(
                    f"modality '{modality_name}' does not align with time_axis."
                )
            modality_arrays[modality_name] = values
            modality_masks[modality_name] = mask
        extras: dict[str, np.ndarray] = {}
        extra_names = tuple(str(value) for value in payload["extra_names"].tolist())
        for extra_name in extra_names:
            key = f"extra__{extra_name}"
            if key not in payload.files:
                raise ValueError(f"sequence bundle missing extra payload '{extra_name}'.")
            value = np.asarray(payload[key])
            if value.shape and value.shape[0] == len(sample_ids):
                _validate_sample_axis(
                    sample_ids=sample_ids,
                    metadata_json=metadata_json,
                    arrays={key: value},
                )
            extras[extra_name] = value
        return StageISequenceBundle(
            sample_ids=sample_ids,
            time_axis=time_axis,
            modality_arrays=modality_arrays,
            modality_masks=modality_masks,
            objective_label_values=np.asarray(
                payload["objective_label_values"],
                dtype=np.float32,
            ),
            objective_label_mask=np.asarray(
                payload["objective_label_mask"],
                dtype=np.uint8,
            ),
            subjective_target_values=np.asarray(
                payload["subjective_target_values"],
                dtype=np.float32,
            ),
            subjective_target_mask=np.asarray(
                payload["subjective_target_mask"],
                dtype=np.uint8,
            ),
            metadata_json=metadata_json,
            extras=extras,
        )


def _validate_sequence_entry_row(row: Mapping[str, object]) -> None:
    required = (
        "sample_id",
        "dataset_id",
        "subset_id",
        "subject_id",
        "session_id",
        "split_group",
        "training_role",
        "sequence_bundle_path",
        "sequence_length",
        "modality_schema",
        "source_origin",
    )
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(
            "Stage I sequence entry is missing required fields: "
            + ", ".join(missing)
        )


def _validate_sample_axis(
    *,
    sample_ids: Sequence[str],
    metadata_json: Sequence[str],
    arrays: Mapping[str, object],
) -> None:
    entry_count = len(sample_ids)
    if len(metadata_json) != entry_count:
        raise ValueError("sequence bundle metadata_json length does not match sample_ids.")
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] != entry_count:
            raise ValueError(
                f"sequence bundle payload '{key}' has inconsistent sample axis."
            )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
