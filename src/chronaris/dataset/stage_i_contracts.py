"""Contracts and JSON helpers for Stage I public-dataset tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StageITaskEntry:
    """One Stage I task sample backed by a public-dataset recording."""

    sample_id: str
    dataset_id: str
    subset_id: str
    subject_id: str
    session_id: str
    split_group: str
    training_role: str
    window_start_utc: str
    window_end_utc: str
    source_refs: Mapping[str, str]
    sample_granularity: str = "session"
    recording_id: str | None = None
    window_index: int | None = None
    window_duration_s: float | None = None
    task_family: str | None = None
    label_namespace: str | None = None
    objective_label_name: str | None = None
    objective_label_value: float | int | None = None
    subjective_target_name: str | None = None
    subjective_target_value: float | None = None
    context_payload: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "subset_id": self.subset_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "split_group": self.split_group,
            "training_role": self.training_role,
            "sample_granularity": self.sample_granularity,
            "recording_id": self.recording_id or self.session_id,
            "window_index": self.window_index,
            "window_duration_s": self.window_duration_s,
            "task_family": self.task_family,
            "label_namespace": self.label_namespace,
            "window_start_utc": self.window_start_utc,
            "window_end_utc": self.window_end_utc,
            "source_refs": dict(self.source_refs),
            "objective_label_name": self.objective_label_name,
            "objective_label_value": self.objective_label_value,
            "subjective_target_name": self.subjective_target_name,
            "subjective_target_value": self.subjective_target_value,
            "context_payload": dict(self.context_payload),
        }


@dataclass(frozen=True, slots=True)
class StageIDatasetSummary:
    """Compact machine-readable summary for one prepared Stage I dataset."""

    dataset_id: str
    generated_at_utc: str
    entry_count: int
    recording_count: int
    window_count: int
    sample_granularity_counts: Mapping[str, int]
    subset_counts: Mapping[str, int]
    subset_source_counts: Mapping[str, Mapping[str, int]]
    training_role_counts: Mapping[str, int]
    split_group_count: int
    task_family_counts: Mapping[str, int]
    label_distribution: Mapping[str, Mapping[str, int]]
    objective_label_distributions: Mapping[str, Mapping[str, int]]
    subjective_target_counts: Mapping[str, int]
    feature_count: int
    feature_group_counts: Mapping[str, int]
    eeg_feature_count: int
    ecg_feature_count: int
    peripheral_feature_count: int
    missing_ecg_session_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "generated_at_utc": self.generated_at_utc,
            "entry_count": self.entry_count,
            "recording_count": self.recording_count,
            "window_count": self.window_count,
            "sample_granularity_counts": dict(self.sample_granularity_counts),
            "subset_counts": dict(self.subset_counts),
            "subset_source_counts": {
                key: dict(value) for key, value in self.subset_source_counts.items()
            },
            "training_role_counts": dict(self.training_role_counts),
            "split_group_count": self.split_group_count,
            "task_family_counts": dict(self.task_family_counts),
            "label_distribution": {
                key: dict(value) for key, value in self.label_distribution.items()
            },
            "objective_label_distributions": {
                key: dict(value) for key, value in self.objective_label_distributions.items()
            },
            "subjective_target_counts": dict(self.subjective_target_counts),
            "feature_count": self.feature_count,
            "feature_group_counts": dict(self.feature_group_counts),
            "eeg_feature_count": self.eeg_feature_count,
            "ecg_feature_count": self.ecg_feature_count,
            "peripheral_feature_count": self.peripheral_feature_count,
            "missing_ecg_session_counts": dict(self.missing_ecg_session_counts),
        }


def dump_stage_i_task_entries(
    entries: Sequence[StageITaskEntry],
    *,
    path: str | Path,
) -> None:
    """Write Stage I task entries as UTF-8 JSON Lines."""

    Path(path).write_text(
        "".join(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def load_stage_i_task_entries(path: str | Path) -> tuple[StageITaskEntry, ...]:
    """Load Stage I task entries from JSONL.

    Older Phase 0/1 artifacts did not carry the window-level fields. Those rows
    are upgraded in-memory so the reader remains backwards compatible.
    """

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(
        StageITaskEntry(
            sample_id=row["sample_id"],
            dataset_id=row["dataset_id"],
            subset_id=row["subset_id"],
            subject_id=row["subject_id"],
            session_id=row["session_id"],
            split_group=row["split_group"],
            training_role=row["training_role"],
            sample_granularity=str(row.get("sample_granularity") or "session"),
            recording_id=row.get("recording_id") or row["session_id"],
            window_index=_coerce_int(row.get("window_index"), default=0 if row.get("sample_granularity") != "window" else None),
            window_duration_s=_coerce_float(row.get("window_duration_s")),
            task_family=row.get("task_family") or _infer_task_family(row),
            label_namespace=row.get("label_namespace") or _infer_label_namespace(row),
            window_start_utc=row["window_start_utc"],
            window_end_utc=row["window_end_utc"],
            source_refs=dict(row.get("source_refs", {})),
            objective_label_name=row.get("objective_label_name"),
            objective_label_value=row.get("objective_label_value"),
            subjective_target_name=row.get("subjective_target_name"),
            subjective_target_value=row.get("subjective_target_value"),
            context_payload=dict(row.get("context_payload", {})),
        )
        for row in rows
    )


def dump_stage_i_summary(summary: StageIDatasetSummary, *, path: str | Path) -> None:
    """Write a Stage I dataset summary as JSON."""

    Path(path).write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def isoformat_utc(value: datetime) -> str:
    """Render one datetime as an ISO-8601 UTC string."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


def _coerce_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _infer_task_family(row: Mapping[str, object]) -> str:
    objective_name = str(row.get("objective_label_name") or "")
    label_namespace = str(row.get("label_namespace") or "")
    if "attention" in objective_name or "attention" in label_namespace:
        return "attention_state"
    return "workload"


def _infer_label_namespace(row: Mapping[str, object]) -> str | None:
    if row.get("objective_label_name"):
        return str(row["objective_label_name"])
    if row.get("subjective_target_name"):
        return str(row["subjective_target_name"])
    return None
