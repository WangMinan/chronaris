"""Contracts and JSON helpers for Stage I private-task benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StageIPrivateTaskEntry:
    """One private-task sample derived from Stage H all-window assets."""

    sample_id: str
    sortie_id: str
    pilot_id: int
    view_id: str
    window_index: int
    sample_partition: str | None
    task_name: str
    task_type: str
    label_name: str
    label_value: float | int | str | None
    label_source: str
    source_refs: Mapping[str, str]
    paired_sample_id: str | None = None
    context_payload: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "sortie_id": self.sortie_id,
            "pilot_id": self.pilot_id,
            "view_id": self.view_id,
            "window_index": self.window_index,
            "sample_partition": self.sample_partition,
            "task_name": self.task_name,
            "task_type": self.task_type,
            "label_name": self.label_name,
            "label_value": self.label_value,
            "label_source": self.label_source,
            "source_refs": dict(self.source_refs),
            "paired_sample_id": self.paired_sample_id,
            "context_payload": dict(self.context_payload),
        }


def dump_stage_i_private_task_entries(
    entries: Sequence[StageIPrivateTaskEntry],
    *,
    path: str | Path,
) -> None:
    Path(path).write_text(
        "".join(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def load_stage_i_private_task_entries(path: str | Path) -> tuple[StageIPrivateTaskEntry, ...]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tuple(
        StageIPrivateTaskEntry(
            sample_id=str(row["sample_id"]),
            sortie_id=str(row["sortie_id"]),
            pilot_id=int(row["pilot_id"]),
            view_id=str(row["view_id"]),
            window_index=int(row["window_index"]),
            sample_partition=_optional_str(row.get("sample_partition")),
            task_name=str(row["task_name"]),
            task_type=str(row["task_type"]),
            label_name=str(row["label_name"]),
            label_value=row.get("label_value"),
            label_source=str(row["label_source"]),
            source_refs=dict(row.get("source_refs", {})),
            paired_sample_id=_optional_str(row.get("paired_sample_id")),
            context_payload=dict(row.get("context_payload", {})),
        )
        for row in rows
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
