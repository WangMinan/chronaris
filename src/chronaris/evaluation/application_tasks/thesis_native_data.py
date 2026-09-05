"""Frozen native-time public task selection for thesis outer evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chronaris.dataset.clare_native import build_clare_native_dataset
from chronaris.dataset.cogpilot_native import (
    build_cogpilot_difficulty_dataset,
    build_cogpilot_event_response_dataset,
)


COGPILOT_ROOT = Path(
    "/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/"
    "virtual-reality-piloting/1.0.0/dataPackage/task-ils"
)
CLARE_ROOT = Path("/home/wangminan/dataset/chronaris/clare")


@dataclass(frozen=True, slots=True)
class FrozenNativeTask:
    task_id: str
    dataset: object
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    class_targets: tuple[int, ...] | None
    regression_targets: tuple[float, ...] | None
    class_values: tuple[int, ...]
    lineage_sha256: str


def load_frozen_native_task(task_id: str, *, cache_root: str | Path):
    cache_root = Path(cache_root)
    if task_id == "cogpilot_difficulty":
        dataset = build_cogpilot_difficulty_dataset(
            COGPILOT_ROOT,
            subject_limit=20,
            context_duration_s=30.0,
            cache_root=cache_root / task_id,
        )
        selected = first_record_indices(
            dataset.records,
            lambda row: (row.group_id, int(row.label)),
        )
        return _task(
            task_id,
            dataset,
            selected,
            class_targets=tuple(int(dataset.labels[index]) for index in selected),
            regression_targets=None,
            class_values=(0, 1, 2, 3),
        )
    if task_id == "cogpilot_event_response":
        dataset = build_cogpilot_event_response_dataset(
            COGPILOT_ROOT,
            subject_limit=20,
            context_duration_s=12.0,
            cache_root=cache_root / task_id,
        )
        selected = middle_record_indices(
            dataset.records,
            lambda row: row.sample_id.rsplit("::", 1)[0],
        )
        return _task(
            task_id,
            dataset,
            selected,
            class_targets=None,
            regression_targets=tuple(
                float(dataset.labels[index]) for index in selected
            ),
            class_values=(),
        )
    if task_id == "clare_cognitive_load":
        dataset = build_clare_native_dataset(
            CLARE_ROOT,
            subject_limit=16,
            context_duration_s=10.0,
            window_stride=2,
            cache_root=cache_root / task_id,
        )
        selected = middle_dual_stream_indices(
            dataset,
            lambda row: (
                row.group_id,
                row.sample_id.split("__", 1)[1].split("_", 1)[0],
            ),
        )
        scores = tuple(float(dataset.labels[index]) for index in selected)
        return _task(
            task_id,
            dataset,
            selected,
            class_targets=tuple(int(value >= 7) for value in scores),
            regression_targets=scores,
            class_values=(0, 1),
        )
    raise ValueError(f"unsupported frozen native task: {task_id}")


def first_record_indices(records, key):
    seen = set()
    selected = []
    for index, record in enumerate(records):
        value = key(record)
        if value not in seen:
            selected.append(index)
            seen.add(value)
    return tuple(selected)


def middle_record_indices(records, key):
    grouped = {}
    for index, record in enumerate(records):
        grouped.setdefault(key(record), []).append(index)
    return tuple(indices[(len(indices) - 1) // 2] for indices in grouped.values())


def middle_dual_stream_indices(dataset, key):
    grouped = {}
    for index, record in enumerate(dataset.records):
        grouped.setdefault(key(record), []).append(index)
    selected = []
    for indices in grouped.values():
        middle = (len(indices) - 1) / 2
        for position in sorted(
            range(len(indices)), key=lambda value: abs(value - middle)
        ):
            index = indices[position]
            sample = dataset.load_sample(dataset.sample_ids[index])
            if (
                np.asarray(sample.physiology_feature_mask).any()
                and np.asarray(sample.vehicle_feature_mask).any()
            ):
                selected.append(index)
                break
    return tuple(selected)


def _task(
    task_id,
    dataset,
    selected,
    *,
    class_targets,
    regression_targets,
    class_values,
):
    if len({dataset.group_ids[index] for index in selected}) < 5:
        raise ValueError(f"{task_id} requires at least five subject groups")
    sample_ids = tuple(dataset.sample_ids[index] for index in selected)
    group_ids = tuple(dataset.group_ids[index] for index in selected)
    lineage = [
        {
            "sample_id": dataset.records[index].sample_id,
            "group_id": dataset.records[index].group_id,
            "label": dataset.records[index].label,
            "context_duration_s": dataset.records[index].context_duration_s,
            "source_sample_hash": dataset.records[index].source_sample_hash,
        }
        for index in selected
    ]
    digest = hashlib.sha256(
        json.dumps(lineage, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return FrozenNativeTask(
        task_id=task_id,
        dataset=dataset,
        sample_ids=sample_ids,
        group_ids=group_ids,
        class_targets=class_targets,
        regression_targets=regression_targets,
        class_values=class_values,
        lineage_sha256=digest,
    )
