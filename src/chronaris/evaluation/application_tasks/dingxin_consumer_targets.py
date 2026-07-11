"""Fold-role targets for Dingxin frozen-representation consumer smoke."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


MANEUVER_TASK = "maneuver_intensity_classification"
RESPONSE_TASK = "physiology_response_prediction"


@dataclass(frozen=True, slots=True)
class DingxinFoldConsumerTargets:
    fold_id: str
    role_by_sample_id: Mapping[str, str]
    maneuver_class_by_sample_id: Mapping[str, int]
    response_value_by_sample_id: Mapping[str, float]
    high_response_by_sample_id: Mapping[str, int]
    target_source_sha256: str
    threshold_scope: str = "outer_train_smoke_only"

    def sample_ids(self, *, role: str, task: str) -> tuple[str, ...]:
        source = (
            self.maneuver_class_by_sample_id
            if task == MANEUVER_TASK
            else self.response_value_by_sample_id
            if task == RESPONSE_TASK
            else None
        )
        if source is None:
            raise ValueError(f"unsupported Dingxin consumer task: {task}")
        return tuple(
            sample_id
            for sample_id, sample_role in self.role_by_sample_id.items()
            if sample_role == role and sample_id in source
        )

    def maneuver_classes(self, sample_ids: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self.maneuver_class_by_sample_id[sample_id] for sample_id in sample_ids],
            dtype=np.int64,
        )

    def response_values(self, sample_ids: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self.response_value_by_sample_id[sample_id] for sample_id in sample_ids],
            dtype=np.float64,
        )

    def high_response_classes(self, sample_ids: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [self.high_response_by_sample_id[sample_id] for sample_id in sample_ids],
            dtype=np.int64,
        )

    def to_manifest(self):
        return {
            "fold_id": self.fold_id,
            "threshold_scope": self.threshold_scope,
            "target_source_sha256": self.target_source_sha256,
            "role_counts": {
                role: sum(value == role for value in self.role_by_sample_id.values())
                for role in ("train", "validation", "held_out")
            },
            "maneuver_counts": {
                role: len(self.sample_ids(role=role, task=MANEUVER_TASK))
                for role in ("train", "validation", "held_out")
            },
            "response_counts": {
                role: len(self.sample_ids(role=role, task=RESPONSE_TASK))
                for role in ("train", "validation", "held_out")
            },
        }


def load_dingxin_fold_consumer_targets(
    *,
    fold_id: str,
    split_manifest_path: str | Path,
    binding_path: str | Path,
) -> DingxinFoldConsumerTargets:
    split_path = Path(split_manifest_path)
    target_path = Path(binding_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if str(split["fold_id"]) != fold_id:
        raise ValueError("Dingxin consumer split fold mismatch")
    role_by_id = {
        str(sample_id): role
        for role, key in (
            ("train", "train_sample_ids"),
            ("validation", "validation_sample_ids"),
            ("held_out", "held_out_sample_ids"),
        )
        for sample_id in split[key]
    }
    bindings = pd.read_csv(target_path)
    fold = bindings[
        (bindings["fold_id"].astype(str) == fold_id)
        & (bindings["binding_status"].astype(str) == "available")
        & (bindings["context_id"].astype(str).isin(role_by_id))
    ]
    maneuver = fold[fold["task_slug"] == MANEUVER_TASK]
    response = fold[fold["task_slug"] == RESPONSE_TASK]
    if maneuver["context_id"].duplicated().any() or response["context_id"].duplicated().any():
        raise ValueError("Dingxin consumer target IDs are duplicated")
    maneuver_values = {
        str(row.context_id): int(row.class_target)
        for row in maneuver.itertuples(index=False)
    }
    response_values = {
        str(row.context_id): float(row.continuous_target)
        for row in response.itertuples(index=False)
        if np.isfinite(row.continuous_target)
    }
    high_values = {
        str(row.context_id): int(row.binary_target)
        for row in response.itertuples(index=False)
        if np.isfinite(row.continuous_target) and int(row.binary_target) in {0, 1}
    }
    if set(response_values) != set(high_values):
        raise ValueError("Dingxin response continuous/binary targets are misaligned")
    for role in ("train", "validation", "held_out"):
        maneuver_role = [
            value
            for sample_id, value in maneuver_values.items()
            if role_by_id[sample_id] == role
        ]
        response_role = [
            high_values[sample_id]
            for sample_id in response_values
            if role_by_id[sample_id] == role
        ]
        if set(maneuver_role) != {0, 1, 2}:
            raise ValueError(f"Dingxin maneuver classes incomplete for {fold_id}/{role}")
        if set(response_role) != {0, 1}:
            raise ValueError(f"Dingxin response classes incomplete for {fold_id}/{role}")
    source_hash = hashlib.sha256(
        (
            target_path.read_bytes()
            + split_path.read_bytes()
            + json.dumps(
                {
                    "fold_id": fold_id,
                    "role_by_id": role_by_id,
                    "maneuver": maneuver_values,
                    "response": response_values,
                    "high": high_values,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        )
    ).hexdigest()
    return DingxinFoldConsumerTargets(
        fold_id=fold_id,
        role_by_sample_id=role_by_id,
        maneuver_class_by_sample_id=maneuver_values,
        response_value_by_sample_id=response_values,
        high_response_by_sample_id=high_values,
        target_source_sha256=source_hash,
    )
