"""Split-local Dingxin labels for the core-task recovery audit and training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    FieldRoleRecord,
    stable_sample_hash,
)
from chronaris.dataset.application_evaluation.labels import (
    build_maneuver_labels_for_split,
)
from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    build_response_targets_for_split,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    build_raw_median_response_targets,
    load_dingxin_target_source_data,
)
from chronaris.evaluation.application_tasks.core_recovery_tasks import (
    TaskAwareTargetBundle,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    load_aligned_private_records,
)


@dataclass(frozen=True, slots=True)
class CoreRecoveryTargetSource:
    contexts: tuple[ApplicationContextRecord, ...]
    context_by_id: Mapping[str, ApplicationContextRecord]
    record_by_sample: Mapping[str, object]
    field_roles: tuple[FieldRoleRecord, ...]
    response_delta_index: Mapping[tuple[str, str], float]
    response_candidate_fields: tuple[str, ...]
    source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CoreRecoverySplitTargets:
    maneuver: pd.DataFrame
    response: pd.DataFrame
    maneuver_thresholds: pd.DataFrame
    response_thresholds: pd.DataFrame
    maneuver_target_mode: str
    fit_sample_hash: str


def load_core_recovery_target_source(
    *,
    fixed_audit_root: str | Path,
    snapshot_root: str | Path,
    e_run_manifest_path: str | Path,
    f_run_manifest_path: str | Path,
) -> CoreRecoveryTargetSource:
    fixed_root = Path(fixed_audit_root)
    context_path = fixed_root / "context_sample_manifest.jsonl"
    role_path = fixed_root / "field_role_manifest.csv"
    if not context_path.is_file() or not role_path.is_file():
        raise FileNotFoundError("fixed Dingxin context or field-role source is missing")
    contexts = _load_contexts(context_path)
    roles = _load_field_roles(role_path)
    records = load_aligned_private_records(
        e_run_manifest_path=str(e_run_manifest_path),
        f_run_manifest_path=str(f_run_manifest_path),
    )
    response_source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
    )
    response_raw = build_raw_median_response_targets(
        response_source,
        snapshot_root=snapshot_root,
    )
    delta_index = {
        (str(row.context_id), str(row.feature_name)): float(row.absolute_delta)
        for row in response_raw.field_delta_rows.itertuples(index=False)
        if np.isfinite(row.absolute_delta)
    }
    candidate_fields = tuple(
        sorted(response_raw.field_delta_rows["feature_name"].astype(str).unique())
    )
    return CoreRecoveryTargetSource(
        contexts=contexts,
        context_by_id={row.context_id: row for row in contexts},
        record_by_sample={str(row.sample_id): row for row in records.itertuples(index=False)},
        field_roles=roles,
        response_delta_index=delta_index,
        response_candidate_fields=candidate_fields,
        source_hashes=response_source.source_hashes,
    )


def fit_core_recovery_split_targets(
    source: CoreRecoveryTargetSource,
    *,
    train_context_ids: Sequence[str],
    evaluation_context_ids: Sequence[str],
    maneuver_target_mode: str,
) -> CoreRecoverySplitTargets:
    train_ids = tuple(str(value) for value in train_context_ids)
    evaluation_ids = tuple(str(value) for value in evaluation_context_ids)
    if not train_ids or set(train_ids) & set(evaluation_ids):
        raise ValueError("core-recovery target train and evaluation roles must be disjoint")
    fit_hash = stable_sample_hash(train_ids)
    maneuver, maneuver_thresholds = fit_core_recovery_maneuver_targets(
        source,
        train_context_ids=train_ids,
        evaluation_context_ids=evaluation_ids,
        maneuver_target_mode=maneuver_target_mode,
    )
    response, response_thresholds = fit_core_recovery_response_targets(
        source,
        train_context_ids=train_ids,
        evaluation_context_ids=evaluation_ids,
    )
    return CoreRecoverySplitTargets(
        maneuver=maneuver,
        response=response,
        maneuver_thresholds=maneuver_thresholds,
        response_thresholds=response_thresholds,
        maneuver_target_mode=maneuver_target_mode,
        fit_sample_hash=fit_hash,
    )


def fit_core_recovery_maneuver_targets(
    source: CoreRecoveryTargetSource,
    *,
    train_context_ids: Sequence[str],
    evaluation_context_ids: Sequence[str],
    maneuver_target_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = tuple(str(value) for value in train_context_ids)
    evaluation_ids = tuple(str(value) for value in evaluation_context_ids)
    fit_hash = stable_sample_hash(train_ids)
    rows, thresholds = build_maneuver_labels_for_split(
        record_by_sample=source.record_by_sample,
        context_by_id=source.context_by_id,
        train_context_ids=train_ids,
        evaluation_context_ids=evaluation_ids,
        roles=source.field_roles,
        fit_sample_hash=fit_hash,
        target_mode=maneuver_target_mode,
    )
    return (
        pd.DataFrame(rows).rename(columns={"split_role": "role"}),
        pd.DataFrame(thresholds),
    )


def fit_core_recovery_response_targets(
    source: CoreRecoveryTargetSource,
    *,
    train_context_ids: Sequence[str],
    evaluation_context_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ids = tuple(str(value) for value in train_context_ids)
    evaluation_ids = tuple(str(value) for value in evaluation_context_ids)
    rows, thresholds, response_hash = build_response_targets_for_split(
        train_context_ids=train_ids,
        evaluation_context_ids=evaluation_ids,
        delta_index=source.response_delta_index,
        candidate_fields=source.response_candidate_fields,
    )
    if response_hash != stable_sample_hash(
        tuple(
            context_id
            for context_id in train_ids
            if any(
                (context_id, field) in source.response_delta_index
                for field in source.response_candidate_fields
            )
        )
    ):
        raise ValueError("response target fit hash is inconsistent")
    return (
        pd.DataFrame(rows).rename(columns={"continuous_target": "score"}),
        pd.DataFrame(thresholds),
    )


def build_task_aware_target_bundle(
    source: CoreRecoveryTargetSource,
    targets: CoreRecoverySplitTargets,
    *,
    sample_ids: Sequence[str],
) -> tuple[TaskAwareTargetBundle, tuple[str, ...]]:
    ordered = tuple(str(value) for value in sample_ids)
    maneuver_index = targets.maneuver.set_index("context_id")
    response_index = targets.response.set_index("context_id")
    missing = sorted(set(ordered) - set(maneuver_index.index))
    if missing:
        raise ValueError(f"task-aware maneuver targets are missing: {missing[:5]}")
    class_mapping = {"low": 0, "medium": 1, "high": 2}
    maneuver_classes = []
    maneuver_scores = []
    for sample_id in ordered:
        row = maneuver_index.loc[sample_id]
        if row.status != "completed":
            raise ValueError(f"maneuver target is unavailable: {sample_id}")
        maneuver_classes.append(class_mapping[str(row.class_label)])
        maneuver_scores.append(float(row.score))
    selected_rows = targets.response_thresholds[
        targets.response_thresholds["parameter_type"] == "response_field_scale"
    ]
    field_scales = {
        str(row.parameter_name): float(row.iqr_delta)
        for row in selected_rows.itertuples(index=False)
    }
    field_names = tuple(sorted(field_scales))
    if not field_names:
        raise ValueError("task-aware response targets have no selected physiology fields")
    response_values = []
    high_response = []
    response_available = []
    field_deltas = []
    field_masks = []
    for sample_id in ordered:
        if sample_id not in response_index.index or response_index.loc[sample_id].status != "completed":
            response_values.append(0.0)
            high_response.append(0.0)
            response_available.append(False)
            field_deltas.append([0.0] * len(field_names))
            field_masks.append([False] * len(field_names))
            continue
        row = response_index.loc[sample_id]
        response_values.append(float(row.score))
        high_response.append(float(row.binary_target))
        response_available.append(True)
        values = []
        masks = []
        for field_name in field_names:
            raw = source.response_delta_index.get((sample_id, field_name))
            if raw is None:
                values.append(0.0)
                masks.append(False)
            else:
                values.append(float(np.clip(raw / field_scales[field_name], 0, 10)))
                masks.append(True)
        field_deltas.append(values)
        field_masks.append(masks)
    maneuver_scores_tensor = torch.tensor(maneuver_scores, dtype=torch.float32)
    return (
        TaskAwareTargetBundle(
            sample_ids=ordered,
            maneuver_class=torch.tensor(maneuver_classes, dtype=torch.long),
            maneuver_score=maneuver_scores_tensor,
            response_value=torch.tensor(response_values, dtype=torch.float32),
            high_response=torch.tensor(high_response, dtype=torch.float32),
            response_available=torch.tensor(response_available, dtype=torch.bool),
            field_deltas=torch.tensor(field_deltas, dtype=torch.float32),
            field_delta_mask=torch.tensor(field_masks, dtype=torch.bool),
            maneuver_soft_targets=_maneuver_soft_targets(
                targets.maneuver,
                maneuver_scores_tensor,
                torch.tensor(maneuver_classes, dtype=torch.long),
                targets.maneuver_thresholds,
            ),
            maneuver_target_mode=targets.maneuver_target_mode,
        ),
        field_names,
    )


def _maneuver_soft_targets(frame, scores, classes, thresholds):
    bounds = thresholds[thresholds["parameter_type"] == "class_bounds"]
    if len(bounds) != 1:
        raise ValueError("maneuver target requires one train-fitted class boundary row")
    lower = float(bounds.iloc[0]["lower_bound"])
    upper = float(bounds.iloc[0]["upper_bound"])
    train_scores = frame[frame["role"] == "train"]["score"].astype(float).to_numpy()
    width = 0.05 * float(np.quantile(train_scores, 0.75) - np.quantile(train_scores, 0.25))
    soft = torch.nn.functional.one_hot(classes, num_classes=3).to(torch.float32)
    if width <= 1e-8:
        return soft
    for index, score in enumerate(scores.tolist()):
        if abs(score - lower) <= width:
            alpha = float(np.clip((score - (lower - width)) / (2 * width), 0, 1))
            soft[index] = torch.tensor([1 - alpha, alpha, 0.0])
        elif abs(score - upper) <= width:
            alpha = float(np.clip((score - (upper - width)) / (2 * width), 0, 1))
            soft[index] = torch.tensor([0.0, 1 - alpha, alpha])
    return soft


def _load_contexts(path: Path) -> tuple[ApplicationContextRecord, ...]:
    rows = []
    for payload in pd.read_json(path, lines=True).to_dict("records"):
        resolved = dict(payload)
        resolved["source_sample_ids"] = tuple(resolved["source_sample_ids"])
        rows.append(ApplicationContextRecord(**resolved))
    return tuple(rows)


def _load_field_roles(path: Path) -> tuple[FieldRoleRecord, ...]:
    rows = []
    for payload in pd.read_csv(path).to_dict("records"):
        resolved = {key: value for key, value in payload.items() if key != "valid_window_ratio"}
        for key in ("display_label", "unit_hint", "semantic_key", "exclusion_reason"):
            if pd.isna(resolved[key]):
                resolved[key] = None
        for key in (
            "selected_for_maneuver_label",
            "selected_for_response_target",
            "allowed_in_maneuver_input",
        ):
            resolved[key] = bool(resolved[key])
        rows.append(FieldRoleRecord(**resolved))
    return tuple(rows)
