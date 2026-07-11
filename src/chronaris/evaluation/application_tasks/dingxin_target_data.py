"""Load fixed Dingxin lineage and derive raw-median physiology targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation import stable_sample_hash
from chronaris.dataset.application_evaluation.snapshot_io import (
    iter_raw_point_snapshot,
    sha256_file,
)
from chronaris.dataset.application_evaluation.labels import RESPONSE_TASK_ID


@dataclass(frozen=True, slots=True)
class DingxinTargetSourceData:
    contexts: pd.DataFrame
    audit_labels: pd.DataFrame
    audit_thresholds: pd.DataFrame
    field_roles: pd.DataFrame
    split_manifest: dict[str, object]
    snapshot_manifest: dict[str, object]
    source_hashes: dict[str, str]
    verified_snapshot_files: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class RawMedianResponseResult:
    label_rows: pd.DataFrame
    threshold_rows: pd.DataFrame
    field_delta_rows: pd.DataFrame
    availability_rows: pd.DataFrame


def load_dingxin_target_source_data(
    *,
    fixed_audit_root: str | Path,
    snapshot_root: str | Path,
) -> DingxinTargetSourceData:
    audit_root = Path(fixed_audit_root)
    raw_root = Path(snapshot_root)
    required = {
        "context_sample_manifest.jsonl",
        "fold_task_labels.csv",
        "fold_label_thresholds.csv",
        "field_role_manifest.csv",
        "split_manifest.json",
    }
    missing = sorted(name for name in required if not (audit_root / name).is_file())
    if missing:
        raise FileNotFoundError(f"fixed audit inputs missing: {missing}")
    snapshot_path = raw_root / "snapshot_manifest.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    contexts = pd.read_json(
        audit_root / "context_sample_manifest.jsonl",
        lines=True,
    )
    labels = pd.read_csv(audit_root / "fold_task_labels.csv")
    thresholds = pd.read_csv(audit_root / "fold_label_thresholds.csv")
    roles = pd.read_csv(audit_root / "field_role_manifest.csv")
    split_manifest = json.loads(
        (audit_root / "split_manifest.json").read_text(encoding="utf-8")
    )
    snapshot_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot_manifest.get("status") != "completed":
        raise ValueError("Dingxin raw snapshot is not completed")
    verified = []
    for item in snapshot_manifest["files"]:
        path = raw_root / item["relative_path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"raw snapshot hash mismatch: {path}")
        verified.append(
            {
                "stream_kind": item["stream_kind"],
                "sortie_id": item["sortie_id"],
                "view_id": item["view_id"],
                "path": str(path),
                "sha256": actual,
                "point_count": item["point_count"],
            }
        )
    source_paths = [
        audit_root / "context_sample_manifest.jsonl",
        audit_root / "fold_task_labels.csv",
        audit_root / "fold_label_thresholds.csv",
        audit_root / "field_role_manifest.csv",
        audit_root / "split_manifest.json",
        snapshot_path,
    ]
    return DingxinTargetSourceData(
        contexts=contexts,
        audit_labels=labels,
        audit_thresholds=thresholds,
        field_roles=roles,
        split_manifest=split_manifest,
        snapshot_manifest=snapshot_manifest,
        source_hashes={path.name: sha256_file(path) for path in source_paths},
        verified_snapshot_files=tuple(verified),
    )


def build_raw_median_response_targets(
    source: DingxinTargetSourceData,
    *,
    snapshot_root: str | Path,
    train_valid_ratio: float = 0.80,
    minimum_field_count: int = 2,
    eps: float = 1e-6,
) -> RawMedianResponseResult:
    deltas, availability = _build_context_field_deltas(
        source,
        snapshot_root=Path(snapshot_root),
    )
    delta_index = {
        (row.context_id, row.feature_name): row
        for row in deltas.itertuples(index=False)
    }
    candidate_fields = tuple(sorted(deltas["feature_name"].unique()))
    label_rows = []
    threshold_rows = []
    context_by_id = source.contexts.set_index("context_id")
    for fold in source.split_manifest["split_protocols"]:
        fold_id = str(fold["fold_id"])
        train_ids = tuple(str(value) for value in fold["response_train_context_ids"])
        test_ids = tuple(str(value) for value in fold["response_test_context_ids"])
        fully_observed_train = tuple(
            context_id
            for context_id in train_ids
            if bool(availability.set_index("context_id").loc[context_id, "fully_observed"])
        )
        fit_hash = stable_sample_hash(fully_observed_train)
        selected = []
        field_iqrs = {}
        for field_name in candidate_fields:
            values = [
                float(delta_index[(context_id, field_name)].absolute_delta)
                for context_id in fully_observed_train
                if (context_id, field_name) in delta_index
                and np.isfinite(delta_index[(context_id, field_name)].absolute_delta)
            ]
            valid_ratio = len(values) / max(len(fully_observed_train), 1)
            if valid_ratio < train_valid_ratio or not values:
                threshold_rows.append(
                    _response_threshold_row(
                        fold_id,
                        fit_hash,
                        "response_field_excluded",
                        field_name,
                        valid_ratio=valid_ratio,
                        exclusion_reason="train_valid_ratio_below_threshold",
                    )
                )
                continue
            iqr = _iqr(values)
            if iqr <= eps:
                threshold_rows.append(
                    _response_threshold_row(
                        fold_id,
                        fit_hash,
                        "response_field_excluded",
                        field_name,
                        valid_ratio=valid_ratio,
                        iqr_delta=iqr,
                        exclusion_reason="train_iqr_is_zero",
                    )
                )
                continue
            selected.append(field_name)
            field_iqrs[field_name] = iqr
            threshold_rows.append(
                _response_threshold_row(
                    fold_id,
                    fit_hash,
                    "response_field_scale",
                    field_name,
                    valid_ratio=valid_ratio,
                    iqr_delta=iqr,
                )
            )
        if len(selected) < minimum_field_count:
            raise ValueError(
                f"{fold_id} has only {len(selected)} raw-median response fields"
            )
        scores = {
            context_id: _response_score(
                context_id,
                selected,
                field_iqrs,
                delta_index,
                minimum_field_count,
            )
            for context_id in train_ids + test_ids
        }
        train_scores = [
            score
            for context_id in fully_observed_train
            if (score := scores[context_id][0]) is not None
        ]
        if len(train_scores) < 4:
            raise ValueError(f"{fold_id} has fewer than four raw-median train targets")
        high_threshold = float(np.quantile(train_scores, 0.75))
        threshold_rows.append(
            _response_threshold_row(
                fold_id,
                fit_hash,
                "high_response_bound",
                "train_q75",
                high_response_threshold=high_threshold,
            )
        )
        for role, context_ids in (("train", train_ids), ("test", test_ids)):
            for context_id in context_ids:
                score, valid_count = scores[context_id]
                context = context_by_id.loc[context_id]
                full = bool(
                    availability.set_index("context_id").loc[
                        context_id, "fully_observed"
                    ]
                )
                status = "completed" if score is not None and full else (
                    "future_interval_not_fully_observed"
                    if not full
                    else "insufficient_fields"
                )
                label_rows.append(
                    {
                        "fold_id": fold_id,
                        "split_strategy": fold["split_strategy"],
                        "held_out_group": fold["held_out_group"],
                        "task_id": RESPONSE_TASK_ID + "_raw_median_v1",
                        "task_name": "机动诱发生理响应预测",
                        "task_type": "regression_and_binary",
                        "split_role": role,
                        "context_id": context_id,
                        "continuous_target": score,
                        "high_response_label": (
                            None if score is None else int(score >= high_threshold)
                        ),
                        "valid_field_count": valid_count,
                        "status": status,
                        "fit_sample_hash": fit_hash,
                        "representative_statistic": "raw_window_median",
                        "input_start_offset_ms": int(context.start_offset_ms),
                        "input_end_exclusive_ms": int(context.end_offset_ms),
                        "target_start_offset_ms": int(context.end_offset_ms),
                        "target_end_exclusive_ms": int(context.end_offset_ms) + 5_000,
                    }
                )
    return RawMedianResponseResult(
        label_rows=pd.DataFrame(label_rows),
        threshold_rows=pd.DataFrame(threshold_rows),
        field_delta_rows=deltas,
        availability_rows=availability,
    )


def _build_context_field_deltas(source, *, snapshot_root):
    response_contexts = source.contexts[source.contexts["response_eligible"]].copy()
    response_fields = source.field_roles[
        source.field_roles["selected_for_response_target"]
    ][["feature_name", "measurement", "source_field"]].drop_duplicates()
    field_specs = {
        row.feature_name: (row.measurement, row.source_field)
        for row in response_fields.itertuples(index=False)
    }
    plans = {item["view_id"]: item for item in source.snapshot_manifest["plans"]}
    files = {
        item["view_id"]: item
        for item in source.snapshot_manifest["files"]
        if item["stream_kind"] == "physiology"
    }
    delta_rows = []
    availability_rows = []
    for view_id, contexts in response_contexts.groupby("view_id", sort=True):
        plan = plans[view_id]
        start = datetime.fromisoformat(plan["start_utc"])
        stop_offset_ms = int(
            round((datetime.fromisoformat(plan["stop_utc"]) - start).total_seconds() * 1000)
        )
        file_record = files[view_id]
        path = snapshot_root / file_record["relative_path"]
        observed = {field_name: ([], []) for field_name in field_specs}
        for point in iter_raw_point_snapshot(path):
            offset_ms = (point.timestamp - start).total_seconds() * 1000.0
            for field_name, (measurement, source_field) in field_specs.items():
                if point.measurement != measurement or source_field not in point.values:
                    continue
                value = _finite_float(point.values[source_field])
                if value is not None:
                    observed[field_name][0].append(offset_ms)
                    observed[field_name][1].append(value)
        arrays = {
            field_name: (
                np.asarray(times, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
            for field_name, (times, values) in observed.items()
        }
        for context in contexts.itertuples(index=False):
            target_end = int(context.end_offset_ms) + 5_000
            fully_observed = target_end <= stop_offset_ms
            availability_rows.append(
                {
                    "context_id": context.context_id,
                    "sortie_id": context.sortie_id,
                    "view_id": view_id,
                    "input_end_exclusive_ms": int(context.end_offset_ms),
                    "target_start_offset_ms": int(context.end_offset_ms),
                    "target_end_exclusive_ms": target_end,
                    "snapshot_stop_offset_ms": stop_offset_ms,
                    "fully_observed": fully_observed,
                    "unavailable_reason": (
                        None if fully_observed else "future_interval_not_fully_observed"
                    ),
                }
            )
            for field_name, (times, values) in arrays.items():
                current = (times >= context.end_offset_ms - 5_000) & (
                    times < context.end_offset_ms
                )
                future = (times >= context.end_offset_ms) & (times < target_end)
                current_median = float(np.median(values[current])) if current.any() else None
                future_median = (
                    float(np.median(values[future]))
                    if future.any() and fully_observed
                    else None
                )
                delta_rows.append(
                    {
                        "context_id": context.context_id,
                        "sortie_id": context.sortie_id,
                        "view_id": view_id,
                        "feature_name": field_name,
                        "current_count": int(current.sum()),
                        "future_count": int(future.sum()),
                        "current_median": current_median,
                        "future_median": future_median,
                        "absolute_delta": (
                            None
                            if current_median is None or future_median is None
                            else abs(future_median - current_median)
                        ),
                        "snapshot_sha256": file_record["sha256"],
                    }
                )
    return pd.DataFrame(delta_rows), pd.DataFrame(availability_rows)


def _response_score(context_id, selected, iqrs, index, minimum):
    values = []
    for field_name in selected:
        row = index.get((context_id, field_name))
        if row is None or not np.isfinite(row.absolute_delta):
            continue
        values.append(float(row.absolute_delta) / iqrs[field_name])
    if len(values) < minimum:
        return None, len(values)
    return float(np.mean(values)), len(values)


def _response_threshold_row(
    fold_id,
    fit_hash,
    parameter_type,
    parameter_name,
    *,
    valid_ratio=None,
    iqr_delta=None,
    high_response_threshold=None,
    exclusion_reason=None,
):
    return {
        "fold_id": fold_id,
        "task_id": RESPONSE_TASK_ID + "_raw_median_v1",
        "parameter_type": parameter_type,
        "parameter_name": parameter_name,
        "train_valid_ratio": valid_ratio,
        "iqr_delta": iqr_delta,
        "high_response_threshold": high_response_threshold,
        "exclusion_reason": exclusion_reason,
        "fit_sample_hash": fit_hash,
        "representative_statistic": "raw_window_median",
    }


def _iqr(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def _finite_float(value):
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None
