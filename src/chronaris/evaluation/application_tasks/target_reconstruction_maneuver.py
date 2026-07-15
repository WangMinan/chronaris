"""Train-only future-maneuver score and trend targets for Dingxin task 1C."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.dataset.application_evaluation.labels import (
    _maneuver_raw_values,
    _maneuver_score,
)
from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    _load_contexts,
    _load_field_roles,
)
from chronaris.evaluation.dingxin.pipelines.benchmark_data import (
    load_aligned_private_records,
)


def build_future_maneuver_targets(
    *,
    plans: Sequence[Mapping[str, object]],
    fixed_root: str | Path,
    e_run_manifest_path: str,
    f_run_manifest_path: str,
    minimum_semantic_count: int = 4,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit maneuver scales on inner-train and score the next five-second window."""

    fixed = Path(fixed_root)
    contexts = _load_contexts(fixed / "context_sample_manifest.jsonl")
    context_by_id = {row.context_id: row for row in contexts}
    roles = _load_field_roles(fixed / "field_role_manifest.csv")
    roles_by_sortie = defaultdict(list)
    for role in roles:
        if role.selected_for_maneuver_label and role.semantic_key:
            roles_by_sortie[role.sortie_id].append(role)
    records = load_aligned_private_records(
        e_run_manifest_path=e_run_manifest_path,
        f_run_manifest_path=f_run_manifest_path,
    )
    record_by_sample = {
        str(row.sample_id): row for row in records.itertuples(index=False)
    }
    target_rows = []
    threshold_rows = []
    for plan in plans:
        split_id = str(plan["fold_id"])
        role_ids = {
            "train": tuple(str(value) for value in plan["train_sample_ids"]),
            "validation": tuple(str(value) for value in plan["validation_sample_ids"]),
        }
        fit_hash = stable_sample_hash(role_ids["train"])
        scalers, scaler_rows = _fit_semantic_scalers(
            train_ids=role_ids["train"],
            context_by_id=context_by_id,
            record_by_sample=record_by_sample,
            roles_by_sortie=roles_by_sortie,
            minimum_semantic_count=minimum_semantic_count,
            eps=eps,
        )
        threshold_rows.extend(
            {"split_id": split_id, "fit_sample_hash": fit_hash, **row}
            for row in scaler_rows
        )
        scored = {}
        for role, sample_ids in role_ids.items():
            for context_id in sample_ids:
                context = context_by_id[context_id]
                current = _score_record(
                    record_by_sample.get(context.end_sample_id),
                    roles_by_sortie.get(context.sortie_id, ()),
                    scalers,
                    minimum_semantic_count,
                    eps,
                )
                future = _score_record(
                    record_by_sample.get(context.target_sample_id),
                    roles_by_sortie.get(context.sortie_id, ()),
                    scalers,
                    minimum_semantic_count,
                    eps,
                )
                scored[(role, context_id)] = (current, future)
        train_deltas = [
            future[0] - current[0]
            for (role, _context_id), (current, future) in scored.items()
            if role == "train" and current[0] is not None and future[0] is not None
        ]
        if len(train_deltas) < 4:
            raise ValueError(f"{split_id} has too few future maneuver targets")
        trend_tolerance = float(np.quantile(np.abs(train_deltas), 1.0 / 3.0))
        trend_tolerance = max(trend_tolerance, eps)
        threshold_rows.append(
            {
                "split_id": split_id,
                "parameter_type": "future_trend_tolerance",
                "parameter_name": "train_abs_delta_q33",
                "parameter_value": trend_tolerance,
                "fit_sample_hash": fit_hash,
            }
        )
        for (role, context_id), (current, future) in scored.items():
            available = current[0] is not None and future[0] is not None
            delta = None if not available else float(future[0] - current[0])
            trend = None if delta is None else _trend_class(delta, trend_tolerance)
            context = context_by_id[context_id]
            target_rows.append(
                {
                    "split_id": split_id,
                    "outer_pool_id": str(plan["outer_pool_id"]),
                    "split_kind": str(plan["split_kind"]),
                    "main_selection": bool(plan["main_selection"]),
                    "role": role,
                    "context_id": context_id,
                    "current_maneuver_score": current[0],
                    "future_maneuver_score": future[0],
                    "maneuver_score_delta": delta,
                    "maneuver_trend_class": trend,
                    "current_valid_semantic_count": current[1],
                    "future_valid_semantic_count": future[1],
                    "target_sample_id": context.target_sample_id,
                    "fit_sample_hash": fit_hash,
                    "status": "completed" if available else "future_score_unavailable",
                    "outer_test_opened": False,
                }
            )
    return pd.DataFrame(target_rows), pd.DataFrame(threshold_rows)


def _fit_semantic_scalers(
    *, train_ids, context_by_id, record_by_sample, roles_by_sortie,
    minimum_semantic_count, eps,
):
    raw = {}
    for context_id in train_ids:
        context = context_by_id[context_id]
        raw[context_id] = _maneuver_raw_values(
            record_by_sample[context.end_sample_id].raw_vehicle_stats,
            roles_by_sortie.get(context.sortie_id, ()),
        )
    semantic_keys = sorted({key for values in raw.values() for key in values})
    scalers = {}
    rows = []
    for key in semantic_keys:
        std_values = [values[key][0] for values in raw.values() if key in values]
        delta_values = [values[key][1] for values in raw.values() if key in values]
        median_std, iqr_std = _median_iqr(std_values)
        median_delta, iqr_delta = _median_iqr(delta_values)
        selected = bool(iqr_std > eps or iqr_delta > eps)
        if selected:
            scalers[key] = (median_std, iqr_std, median_delta, iqr_delta)
        rows.append(
            {
                "parameter_type": "semantic_scaler" if selected else "semantic_scaler_excluded",
                "parameter_name": key,
                "median_std": median_std,
                "iqr_std": iqr_std,
                "median_abs_delta": median_delta,
                "iqr_abs_delta": iqr_delta,
                "exclusion_reason": None if selected else "both_train_iqrs_are_zero",
            }
        )
    if len(scalers) < minimum_semantic_count:
        raise ValueError("future maneuver target has too few train semantic groups")
    return scalers, rows


def _score_record(record, roles, scalers, minimum_semantic_count, eps):
    if record is None:
        return None, 0
    raw = _maneuver_raw_values(record.raw_vehicle_stats, roles)
    return _maneuver_score(
        raw,
        scalers,
        minimum_semantic_count=minimum_semantic_count,
        eps=eps,
    )


def _trend_class(delta: float, tolerance: float) -> int:
    if delta < -tolerance:
        return 0
    if delta > tolerance:
        return 2
    return 1


def _median_iqr(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.quantile(array, 0.75) - np.quantile(array, 0.25))
