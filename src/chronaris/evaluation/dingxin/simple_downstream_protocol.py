"""Frozen future-task contract for the simplified Dingxin evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from chronaris.dataset.application_evaluation import stable_sample_hash
from chronaris.dataset.application_evaluation.snapshot_io import (
    iter_raw_point_snapshot,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    DingxinTargetSourceData,
)


FUTURE_MANEUVER_SCORE_TASK_ID = "dingxin_future_maneuver_score_v1"
FUTURE_MANEUVER_CLASS_TASK_ID = "dingxin_future_maneuver_class_v1"
FUTURE_PHYSIOLOGY_FIELDS_TASK_ID = "dingxin_future_physiology_fields_v1"
SIMPLE_DOWNSTREAM_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)
PRIMARY_SPLIT_STRATEGY = "leave_one_sortie_out"
DIAGNOSTIC_SPLIT_STRATEGY = "leave_one_view_out"


@dataclass(frozen=True, slots=True)
class SimpleRawTargetBundle:
    """Method-independent raw target statistics before fold fitting."""

    contexts: pd.DataFrame
    maneuver_statistics: pd.DataFrame
    physiology_statistics: pd.DataFrame
    source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SimpleFoldTargetBundle:
    """Fold-fitted targets for the two leave-one-sortie-out folds."""

    fold_manifest: pd.DataFrame
    maneuver_targets: pd.DataFrame
    physiology_targets: pd.DataFrame
    threshold_rows: pd.DataFrame


def build_vehicle_context_id(sortie_id: str, target_start_offset_ms: int) -> str:
    return f"{sortie_id}::target_{int(target_start_offset_ms):09d}"


def extract_simple_raw_targets(
    source: DingxinTargetSourceData,
    *,
    snapshot_root: str | Path,
    strict_counts: bool = True,
) -> SimpleRawTargetBundle:
    """Extract complete five-second future targets from the frozen raw snapshot."""

    root = Path(snapshot_root)
    contexts = _complete_contexts(source)
    if strict_counts:
        _require_count("complete view-context", len(contexts), 90)
        _require_count(
            "unique vehicle-context",
            contexts["vehicle_context_id"].nunique(),
            60,
        )
    maneuver = _extract_maneuver_statistics(source, contexts, root)
    physiology = _extract_physiology_statistics(source, contexts, root)
    _validate_raw_bundle(contexts, maneuver, physiology)
    return SimpleRawTargetBundle(
        contexts=contexts,
        maneuver_statistics=maneuver,
        physiology_statistics=physiology,
        source_hashes=dict(source.source_hashes),
    )


def fit_simple_loso_targets(
    raw: SimpleRawTargetBundle,
    *,
    minimum_maneuver_semantic_count: int = 4,
    physiology_train_valid_ratio: float = 0.80,
    eps: float = 1e-6,
) -> SimpleFoldTargetBundle:
    """Fit target scales on each training sortie and apply them once to its holdout."""

    contexts = raw.contexts.copy()
    sorties = tuple(sorted(contexts["sortie_id"].astype(str).unique()))
    if len(sorties) != 2:
        raise ValueError(f"simple Dingxin protocol requires two sorties, got {len(sorties)}")
    fold_rows: list[dict[str, object]] = []
    maneuver_rows: list[dict[str, object]] = []
    physiology_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for fold_number, held_out_sortie in enumerate(sorties, start=1):
        fold_id = f"leave_one_sortie_out__fold{fold_number:02d}"
        role_by_context = {
            str(row.context_id): (
                "held_out" if str(row.sortie_id) == held_out_sortie else "train"
            )
            for row in contexts.itertuples(index=False)
        }
        train_context_ids = tuple(
            context_id
            for context_id, role in role_by_context.items()
            if role == "train"
        )
        held_out_context_ids = tuple(
            context_id
            for context_id, role in role_by_context.items()
            if role == "held_out"
        )
        fold_rows.append(
            {
                "fold_id": fold_id,
                "split_strategy": PRIMARY_SPLIT_STRATEGY,
                "held_out_sortie": held_out_sortie,
                "train_context_count": len(train_context_ids),
                "held_out_context_count": len(held_out_context_ids),
                "train_vehicle_context_count": int(
                    contexts[contexts["context_id"].isin(train_context_ids)][
                        "vehicle_context_id"
                    ].nunique()
                ),
                "held_out_vehicle_context_count": int(
                    contexts[contexts["context_id"].isin(held_out_context_ids)][
                        "vehicle_context_id"
                    ].nunique()
                ),
                "fit_sample_hash": stable_sample_hash(train_context_ids),
            }
        )
        fold_maneuver, maneuver_thresholds = _fit_maneuver_fold(
            fold_id=fold_id,
            contexts=contexts,
            statistics=raw.maneuver_statistics,
            role_by_context=role_by_context,
            minimum_semantic_count=minimum_maneuver_semantic_count,
            eps=eps,
        )
        fold_physiology, physiology_thresholds = _fit_physiology_fold(
            fold_id=fold_id,
            statistics=raw.physiology_statistics,
            role_by_context=role_by_context,
            train_valid_ratio=physiology_train_valid_ratio,
            eps=eps,
        )
        maneuver_rows.extend(fold_maneuver)
        physiology_rows.extend(fold_physiology)
        threshold_rows.extend(maneuver_thresholds)
        threshold_rows.extend(physiology_thresholds)
    result = SimpleFoldTargetBundle(
        fold_manifest=pd.DataFrame(fold_rows),
        maneuver_targets=pd.DataFrame(maneuver_rows),
        physiology_targets=pd.DataFrame(physiology_rows),
        threshold_rows=pd.DataFrame(threshold_rows),
    )
    validate_simple_fold_targets(result)
    return result


def validate_simple_fold_targets(bundle: SimpleFoldTargetBundle) -> None:
    folds = bundle.fold_manifest["fold_id"].astype(str).tolist()
    if len(folds) != 2 or len(set(folds)) != 2:
        raise ValueError("simple protocol must contain two unique LOSO folds")
    for fold_id in folds:
        maneuver = bundle.maneuver_targets[
            bundle.maneuver_targets["fold_id"].astype(str) == fold_id
        ]
        if len(maneuver) != 90 or maneuver["context_id"].duplicated().any():
            raise ValueError(f"{fold_id} maneuver targets do not cover 90 contexts")
        if maneuver["vehicle_context_id"].nunique() != 60:
            raise ValueError(f"{fold_id} maneuver targets do not cover 60 vehicle contexts")
        weight_sums = maneuver.groupby(["split_role", "vehicle_context_id"])[
            "sample_weight"
        ].sum()
        if not np.allclose(weight_sums.to_numpy(dtype=float), 1.0):
            raise ValueError(f"{fold_id} maneuver weights do not sum to one")
        physiology = bundle.physiology_targets[
            bundle.physiology_targets["fold_id"].astype(str) == fold_id
        ]
        selected = physiology[physiology["selected"].astype(bool)]
        if selected.empty or selected["context_id"].nunique() != 90:
            raise ValueError(f"{fold_id} selected physiology targets are incomplete")
        if not (
            maneuver["input_end_exclusive_ms"].to_numpy()
            == maneuver["target_start_offset_ms"].to_numpy()
        ).all():
            raise ValueError(f"{fold_id} input and target boundary mismatch")
        if not (
            maneuver["target_end_exclusive_ms"].to_numpy()
            - maneuver["target_start_offset_ms"].to_numpy()
            == 5_000
        ).all():
            raise ValueError(f"{fold_id} future target duration is not five seconds")


def protocol_payload() -> dict[str, object]:
    return {
        "format": "chronaris.simple_downstream_protocol.v1",
        "task_ids": [
            FUTURE_MANEUVER_SCORE_TASK_ID,
            FUTURE_MANEUVER_CLASS_TASK_ID,
            FUTURE_PHYSIOLOGY_FIELDS_TASK_ID,
        ],
        "methods": list(SIMPLE_DOWNSTREAM_METHODS),
        "history_duration_s": 30.0,
        "future_duration_s": 5.0,
        "expected_view_context_count": 90,
        "expected_vehicle_context_count": 60,
        "primary_split_strategy": PRIMARY_SPLIT_STRATEGY,
        "diagnostic_split_strategy": DIAGNOSTIC_SPLIT_STRATEGY,
        "representation_output_dim": 64,
        "query_point_count": 96,
        "maneuver_regression_alpha": 1.0,
        "maneuver_classification_c": 1.0,
        "physiology_regression_alpha": 1.0,
        "seeds": [17, 29, 43],
        "outer_result_driven_tuning_allowed": False,
        "end_to_end_retraining_allowed": False,
    }


def _complete_contexts(source: DingxinTargetSourceData) -> pd.DataFrame:
    contexts = source.contexts[source.contexts["response_eligible"].astype(bool)].copy()
    plans = {str(item["view_id"]): item for item in source.snapshot_manifest["plans"]}
    rows = []
    for row in contexts.itertuples(index=False):
        plan = plans[str(row.view_id)]
        start = datetime.fromisoformat(str(plan["start_utc"]))
        stop = datetime.fromisoformat(str(plan["stop_utc"]))
        stop_offset_ms = int(round((stop - start).total_seconds() * 1_000.0))
        target_start = int(row.end_offset_ms)
        target_end = target_start + 5_000
        if target_end > stop_offset_ms:
            continue
        rows.append(
            {
                "context_id": str(row.context_id),
                "sortie_id": str(row.sortie_id),
                "view_id": str(row.view_id),
                "pilot_id": int(row.pilot_id),
                "input_start_offset_ms": int(row.start_offset_ms),
                "input_end_exclusive_ms": target_start,
                "target_start_offset_ms": target_start,
                "target_end_exclusive_ms": target_end,
                "vehicle_context_id": build_vehicle_context_id(
                    str(row.sortie_id), target_start
                ),
                "snapshot_stop_offset_ms": stop_offset_ms,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sortie_id", "view_id", "target_start_offset_ms"], kind="mergesort"
    ).reset_index(drop=True)


def _extract_maneuver_statistics(source, contexts, root):
    roles = source.field_roles[
        source.field_roles["selected_for_maneuver_label"].astype(bool)
    ]
    roles_by_sortie: dict[str, dict[str, str]] = defaultdict(dict)
    for row in roles.itertuples(index=False):
        roles_by_sortie[str(row.sortie_id)][
            f"{row.measurement}.{row.source_field}"
        ] = str(row.semantic_key)
    plans = {}
    for item in source.snapshot_manifest["plans"]:
        plans.setdefault(str(item["sortie_id"]), item)
    files = {
        str(item["sortie_id"]): item
        for item in source.snapshot_manifest["files"]
        if item["stream_kind"] == "vehicle"
    }
    rows = []
    unique_contexts = contexts.drop_duplicates("vehicle_context_id")
    for sortie_id, frame in unique_contexts.groupby("sortie_id", sort=True):
        plan = plans[str(sortie_id)]
        start = datetime.fromisoformat(str(plan["start_utc"]))
        observed: dict[str, tuple[list[float], list[float]]] = {
            key: ([], []) for key in set(roles_by_sortie[str(sortie_id)].values())
        }
        file_record = files[str(sortie_id)]
        for point in iter_raw_point_snapshot(root / str(file_record["relative_path"])):
            offset_ms = (point.timestamp - start).total_seconds() * 1_000.0
            for field_name, raw_value in point.values.items():
                semantic = roles_by_sortie[str(sortie_id)].get(
                    f"{point.measurement}.{field_name}"
                )
                value = _finite_float(raw_value)
                if semantic is not None and value is not None:
                    observed[semantic][0].append(offset_ms)
                    observed[semantic][1].append(value)
        arrays = {
            key: (
                np.asarray(times, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
            for key, (times, values) in observed.items()
        }
        for context in frame.itertuples(index=False):
            current_bounds = (
                int(context.target_start_offset_ms) - 5_000,
                int(context.target_start_offset_ms),
            )
            future_bounds = (
                int(context.target_start_offset_ms),
                int(context.target_end_exclusive_ms),
            )
            for semantic_key, (times, values) in arrays.items():
                current = _window_summary(times, values, current_bounds)
                future = _window_summary(times, values, future_bounds)
                rows.append(
                    {
                        "vehicle_context_id": str(context.vehicle_context_id),
                        "sortie_id": str(sortie_id),
                        "target_start_offset_ms": int(context.target_start_offset_ms),
                        "semantic_key": semantic_key,
                        "current_count": current["count"],
                        "current_std": current["std"],
                        "current_abs_delta": current["abs_delta"],
                        "future_count": future["count"],
                        "future_std": future["std"],
                        "future_abs_delta": future["abs_delta"],
                        "snapshot_sha256": str(file_record["sha256"]),
                    }
                )
    return pd.DataFrame(rows)


def _extract_physiology_statistics(source, contexts, root):
    roles = source.field_roles[
        source.field_roles["selected_for_response_target"].astype(bool)
    ].drop_duplicates("feature_name")
    field_specs = {
        str(row.feature_name): (
            str(row.measurement),
            str(row.source_field),
            str(row.semantic_category),
        )
        for row in roles.itertuples(index=False)
    }
    plans = {str(item["view_id"]): item for item in source.snapshot_manifest["plans"]}
    files = {
        str(item["view_id"]): item
        for item in source.snapshot_manifest["files"]
        if item["stream_kind"] == "physiology"
    }
    rows = []
    for view_id, frame in contexts.groupby("view_id", sort=True):
        start = datetime.fromisoformat(str(plans[str(view_id)]["start_utc"]))
        file_record = files[str(view_id)]
        observed = {name: ([], []) for name in field_specs}
        for point in iter_raw_point_snapshot(root / str(file_record["relative_path"])):
            offset_ms = (point.timestamp - start).total_seconds() * 1_000.0
            for name, (measurement, source_field, _category) in field_specs.items():
                if point.measurement != measurement or source_field not in point.values:
                    continue
                value = _finite_float(point.values[source_field])
                if value is not None:
                    observed[name][0].append(offset_ms)
                    observed[name][1].append(value)
        arrays = {
            name: (
                np.asarray(times, dtype=np.float64),
                np.asarray(values, dtype=np.float64),
            )
            for name, (times, values) in observed.items()
        }
        for context in frame.itertuples(index=False):
            for field_name, (times, values) in arrays.items():
                current = _window_median(
                    times,
                    values,
                    (
                        int(context.target_start_offset_ms) - 5_000,
                        int(context.target_start_offset_ms),
                    ),
                )
                future = _window_median(
                    times,
                    values,
                    (
                        int(context.target_start_offset_ms),
                        int(context.target_end_exclusive_ms),
                    ),
                )
                rows.append(
                    {
                        "context_id": str(context.context_id),
                        "sortie_id": str(context.sortie_id),
                        "view_id": str(view_id),
                        "field_name": field_name,
                        "semantic_category": field_specs[field_name][2],
                        "current_count": current[0],
                        "current_median": current[1],
                        "future_count": future[0],
                        "future_median": future[1],
                        "snapshot_sha256": str(file_record["sha256"]),
                    }
                )
    return pd.DataFrame(rows)


def _fit_maneuver_fold(
    *, fold_id, contexts, statistics, role_by_context, minimum_semantic_count, eps
):
    vehicle_roles = contexts[["vehicle_context_id", "context_id"]].copy()
    vehicle_roles["split_role"] = vehicle_roles["context_id"].map(role_by_context)
    role_by_vehicle = vehicle_roles.groupby("vehicle_context_id")["split_role"].first()
    train_vehicle_ids = tuple(
        str(value) for value in role_by_vehicle[role_by_vehicle == "train"].index
    )
    train = statistics[statistics["vehicle_context_id"].isin(train_vehicle_ids)]
    scalers = {}
    thresholds = []
    for semantic_key, frame in train.groupby("semantic_key", sort=True):
        std_values = frame["future_std"].dropna().to_numpy(dtype=float)
        delta_values = frame["future_abs_delta"].dropna().to_numpy(dtype=float)
        if not len(std_values) or not len(delta_values):
            continue
        median_std, iqr_std = _median_iqr(std_values)
        median_delta, iqr_delta = _median_iqr(delta_values)
        selected = iqr_std > eps or iqr_delta > eps
        if selected:
            scalers[str(semantic_key)] = (
                median_std,
                iqr_std,
                median_delta,
                iqr_delta,
            )
        thresholds.append(
            {
                "fold_id": fold_id,
                "task_id": FUTURE_MANEUVER_SCORE_TASK_ID,
                "parameter_type": "semantic_scaler" if selected else "semantic_excluded",
                "parameter_name": str(semantic_key),
                "median_std": median_std,
                "iqr_std": iqr_std,
                "median_abs_delta": median_delta,
                "iqr_abs_delta": iqr_delta,
                "selected": selected,
                "fit_sample_hash": stable_sample_hash(train_vehicle_ids),
            }
        )
    if len(scalers) < minimum_semantic_count:
        raise ValueError(f"{fold_id} has only {len(scalers)} maneuver semantics")
    score_by_vehicle = {}
    for vehicle_id, frame in statistics.groupby("vehicle_context_id", sort=True):
        current = _maneuver_score_from_frame(
            frame, scalers, prefix="current", minimum=minimum_semantic_count, eps=eps
        )
        future = _maneuver_score_from_frame(
            frame, scalers, prefix="future", minimum=minimum_semantic_count, eps=eps
        )
        score_by_vehicle[str(vehicle_id)] = (current, future)
    train_scores = np.asarray(
        [score_by_vehicle[value][1] for value in train_vehicle_ids], dtype=float
    )
    if not np.isfinite(train_scores).all() or len(train_scores) < 3:
        raise ValueError(f"{fold_id} has invalid train maneuver scores")
    lower, upper = _strict_tertile_bounds(train_scores)
    score_iqr = _iqr(train_scores)
    thresholds.append(
        {
            "fold_id": fold_id,
            "task_id": FUTURE_MANEUVER_CLASS_TASK_ID,
            "parameter_type": "class_bounds",
            "parameter_name": "train_tertiles",
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "train_target_iqr": score_iqr,
            "selected": True,
            "fit_sample_hash": stable_sample_hash(train_vehicle_ids),
        }
    )
    duplicate_counts = contexts.groupby("vehicle_context_id")["context_id"].count()
    rows = []
    for context in contexts.itertuples(index=False):
        current, future = score_by_vehicle[str(context.vehicle_context_id)]
        rows.append(
            {
                "fold_id": fold_id,
                "score_task_id": FUTURE_MANEUVER_SCORE_TASK_ID,
                "class_task_id": FUTURE_MANEUVER_CLASS_TASK_ID,
                "split_role": role_by_context[str(context.context_id)],
                "context_id": str(context.context_id),
                "sortie_id": str(context.sortie_id),
                "view_id": str(context.view_id),
                "vehicle_context_id": str(context.vehicle_context_id),
                "sample_weight": 1.0
                / int(duplicate_counts.loc[str(context.vehicle_context_id)]),
                "current_maneuver_score": current,
                "future_maneuver_score": future,
                "future_maneuver_class": _bucketize(future, lower, upper),
                "train_target_iqr": score_iqr,
                "input_start_offset_ms": int(context.input_start_offset_ms),
                "input_end_exclusive_ms": int(context.input_end_exclusive_ms),
                "target_start_offset_ms": int(context.target_start_offset_ms),
                "target_end_exclusive_ms": int(context.target_end_exclusive_ms),
                "status": "completed",
            }
        )
    return rows, thresholds


def _fit_physiology_fold(
    *, fold_id, statistics, role_by_context, train_valid_ratio, eps
):
    frame = statistics.copy()
    frame["split_role"] = frame["context_id"].map(role_by_context)
    train = frame[frame["split_role"] == "train"]
    train_context_count = train["context_id"].nunique()
    field_contract = {}
    thresholds = []
    for field_name, values in train.groupby("field_name", sort=True):
        future = values["future_median"].to_numpy(dtype=float)
        valid = future[np.isfinite(future)]
        valid_ratio = len(valid) / max(train_context_count, 1)
        median = float(np.median(valid)) if len(valid) else None
        iqr = _iqr(valid) if len(valid) else None
        selected = bool(
            len(valid)
            and valid_ratio >= train_valid_ratio
            and iqr is not None
            and iqr > eps
        )
        if selected:
            field_contract[str(field_name)] = (float(median), float(iqr))
        thresholds.append(
            {
                "fold_id": fold_id,
                "task_id": FUTURE_PHYSIOLOGY_FIELDS_TASK_ID,
                "parameter_type": "field_scale" if selected else "field_excluded",
                "parameter_name": str(field_name),
                "train_median": median,
                "train_iqr": iqr,
                "train_valid_ratio": valid_ratio,
                "selected": selected,
                "fit_sample_hash": stable_sample_hash(
                    tuple(sorted(train["context_id"].astype(str).unique()))
                ),
            }
        )
    if len(field_contract) < 2:
        raise ValueError(f"{fold_id} has only {len(field_contract)} physiology fields")
    rows = []
    for row in frame.itertuples(index=False):
        selected = str(row.field_name) in field_contract
        center, scale = field_contract.get(str(row.field_name), (None, None))
        rows.append(
            {
                "fold_id": fold_id,
                "task_id": FUTURE_PHYSIOLOGY_FIELDS_TASK_ID,
                "split_role": str(row.split_role),
                "context_id": str(row.context_id),
                "sortie_id": str(row.sortie_id),
                "view_id": str(row.view_id),
                "field_name": str(row.field_name),
                "semantic_category": str(row.semantic_category),
                "selected": selected,
                "train_center": center,
                "train_scale": scale,
                "current_value": row.current_median,
                "future_value": row.future_median,
                "current_standardized": _standardize(row.current_median, center, scale),
                "future_standardized": _standardize(row.future_median, center, scale),
                "status": (
                    "completed"
                    if selected
                    and np.isfinite(row.current_median)
                    and np.isfinite(row.future_median)
                    else "field_excluded"
                    if not selected
                    else "missing_window_value"
                ),
            }
        )
    return rows, thresholds


def _maneuver_score_from_frame(frame, scalers, *, prefix, minimum, eps):
    values = []
    for row in frame.itertuples(index=False):
        scaler = scalers.get(str(row.semantic_key))
        std = getattr(row, f"{prefix}_std")
        delta = getattr(row, f"{prefix}_abs_delta")
        if scaler is None or not np.isfinite(std) or not np.isfinite(delta):
            continue
        median_std, iqr_std, median_delta, iqr_delta = scaler
        z_std = np.clip((float(std) - median_std) / (iqr_std + eps), -5.0, 5.0)
        z_delta = np.clip(
            (float(delta) - median_delta) / (iqr_delta + eps), -5.0, 5.0
        )
        values.append(max(0.0, float(z_std)) + max(0.0, float(z_delta)))
    if len(values) < minimum:
        raise ValueError(f"maneuver score has only {len(values)} valid semantics")
    return float(np.mean(values))


def _window_summary(times, values, bounds):
    mask = (times >= bounds[0]) & (times < bounds[1])
    selected = values[mask]
    if len(selected) < 2:
        return {"count": len(selected), "std": None, "abs_delta": None}
    return {
        "count": len(selected),
        "std": float(np.std(selected)),
        "abs_delta": float(abs(selected[-1] - selected[0])),
    }


def _window_median(times, values, bounds):
    mask = (times >= bounds[0]) & (times < bounds[1])
    selected = values[mask]
    return len(selected), None if not len(selected) else float(np.median(selected))


def _validate_raw_bundle(contexts, maneuver, physiology):
    if contexts["context_id"].duplicated().any():
        raise ValueError("simple target contexts are duplicated")
    if not (
        contexts["input_end_exclusive_ms"].to_numpy()
        == contexts["target_start_offset_ms"].to_numpy()
    ).all():
        raise ValueError("simple targets overlap their inputs")
    if maneuver["vehicle_context_id"].nunique() != contexts["vehicle_context_id"].nunique():
        raise ValueError("maneuver statistics do not cover every vehicle context")
    if physiology["context_id"].nunique() != len(contexts):
        raise ValueError("physiology statistics do not cover every view context")


def _bucketize(value, lower, upper):
    if value < lower:
        return 0
    if value < upper:
        return 1
    return 2


def _strict_tertile_bounds(values):
    """Choose two between-value boundaries with non-empty training classes."""

    array = np.sort(np.asarray(values, dtype=np.float64))
    unique = np.unique(array)
    if len(unique) < 3:
        raise ValueError("maneuver classification requires three distinct train scores")
    target = len(array) / 3.0
    best = None
    for lower_index in range(len(unique) - 2):
        lower = float((unique[lower_index] + unique[lower_index + 1]) / 2.0)
        low_count = int(np.searchsorted(array, lower, side="left"))
        for upper_index in range(lower_index + 1, len(unique) - 1):
            upper = float((unique[upper_index] + unique[upper_index + 1]) / 2.0)
            middle_stop = int(np.searchsorted(array, upper, side="left"))
            counts = (low_count, middle_stop - low_count, len(array) - middle_stop)
            if min(counts) <= 0:
                continue
            objective = sum((count - target) ** 2 for count in counts)
            candidate = (objective, lower, upper)
            if best is None or candidate < best:
                best = candidate
    if best is None:  # pragma: no cover - guarded by three distinct values.
        raise ValueError("unable to construct non-empty maneuver train classes")
    return float(best[1]), float(best[2])


def _standardize(value, center, scale):
    if value is None or center is None or scale is None:
        return None
    if not np.isfinite(value) or not np.isfinite(center) or not np.isfinite(scale):
        return None
    return float((value - center) / scale)


def _median_iqr(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), _iqr(array)


def _iqr(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.75) - np.quantile(array, 0.25))


def _finite_float(value):
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if np.isfinite(resolved) else None


def _require_count(label, actual, expected):
    if int(actual) != int(expected):
        raise ValueError(f"expected {expected} {label}s, got {actual}")
