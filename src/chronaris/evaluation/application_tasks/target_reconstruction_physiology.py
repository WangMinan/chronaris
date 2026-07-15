"""Robust physiology states and cross-fitted inertia-residual targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from chronaris.dataset.application_evaluation.contracts import stable_sample_hash
from chronaris.dataset.application_evaluation.snapshot_io import iter_raw_point_snapshot
from chronaris.evaluation.application_tasks.core_feasibility_features import (
    DingxinFeatureCache,
    summary_features,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    DingxinTargetSourceData,
)


def build_robust_physiology_states(
    source: DingxinTargetSourceData,
    *,
    snapshot_root: str | Path,
) -> pd.DataFrame:
    """Extract DC-robust EEG states and slope-aware slow physiology states."""

    root = Path(snapshot_root)
    contexts = source.contexts[source.contexts["response_eligible"].astype(bool)]
    roles = source.field_roles[source.field_roles["selected_for_response_target"].astype(bool)]
    field_specs = {
        str(row.feature_name): (
            str(row.measurement),
            str(row.source_field),
            str(row.semantic_category),
        )
        for row in roles.drop_duplicates("feature_name").itertuples(index=False)
    }
    plans = {str(item["view_id"]): item for item in source.snapshot_manifest["plans"]}
    files = {
        str(item["view_id"]): item
        for item in source.snapshot_manifest["files"]
        if item["stream_kind"] == "physiology"
    }
    rows = []
    for view_id, frame in contexts.groupby("view_id", sort=True):
        plan = plans[str(view_id)]
        start = datetime.fromisoformat(str(plan["start_utc"]))
        file_record = files[str(view_id)]
        observed = {name: ([], []) for name in field_specs}
        for point in iter_raw_point_snapshot(root / str(file_record["relative_path"])):
            offset_ms = (point.timestamp - start).total_seconds() * 1_000.0
            for name, (measurement, source_field, _category) in field_specs.items():
                if point.measurement != measurement or source_field not in point.values:
                    continue
                value = _finite(point.values[source_field])
                if value is not None:
                    observed[name][0].append(offset_ms)
                    observed[name][1].append(value)
        arrays = {
            name: (np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64))
            for name, (times, values) in observed.items()
        }
        for context in frame.itertuples(index=False):
            current_bounds = (int(context.end_offset_ms) - 5_000, int(context.end_offset_ms))
            future_bounds = (int(context.end_offset_ms), int(context.end_offset_ms) + 5_000)
            for field_name, (times, values) in arrays.items():
                category = field_specs[field_name][2]
                current = _window_values(times, values, current_bounds)
                future = _window_values(times, values, future_bounds)
                for descriptor, current_value in _descriptors(current, category).items():
                    future_value = _descriptors(future, category).get(descriptor)
                    rows.append(
                        {
                            "context_id": str(context.context_id),
                            "sortie_id": str(context.sortie_id),
                            "view_id": str(view_id),
                            "feature_name": field_name,
                            "semantic_category": category,
                            "descriptor": descriptor,
                            "descriptor_key": f"{field_name}::{descriptor}",
                            "current_value": current_value,
                            "future_value": future_value,
                            "current_count": len(current[1]),
                            "future_count": len(future[1]),
                            "status": (
                                "completed"
                                if current_value is not None and future_value is not None
                                else "insufficient_window_observations"
                            ),
                            "snapshot_sha256": str(file_record["sha256"]),
                        }
                    )
    return pd.DataFrame(rows)


def build_physiology_residual_targets(
    *,
    plans: Sequence[Mapping[str, object]],
    states: pd.DataFrame,
    cache: DingxinFeatureCache,
    contexts: pd.DataFrame,
    minimum_train_valid_ratio: float = 0.80,
    minimum_descriptor_count: int = 4,
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-fit a physiology-only inertia model and score unexplained response."""

    target_rows = []
    descriptor_rows = []
    audit_rows = []
    for plan in plans:
        split_id = str(plan["fold_id"])
        train_ids = tuple(str(value) for value in plan["train_sample_ids"])
        validation_ids = tuple(str(value) for value in plan["validation_sample_ids"])
        pivot_current, pivot_future = _state_matrices(states, train_ids + validation_ids)
        descriptor_keys = _select_descriptors(
            pivot_future.loc[list(train_ids)],
            minimum_train_valid_ratio=minimum_train_valid_ratio,
            eps=eps,
        )
        if len(descriptor_keys) < minimum_descriptor_count:
            raise ValueError(f"{split_id} has too few robust physiology descriptors")
        train_future, validation_future, fill = _filled_pair(
            pivot_future, train_ids, validation_ids, descriptor_keys
        )
        train_current, validation_current, _ = _filled_pair(
            pivot_current, train_ids, validation_ids, descriptor_keys, fill=fill
        )
        train_history = summary_features(
            cache, sample_ids=train_ids, modality="physiology", history_s=30.0
        )
        validation_history = summary_features(
            cache, sample_ids=validation_ids, modality="physiology", history_s=30.0
        )
        train_features = np.concatenate((train_history, train_current), axis=1)
        validation_features = np.concatenate((validation_history, validation_current), axis=1)
        train_prediction, purge_counts = _purged_oof_prediction(
            train_features,
            train_future,
            sample_ids=train_ids,
            contexts=contexts,
        )
        validation_prediction = _multioutput_predict(
            train_features, train_future, validation_features
        )
        train_residual = train_future - train_prediction
        validation_residual = validation_future - validation_prediction
        scale, reliability, selected = _residual_contract(
            train_future, train_residual, descriptor_keys, eps=eps
        )
        if int(selected.sum()) < minimum_descriptor_count:
            raise ValueError(f"{split_id} has too few stable residual descriptors")
        selected_keys = tuple(np.asarray(descriptor_keys)[selected])
        train_score = _aggregate_residual(train_residual[:, selected], scale[selected], reliability[selected])
        validation_score = _aggregate_residual(
            validation_residual[:, selected], scale[selected], reliability[selected]
        )
        high_threshold = float(np.quantile(train_score, 0.75))
        fit_hash = stable_sample_hash(train_ids)
        for role, ids, score in (
            ("train", train_ids, train_score),
            ("validation", validation_ids, validation_score),
        ):
            for sample_id, value in zip(ids, score, strict=True):
                target_rows.append(
                    {
                        "split_id": split_id,
                        "outer_pool_id": str(plan["outer_pool_id"]),
                        "split_kind": str(plan["split_kind"]),
                        "main_selection": bool(plan["main_selection"]),
                        "role": role,
                        "context_id": sample_id,
                        "residual_response": float(value),
                        "high_residual_response": int(value >= high_threshold),
                        "high_residual_threshold": high_threshold,
                        "selected_descriptor_count": int(selected.sum()),
                        "fit_sample_hash": fit_hash,
                        "status": "completed",
                        "outer_test_opened": False,
                    }
                )
        for index, key in enumerate(descriptor_keys):
            descriptor_rows.append(
                {
                    "split_id": split_id,
                    "descriptor_key": key,
                    "selected": bool(selected[index]),
                    "residual_iqr_scale": float(scale[index]),
                    "reliability_weight": float(reliability[index]),
                    "train_future_iqr": float(_iqr(train_future[:, index])),
                    "train_oof_rmse": float(np.sqrt(np.mean(train_residual[:, index] ** 2))),
                    "fit_sample_hash": fit_hash,
                }
            )
        audit_rows.append(
            {
                "split_id": split_id,
                "outer_pool_id": str(plan["outer_pool_id"]),
                "main_selection": bool(plan["main_selection"]),
                "candidate_descriptor_count": len(descriptor_keys),
                "selected_descriptor_count": len(selected_keys),
                "selected_descriptors": "|".join(selected_keys),
                "minimum_oof_fit_count": int(min(purge_counts)),
                "median_oof_fit_count": float(np.median(purge_counts)),
                "cross_fitted_train_predictions": True,
                "validation_future_used_for_fit": False,
                "outer_test_opened": False,
            }
        )
    return pd.DataFrame(target_rows), pd.DataFrame(descriptor_rows), pd.DataFrame(audit_rows)


def _window_values(times, values, bounds):
    mask = (times >= bounds[0]) & (times < bounds[1])
    return times[mask] / 1_000.0, values[mask]


def _descriptors(window, category):
    times, values = window
    minimum = 8 if category == "eeg" else 3
    if len(values) < minimum or not np.isfinite(values).all():
        names = ("centered_rms", "mad", "line_length", "spectral_entropy") if category == "eeg" else ("median", "slope", "mad")
        return {name: None for name in names}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if category == "eeg":
        centered = values - median
        return {
            "centered_rms": float(np.sqrt(np.mean(centered**2))),
            "mad": mad,
            "line_length": float(np.median(np.abs(np.diff(values)))),
            "spectral_entropy": _spectral_entropy(centered),
        }
    centered_t = times - float(np.mean(times))
    denominator = float(np.dot(centered_t, centered_t))
    slope = 0.0 if denominator <= 1e-12 else float(np.dot(centered_t, values - np.mean(values)) / denominator)
    return {"median": median, "slope": slope, "mad": mad}


def _spectral_entropy(centered):
    power = np.abs(np.fft.rfft(np.asarray(centered, dtype=np.float64))) ** 2
    power = power[1:]
    total = float(power.sum())
    if total <= 1e-12 or len(power) <= 1:
        return 0.0
    probability = power / total
    return float(-np.sum(probability * np.log(probability + 1e-12)) / np.log(len(power)))


def _state_matrices(states, sample_ids):
    subset = states[states["context_id"].isin(sample_ids)].copy()
    current = subset.pivot(index="context_id", columns="descriptor_key", values="current_value")
    future = subset.pivot(index="context_id", columns="descriptor_key", values="future_value")
    return current.reindex(sample_ids), future.reindex(sample_ids)


def _select_descriptors(train_future, *, minimum_train_valid_ratio, eps):
    selected = []
    for key in train_future.columns:
        values = train_future[key].to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        if len(valid) / max(len(values), 1) >= minimum_train_valid_ratio and _iqr(valid) > eps:
            selected.append(str(key))
    return tuple(selected)


def _filled_pair(frame, train_ids, validation_ids, columns, fill=None):
    train = frame.loc[list(train_ids), list(columns)].to_numpy(dtype=np.float64)
    validation = frame.loc[list(validation_ids), list(columns)].to_numpy(dtype=np.float64)
    if fill is None:
        fill = np.nanmedian(train, axis=0)
    return np.where(np.isfinite(train), train, fill), np.where(np.isfinite(validation), validation, fill), fill


def _purged_oof_prediction(features, target, *, sample_ids, contexts):
    lookup = contexts.set_index("context_id")
    predictions = np.empty_like(target, dtype=np.float64)
    counts = []
    for index, sample_id in enumerate(sample_ids):
        held = lookup.loc[str(sample_id)]
        fit = []
        for other_index, other_id in enumerate(sample_ids):
            if other_index == index:
                continue
            other = lookup.loc[str(other_id)]
            separated = (
                str(other["view_id"]) != str(held["view_id"])
                or abs(int(other["start_offset_ms"]) - int(held["start_offset_ms"])) >= 35_000
            )
            if separated:
                fit.append(other_index)
        if len(fit) < 4:
            fit = [value for value in range(len(sample_ids)) if value != index]
        predictions[index] = _multioutput_predict(features[fit], target[fit], features[index : index + 1])[0]
        counts.append(len(fit))
    return predictions, counts


def _multioutput_predict(train, target, validation):
    left, right, keep = _drop_constant_columns(train, validation)
    if not np.any(keep):
        return np.repeat(np.mean(target, axis=0, keepdims=True), len(validation), axis=0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(left, target)
    return np.asarray(model.predict(right), dtype=np.float64)


def _drop_constant_columns(train, validation):
    left = np.asarray(train, dtype=np.float64).reshape(len(train), -1)
    right = np.asarray(validation, dtype=np.float64).reshape(len(validation), -1)
    keep = np.isfinite(left).all(axis=0) & np.isfinite(right).all(axis=0) & (np.ptp(left, axis=0) > 1e-10)
    return left[:, keep], right[:, keep], keep


def _residual_contract(future, residual, keys, *, eps):
    scale = np.asarray([_iqr(np.abs(residual[:, index])) for index in range(residual.shape[1])])
    future_iqr = np.asarray([_iqr(future[:, index]) for index in range(future.shape[1])])
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    reliability = 1.0 / (1.0 + rmse / np.maximum(future_iqr, eps))
    selected = np.isfinite(scale) & (scale > eps) & np.isfinite(reliability)
    return scale, reliability, selected


def _aggregate_residual(residual, scale, reliability):
    standardized = np.clip(np.abs(residual) / np.maximum(scale, 1e-6), 0.0, 10.0)
    return np.asarray([_weighted_median(row, reliability) for row in standardized])


def _weighted_median(values, weights):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _iqr(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None
