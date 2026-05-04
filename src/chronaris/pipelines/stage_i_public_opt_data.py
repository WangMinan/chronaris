"""Feature-frame builder for the Stage I public-opt UAB regression path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset import StageISequenceBundle, StageISequenceEntry

PUBLIC_OPT_DATASET_ID = "uab_workload_dataset"
PUBLIC_OPT_PROFILE = "window_v2"
PUBLIC_OPT_SUBSET_ORDER = ("n_back", "heat_the_chair")
PUBLIC_OPT_REQUIRED_MODALITIES = ("physiology", "task_context")
PUBLIC_OPT_HEAD_FEATURES = {
    "physiology_persistence": ("residual__physiology_intensity_mean",),
    "ridge_residual": (
        "ctx__sequence_length_steps",
        "ctx__physiology_valid_ratio",
        "ctx__task_context_valid_ratio",
        "residual__physiology_intensity_mean",
        "residual__physiology_delta_l2",
        "residual__physiology_diff_l2_mean",
        "residual__task_context_intensity_mean",
    ),
}


@dataclass(frozen=True, slots=True)
class StageIPublicOptFeatureFrameResult:
    """Materialized public-opt feature table plus column groups."""

    feature_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    head_feature_columns: Mapping[str, tuple[str, ...]]
    subset_order: tuple[str, ...]


def build_stage_i_public_opt_feature_frame(
    entries: Sequence[StageISequenceEntry],
    bundle: StageISequenceBundle,
    *,
    dataset_id: str,
    profile: str,
) -> StageIPublicOptFeatureFrameResult:
    """Build one sample-level feature frame from prepared Stage I sequence assets."""

    _validate_public_opt_inputs(entries=entries, bundle=bundle, dataset_id=dataset_id, profile=profile)
    physiology_feature_names = _feature_names(entries, "physiology")
    task_context_feature_names = _feature_names(entries, "task_context")
    rows: list[dict[str, object]] = []
    feature_columns: list[str] = []
    feature_column_seen: set[str] = set()
    entry_index_by_sample_id = {
        sample_id: index
        for index, sample_id in enumerate(bundle.sample_ids)
    }

    for entry in entries:
        if not _should_include_entry(entry):
            continue
        bundle_index = entry_index_by_sample_id.get(entry.sample_id)
        if bundle_index is None:
            raise ValueError(f"prepared bundle is missing sample_id={entry.sample_id}")

        physiology_values = np.asarray(
            bundle.modality_arrays["physiology"][bundle_index],
            dtype=np.float32,
        )
        physiology_mask = np.asarray(
            bundle.modality_masks["physiology"][bundle_index],
            dtype=np.uint8,
        )
        task_context_values = np.asarray(
            bundle.modality_arrays["task_context"][bundle_index],
            dtype=np.float32,
        )
        task_context_mask = np.asarray(
            bundle.modality_masks["task_context"][bundle_index],
            dtype=np.uint8,
        )

        row: dict[str, object] = {
            "sample_id": entry.sample_id,
            "dataset_id": entry.dataset_id,
            "profile": profile,
            "subset_id": entry.subset_id,
            "subject_id": entry.subject_id,
            "split_group": entry.split_group,
            "session_id": entry.session_id,
            "window_index": int(entry.window_index or 0),
            "y_true": float(entry.subjective_target_value),
        }
        _append_modality_stats(
            row=row,
            feature_columns=feature_columns,
            feature_column_seen=feature_column_seen,
            prefix="physiology",
            feature_names=physiology_feature_names,
            values=physiology_values,
            mask=physiology_mask,
        )
        _append_modality_stats(
            row=row,
            feature_columns=feature_columns,
            feature_column_seen=feature_column_seen,
            prefix="task_context",
            feature_names=task_context_feature_names,
            values=task_context_values,
            mask=task_context_mask,
        )
        explicit_scalars = _build_explicit_scalar_features(
            entry=entry,
            physiology_values=physiology_values,
            physiology_mask=physiology_mask,
            task_context_values=task_context_values,
            task_context_mask=task_context_mask,
        )
        row.update(explicit_scalars)
        for feature_name in explicit_scalars:
            if feature_name not in feature_column_seen:
                feature_columns.append(feature_name)
                feature_column_seen.add(feature_name)
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ValueError("public opt feature frame is empty after filtering primary UAB subjective windows.")
    feature_frame = feature_frame.sort_values(
        ["subset_id", "split_group", "sample_id"],
    ).reset_index(drop=True)
    return StageIPublicOptFeatureFrameResult(
        feature_frame=feature_frame,
        feature_columns=tuple(feature_columns),
        head_feature_columns={
            head_name: tuple(columns)
            for head_name, columns in PUBLIC_OPT_HEAD_FEATURES.items()
        },
        subset_order=PUBLIC_OPT_SUBSET_ORDER,
    )


def _validate_public_opt_inputs(
    *,
    entries: Sequence[StageISequenceEntry],
    bundle: StageISequenceBundle,
    dataset_id: str,
    profile: str,
) -> None:
    if dataset_id != PUBLIC_OPT_DATASET_ID:
        raise ValueError(
            f"public opt only supports dataset_id={PUBLIC_OPT_DATASET_ID}, got {dataset_id}"
        )
    if profile != PUBLIC_OPT_PROFILE:
        raise ValueError(
            f"public opt only supports profile={PUBLIC_OPT_PROFILE}, got {profile}"
        )
    if not entries:
        raise ValueError("public opt requires non-empty prepared sequence entries.")
    if tuple(bundle.sample_ids) != tuple(entry.sample_id for entry in entries):
        raise ValueError("prepared sequence entries must align with sequence bundle sample order.")
    entry_modalities = tuple(entries[0].modality_schema)
    if entry_modalities != PUBLIC_OPT_REQUIRED_MODALITIES:
        raise ValueError(
            "public opt expects Chronaris-public UAB modalities "
            f"{PUBLIC_OPT_REQUIRED_MODALITIES}, got {entry_modalities}. "
            "Re-prepare the UAB sequence assets with the current Chronaris-public adapter."
        )
    if tuple(bundle.modality_arrays) != PUBLIC_OPT_REQUIRED_MODALITIES:
        raise ValueError(
            "public opt expects bundle modalities "
            f"{PUBLIC_OPT_REQUIRED_MODALITIES}, got {tuple(bundle.modality_arrays)}. "
            "Re-prepare the UAB sequence assets with the current Chronaris-public adapter."
        )


def _should_include_entry(entry: StageISequenceEntry) -> bool:
    return (
        entry.dataset_id == PUBLIC_OPT_DATASET_ID
        and entry.training_role == "primary"
        and entry.subset_id in PUBLIC_OPT_SUBSET_ORDER
        and entry.subset_id != "flight_simulator"
        and entry.subjective_target_value is not None
    )


def _feature_names(
    entries: Sequence[StageISequenceEntry],
    modality_name: str,
) -> tuple[str, ...]:
    schema = entries[0].modality_schema.get(modality_name)
    if not isinstance(schema, Mapping):
        raise ValueError(f"missing modality schema for {modality_name}")
    feature_names = tuple(str(value) for value in schema.get("feature_names", ()))
    if not feature_names:
        raise ValueError(f"modality {modality_name} is missing feature_names")
    return feature_names


def _append_modality_stats(
    *,
    row: dict[str, object],
    feature_columns: list[str],
    feature_column_seen: set[str],
    prefix: str,
    feature_names: Sequence[str],
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    stats_by_feature = _compute_masked_feature_stats(values=values, mask=mask)
    stat_names = ("mean", "std", "min", "max", "delta")
    for feature_index, feature_name in enumerate(feature_names):
        for stat_name, stat_values in zip(stat_names, stats_by_feature, strict=True):
            column_name = f"{prefix}__{feature_name}__{stat_name}"
            row[column_name] = float(stat_values[feature_index])
            if column_name not in feature_column_seen:
                feature_columns.append(column_name)
                feature_column_seen.add(column_name)


def _compute_masked_feature_stats(
    *,
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_dim = int(values.shape[-1])
    active = np.flatnonzero(mask.astype(bool))
    if active.size == 0:
        zeros = np.zeros((feature_dim,), dtype=np.float32)
        return zeros, zeros, zeros, zeros, zeros

    valid_values = np.asarray(values[active], dtype=np.float32)
    valid_values = np.nan_to_num(valid_values, nan=0.0, posinf=0.0, neginf=0.0)
    means = valid_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = valid_values.std(axis=0, dtype=np.float64).astype(np.float32)
    mins = valid_values.min(axis=0).astype(np.float32)
    maxs = valid_values.max(axis=0).astype(np.float32)
    deltas = (valid_values[-1] - valid_values[0]).astype(np.float32)
    return means, stds, mins, maxs, deltas


def _build_explicit_scalar_features(
    *,
    entry: StageISequenceEntry,
    physiology_values: np.ndarray,
    physiology_mask: np.ndarray,
    task_context_values: np.ndarray,
    task_context_mask: np.ndarray,
) -> dict[str, float]:
    return {
        "ctx__sequence_length_steps": float(entry.sequence_length),
        "ctx__physiology_valid_ratio": _valid_ratio(physiology_mask),
        "ctx__task_context_valid_ratio": _valid_ratio(task_context_mask),
        "residual__physiology_intensity_mean": _mean_abs_intensity(
            physiology_values,
            physiology_mask,
        ),
        "residual__physiology_delta_l2": _delta_l2(physiology_values, physiology_mask),
        "residual__physiology_diff_l2_mean": _diff_l2_mean(
            physiology_values,
            physiology_mask,
        ),
        "residual__task_context_intensity_mean": _mean_abs_intensity(
            task_context_values,
            task_context_mask,
        ),
    }


def _valid_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float32)))


def _mean_abs_intensity(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.flatnonzero(mask.astype(bool))
    if active.size == 0:
        return 0.0
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.mean(np.abs(valid_values), dtype=np.float64))


def _delta_l2(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.flatnonzero(mask.astype(bool))
    if active.size == 0:
        return 0.0
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(valid_values[-1] - valid_values[0], ord=2))


def _diff_l2_mean(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.flatnonzero(mask.astype(bool))
    if active.size <= 1:
        return 0.0
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    diffs = np.diff(valid_values, axis=0)
    if diffs.size == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(diffs, ord=2, axis=1), dtype=np.float64))
