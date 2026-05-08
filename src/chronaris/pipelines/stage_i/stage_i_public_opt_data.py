"""Feature-frame builders for Stage I public-opt runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset import StageISequenceBundle, StageISequenceEntry

PUBLIC_OPT_DEFAULT_DATASET_ID = "uab_workload_dataset"
PUBLIC_OPT_PROFILE = "window_v2"
PUBLIC_OPT_SUBSET_ORDER = ("n_back", "heat_the_chair")
PUBLIC_OPT_REQUIRED_MODALITIES = ("physiology", "task_context")
PUBLIC_OPT_FEATURE_PROFILES = (
    "full",
    "physiology_only",
    "physiology_lowdim",
    "physiology_scalar_only",
    "context_only",
    "residual_only",
)

PUBLIC_OPT_DATASET_SPECS: Mapping[str, Mapping[str, object]] = {
    "uab_workload_dataset": {
        "dataset_id": "uab_workload_dataset",
        "profile": PUBLIC_OPT_PROFILE,
        "task_type": "regression",
        "track": "subjective",
        "required_modalities": ("physiology", "task_context"),
        "subset_order": ("n_back", "heat_the_chair"),
        "source_subset_ids": ("n_back", "heat_the_chair"),
        "evaluation_groups": {
            "n_back": ("n_back",),
            "heat_the_chair": ("heat_the_chair",),
        },
        "label_order": None,
    },
    "nasa_csm": {
        "dataset_id": "nasa_csm",
        "profile": PUBLIC_OPT_PROFILE,
        "task_type": "classification",
        "track": "objective",
        "required_modalities": ("physiology", "scenario_context"),
        "subset_order": ("benchmark_only", "loft_only", "combined"),
        "source_subset_ids": ("benchmark", "loft"),
        "evaluation_groups": {
            "benchmark_only": ("benchmark",),
            "loft_only": ("loft",),
            "combined": ("benchmark", "loft"),
        },
        "label_order": (1, 2, 5),
    },
}

PUBLIC_OPT_DATASET_ALIASES = {
    "uab": "uab_workload_dataset",
    "uab_workload_dataset": "uab_workload_dataset",
    "nasa": "nasa_csm",
    "nasa_csm": "nasa_csm",
}


@dataclass(frozen=True, slots=True)
class StageIPublicOptFeatureFrameResult:
    """Materialized public-opt feature table plus dataset-specific metadata."""

    dataset_id: str
    profile: str
    task_type: str
    track: str
    label_order: tuple[int | float, ...] | None
    ordered_modalities: tuple[str, str]
    feature_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    head_feature_columns: Mapping[str, tuple[str, ...]]
    feature_groups: Mapping[str, tuple[str, ...]]
    subset_order: tuple[str, ...]
    evaluation_groups: Mapping[str, tuple[str, ...]]


def normalize_public_opt_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id.strip().lower()
    return PUBLIC_OPT_DATASET_ALIASES.get(normalized, normalized)


def get_public_opt_spec(dataset_id: str) -> Mapping[str, object]:
    normalized = normalize_public_opt_dataset_id(dataset_id)
    if normalized not in PUBLIC_OPT_DATASET_SPECS:
        raise ValueError(f"unsupported public opt dataset: {dataset_id}")
    return PUBLIC_OPT_DATASET_SPECS[normalized]


def build_stage_i_public_opt_feature_frame(
    entries: Sequence[StageISequenceEntry],
    bundle: StageISequenceBundle,
    *,
    dataset_id: str,
    profile: str,
) -> StageIPublicOptFeatureFrameResult:
    """Build one sample-level feature frame from prepared Stage I sequence assets."""

    spec = get_public_opt_spec(dataset_id)
    canonical_dataset_id = str(spec["dataset_id"])
    _validate_public_opt_inputs(
        entries=entries,
        bundle=bundle,
        dataset_id=canonical_dataset_id,
        profile=profile,
    )
    required_modalities = tuple(str(value) for value in spec["required_modalities"])
    first_modality, second_modality = required_modalities
    first_feature_names = _feature_names(entries, first_modality)
    second_feature_names = _feature_names(entries, second_modality)
    session_window_context = _build_session_window_context(entries, spec=spec)
    rows: list[dict[str, object]] = []
    feature_columns: list[str] = []
    feature_column_seen: set[str] = set()
    entry_index_by_sample_id = {
        sample_id: index
        for index, sample_id in enumerate(bundle.sample_ids)
    }

    for entry in entries:
        if not _should_include_entry(entry, spec=spec):
            continue
        bundle_index = entry_index_by_sample_id.get(entry.sample_id)
        if bundle_index is None:
            raise ValueError(f"prepared bundle is missing sample_id={entry.sample_id}")

        first_values = np.asarray(
            bundle.modality_arrays[first_modality][bundle_index],
            dtype=np.float32,
        )
        first_mask = np.asarray(
            bundle.modality_masks[first_modality][bundle_index],
            dtype=np.uint8,
        )
        second_values = np.asarray(
            bundle.modality_arrays[second_modality][bundle_index],
            dtype=np.float32,
        )
        second_mask = np.asarray(
            bundle.modality_masks[second_modality][bundle_index],
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
            "y_true": _target_value(entry, spec=spec),
        }
        _append_modality_stats(
            row=row,
            feature_columns=feature_columns,
            feature_column_seen=feature_column_seen,
            prefix=first_modality,
            feature_names=first_feature_names,
            values=first_values,
            mask=first_mask,
        )
        _append_modality_stats(
            row=row,
            feature_columns=feature_columns,
            feature_column_seen=feature_column_seen,
            prefix=second_modality,
            feature_names=second_feature_names,
            values=second_values,
            mask=second_mask,
        )
        explicit_scalars = _build_explicit_scalar_features(
            entry=entry,
            first_modality_name=first_modality,
            first_values=first_values,
            first_mask=first_mask,
            second_modality_name=second_modality,
            second_values=second_values,
            second_mask=second_mask,
            max_window_index=session_window_context.get(_window_context_key(entry), 0),
        )
        row.update(explicit_scalars)
        for feature_name in explicit_scalars:
            if feature_name not in feature_column_seen:
                feature_columns.append(feature_name)
                feature_column_seen.add(feature_name)
        segment_features = _build_segment_scalar_features(
            first_modality_name=first_modality,
            first_values=first_values,
            first_mask=first_mask,
            second_modality_name=second_modality,
            second_values=second_values,
            second_mask=second_mask,
        )
        row.update(segment_features)
        for feature_name in segment_features:
            if feature_name not in feature_column_seen:
                feature_columns.append(feature_name)
                feature_column_seen.add(feature_name)
        cross_modal_features = _build_cross_modal_scalar_features(
            first_modality_name=first_modality,
            first_values=first_values,
            first_mask=first_mask,
            second_modality_name=second_modality,
            second_values=second_values,
            second_mask=second_mask,
        )
        row.update(cross_modal_features)
        for feature_name in cross_modal_features:
            if feature_name not in feature_column_seen:
                feature_columns.append(feature_name)
                feature_column_seen.add(feature_name)
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ValueError(
            f"public opt feature frame is empty after filtering primary {canonical_dataset_id} entries."
        )
    feature_frame = feature_frame.sort_values(
        ["subset_id", "split_group", "sample_id"],
    ).reset_index(drop=True)
    feature_groups = _build_feature_groups(
        feature_columns=feature_columns,
        first_modality_name=first_modality,
        second_modality_name=second_modality,
    )
    return StageIPublicOptFeatureFrameResult(
        dataset_id=canonical_dataset_id,
        profile=str(spec["profile"]),
        task_type=str(spec["task_type"]),
        track=str(spec["track"]),
        label_order=(
            tuple(spec["label_order"]) if spec["label_order"] is not None else None
        ),
        ordered_modalities=(first_modality, second_modality),
        feature_frame=feature_frame,
        feature_columns=tuple(feature_columns),
        head_feature_columns=_build_default_head_feature_columns(
            dataset_id=canonical_dataset_id,
            feature_groups=feature_groups,
        ),
        feature_groups=feature_groups,
        subset_order=tuple(str(value) for value in spec["subset_order"]),
        evaluation_groups={
            group_name: tuple(str(value) for value in subset_ids)
            for group_name, subset_ids in dict(spec["evaluation_groups"]).items()
        },
    )


def _validate_public_opt_inputs(
    *,
    entries: Sequence[StageISequenceEntry],
    bundle: StageISequenceBundle,
    dataset_id: str,
    profile: str,
) -> None:
    spec = get_public_opt_spec(dataset_id)
    if profile != str(spec["profile"]):
        raise ValueError(
            f"public opt only supports profile={spec['profile']}, got {profile}"
        )
    if not entries:
        raise ValueError("public opt requires non-empty prepared sequence entries.")
    if tuple(bundle.sample_ids) != tuple(entry.sample_id for entry in entries):
        raise ValueError("prepared sequence entries must align with sequence bundle sample order.")
    required_modalities = tuple(str(value) for value in spec["required_modalities"])
    entry_modalities = tuple(entries[0].modality_schema)
    if entry_modalities != required_modalities:
        raise ValueError(
            "public opt expects Chronaris-public modalities "
            f"{required_modalities}, got {entry_modalities}. "
            "Re-prepare the sequence assets with the current Chronaris-public adapter."
        )
    if tuple(bundle.modality_arrays) != required_modalities:
        raise ValueError(
            "public opt expects bundle modalities "
            f"{required_modalities}, got {tuple(bundle.modality_arrays)}. "
            "Re-prepare the sequence assets with the current Chronaris-public adapter."
        )


def _should_include_entry(
    entry: StageISequenceEntry,
    *,
    spec: Mapping[str, object],
) -> bool:
    source_subset_ids = {
        str(value)
        for value in tuple(spec["source_subset_ids"])
    }
    if entry.dataset_id != str(spec["dataset_id"]):
        return False
    if entry.training_role != "primary":
        return False
    if entry.subset_id not in source_subset_ids:
        return False
    if str(spec["task_type"]) == "regression":
        return entry.subjective_target_value is not None
    return entry.objective_label_value is not None


def _target_value(
    entry: StageISequenceEntry,
    *,
    spec: Mapping[str, object],
) -> float | int:
    if str(spec["task_type"]) == "regression":
        if entry.subjective_target_value is None:
            raise ValueError(f"missing subjective target for sample_id={entry.sample_id}")
        return float(entry.subjective_target_value)
    if entry.objective_label_value is None:
        raise ValueError(f"missing objective label for sample_id={entry.sample_id}")
    return int(entry.objective_label_value)


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


def _build_session_window_context(
    entries: Sequence[StageISequenceEntry],
    *,
    spec: Mapping[str, object],
) -> dict[tuple[str, str, str], int]:
    context: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        if not _should_include_entry(entry, spec=spec):
            continue
        key = _window_context_key(entry)
        context[key] = max(context.get(key, 0), int(entry.window_index or 0))
    return context


def _window_context_key(entry: StageISequenceEntry) -> tuple[str, str, str]:
    return (
        str(entry.dataset_id),
        str(entry.session_id),
        str(entry.split_group),
    )


def _build_explicit_scalar_features(
    *,
    entry: StageISequenceEntry,
    first_modality_name: str,
    first_values: np.ndarray,
    first_mask: np.ndarray,
    second_modality_name: str,
    second_values: np.ndarray,
    second_mask: np.ndarray,
    max_window_index: int,
) -> dict[str, float]:
    return {
        "ctx__sequence_length_steps": float(entry.sequence_length),
        "ctx__window_fraction": _window_fraction(entry, max_window_index=max_window_index),
        f"ctx__{first_modality_name}_valid_ratio": _valid_ratio(first_mask),
        f"ctx__{second_modality_name}_valid_ratio": _valid_ratio(second_mask),
        f"residual__{first_modality_name}_intensity_mean": _mean_abs_intensity(
            first_values,
            first_mask,
        ),
        f"residual__{first_modality_name}_delta_l2": _delta_l2(first_values, first_mask),
        f"residual__{first_modality_name}_diff_l2_mean": _diff_l2_mean(
            first_values,
            first_mask,
        ),
        f"residual__{second_modality_name}_intensity_mean": _mean_abs_intensity(
            second_values,
            second_mask,
        ),
        f"residual__{second_modality_name}_delta_l2": _delta_l2(second_values, second_mask),
        f"residual__{second_modality_name}_diff_l2_mean": _diff_l2_mean(
            second_values,
            second_mask,
        ),
        f"residual__{second_modality_name}_peak_rate": _peak_rate(second_values, second_mask),
        f"residual__{second_modality_name}_jump_rate": _jump_rate(second_values, second_mask),
    }


def _valid_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.mean(mask.astype(np.float32)))


def _window_fraction(entry: StageISequenceEntry, *, max_window_index: int) -> float:
    if max_window_index <= 0:
        return 0.0
    return float(int(entry.window_index or 0) / float(max_window_index))


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


def _peak_rate(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.flatnonzero(mask.astype(bool))
    if active.size == 0:
        return 0.0
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    amplitudes = np.linalg.norm(valid_values, ord=2, axis=1)
    if amplitudes.size == 0:
        return 0.0
    threshold = float(np.mean(amplitudes, dtype=np.float64) + np.std(amplitudes, dtype=np.float64))
    if threshold <= 0:
        return 0.0
    return float(np.mean(amplitudes >= threshold, dtype=np.float64))


def _jump_rate(values: np.ndarray, mask: np.ndarray) -> float:
    active = np.flatnonzero(mask.astype(bool))
    if active.size <= 1:
        return 0.0
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    diffs = np.linalg.norm(np.diff(valid_values, axis=0), ord=2, axis=1)
    if diffs.size == 0:
        return 0.0
    threshold = float(np.mean(diffs, dtype=np.float64) + np.std(diffs, dtype=np.float64))
    if threshold <= 0:
        return 0.0
    return float(np.mean(diffs >= threshold, dtype=np.float64))


def _build_segment_scalar_features(
    *,
    first_modality_name: str,
    first_values: np.ndarray,
    first_mask: np.ndarray,
    second_modality_name: str,
    second_values: np.ndarray,
    second_mask: np.ndarray,
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for modality_name, values, mask in (
        (first_modality_name, first_values, first_mask),
        (second_modality_name, second_values, second_mask),
    ):
        segment_stats = _segment_profile(values, mask)
        for segment_name, payload in segment_stats.items():
            rows[f"segment__{modality_name}__{segment_name}__intensity_mean"] = payload[
                "intensity_mean"
            ]
            rows[f"segment__{modality_name}__{segment_name}__diff_l2_mean"] = payload[
                "diff_l2_mean"
            ]
        rows[f"segment__{modality_name}__late_minus_early_intensity"] = (
            segment_stats["late"]["intensity_mean"] - segment_stats["early"]["intensity_mean"]
        )
        rows[f"segment__{modality_name}__late_minus_early_diff_l2"] = (
            segment_stats["late"]["diff_l2_mean"] - segment_stats["early"]["diff_l2_mean"]
        )
    return rows


def _segment_profile(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, dict[str, float]]:
    active = np.flatnonzero(mask.astype(bool))
    if active.size == 0:
        zero = {"intensity_mean": 0.0, "diff_l2_mean": 0.0}
        return {"early": dict(zero), "middle": dict(zero), "late": dict(zero)}
    valid_values = np.nan_to_num(values[active], nan=0.0, posinf=0.0, neginf=0.0)
    parts = np.array_split(valid_values, 3)
    result: dict[str, dict[str, float]] = {}
    for segment_name, segment_values in zip(("early", "middle", "late"), parts, strict=True):
        if len(segment_values) == 0:
            result[segment_name] = {"intensity_mean": 0.0, "diff_l2_mean": 0.0}
            continue
        intensity_mean = float(np.mean(np.abs(segment_values), dtype=np.float64))
        diffs = np.diff(segment_values, axis=0)
        diff_l2_mean = (
            float(np.mean(np.linalg.norm(diffs, ord=2, axis=1), dtype=np.float64))
            if diffs.size
            else 0.0
        )
        result[segment_name] = {
            "intensity_mean": intensity_mean,
            "diff_l2_mean": diff_l2_mean,
        }
    return result


def _build_cross_modal_scalar_features(
    *,
    first_modality_name: str,
    first_values: np.ndarray,
    first_mask: np.ndarray,
    second_modality_name: str,
    second_values: np.ndarray,
    second_mask: np.ndarray,
) -> dict[str, float]:
    first_intensity = _mean_abs_intensity(first_values, first_mask)
    second_intensity = _mean_abs_intensity(second_values, second_mask)
    first_delta = _delta_l2(first_values, first_mask)
    second_delta = _delta_l2(second_values, second_mask)
    first_diff = _diff_l2_mean(first_values, first_mask)
    second_diff = _diff_l2_mean(second_values, second_mask)
    first_valid_ratio = _valid_ratio(first_mask)
    second_valid_ratio = _valid_ratio(second_mask)
    first_segments = _segment_profile(first_values, first_mask)
    second_segments = _segment_profile(second_values, second_mask)
    second_jump_rate = _jump_rate(second_values, second_mask)
    second_peak_rate = _peak_rate(second_values, second_mask)
    return {
        "cross__valid_ratio_gap": first_valid_ratio - second_valid_ratio,
        "cross__intensity_gap": first_intensity - second_intensity,
        "cross__intensity_ratio": _safe_ratio(first_intensity, second_intensity),
        "cross__delta_l2_gap": first_delta - second_delta,
        "cross__diff_l2_gap": first_diff - second_diff,
        "cross__late_shift_gap": (
            first_segments["late"]["intensity_mean"] - first_segments["early"]["intensity_mean"]
        )
        - (
            second_segments["late"]["intensity_mean"]
            - second_segments["early"]["intensity_mean"]
        ),
        f"cross__{first_modality_name}_{second_modality_name}_jump_peak_product": (
            second_jump_rate * second_peak_rate
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-6:
        return 0.0
    return float(numerator / denominator)


def _build_feature_groups(
    *,
    feature_columns: Sequence[str],
    first_modality_name: str,
    second_modality_name: str,
) -> dict[str, tuple[str, ...]]:
    generic_context_columns = {
        "ctx__sequence_length_steps",
        "ctx__window_fraction",
    }
    physiology_only = []
    physiology_lowdim = []
    context_only = []
    residual_only = []
    physiology_scalar_columns = {
        f"ctx__{first_modality_name}_valid_ratio",
        f"residual__{first_modality_name}_intensity_mean",
        f"residual__{first_modality_name}_delta_l2",
        f"residual__{first_modality_name}_diff_l2_mean",
    }
    for column in feature_columns:
        if (
            column.startswith(f"{first_modality_name}__")
            or column.startswith(f"segment__{first_modality_name}__")
            or column.startswith(f"ctx__{first_modality_name}_")
            or column.startswith(f"residual__{first_modality_name}_")
        ):
            physiology_only.append(column)
        if column in physiology_scalar_columns:
            physiology_lowdim.append(column)
        if (
            column.startswith(f"{second_modality_name}__")
            or column.startswith(f"segment__{second_modality_name}__")
            or column.startswith(f"ctx__{second_modality_name}_")
            or column.startswith(f"residual__{second_modality_name}_")
        ):
            context_only.append(column)
        if (
            column.startswith("residual__")
            or column.startswith("cross__")
            or column in generic_context_columns
        ):
            residual_only.append(column)
    for column in generic_context_columns:
        if column in feature_columns and column not in physiology_only:
            physiology_only.append(column)
        if column in feature_columns and column not in physiology_lowdim:
            physiology_lowdim.append(column)
        if column in feature_columns and column not in context_only:
            context_only.append(column)
    return {
        "full": tuple(feature_columns),
        "physiology_only": tuple(physiology_only),
        "physiology_lowdim": tuple(physiology_lowdim),
        "physiology_scalar_only": tuple(physiology_lowdim),
        "context_only": tuple(context_only),
        "residual_only": tuple(residual_only),
    }


def _build_default_head_feature_columns(
    *,
    dataset_id: str,
    feature_groups: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    full = tuple(feature_groups["full"])
    physiology_only = tuple(feature_groups["physiology_only"])
    physiology_lowdim = _uab_physiology_lowdim_columns(full)
    residual_only = tuple(feature_groups["residual_only"])
    if dataset_id == "uab_workload_dataset":
        return {
            "physiology_persistence": ("residual__physiology_intensity_mean",),
            "ridge_residual_cv": full,
            "elasticnet_residual": residual_only,
            "huber_residual": full,
            "ridge_heat_physiology_lowdim": physiology_lowdim,
            "huber_heat_physiology_lowdim": physiology_lowdim,
        }
    if dataset_id == "nasa_csm":
        return {
            "physiology_margin_balanced_logistic": physiology_only,
            "balanced_logistic_context": full,
            "balanced_linear_svc_context": full,
        }
    raise ValueError(f"unsupported public opt dataset for head features: {dataset_id}")


def _uab_physiology_lowdim_columns(
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    selected = []
    for column in feature_columns:
        if column in {
            "ctx__window_fraction",
            "ctx__physiology_valid_ratio",
            "residual__physiology_intensity_mean",
            "residual__physiology_delta_l2",
            "residual__physiology_diff_l2_mean",
        } or column.startswith("segment__physiology__"):
            selected.append(column)
    if not selected:
        raise ValueError("UAB physiology low-dimensional head has no feature columns.")
    return tuple(selected)
