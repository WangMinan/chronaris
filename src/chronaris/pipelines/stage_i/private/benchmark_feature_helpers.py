"""Feature-frame helper functions for Stage I private benchmark variants."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.pipelines.alignment_preview import (
    AlignmentPreviewIntermediateExport,
    AlignmentPreviewSampleIntermediate,
    StreamIntermediateSnapshot,
)
from chronaris.pipelines.causal_fusion import (
    StageGCausalFusionConfig,
    export_stage_g_causal_fusion_tensors,
    run_stage_g_causal_fusion,
)
from chronaris.pipelines.stage_i.private.feature_utils import (
    cosine_similarity_numpy,
    mean_cosine,
    row_l2_mean,
    sanitize_feature_name,
)


def base_feature_row(row, features: Mapping[str, float]) -> dict[str, object]:
    if isinstance(row, pd.Series):
        sample_id = row["sample_id"]
        sortie_id = row["sortie_id"]
        pilot_id = row["pilot_id"]
        view_id = row["view_id"]
        window_index = row["window_index"]
        sample_partition = row["sample_partition"]
    else:
        sample_id = row.sample_id
        sortie_id = row.sortie_id
        pilot_id = row.pilot_id
        view_id = row.view_id
        window_index = row.window_index
        sample_partition = row.sample_partition
    return {
        "sample_id": sample_id,
        "sortie_id": sortie_id,
        "pilot_id": int(pilot_id),
        "view_id": view_id,
        "window_index": int(window_index),
        "sample_partition": sample_partition,
        "feature_values": dict(features),
    }


def flatten_stream_stats(stats: Mapping[str, object], *, prefix: str) -> dict[str, float]:
    feature_rows = stats.get("features", {}) if isinstance(stats, Mapping) else {}
    flattened: dict[str, float] = {}
    for feature_name, payload in feature_rows.items():
        if not isinstance(payload, Mapping):
            continue
        safe_name = sanitize_feature_name(str(feature_name))
        for stat_name in ("mean", "std", "min", "max", "delta"):
            value = payload.get(stat_name)
            if value is None:
                continue
            flattened[f"{prefix}__{safe_name}__{stat_name}"] = float(value)
    return flattened


def pool_sequence_features(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    if values.ndim != 2:
        raise ValueError("sequence features must have shape [T, D].")
    for feature_index in range(values.shape[1]):
        column = values[:, feature_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        base = f"{prefix}__dim_{feature_index:03d}"
        rows[f"{base}__mean"] = float(finite.mean())
        rows[f"{base}__std"] = float(finite.std())
        rows[f"{base}__min"] = float(finite.min())
        rows[f"{base}__max"] = float(finite.max())
        rows[f"{base}__delta"] = float(finite[-1] - finite[0])
    return rows


def build_intermediate_export_from_view(view) -> AlignmentPreviewIntermediateExport:
    if view.physiology_reference_hidden is None or view.vehicle_reference_hidden is None:
        raise ValueError("Stage H feature bundle is missing reference hidden states.")
    sample_ids = view.sample_ids or tuple(f"{view.view_id}:{index:04d}" for index in range(view.reference_offsets_s.shape[0]))
    samples: list[AlignmentPreviewSampleIntermediate] = []
    for sample_index, sample_id in enumerate(sample_ids):
        physiology_hidden = np.asarray(view.physiology_reference_hidden[sample_index], dtype=np.float32)
        vehicle_hidden = np.asarray(view.vehicle_reference_hidden[sample_index], dtype=np.float32)
        physiology_projection = np.asarray(view.physiology_reference_projection[sample_index], dtype=np.float32)
        vehicle_projection = np.asarray(view.vehicle_reference_projection[sample_index], dtype=np.float32)
        offsets = np.asarray(view.reference_offsets_s[sample_index], dtype=np.float32)
        samples.append(
            AlignmentPreviewSampleIntermediate(
                sample_id=sample_id,
                physiology=build_stream_snapshot(
                    feature_prefix="phys",
                    hidden=physiology_hidden,
                    projection=physiology_projection,
                    offsets=offsets,
                ),
                vehicle=build_stream_snapshot(
                    feature_prefix="veh",
                    hidden=vehicle_hidden,
                    projection=vehicle_projection,
                    offsets=offsets,
                ),
                mean_reference_projection_cosine=mean_cosine(
                    physiology_projection,
                    vehicle_projection,
                ),
            )
        )
    return AlignmentPreviewIntermediateExport(
        partition="all",
        sample_count=len(samples),
        reference_point_count=int(view.reference_offsets_s.shape[1]),
        samples=tuple(samples),
    )


def build_stream_snapshot(
    *,
    feature_prefix: str,
    hidden: np.ndarray,
    projection: np.ndarray,
    offsets: np.ndarray,
) -> StreamIntermediateSnapshot:
    feature_names = tuple(f"{feature_prefix}_{index:03d}" for index in range(projection.shape[1]))
    return StreamIntermediateSnapshot(
        feature_names=feature_names,
        point_count=int(offsets.shape[0]),
        observation_offsets_s=tuple(float(value) for value in offsets),
        reference_offsets_s=tuple(float(value) for value in offsets),
        observation_hidden_states=tuple(tuple(float(value) for value in row) for row in hidden),
        reference_hidden_states=tuple(tuple(float(value) for value in row) for row in hidden),
        reference_projected_states=tuple(tuple(float(value) for value in row) for row in projection),
        mean_observation_hidden_l2=row_l2_mean(hidden),
        mean_reference_hidden_l2=row_l2_mean(hidden),
        mean_reference_projection_l2=row_l2_mean(projection),
    )


def compute_event_mask_interference(
    intermediate: AlignmentPreviewIntermediateExport,
    *,
    config: StageGCausalFusionConfig,
) -> dict[str, float]:
    baseline = export_stage_g_causal_fusion_tensors(intermediate, config=config)
    result = run_stage_g_causal_fusion(intermediate, config=config)
    vehicle_states = np.asarray(
        [sample.vehicle.reference_hidden_states if config.state_source == "hidden" else sample.vehicle.reference_projected_states for sample in intermediate.samples],
        dtype=np.float32,
    )
    physiology_states = np.asarray(
        [sample.physiology.reference_hidden_states if config.state_source == "hidden" else sample.physiology.reference_projected_states for sample in intermediate.samples],
        dtype=np.float32,
    )
    projection_states = np.asarray(
        [sample.physiology.reference_projected_states for sample in intermediate.samples],
        dtype=np.float32,
    )
    vehicle_projection = np.asarray(
        [sample.vehicle.reference_projected_states for sample in intermediate.samples],
        dtype=np.float32,
    )
    offsets = np.asarray([sample.physiology.reference_offsets_s for sample in intermediate.samples], dtype=np.float32)
    baseline_pooled = np.asarray([np.asarray(item, dtype=np.float32).mean(axis=0) for item in baseline.fused_states], dtype=np.float32)
    event_scores = np.asarray([sample.vehicle_event_scores for sample in result.samples], dtype=np.float32)
    perturbed_vehicle = vehicle_states.copy()
    for sample_index in range(event_scores.shape[0]):
        threshold = float(np.quantile(event_scores[sample_index], 0.75))
        active = event_scores[sample_index] >= threshold
        perturbed_vehicle[sample_index, active, :] = 0.0
    perturbed_export = build_intermediate_export_from_arrays(
        sample_ids=tuple(sample.sample_id for sample in intermediate.samples),
        physiology_states=physiology_states,
        vehicle_states=perturbed_vehicle,
        physiology_projection=projection_states,
        vehicle_projection=vehicle_projection,
        offsets=offsets,
    )
    perturbed = export_stage_g_causal_fusion_tensors(perturbed_export, config=config)
    perturbed_pooled = np.asarray([np.asarray(item, dtype=np.float32).mean(axis=0) for item in perturbed.fused_states], dtype=np.float32)
    similarities = cosine_similarity_numpy(baseline_pooled, perturbed_pooled)
    return {
        sample.sample_id: float(1.0 - similarities[index])
        for index, sample in enumerate(result.samples)
    }


def build_intermediate_export_from_arrays(
    *,
    sample_ids: Sequence[str],
    physiology_states: np.ndarray,
    vehicle_states: np.ndarray,
    physiology_projection: np.ndarray,
    vehicle_projection: np.ndarray,
    offsets: np.ndarray,
) -> AlignmentPreviewIntermediateExport:
    samples = []
    for sample_index, sample_id in enumerate(sample_ids):
        samples.append(
            AlignmentPreviewSampleIntermediate(
                sample_id=str(sample_id),
                physiology=build_stream_snapshot(
                    feature_prefix="phys",
                    hidden=physiology_states[sample_index],
                    projection=physiology_projection[sample_index],
                    offsets=offsets[sample_index],
                ),
                vehicle=build_stream_snapshot(
                    feature_prefix="veh",
                    hidden=vehicle_states[sample_index],
                    projection=vehicle_projection[sample_index],
                    offsets=offsets[sample_index],
                ),
                mean_reference_projection_cosine=mean_cosine(
                    physiology_projection[sample_index],
                    vehicle_projection[sample_index],
                ),
            )
        )
    return AlignmentPreviewIntermediateExport(
        partition="all",
        sample_count=len(samples),
        reference_point_count=int(offsets.shape[1]),
        samples=tuple(samples),
    )


def select_feature_names(
    stats_series: Sequence[Mapping[str, object]],
    *,
    preferred_keywords: Sequence[str],
) -> tuple[str, ...]:
    feature_names: set[str] = set()
    for stats in stats_series:
        feature_map = stats.get("features", {}) if isinstance(stats, Mapping) else {}
        feature_names.update(str(name) for name in feature_map)
    if not feature_names:
        return ()
    lowered = tuple(keyword.lower() for keyword in preferred_keywords)
    preferred = tuple(
        name for name in sorted(feature_names)
        if any(keyword in name.lower() for keyword in lowered)
    )
    if preferred:
        return preferred
    return tuple(sorted(feature_names))


def aggregate_field_score(stats: Mapping[str, object], selected_fields: Sequence[str]) -> float | None:
    feature_map = stats.get("features", {}) if isinstance(stats, Mapping) else {}
    values: list[float] = []
    for field_name in selected_fields:
        payload = feature_map.get(field_name)
        if not isinstance(payload, Mapping) or payload.get("count", 0) <= 0:
            continue
        delta = abs(float(payload.get("delta") or 0.0))
        std = abs(float(payload.get("std") or 0.0))
        span = abs(float(payload.get("max") or 0.0) - float(payload.get("min") or 0.0))
        values.append(delta + std + span)
    if not values:
        return None
    return float(sum(values) / len(values))


def resolve_quantile_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lower_q = float(np.quantile(values, 1.0 / 3.0))
    upper_q = float(np.quantile(values, 2.0 / 3.0))
    if lower_q > upper_q:
        lower_q, upper_q = upper_q, lower_q
    return lower_q, upper_q
