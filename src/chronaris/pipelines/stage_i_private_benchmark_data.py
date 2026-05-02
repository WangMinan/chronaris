"""Data loading and task construction helpers for the private benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset.stage_i_private_contracts import StageIPrivateTaskEntry
from chronaris.features import load_stage_h_feature_run
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
from chronaris.pipelines.stage_i_private_feature_utils import (
    bucketize_score,
    cosine_similarity_numpy,
    mean_cosine,
    none_if_empty,
    read_jsonl,
    row_l2_mean,
    safe_mean,
    sanitize_feature_name,
)

PRIVATE_DATASET_ID = "private_stage_h_benchmark"
VARIANT_ORDER = (
    "naive_sync",
    "e_baseline",
    "f_full",
    "g_min",
    "g_no_causal_mask",
)
TASK_MANEUVER = "T1_maneuver_intensity_class"
TASK_RESPONSE = "T2_next_window_physiology_response"
TASK_RETRIEVAL = "T3_paired_pilot_window_retrieval"
CLASS_LABEL_TO_ID = {"low": 0, "medium": 1, "high": 2}
CLASS_ID_TO_LABEL = {value: key for key, value in CLASS_LABEL_TO_ID.items()}


def load_aligned_private_records(
    *,
    e_run_manifest_path: str,
    f_run_manifest_path: str,
) -> pd.DataFrame:
    e_run = load_stage_h_feature_run(e_run_manifest_path)
    f_run = load_stage_h_feature_run(f_run_manifest_path)
    validate_private_stage_h_run_contract(e_run.run_manifest, stage_name="E")
    validate_private_stage_h_run_contract(f_run.run_manifest, stage_name="F")
    e_views = {view.view_id: view for view in e_run.views}
    f_views = {view.view_id: view for view in f_run.views}
    if set(e_views) != set(f_views):
        raise ValueError("E/F Stage H runs must export the same view ids.")

    rows: list[dict[str, object]] = []
    for view_id in sorted(e_views):
        e_view = e_views[view_id]
        f_view = f_views[view_id]
        e_window_rows = read_jsonl(
            e_view.view_manifest["artifact_paths"]["window_manifest_jsonl"],
        )
        f_window_rows = read_jsonl(
            f_view.view_manifest["artifact_paths"]["window_manifest_jsonl"],
        )
        if [row["sample_id"] for row in e_window_rows] != [row["sample_id"] for row in f_window_rows]:
            raise ValueError(f"E/F window orders differ for view {view_id}.")
        raw_rows = {
            row["sample_id"]: row
            for row in read_jsonl(
                f_view.view_manifest["artifact_paths"]["raw_window_summary_jsonl"],
            )
        }
        e_index = {sample_id: index for index, sample_id in enumerate(e_view.sample_ids)}
        f_index = {sample_id: index for index, sample_id in enumerate(f_view.sample_ids)}
        for row in f_window_rows:
            raw_sample_id = str(row["sample_id"])
            sample_id = f"{view_id}::{raw_sample_id}"
            raw_row = raw_rows.get(raw_sample_id, {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "raw_sample_id": raw_sample_id,
                    "sortie_id": str(row["sortie_id"]),
                    "view_id": view_id,
                    "pilot_id": int(f_view.pilot_id),
                    "window_index": int(row["window_index"]),
                    "sample_partition": row.get("sample_partition"),
                    "start_offset_ms": int(row["start_offset_ms"]),
                    "end_offset_ms": int(row["end_offset_ms"]),
                    "e_view": e_view,
                    "f_view": f_view,
                    "e_index": e_index.get(raw_sample_id),
                    "f_index": f_index.get(raw_sample_id),
                    "raw_physiology_stats": raw_row.get("physiology_feature_stats", {}),
                    "raw_vehicle_stats": raw_row.get("vehicle_feature_stats", {}),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("private benchmark requires at least one aligned view record.")
    return frame.sort_values(["sortie_id", "pilot_id", "window_index"]).reset_index(drop=True)


def validate_private_stage_h_run_contract(
    run_manifest: Mapping[str, object],
    *,
    stage_name: str,
) -> None:
    """Reject Stage H runs that cannot support the private E/F comparison."""

    config = run_manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"{stage_name} Stage H run manifest is missing config.")
    run_id = str(run_manifest.get("run_id") or stage_name)
    if config.get("intermediate_partition") != "all":
        raise ValueError(f"{stage_name} Stage H run {run_id} must use intermediate_partition='all'.")
    if run_manifest.get("partial_data") is not None:
        raise ValueError(f"{stage_name} Stage H run {run_id} must disable partial-data sidecar.")
    if config.get("causal_fusion_enabled") is not False:
        raise ValueError(f"{stage_name} Stage H run {run_id} must disable causal fusion.")
    if config.get("physiology_point_limit_per_measurement") is not None:
        raise ValueError(f"{stage_name} Stage H run {run_id} must not cap physiology points.")
    if config.get("vehicle_point_limit_per_measurement") is not None:
        raise ValueError(f"{stage_name} Stage H run {run_id} must not cap vehicle points.")

    physics_enabled = config.get("physics_constraints_enabled")
    if stage_name == "E":
        if physics_enabled is not False:
            raise ValueError(f"E Stage H run {run_id} must set physics_constraints_enabled=false.")
        return
    if stage_name == "F":
        if physics_enabled is not True:
            raise ValueError(f"F Stage H run {run_id} must set physics_constraints_enabled=true.")
        if config.get("physics_constraint_family") != "full":
            raise ValueError(f"F Stage H run {run_id} must use physics_constraint_family='full'.")
        return
    raise ValueError(f"unsupported private Stage H stage name: {stage_name}")


def derive_private_task_entries(
    records: pd.DataFrame,
) -> dict[str, object]:
    maneuver_fields = select_feature_names(records["raw_vehicle_stats"], preferred_keywords=(
        "speed",
        "acc",
        "pitch",
        "roll",
        "yaw",
        "rate",
        "overload",
        "rudder",
        "stick",
        "angle",
        "heading",
    ))
    maneuver_scores = records["raw_vehicle_stats"].apply(
        lambda stats: aggregate_field_score(stats, maneuver_fields),
    )
    lower_q, upper_q = resolve_quantile_bounds(maneuver_scores.dropna().to_numpy(dtype=float))

    physiology_fields = select_feature_names(
        records["raw_physiology_stats"],
        preferred_keywords=("eeg", "spo2"),
    )
    task_entries: list[StageIPrivateTaskEntry] = []
    by_task: dict[str, list[StageIPrivateTaskEntry]] = {
        TASK_MANEUVER: [],
        TASK_RESPONSE: [],
        TASK_RETRIEVAL: [],
    }
    response_valid_count = 0
    maneuver_valid_count = 0
    retrieval_valid_count = 0

    response_labels: dict[str, float | None] = {}
    response_refs: dict[str, str | None] = {}
    for _view_id, frame in records.groupby("view_id", sort=False):
        ordered = frame.sort_values("window_index").reset_index(drop=True)
        for row_index, row in ordered.iterrows():
            if row_index + 1 >= len(ordered):
                response_labels[str(row["sample_id"])] = None
                response_refs[str(row["sample_id"])] = None
                continue
            next_row = ordered.iloc[row_index + 1]
            score = aggregate_field_score(next_row["raw_physiology_stats"], physiology_fields)
            response_labels[str(row["sample_id"])] = score
            response_refs[str(row["sample_id"])] = str(next_row["sample_id"])

    paired_lookup: dict[tuple[str, int, int], str] = {}
    for (_sortie_id, _window_index), frame in records.groupby(["sortie_id", "window_index"], sort=False):
        if len(frame) < 2:
            continue
        ordered = frame.sort_values("pilot_id").reset_index(drop=True)
        if len(ordered) != 2:
            continue
        left = ordered.iloc[0]
        right = ordered.iloc[1]
        paired_lookup[(str(left["sortie_id"]), int(left["pilot_id"]), int(left["window_index"]))] = str(right["sample_id"])
        paired_lookup[(str(right["sortie_id"]), int(right["pilot_id"]), int(right["window_index"]))] = str(left["sample_id"])

    for row in records.itertuples(index=False):
        maneuver_score = aggregate_field_score(row.raw_vehicle_stats, maneuver_fields)
        maneuver_label = None if maneuver_score is None else bucketize_score(maneuver_score, lower_q, upper_q)
        if maneuver_label is not None:
            maneuver_valid_count += 1
        maneuver_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_MANEUVER,
            task_type="classification",
            label_name="maneuver_intensity_class",
            label_value=maneuver_label,
            label_source="raw_vehicle_window_stats",
            source_refs={"window_summary": "raw_window_summary.jsonl"},
            context_payload={
                "score": maneuver_score,
                "selected_vehicle_fields": list(maneuver_fields),
            },
        )
        task_entries.append(maneuver_entry)
        by_task[TASK_MANEUVER].append(maneuver_entry)

        response_value = response_labels[row.sample_id]
        if response_value is not None:
            response_valid_count += 1
        response_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_RESPONSE,
            task_type="regression",
            label_name="next_window_physiology_response",
            label_value=response_value,
            label_source="next_window_raw_physiology_stats",
            source_refs={"window_summary": "raw_window_summary.jsonl"},
            context_payload={
                "selected_physiology_fields": list(physiology_fields),
                "next_sample_id": response_refs[row.sample_id],
            },
        )
        task_entries.append(response_entry)
        by_task[TASK_RESPONSE].append(response_entry)

        paired_sample_id = paired_lookup.get((row.sortie_id, int(row.pilot_id), int(row.window_index)))
        if paired_sample_id is not None:
            retrieval_valid_count += 1
        retrieval_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_RETRIEVAL,
            task_type="retrieval",
            label_name="paired_sample_id",
            label_value=paired_sample_id,
            label_source="same_sortie_dual_pilot_window_index",
            source_refs={"window_manifest": "window_manifest.jsonl"},
            paired_sample_id=paired_sample_id,
            context_payload={},
        )
        task_entries.append(retrieval_entry)
        by_task[TASK_RETRIEVAL].append(retrieval_entry)

    summary = {
        "entry_count": len(task_entries),
        "task_counts": {task_name: len(entries) for task_name, entries in by_task.items()},
        "coverage": {
            TASK_MANEUVER: {
                "valid_label_count": maneuver_valid_count,
                "total_count": len(by_task[TASK_MANEUVER]),
            },
            TASK_RESPONSE: {
                "valid_label_count": response_valid_count,
                "total_count": len(by_task[TASK_RESPONSE]),
            },
            TASK_RETRIEVAL: {
                "valid_label_count": retrieval_valid_count,
                "total_count": len(by_task[TASK_RETRIEVAL]),
            },
        },
        "selected_vehicle_fields": list(maneuver_fields),
        "selected_physiology_fields": list(physiology_fields),
        "maneuver_label_distribution": dict(Counter(
            entry.label_value for entry in by_task[TASK_MANEUVER] if entry.label_value is not None
        )),
    }
    return {
        "entries": tuple(task_entries),
        "by_task": {task_name: tuple(entries) for task_name, entries in by_task.items()},
        "summary": summary,
    }


def build_variant_feature_frames(
    records: pd.DataFrame,
    *,
    enable_optimized_chronaris: bool = False,
    target_variant_name: str = "chronaris_opt",
    lag_window_points: int = 3,
    residual_mode: str = "raw_window_stats",
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    variant_frames: dict[str, pd.DataFrame] = {
        "naive_sync": build_naive_feature_frame(records),
        "e_baseline": build_projection_feature_frame(records, stream_key="e"),
        "f_full": build_projection_feature_frame(records, stream_key="f"),
    }
    g_min_frame, g_min_summary = build_g_variant_feature_frame(
        records,
        config=StageGCausalFusionConfig(),
    )
    g_nomask_frame, g_nomask_summary = build_g_variant_feature_frame(
        records,
        config=StageGCausalFusionConfig(use_causal_mask=False),
    )
    variant_frames["g_min"] = g_min_frame
    variant_frames["g_no_causal_mask"] = g_nomask_frame
    diagnostics = {
        "f_full": {
            "mean_attention_entropy": 0.0,
            "mean_top_event_concentration": 0.0,
            "mean_event_mask_interference": 0.0,
        },
        "g_min": g_min_summary,
        "g_no_causal_mask": g_nomask_summary,
    }
    if enable_optimized_chronaris:
        from chronaris.pipelines.stage_i_private_optimization import (
            build_optimized_chronaris_feature_frames,
        )

        selected_vehicle_fields = select_feature_names(
            records["raw_vehicle_stats"],
            preferred_keywords=(
                "speed",
                "acc",
                "pitch",
                "roll",
                "yaw",
                "rate",
                "overload",
                "rudder",
                "stick",
                "angle",
                "heading",
            ),
        )
        selected_physiology_fields = select_feature_names(
            records["raw_physiology_stats"],
            preferred_keywords=("eeg", "spo2"),
        )
        optimized_frames, optimized_diagnostics = build_optimized_chronaris_feature_frames(
            records,
            target_variant_name=target_variant_name,
            lag_window_points=lag_window_points,
            residual_mode=residual_mode,
            selected_vehicle_fields=selected_vehicle_fields,
            selected_physiology_fields=selected_physiology_fields,
        )
        variant_frames.update(optimized_frames)
        diagnostics.update(optimized_diagnostics)
    return variant_frames, diagnostics


def build_naive_feature_frame(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in records.itertuples(index=False):
        features = {}
        features.update(flatten_stream_stats(row.raw_physiology_stats, prefix="phys"))
        features.update(flatten_stream_stats(row.raw_vehicle_stats, prefix="veh"))
        rows.append(base_feature_row(row, features))
    return pd.DataFrame(rows)


def build_projection_feature_frame(records: pd.DataFrame, *, stream_key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in records.itertuples(index=False):
        sample_index = getattr(row, f"{stream_key}_index")
        if sample_index is None:
            continue
        view = getattr(row, f"{stream_key}_view")
        physiology = np.asarray(view.physiology_reference_projection[sample_index], dtype=np.float32)
        vehicle = np.asarray(view.vehicle_reference_projection[sample_index], dtype=np.float32)
        features = {}
        features.update(pool_sequence_features(physiology, prefix="phys"))
        features.update(pool_sequence_features(vehicle, prefix="veh"))
        rows.append(base_feature_row(row, features))
    return pd.DataFrame(rows)


def build_g_variant_feature_frame(
    records: pd.DataFrame,
    *,
    config: StageGCausalFusionConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    attention_entropies: list[float] = []
    top_concentrations: list[float] = []
    event_interference: list[float] = []
    for _view_id, frame in records.groupby("view_id", sort=False):
        f_view = frame["f_view"].iloc[0]
        intermediate = build_intermediate_export_from_view(f_view)
        fusion_result = run_stage_g_causal_fusion(intermediate, config=config)
        tensor_export = export_stage_g_causal_fusion_tensors(intermediate, config=config)
        interference_by_sample = compute_event_mask_interference(intermediate, config=config)
        for sample_index, sample in enumerate(fusion_result.samples):
            record = frame.loc[frame["raw_sample_id"] == sample.sample_id]
            if record.empty:
                continue
            base_row = record.iloc[0]
            fused = np.asarray(tensor_export.fused_states[sample_index], dtype=np.float32)
            features = pool_sequence_features(fused, prefix="fused")
            top_concentration = float(
                np.asarray(sample.attention_weights, dtype=np.float32).max(axis=-1).mean()
            )
            features["diag_attention_entropy"] = sample.mean_attention_entropy
            features["diag_top_event_concentration"] = top_concentration
            features["diag_event_mask_interference"] = interference_by_sample.get(sample.sample_id, 0.0)
            rows.append(base_feature_row(base_row, features))
            attention_entropies.append(sample.mean_attention_entropy)
            top_concentrations.append(top_concentration)
            event_interference.append(features["diag_event_mask_interference"])
    frame = pd.DataFrame(rows)
    summary = {
        "mean_attention_entropy": safe_mean(attention_entropies),
        "mean_top_event_concentration": safe_mean(top_concentrations),
        "mean_event_mask_interference": safe_mean(event_interference),
    }
    return frame, summary


def merge_task_features(
    task_entries: Sequence[StageIPrivateTaskEntry],
    frame: pd.DataFrame,
    *,
    task_type: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    frame_by_sample = frame.set_index("sample_id", drop=False)
    for entry in task_entries:
        if entry.sample_id not in frame_by_sample.index:
            continue
        if task_type != "retrieval" and entry.label_value is None:
            continue
        source_row = frame_by_sample.loc[entry.sample_id]
        features = {
            key: float(value)
            for key, value in source_row["feature_values"].items()
        }
        rows.append(
            {
                "sample_id": entry.sample_id,
                "sortie_id": entry.sortie_id,
                "pilot_id": entry.pilot_id,
                "view_id": entry.view_id,
                "split_group": entry.view_id,
                "paired_sample_id": entry.paired_sample_id,
                "task_entry": entry,
                "y_label": (
                    CLASS_LABEL_TO_ID[str(entry.label_value)]
                    if task_type == "classification"
                    else float(entry.label_value)
                    if task_type == "regression" and entry.label_value is not None
                    else np.nan
                ),
                **{f"feat__{key}": value for key, value in features.items()},
            }
        )
    merged = pd.DataFrame(rows)
    if merged.empty:
        return merged
    feature_columns = sorted(column for column in merged.columns if column.startswith("feat__"))
    merged.loc[:, feature_columns] = merged.loc[:, feature_columns].fillna(0.0)
    merged["feature_vector"] = [
        row.to_numpy(dtype=np.float32)
        for _, row in merged.loc[:, feature_columns].iterrows()
    ]
    return merged


def build_private_sequence_frame(
    task_entries: Sequence[StageIPrivateTaskEntry],
    records: pd.DataFrame,
    *,
    task_type: str,
) -> pd.DataFrame:
    record_by_sample = records.set_index("sample_id", drop=False)
    rows = []
    for entry in task_entries:
        if entry.label_value is None:
            continue
        if entry.sample_id not in record_by_sample.index:
            continue
        record = record_by_sample.loc[entry.sample_id]
        if record["f_index"] is None:
            continue
        f_view = record["f_view"]
        sample_index = int(record["f_index"])
        physiology = np.asarray(f_view.physiology_reference_projection[sample_index], dtype=np.float32)
        vehicle = np.asarray(f_view.vehicle_reference_projection[sample_index], dtype=np.float32)
        time_axis = np.asarray(f_view.reference_offsets_s[sample_index], dtype=np.float32)
        rows.append(
            {
                "sample_id": entry.sample_id,
                "split_group": entry.view_id,
                "task_entry": entry,
                "physiology_sequence": physiology,
                "vehicle_sequence": vehicle,
                "physiology_mask": np.isfinite(physiology).any(axis=1).astype(np.uint8),
                "vehicle_mask": np.isfinite(vehicle).any(axis=1).astype(np.uint8),
                "time_axis": time_axis,
                "y_label": (
                    CLASS_LABEL_TO_ID[str(entry.label_value)]
                    if task_type == "classification"
                    else float(entry.label_value)
                ),
            }
        )
    return pd.DataFrame(rows)


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

