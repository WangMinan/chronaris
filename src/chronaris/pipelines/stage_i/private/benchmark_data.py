"""Data loading and task construction helpers for the private benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset.stage_i_private_contracts import StageIPrivateTaskEntry
from chronaris.features import load_stage_h_feature_run
from chronaris.pipelines.causal_fusion import (
    StageGCausalFusionConfig,
    export_stage_g_causal_fusion_tensors,
    run_stage_g_causal_fusion,
)
from chronaris.pipelines.stage_i.private.benchmark_feature_helpers import (
    aggregate_field_score,
    base_feature_row,
    build_intermediate_export_from_view,
    compute_event_mask_interference,
    flatten_stream_stats,
    pool_sequence_features,
    resolve_quantile_bounds,
    select_feature_names,
)
from chronaris.pipelines.stage_i.private.feature_utils import (
    bucketize_score,
    none_if_empty,
    read_jsonl,
    safe_mean,
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
PRIVATE_PROXY_BENCHMARK_ROLE = "private_proxy_benchmark"
PROXY_TASK_ROLE = "proxy_task"
PROXY_TASK_METADATA = {
    TASK_MANEUVER: {
        "proxy_task_id": "maneuver_intensity_proxy",
        "proxy_description": "window-level maneuver-intensity proxy derived from raw vehicle stats",
    },
    TASK_RESPONSE: {
        "proxy_task_id": "next_window_physiology_response_proxy",
        "proxy_description": "next-window physiology response proxy derived from adjacent windows",
    },
    TASK_RETRIEVAL: {
        "proxy_task_id": "paired_pilot_window_retrieval_proxy",
        "proxy_description": "same-sortie paired-pilot retrieval proxy at matched window index",
    },
}


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


def derive_private_proxy_task_entries(
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
        maneuver_proxy = PROXY_TASK_METADATA[TASK_MANEUVER]
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
            benchmark_role=PRIVATE_PROXY_BENCHMARK_ROLE,
            task_role=PROXY_TASK_ROLE,
            context_payload={
                **maneuver_proxy,
                "thesis_task_boundary": "proxy_task_not_direct_thesis_task",
                "score": maneuver_score,
                "selected_vehicle_fields": list(maneuver_fields),
            },
        )
        task_entries.append(maneuver_entry)
        by_task[TASK_MANEUVER].append(maneuver_entry)

        response_proxy = PROXY_TASK_METADATA[TASK_RESPONSE]
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
            benchmark_role=PRIVATE_PROXY_BENCHMARK_ROLE,
            task_role=PROXY_TASK_ROLE,
            context_payload={
                **response_proxy,
                "thesis_task_boundary": "proxy_task_not_direct_thesis_task",
                "selected_physiology_fields": list(physiology_fields),
                "next_sample_id": response_refs[row.sample_id],
            },
        )
        task_entries.append(response_entry)
        by_task[TASK_RESPONSE].append(response_entry)

        retrieval_proxy = PROXY_TASK_METADATA[TASK_RETRIEVAL]
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
            benchmark_role=PRIVATE_PROXY_BENCHMARK_ROLE,
            task_role=PROXY_TASK_ROLE,
            paired_sample_id=paired_sample_id,
            context_payload={
                **retrieval_proxy,
                "thesis_task_boundary": "proxy_task_not_direct_thesis_task",
            },
        )
        task_entries.append(retrieval_entry)
        by_task[TASK_RETRIEVAL].append(retrieval_entry)

    summary = {
        "entry_count": len(task_entries),
        "benchmark_role": PRIVATE_PROXY_BENCHMARK_ROLE,
        "task_role": PROXY_TASK_ROLE,
        "evidence_layer": "proxy_evidence",
        "thesis_task_boundary": "t1_t2_t3_are_proxy_tasks_not_direct_thesis_tasks",
        "task_counts": {task_name: len(entries) for task_name, entries in by_task.items()},
        "task_role_counts": dict(Counter(entry.task_role for entry in task_entries)),
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
        "proxy_task_definitions": PROXY_TASK_METADATA,
        "maneuver_label_distribution": dict(Counter(
            entry.label_value for entry in by_task[TASK_MANEUVER] if entry.label_value is not None
        )),
    }
    return {
        "entries": tuple(task_entries),
        "by_task": {task_name: tuple(entries) for task_name, entries in by_task.items()},
        "summary": summary,
    }


def derive_private_task_entries(
    records: pd.DataFrame,
) -> dict[str, object]:
    """Backward-compatible alias for the historical private proxy task builder."""

    return derive_private_proxy_task_entries(records)


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
        from chronaris.pipelines.stage_i.private.optimization import (
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
