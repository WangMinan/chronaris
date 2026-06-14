"""Weak-label thesis-task builders over private Stage H aligned records."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.dataset.stage_i_private_contracts import StageIPrivateTaskEntry
from chronaris.pipelines.stage_i.private.feature_utils import bucketize_score, none_if_empty

TASK_RISK_PROXY = "risk_proxy"
TASK_WORKLOAD_PROXY = "workload_proxy"
TASK_EVENT_REPLAY_TAG = "event_replay_tag"
THESIS_WEAK_LABEL_BENCHMARK_ROLE = "thesis_task_weak_label_benchmark"
THESIS_WEAK_LABEL_ROLE = "thesis_weak_label_task"
THESIS_WEAK_LABEL_BOUNDARY = "weak_label_proxy_not_manual_ground_truth"

THESIS_TASK_DEFINITIONS = {
    TASK_RISK_PROXY: {
        "task_family": "risk_analysis",
        "label_source": "vehicle_intensity_plus_physiology_variation",
        "weak_label_note": "risk proxy is weakly derived from window stats, not manual incident truth",
    },
    TASK_WORKLOAD_PROXY: {
        "task_family": "cognitive_workload",
        "label_source": "physiology_variation_plus_vehicle_intensity",
        "weak_label_note": "workload proxy is weakly derived from physiology and maneuver intensity",
    },
    TASK_EVENT_REPLAY_TAG: {
        "task_family": "event_replay",
        "label_source": "derived_event_tag_group_pairing",
        "weak_label_note": "event replay tag uses derived event groups, not expert replay annotations",
    },
}


def attach_llm_preprocessing_context_to_task_entries(
    entries: Sequence[StageIPrivateTaskEntry],
    llm_preprocessing_context: Mapping[str, object],
) -> tuple[StageIPrivateTaskEntry, ...]:
    """Attach audited LLM preprocessing context without changing labels."""

    if not entries:
        return ()
    review_by_task = {
        str(row.get("task_name")): dict(row)
        for row in llm_preprocessing_context.get("weak_label_rule_review", [])
        if isinstance(row, Mapping) and row.get("task_name")
    }
    context_ref = {
        "run_id": llm_preprocessing_context.get("run_id"),
        "schema_version": llm_preprocessing_context.get("schema_version"),
        "field_semantic_dictionary_path": llm_preprocessing_context.get("field_semantic_dictionary_path"),
        "context_path": llm_preprocessing_context.get("context_path"),
        "boundary": "llm_preprocessing_context_not_ground_truth",
    }
    attached: list[StageIPrivateTaskEntry] = []
    for entry in entries:
        task_review = review_by_task.get(entry.task_name)
        attached.append(
            replace(
                entry,
                context_payload={
                    **dict(entry.context_payload),
                    "llm_preprocessing_context": context_ref,
                    "llm_weak_label_rule_review": task_review,
                },
            )
        )
    return tuple(attached)


def build_stage_i_real_task_payload(records: pd.DataFrame) -> dict[str, object]:
    """Build thesis-facing weak-label tasks from aligned private Stage H records."""

    if records.empty:
        raise ValueError("Stage I thesis-task builders require at least one aligned record.")

    vehicle_fields = _select_feature_names(
        records["raw_vehicle_stats"],
        preferred_keywords=("speed", "acc", "pitch", "roll", "yaw", "overload", "heading"),
    )
    physiology_fields = _select_feature_names(
        records["raw_physiology_stats"],
        preferred_keywords=("eeg", "spo2", "hr", "heart"),
    )

    risk_scores = records.apply(
        lambda row: _risk_score(
            row["raw_vehicle_stats"],
            row["raw_physiology_stats"],
            vehicle_fields=vehicle_fields,
            physiology_fields=physiology_fields,
        ),
        axis=1,
    )
    workload_raw_scores = records.apply(
        lambda row: _workload_score(
            row["raw_vehicle_stats"],
            row["raw_physiology_stats"],
            vehicle_fields=vehicle_fields,
            physiology_fields=physiology_fields,
        ),
        axis=1,
    )
    workload_scores = _normalize_continuous_scores(workload_raw_scores)
    risk_lower_q, risk_upper_q = _resolve_quantile_bounds(risk_scores.dropna().to_numpy(dtype=float))

    event_tags = {
        str(row.sample_id): _build_event_tag(
            row.raw_vehicle_stats,
            row.raw_physiology_stats,
            vehicle_fields=vehicle_fields,
            physiology_fields=physiology_fields,
        )
        for row in records.itertuples(index=False)
    }
    replay_pairs = _resolve_event_replay_pairs(records, event_tags)

    entries: list[StageIPrivateTaskEntry] = []
    by_task: dict[str, list[StageIPrivateTaskEntry]] = {
        TASK_RISK_PROXY: [],
        TASK_WORKLOAD_PROXY: [],
        TASK_EVENT_REPLAY_TAG: [],
    }
    valid_counts = {task_name: 0 for task_name in by_task}

    for row_index, row in enumerate(records.itertuples(index=False)):
        risk_score = risk_scores.iloc[row_index]
        risk_label = None
        if not pd.isna(risk_score):
            risk_label = bucketize_score(float(risk_score), risk_lower_q, risk_upper_q)
            valid_counts[TASK_RISK_PROXY] += 1
        risk_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_RISK_PROXY,
            task_type="classification",
            label_name="risk_proxy_class",
            label_value=risk_label,
            label_source=THESIS_TASK_DEFINITIONS[TASK_RISK_PROXY]["label_source"],
            source_refs={"window_summary": "raw_window_summary.jsonl"},
            benchmark_role=THESIS_WEAK_LABEL_BENCHMARK_ROLE,
            task_role=THESIS_WEAK_LABEL_ROLE,
            context_payload={
                **THESIS_TASK_DEFINITIONS[TASK_RISK_PROXY],
                "thesis_task_boundary": THESIS_WEAK_LABEL_BOUNDARY,
                "selected_vehicle_fields": list(vehicle_fields),
                "selected_physiology_fields": list(physiology_fields),
                "weak_label_score": None if pd.isna(risk_score) else float(risk_score),
            },
        )
        entries.append(risk_entry)
        by_task[TASK_RISK_PROXY].append(risk_entry)

        workload_raw_value = (
            None
            if pd.isna(workload_raw_scores.iloc[row_index])
            else float(workload_raw_scores.iloc[row_index])
        )
        workload_value = None if pd.isna(workload_scores.iloc[row_index]) else float(workload_scores.iloc[row_index])
        if workload_value is not None:
            valid_counts[TASK_WORKLOAD_PROXY] += 1
        workload_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_WORKLOAD_PROXY,
            task_type="regression",
            label_name="workload_proxy_score",
            label_value=workload_value,
            label_source=THESIS_TASK_DEFINITIONS[TASK_WORKLOAD_PROXY]["label_source"],
            source_refs={"window_summary": "raw_window_summary.jsonl"},
            benchmark_role=THESIS_WEAK_LABEL_BENCHMARK_ROLE,
            task_role=THESIS_WEAK_LABEL_ROLE,
            context_payload={
                **THESIS_TASK_DEFINITIONS[TASK_WORKLOAD_PROXY],
                "thesis_task_boundary": THESIS_WEAK_LABEL_BOUNDARY,
                "selected_vehicle_fields": list(vehicle_fields),
                "selected_physiology_fields": list(physiology_fields),
                "raw_workload_score": workload_raw_value,
                "normalized_workload_score": workload_value,
            },
        )
        entries.append(workload_entry)
        by_task[TASK_WORKLOAD_PROXY].append(workload_entry)

        replay_pair = replay_pairs.get(str(row.sample_id))
        event_tag = event_tags[str(row.sample_id)]
        if replay_pair is not None:
            valid_counts[TASK_EVENT_REPLAY_TAG] += 1
        replay_entry = StageIPrivateTaskEntry(
            sample_id=row.sample_id,
            sortie_id=row.sortie_id,
            pilot_id=int(row.pilot_id),
            view_id=row.view_id,
            window_index=int(row.window_index),
            sample_partition=none_if_empty(row.sample_partition),
            task_name=TASK_EVENT_REPLAY_TAG,
            task_type="retrieval",
            label_name="event_replay_pair_sample_id",
            label_value=replay_pair,
            label_source=THESIS_TASK_DEFINITIONS[TASK_EVENT_REPLAY_TAG]["label_source"],
            source_refs={"window_manifest": "window_manifest.jsonl"},
            benchmark_role=THESIS_WEAK_LABEL_BENCHMARK_ROLE,
            task_role=THESIS_WEAK_LABEL_ROLE,
            paired_sample_id=replay_pair,
            context_payload={
                **THESIS_TASK_DEFINITIONS[TASK_EVENT_REPLAY_TAG],
                "thesis_task_boundary": THESIS_WEAK_LABEL_BOUNDARY,
                "event_replay_tag": event_tag,
            },
        )
        entries.append(replay_entry)
        by_task[TASK_EVENT_REPLAY_TAG].append(replay_entry)

    summary = {
        "entry_count": len(entries),
        "benchmark_role": THESIS_WEAK_LABEL_BENCHMARK_ROLE,
        "task_role": THESIS_WEAK_LABEL_ROLE,
        "thesis_task_boundary": THESIS_WEAK_LABEL_BOUNDARY,
        "task_counts": {task_name: len(task_entries) for task_name, task_entries in by_task.items()},
        "task_role_counts": dict(Counter(entry.task_role for entry in entries)),
        "coverage": {
            task_name: {
                "valid_label_count": valid_counts[task_name],
                "total_count": len(task_entries),
            }
            for task_name, task_entries in by_task.items()
        },
        "selected_vehicle_fields": list(vehicle_fields),
        "selected_physiology_fields": list(physiology_fields),
        "thesis_task_definitions": THESIS_TASK_DEFINITIONS,
        "risk_label_distribution": dict(
            Counter(entry.label_value for entry in by_task[TASK_RISK_PROXY] if entry.label_value is not None)
        ),
        "workload_score_summary": {
            "raw_min": _series_min(workload_raw_scores),
            "raw_max": _series_max(workload_raw_scores),
            "normalized_min": _series_min(workload_scores),
            "normalized_max": _series_max(workload_scores),
        },
        "event_tag_distribution": dict(Counter(event_tags.values())),
    }
    return {
        "entries": tuple(entries),
        "by_task": {task_name: tuple(task_entries) for task_name, task_entries in by_task.items()},
        "summary": summary,
    }


def _risk_score(
    vehicle_stats: Mapping[str, object],
    physiology_stats: Mapping[str, object],
    *,
    vehicle_fields: Sequence[str],
    physiology_fields: Sequence[str],
) -> float | None:
    vehicle_score = _aggregate_field_score(vehicle_stats, vehicle_fields)
    physiology_score = _aggregate_field_score(physiology_stats, physiology_fields)
    if vehicle_score is None and physiology_score is None:
        return None
    return float((vehicle_score or 0.0) + (physiology_score or 0.0))


def _workload_score(
    vehicle_stats: Mapping[str, object],
    physiology_stats: Mapping[str, object],
    *,
    vehicle_fields: Sequence[str],
    physiology_fields: Sequence[str],
) -> float | None:
    physiology_score = _aggregate_field_score(physiology_stats, physiology_fields)
    vehicle_score = _aggregate_field_score(vehicle_stats, vehicle_fields)
    if physiology_score is None and vehicle_score is None:
        return None
    if physiology_score is None:
        return float(vehicle_score or 0.0)
    if vehicle_score is None:
        return float(physiology_score)
    return float((physiology_score * 0.6) + (vehicle_score * 0.4))


def _build_event_tag(
    vehicle_stats: Mapping[str, object],
    physiology_stats: Mapping[str, object],
    *,
    vehicle_fields: Sequence[str],
    physiology_fields: Sequence[str],
) -> str:
    vehicle_score = _aggregate_field_score(vehicle_stats, vehicle_fields) or 0.0
    physiology_score = _aggregate_field_score(physiology_stats, physiology_fields) or 0.0
    if vehicle_score >= physiology_score * 1.25:
        return "maneuver_dominant"
    if physiology_score >= vehicle_score * 1.25:
        return "physiology_dominant"
    return "coupled_transition"


def _resolve_event_replay_pairs(
    records: pd.DataFrame,
    event_tags: Mapping[str, str],
) -> dict[str, str]:
    pairs: dict[str, str] = {}
    working = records.copy()
    working["event_tag"] = working["sample_id"].map(event_tags)
    for (_sortie_id, event_tag), frame in working.groupby(["sortie_id", "event_tag"], sort=False):
        sample_rows = list(frame.sort_values(["window_index", "pilot_id"]).itertuples(index=False))
        if len(sample_rows) < 2:
            continue
        for row in sample_rows:
            same_tag_candidates = [
                candidate
                for candidate in sample_rows
                if candidate.sample_id != row.sample_id
            ]
            if not same_tag_candidates:
                continue
            chosen = min(
                same_tag_candidates,
                key=lambda candidate: (
                    abs(int(candidate.window_index) - int(row.window_index)),
                    1 if int(candidate.pilot_id) == int(row.pilot_id) else 0,
                    int(candidate.window_index),
                    str(candidate.sample_id),
                ),
            )
            pairs[str(row.sample_id)] = str(chosen.sample_id)
    return pairs


def _select_feature_names(
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


def _aggregate_field_score(stats: Mapping[str, object], selected_fields: Sequence[str]) -> float | None:
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


def _resolve_quantile_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    lower_q = float(np.quantile(values, 1.0 / 3.0))
    upper_q = float(np.quantile(values, 2.0 / 3.0))
    if lower_q > upper_q:
        lower_q, upper_q = upper_q, lower_q
    return lower_q, upper_q


def _normalize_continuous_scores(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores
    valid = scores.notna()
    if not bool(valid.any()):
        return scores
    raw = scores.loc[valid].to_numpy(dtype=float)
    log_scaled = np.log1p(np.clip(raw, a_min=0.0, a_max=None))
    lower = float(log_scaled.min())
    upper = float(log_scaled.max())
    if np.isclose(lower, upper):
        normalized = np.full_like(log_scaled, 0.5)
    else:
        normalized = (log_scaled - lower) / (upper - lower)
    resolved = scores.copy()
    resolved.loc[valid] = normalized.astype(float)
    return resolved


def _series_min(values: pd.Series) -> float | None:
    finite = values.dropna()
    if finite.empty:
        return None
    return float(finite.min())


def _series_max(values: pd.Series) -> float | None:
    finite = values.dropna()
    if finite.empty:
        return None
    return float(finite.max())
