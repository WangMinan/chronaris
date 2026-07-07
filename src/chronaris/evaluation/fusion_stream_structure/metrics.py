"""Metrics for E3 fusion stream structure evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from chronaris.evaluation.fusion_stream_structure.contracts import (
    CLAIM_BOUNDARY,
    EVIDENCE_QUADRANT,
    FusionStreamRecord,
    FusionStreamRunConfig,
)


def cp_tolerance_hit_rate(
    change_points: Sequence[int],
    weak_event_boundary: Sequence[object],
    *,
    tol: int | str = "auto",
) -> dict[str, object]:
    T = len(weak_event_boundary)
    tolerance = _resolve_tol(T, tol)
    references = [index for index, value in enumerate(weak_event_boundary) if _truthy(value)]
    if not references:
        return _metric_payload("cp_tolerance_hit_rate", None, "unavailable", reason="no_weak_event_boundaries")
    if not change_points:
        return _metric_payload("cp_tolerance_hit_rate", 0.0, "completed", tolerance=tolerance)
    hits = 0
    for reference in references:
        if any(abs(int(point) - reference) <= tolerance for point in change_points):
            hits += 1
    return _metric_payload("cp_tolerance_hit_rate", hits / len(references), "completed", tolerance=tolerance)


def segment_event_purity(
    change_points: Sequence[int],
    labels: Sequence[object],
) -> dict[str, object]:
    clean_labels = [_clean_label(label) for label in labels]
    if not any(label is not None for label in clean_labels):
        return _metric_payload("segment_event_purity", None, "unavailable", reason="no_maneuver_labels")
    T = len(clean_labels)
    boundaries = [0] + sorted({int(point) for point in change_points if 0 < int(point) < T}) + [T]
    weighted_correct = 0
    valid_total = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment_labels = [label for label in clean_labels[start:end] if label is not None]
        if not segment_labels:
            continue
        count = Counter(segment_labels)
        weighted_correct += count.most_common(1)[0][1]
        valid_total += len(segment_labels)
    if valid_total == 0:
        return _metric_payload("segment_event_purity", None, "unavailable", reason="no_valid_segment_labels")
    return _metric_payload("segment_event_purity", weighted_correct / valid_total, "completed")


def transition_entropy(state_sequence: Sequence[int]) -> dict[str, object]:
    if len(state_sequence) < 2:
        return _metric_payload("transition_entropy", None, "unavailable", reason="too_few_states")
    counts = Counter(zip(state_sequence[:-1], state_sequence[1:]))
    total = sum(counts.values())
    if total == 0:
        return _metric_payload("transition_entropy", 0.0, "completed")
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return _metric_payload("transition_entropy", entropy, "completed")


def state_count(state_sequence: Sequence[int]) -> dict[str, object]:
    if not state_sequence:
        return _metric_payload("state_count", None, "unavailable", reason="missing_state_sequence")
    return _metric_payload("state_count", len(set(int(value) for value in state_sequence)), "completed")


def cross_view_segment_stability(
    change_points_by_view: Mapping[str, Sequence[int]],
    *,
    T: int | Mapping[str, int],
) -> dict[str, object]:
    views = sorted(change_points_by_view)
    if len(views) < 2:
        return _metric_payload("cross_view_segment_stability", None, "unavailable", reason="need_at_least_two_views")
    scores = []
    for left_index, left_view in enumerate(views):
        for right_view in views[left_index + 1:]:
            left_T = T[left_view] if isinstance(T, Mapping) else T
            right_T = T[right_view] if isinstance(T, Mapping) else T
            length = min(int(left_T), int(right_T))
            scores.append(_boundary_f1(change_points_by_view[left_view], change_points_by_view[right_view], length))
    return _metric_payload("cross_view_segment_stability", float(np.mean(scores)), "completed")


def clap_state_replay_consistency(
    states_by_view: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    views = sorted(states_by_view)
    if len(views) < 2:
        return _metric_payload("clap_state_replay_consistency", None, "unavailable", reason="need_at_least_two_views")
    scores = []
    for left_index, left_view in enumerate(views):
        for right_view in views[left_index + 1:]:
            left = np.asarray(states_by_view[left_view])
            right = np.asarray(states_by_view[right_view])
            length = min(left.size, right.size)
            if length == 0:
                continue
            scores.append(float((left[:length] == right[:length]).mean()))
    if not scores:
        return _metric_payload("clap_state_replay_consistency", None, "unavailable", reason="empty_state_sequences")
    return _metric_payload("clap_state_replay_consistency", float(np.mean(scores)), "completed")


def motif_event_consistency(
    motif_pair: Mapping[str, object],
    labels: Sequence[object],
) -> dict[str, object]:
    left = motif_pair.get("left") if isinstance(motif_pair, Mapping) else None
    right = motif_pair.get("right") if isinstance(motif_pair, Mapping) else None
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return _metric_payload("motif_event_consistency", None, "unavailable", reason="missing_motif_pair")
    left_label = _segment_majority_label(left, labels)
    right_label = _segment_majority_label(right, labels)
    if left_label is None or right_label is None:
        return _metric_payload("motif_event_consistency", None, "unavailable", reason="missing_segment_labels")
    return _metric_payload("motif_event_consistency", 1.0 if left_label == right_label else 0.0, "completed")


def discord_interval_overlap(
    discord_segment: Mapping[str, object],
    interval_flags: Sequence[object],
    *,
    metric_name: str,
) -> dict[str, object]:
    if not isinstance(discord_segment, Mapping) or "start_index" not in discord_segment:
        return _metric_payload(metric_name, None, "unavailable", reason="missing_discord_segment")
    if not interval_flags:
        return _metric_payload(metric_name, None, "unavailable", reason="missing_interval_flags")
    start = max(0, int(discord_segment.get("start_index", 0)))
    end = min(len(interval_flags), int(discord_segment.get("end_index", start + 1)))
    if end <= start:
        return _metric_payload(metric_name, None, "unavailable", reason="empty_discord_segment")
    segment = set(range(start, end))
    intervals = {index for index, value in enumerate(interval_flags) if _truthy(value)}
    if not intervals:
        return _metric_payload(metric_name, None, "unavailable", reason="no_reference_intervals")
    union = segment | intervals
    intersection = segment & intervals
    return _metric_payload(metric_name, len(intersection) / len(union), "completed")


def fluss_clasp_agreement(
    fluss_regimes: Sequence[int],
    change_points: Sequence[int],
    *,
    T: int,
    tol: int | str = "auto",
) -> dict[str, object]:
    if not fluss_regimes or not change_points:
        return _metric_payload("fluss_clasp_agreement", None, "unavailable", reason="missing_regimes_or_change_points")
    tolerance = _resolve_tol(T, tol)
    hits = sum(1 for regime in fluss_regimes if any(abs(int(regime) - int(point)) <= tolerance for point in change_points))
    return _metric_payload("fluss_clasp_agreement", hits / len(fluss_regimes), "completed", tolerance=tolerance)


def snippet_coverage(snippets: Sequence[object], *, T: int) -> dict[str, object]:
    if not snippets:
        return _metric_payload("snippet_coverage", None, "unavailable", reason="missing_snippets")
    return _metric_payload("snippet_coverage", min(1.0, len(snippets) / max(T, 1)), "completed")


def compute_metric_rows(
    records: Mapping[tuple[str, str, str], FusionStreamRecord],
    clasp_results: Mapping[tuple[str, str, str], Mapping[str, object]],
    stumpy_results: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    config: FusionStreamRunConfig | None = None,
) -> list[dict[str, object]]:
    config = config or FusionStreamRunConfig()
    rows: list[dict[str, object]] = []
    for key, record in records.items():
        clasp_result = clasp_results.get(key, {})
        stumpy_result = stumpy_results.get(key, {})
        context = _row_context(record)
        if clasp_result.get("status") == "completed":
            labels = record.frame["maneuver_proxy_label"].to_list() if "maneuver_proxy_label" in record.frame else []
            boundaries = record.frame["weak_event_boundary"].to_list() if "weak_event_boundary" in record.frame else []
            change_points = clasp_result.get("change_points", [])
            state_sequence = clasp_result.get("state_sequence", [])
            rows.extend([
                _row(context, "clasp", cp_tolerance_hit_rate(change_points, boundaries, tol=config.tol)),
                _row(context, "clasp", segment_event_purity(change_points, labels)),
                _row(context, "clap", state_count(state_sequence)),
                _row(context, "clap", transition_entropy(state_sequence)),
            ])
        else:
            rows.extend(_unavailable_algorithm_rows(context, "clasp", str(clasp_result.get("status", "not_run")), (
                "cp_tolerance_hit_rate",
                "segment_event_purity",
                "state_count",
                "transition_entropy",
            )))

        if stumpy_result.get("status") == "completed":
            labels = record.frame["maneuver_proxy_label"].to_list() if "maneuver_proxy_label" in record.frame else []
            physio_flags = record.frame["physio_fluctuation_interval"].to_list() if "physio_fluctuation_interval" in record.frame else []
            maneuver_flags = _labels_to_event_flags(labels)
            rows.extend([
                _row(context, "stumpy", motif_event_consistency(stumpy_result.get("motif_pair", {}), labels)),
                _row(context, "stumpy", discord_interval_overlap(stumpy_result.get("discord_segment", {}), maneuver_flags, metric_name="discord_maneuver_overlap")),
                _row(context, "stumpy", discord_interval_overlap(stumpy_result.get("discord_segment", {}), physio_flags, metric_name="discord_physio_overlap")),
                _row(context, "stumpy", fluss_clasp_agreement(stumpy_result.get("fluss_regimes", []), clasp_result.get("change_points", []), T=record.T, tol=config.tol)),
                _row(context, "stumpy", _metric_payload("mp_discord_isolation", stumpy_result.get("mp_discord_isolation_value"), "completed")),
                _row(context, "stumpy", snippet_coverage(stumpy_result.get("snippets", []), T=record.T)),
            ])
        else:
            rows.extend(_unavailable_algorithm_rows(context, "stumpy", str(stumpy_result.get("status", "not_run")), (
                "motif_event_consistency",
                "discord_maneuver_overlap",
                "discord_physio_overlap",
                "fluss_clasp_agreement",
                "mp_discord_isolation",
                "snippet_coverage",
            )))
    rows.extend(_cross_view_rows(records, clasp_results))
    rows.extend(compute_composite_scores(rows, config=config))
    return rows


def compute_composite_scores(
    metric_rows: Sequence[Mapping[str, object]],
    *,
    config: FusionStreamRunConfig | None = None,
) -> list[dict[str, object]]:
    config = config or FusionStreamRunConfig()
    by_method_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        if row.get("status") != "completed":
            continue
        value = row.get("value")
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        by_method_metric[str(row["method_name"])][str(row["metric"])].append(float(value))
    rows = []
    for method_name in sorted(by_method_metric):
        for score_name, weights in config.composite_weights.items():
            missing = [metric for metric in weights if not by_method_metric[method_name].get(metric)]
            if missing:
                rows.append(_summary_row(method_name, score_name, None, "unavailable", reason=f"missing_metrics:{','.join(missing)}"))
                continue
            value = sum(
                float(weight) * float(np.mean(by_method_metric[method_name][metric]))
                for metric, weight in weights.items()
            )
            rows.append(_summary_row(method_name, score_name, value, "completed", weights=dict(weights)))
    return rows


def metric_rows_to_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    columns = [
        "method_name",
        "sortie_id",
        "view_id",
        "algorithm",
        "metric",
        "value",
        "status",
        "evidence_quadrant",
        "claim_boundary",
        "details",
    ]
    return pd.DataFrame(rows, columns=columns)


def _cross_view_rows(
    records: Mapping[tuple[str, str, str], FusionStreamRecord],
    clasp_results: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, tuple[FusionStreamRecord, Mapping[str, object]]]] = defaultdict(dict)
    for key, record in records.items():
        method_name, sortie_id, view_id = key
        grouped[(method_name, sortie_id)][view_id] = (record, clasp_results.get(key, {}))
    rows = []
    for (method_name, sortie_id), view_payload in sorted(grouped.items()):
        cps = {
            view_id: payload["change_points"]
            for view_id, (_record, payload) in view_payload.items()
            if payload.get("status") == "completed"
        }
        lengths = {
            view_id: record.T
            for view_id, (record, payload) in view_payload.items()
            if payload.get("status") == "completed"
        }
        states = {
            view_id: payload["state_sequence"]
            for view_id, (_record, payload) in view_payload.items()
            if payload.get("status") == "completed"
        }
        context = {
            "method_name": method_name,
            "sortie_id": sortie_id,
            "view_id": "__cross_view__",
        }
        rows.append(_row(context, "clasp", cross_view_segment_stability(cps, T=lengths)))
        rows.append(_row(context, "clap", clap_state_replay_consistency(states)))
        rows.append(_row(context, "stumpy", _metric_payload("nn_segment_cross_view_consistency", None, "unavailable", reason="not_enough_completed_stumpy_cross_view_pairs")))
    return rows


def _boundary_f1(left_points: Sequence[int], right_points: Sequence[int], T: int) -> float:
    left = np.zeros(max(T, 1), dtype=bool)
    right = np.zeros(max(T, 1), dtype=bool)
    for point in left_points:
        if 0 <= int(point) < left.size:
            left[int(point)] = True
    for point in right_points:
        if 0 <= int(point) < right.size:
            right[int(point)] = True
    tp = int(np.logical_and(left, right).sum())
    fp = int(np.logical_and(left, ~right).sum())
    fn = int(np.logical_and(~left, right).sum())
    if tp == fp == fn == 0:
        return 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _segment_majority_label(segment: Mapping[str, object], labels: Sequence[object]) -> str | None:
    start = max(0, int(segment.get("start_index", 0)))
    end = min(len(labels), int(segment.get("end_index", start + 1)))
    segment_labels = [_clean_label(label) for label in labels[start:end]]
    segment_labels = [label for label in segment_labels if label is not None]
    if not segment_labels:
        return None
    return Counter(segment_labels).most_common(1)[0][0]


def _labels_to_event_flags(labels: Sequence[object]) -> list[bool]:
    return [_clean_label(label) == "high" for label in labels]


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "high", "event"}
    return bool(value)


def _clean_label(label: object) -> str | None:
    if label is None:
        return None
    if isinstance(label, float) and np.isnan(label):
        return None
    text = str(label)
    if text in {"", "None", "nan"}:
        return None
    return text


def _resolve_tol(T: int, tol: int | str) -> int:
    if tol == "auto":
        return max(1, int(round(0.02 * T)))
    return int(tol)


def _metric_payload(metric: str, value: object, status: str, **details: object) -> dict[str, object]:
    return {
        "metric": metric,
        "value": None if value is None else float(value),
        "status": status,
        "details": details,
    }


def _row(context: Mapping[str, object], algorithm: str, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "method_name": context["method_name"],
        "sortie_id": context["sortie_id"],
        "view_id": context["view_id"],
        "algorithm": algorithm,
        "metric": payload["metric"],
        "value": payload.get("value"),
        "status": payload.get("status"),
        "evidence_quadrant": EVIDENCE_QUADRANT,
        "claim_boundary": CLAIM_BOUNDARY,
        "details": payload.get("details", {}),
    }


def _row_context(record: FusionStreamRecord) -> dict[str, object]:
    return {
        "method_name": record.method_name,
        "sortie_id": record.sortie_id,
        "view_id": record.view_id,
    }


def _unavailable_algorithm_rows(
    context: Mapping[str, object],
    algorithm: str,
    reason: str,
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    return [
        _row(context, algorithm, _metric_payload(metric, None, "unavailable", reason=reason))
        for metric in metrics
    ]


def _summary_row(
    method_name: str,
    metric: str,
    value: object,
    status: str,
    **details: object,
) -> dict[str, object]:
    return {
        "method_name": method_name,
        "sortie_id": "__summary__",
        "view_id": "__summary__",
        "algorithm": "composite_fixed_weights",
        "metric": metric,
        "value": None if value is None else float(value),
        "status": status,
        "evidence_quadrant": EVIDENCE_QUADRANT,
        "claim_boundary": CLAIM_BOUNDARY,
        "details": details,
    }
