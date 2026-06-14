"""Slicing and merge helpers for Stage I LLM preprocessing calls."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

TaskCall = tuple[str, dict[str, object]]


def build_sliced_llm_task_calls(
    *,
    cards: Mapping[str, Mapping[str, object]],
    schema_field_chunk_size: int,
    weak_label_task_chunk_size: int,
    schema_gap_group_chunk_size: int,
    runtime_case_chunk_size: int,
) -> list[TaskCall]:
    """Build bounded LLM calls and keep merge metadata in each payload."""

    calls: list[TaskCall] = []
    calls.extend(
        ("field_semantics", payload)
        for payload in _slice_list_payload(
            cards["field_semantics"],
            list_key="fields",
            chunk_size=schema_field_chunk_size,
        )
    )
    calls.extend(
        ("weak_label_review", payload)
        for payload in _slice_list_payload(
            cards["weak_label_review"],
            list_key="task_summaries",
            chunk_size=weak_label_task_chunk_size,
        )
    )
    calls.append(("semantic_query_hints", _with_unsliced_metadata(cards["semantic_query_hints"])))
    calls.extend(
        ("schema_gap_policy", payload)
        for payload in _slice_mapping_payload(
            cards["schema_gap_policy"],
            mapping_key="missing_measurement_groups",
            chunk_size=schema_gap_group_chunk_size,
        )
    )
    calls.extend(
        ("runtime_explanations", payload)
        for payload in _slice_list_payload(
            cards["runtime_explanations"],
            list_key="runtime_cases",
            chunk_size=runtime_case_chunk_size,
        )
    )
    return calls


def merge_sliced_task_output(task_name: str, current: object, incoming: object) -> object:
    """Merge validated slice outputs without letting duplicate keys multiply rows."""

    if task_name == "schema_gap_policy":
        return _merge_schema_gap_policy(current, incoming)
    if task_name in {"field_semantics", "weak_label_review", "semantic_query_hints", "runtime_explanations"}:
        key_name = {
            "field_semantics": "feature_name",
            "weak_label_review": "task_name",
            "semantic_query_hints": "name",
            "runtime_explanations": "sample_id",
        }[task_name]
        return _merge_row_lists(current, incoming, key_name=key_name)
    return incoming


def build_slicing_summary(audit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize how LLM inputs were sliced before provider calls."""

    slice_rows = []
    task_counts: dict[str, int] = defaultdict(int)
    sliced_tasks: set[str] = set()
    for row in audit_rows:
        if row.get("attempt") != "initial":
            continue
        request = _as_mapping(row.get("request"))
        payload = _as_mapping(request.get("input_payload"))
        slicing = _as_mapping(payload.get("slicing"))
        task_name = str(request.get("task_name"))
        task_counts[task_name] += 1
        if bool(slicing.get("sliced")):
            sliced_tasks.add(task_name)
        slice_rows.append(
            {
                "request_id": request.get("request_id"),
                "task_name": task_name,
                "item_key": slicing.get("item_key"),
                "slice_index": slicing.get("slice_index"),
                "slice_count": slicing.get("slice_count"),
                "item_count": slicing.get("item_count"),
                "total_item_count": slicing.get("total_item_count"),
                "sliced": bool(slicing.get("sliced")),
            }
        )
    return {
        "initial_call_count": sum(task_counts.values()),
        "sliced_task_count": len(sliced_tasks),
        "sliced_tasks": sorted(sliced_tasks),
        "initial_calls_by_task": dict(sorted(task_counts.items())),
        "slice_rows": slice_rows,
        "boundary": "large LLM payloads are split into bounded slices and merged locally by stable identifiers",
    }


def _slice_list_payload(
    card: Mapping[str, object],
    *,
    list_key: str,
    chunk_size: int,
) -> list[dict[str, object]]:
    rows = list(card.get(list_key, [])) if isinstance(card.get(list_key, []), list) else []
    chunks = _chunks(rows, chunk_size)
    payloads = []
    for index, chunk in enumerate(chunks, start=1):
        payload = dict(card)
        payload[list_key] = chunk
        payload["slicing"] = {
            "item_key": list_key,
            "slice_index": index,
            "slice_count": len(chunks),
            "item_count": len(chunk),
            "total_item_count": len(rows),
            "sliced": len(chunks) > 1,
        }
        payloads.append(payload)
    return payloads


def _slice_mapping_payload(
    card: Mapping[str, object],
    *,
    mapping_key: str,
    chunk_size: int,
) -> list[dict[str, object]]:
    values = card.get(mapping_key, {})
    mapping = dict(values) if isinstance(values, Mapping) else {}
    keys = sorted(mapping)
    chunks = _chunks(keys, chunk_size)
    payloads = []
    for index, chunk in enumerate(chunks, start=1):
        payload = dict(card)
        payload[mapping_key] = {key: mapping[key] for key in chunk}
        payload["slicing"] = {
            "item_key": mapping_key,
            "slice_index": index,
            "slice_count": len(chunks),
            "item_count": len(chunk),
            "total_item_count": len(keys),
            "sliced": len(chunks) > 1,
        }
        payloads.append(payload)
    return payloads


def _with_unsliced_metadata(card: Mapping[str, object]) -> dict[str, object]:
    payload = dict(card)
    payload["slicing"] = {
        "item_key": "global_card",
        "slice_index": 1,
        "slice_count": 1,
        "item_count": 1,
        "total_item_count": 1,
        "sliced": False,
    }
    return payload


def _chunks(rows: Sequence[object], chunk_size: int) -> list[list[object]]:
    if chunk_size <= 0:
        raise ValueError("LLM chunk size must be positive.")
    if not rows:
        return [[]]
    return [list(rows[index : index + chunk_size]) for index in range(0, len(rows), chunk_size)]


def _merge_row_lists(current: object, incoming: object, *, key_name: str) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in _as_row_list(current) + _as_row_list(incoming):
        key = str(row.get(key_name))
        if key and key != "None":
            merged[key] = dict(row)
    return list(merged.values())


def _merge_schema_gap_policy(current: object, incoming: object) -> dict[str, object]:
    current_policy = dict(current) if isinstance(current, Mapping) else {}
    incoming_policy = dict(incoming) if isinstance(incoming, Mapping) else {}
    if not incoming_policy:
        return current_policy
    merged = current_policy or incoming_policy
    policies = _merge_row_lists(
        current_policy.get("policies", []),
        incoming_policy.get("policies", []),
        key_name="measurement_group",
    )
    merged["status"] = incoming_policy.get("status") or current_policy.get("status") or "draft_policy"
    merged["canonical_exact_boundary"] = (
        incoming_policy.get("canonical_exact_boundary")
        or current_policy.get("canonical_exact_boundary")
        or "canonical exact is a service payload contract, not native exact evidence"
    )
    merged["native_exact_claim_allowed"] = False
    merged["policies"] = policies
    return merged


def _as_row_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
