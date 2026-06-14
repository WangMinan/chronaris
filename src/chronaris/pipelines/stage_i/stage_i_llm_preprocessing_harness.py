"""Request/repair harness for Stage I LLM preprocessing."""

from __future__ import annotations

from typing import Mapping, Sequence

from chronaris.llm import LLMProvider
from chronaris.llm.prompts import build_llm_task_request
from chronaris.llm.schemas import (
    coerce_field_semantics,
    coerce_runtime_explanations,
    coerce_schema_gap_policy,
    coerce_semantic_query_hints,
    coerce_weak_label_review,
)


def run_preprocessing_llm_task(
    *,
    run_id: str,
    provider: LLMProvider,
    index: int,
    task_name: str,
    input_payload: Mapping[str, object],
) -> tuple[str, object, list[dict[str, object]], list[dict[str, object]]]:
    """Run one task with a single schema-repair retry when validation fails."""

    audits: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    output_key = _output_key(task_name)
    request = build_llm_task_request(
        run_id=run_id,
        request_id=f"{run_id}-{index:02d}-{task_name}",
        task_name=task_name,
        provider=provider.provider_name,
        model=provider.model,
        input_payload=input_payload,
    )
    response = provider.generate(request)
    audits.append({"request": request.to_audit_dict(), "response": response.to_audit_dict()})
    if response.status != "success":
        errors.append(_error_row(request.request_id, task_name, response.status, response.error_summary))
        return output_key, _empty_output(task_name), audits, errors
    parsed = dict(response.parsed_payload or {})
    output = _coerce_output(task_name, parsed, input_payload)
    if _is_valid_output(task_name, output, input_payload):
        return output_key, output, audits, errors

    repair_payload = {
        "schema_repair": {
            "reason": f"{task_name} response failed local schema validation",
            "invalid_response": parsed,
        },
        "original_input_payload": input_payload,
    }
    repair = build_llm_task_request(
        run_id=run_id,
        request_id=f"{run_id}-{index:02d}-{task_name}-repair",
        task_name=task_name,
        provider=provider.provider_name,
        model=provider.model,
        input_payload=repair_payload,
    )
    repair_response = provider.generate(repair)
    audits.append({"request": repair.to_audit_dict(), "response": repair_response.to_audit_dict()})
    if repair_response.status == "success":
        repaired_output = _coerce_output(task_name, dict(repair_response.parsed_payload or {}), input_payload)
        if _is_valid_output(task_name, repaired_output, input_payload):
            return output_key, repaired_output, audits, errors
    errors.append(_error_row(repair.request_id, task_name, "schema_validation_failure", "schema repair did not produce a valid local output"))
    return output_key, output, audits, errors


def _coerce_output(task_name: str, parsed: Mapping[str, object], input_payload: Mapping[str, object]) -> object:
    if task_name == "field_semantics":
        return coerce_field_semantics(parsed)
    if task_name == "weak_label_review":
        return coerce_weak_label_review(parsed)
    if task_name == "semantic_query_hints":
        return coerce_semantic_query_hints(parsed)
    if task_name == "schema_gap_policy":
        return coerce_schema_gap_policy(parsed)
    if task_name == "runtime_explanations":
        return _merge_runtime_source_paths(
            coerce_runtime_explanations(parsed),
            input_payload.get("runtime_cases", []),
        )
    return {}


def _is_valid_output(task_name: str, output: object, input_payload: Mapping[str, object]) -> bool:
    if task_name == "field_semantics":
        return len(output) >= min(1, len(input_payload.get("fields", []))) if isinstance(output, list) else False
    if task_name == "weak_label_review":
        expected = len(input_payload.get("task_summaries", []))
        return isinstance(output, list) and len(output) >= min(1, expected)
    if task_name == "semantic_query_hints":
        return isinstance(output, list) and len(output) >= 1
    if task_name == "schema_gap_policy":
        return isinstance(output, Mapping) and bool(output.get("policies"))
    if task_name == "runtime_explanations":
        expected = len(input_payload.get("runtime_cases", []))
        return isinstance(output, list) and len(output) >= min(1, expected)
    return False


def _merge_runtime_source_paths(
    explanations: Sequence[Mapping[str, object]],
    runtime_cases: object,
) -> list[dict[str, object]]:
    if not isinstance(runtime_cases, list):
        return [dict(row) for row in explanations]
    source_by_sample = {
        str(row.get("sample_id")): [
            row.get("source_path"),
            row.get("support_source_path"),
            row.get("runtime_service_source_path"),
            row.get("runtime_schema_contract_source_path"),
        ]
        for row in runtime_cases
        if isinstance(row, Mapping) and row.get("sample_id")
    }
    resolved = []
    for row in explanations:
        payload = dict(row)
        if not payload.get("source_paths"):
            payload["source_paths"] = [
                path
                for path in source_by_sample.get(str(payload.get("sample_id")), [])
                if path
            ]
        resolved.append(payload)
    return resolved


def _output_key(task_name: str) -> str:
    return {
        "field_semantics": "field_semantics",
        "weak_label_review": "weak_label_rule_review",
        "semantic_query_hints": "semantic_query_hints",
        "schema_gap_policy": "schema_gap_policy",
        "runtime_explanations": "runtime_case_explanations",
    }[task_name]


def _empty_output(task_name: str) -> object:
    return {} if task_name == "schema_gap_policy" else []


def _error_row(request_id: str, task_name: str, status: str, error_summary: str | None) -> dict[str, object]:
    return {
        "request_id": request_id,
        "task_name": task_name,
        "status": status,
        "error_summary": error_summary,
    }
