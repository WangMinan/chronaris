"""Request/repair harness for task evaluation LLM preprocessing."""

from __future__ import annotations

from typing import Mapping, Sequence

from chronaris.llm import LLMProvider
from chronaris.llm.prompts import build_llm_task_request
from chronaris.llm.schemas import (
    ALLOWED_FIELD_ROLES,
    ALLOWED_REVIEW_DECISIONS,
    ALLOWED_SCHEMA_POLICIES,
    ALLOWED_SEMANTIC_RECIPES,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    coerce_field_semantics,
    coerce_runtime_explanations,
    coerce_schema_gap_policy,
    coerce_semantic_query_hints,
    coerce_weak_label_review,
)


def build_harness_summary(audit_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize provider attempts and local gate verdicts."""

    verdict_rows = []
    latest_by_task: dict[str, dict[str, object]] = {}
    for row in audit_rows:
        request = _as_mapping(row.get("request"))
        verdict = _as_mapping(row.get("harness_verdict"))
        if not verdict:
            continue
        checks = [
            dict(check)
            for check in verdict.get("checks", [])
            if isinstance(check, Mapping)
        ]
        failed_gates = [
            str(check.get("name"))
            for check in checks
            if not bool(check.get("passed"))
        ]
        verdict_row = {
            "request_id": request.get("request_id"),
            "task_name": request.get("task_name"),
            "attempt": row.get("attempt"),
            "valid": bool(verdict.get("valid")),
            "output_count": verdict.get("output_count"),
            "failed_gates": failed_gates,
        }
        verdict_rows.append(verdict_row)
        latest_by_task[str(request.get("task_name"))] = verdict_row
    remaining_invalid_tasks = [
        task_name
        for task_name, verdict in sorted(latest_by_task.items())
        if not bool(verdict.get("valid"))
    ]
    return {
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "attempt_count": len(audit_rows),
        "validated_attempt_count": len(verdict_rows),
        "provider_failure_attempt_count": sum(
            1
            for row in audit_rows
            if _as_mapping(row.get("response")).get("status") != "success"
        ),
        "schema_repair_attempt_count": sum(1 for row in audit_rows if row.get("attempt") == "schema_repair"),
        "failed_initial_attempt_count": sum(
            1
            for row in verdict_rows
            if row.get("attempt") == "initial" and not bool(row.get("valid"))
        ),
        "remaining_invalid_task_count": len(remaining_invalid_tasks),
        "remaining_invalid_tasks": remaining_invalid_tasks,
        "task_verdicts": verdict_rows,
        "boundary": "local harness validates LLM JSON shape and identifier coverage before downstream consumption",
    }


def run_preprocessing_llm_task(
    *,
    run_id: str,
    provider: LLMProvider,
    index: int,
    task_name: str,
    input_payload: Mapping[str, object],
) -> tuple[str, object, list[dict[str, object]], list[dict[str, object]]]:
    """Run one task through an agent-style local schema gate and repair step."""

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
    if response.status != "success":
        audits.append(_audit_row(request, response, attempt="initial"))
        errors.append(_error_row(request.request_id, task_name, response.status, response.error_summary))
        return output_key, _empty_output(task_name), audits, errors
    parsed = dict(response.parsed_payload or {})
    output = _coerce_output(task_name, parsed, input_payload)
    verdict = _validate_output(task_name, parsed, output, input_payload)
    audits.append(_audit_row(request, response, attempt="initial", verdict=verdict))
    if verdict["valid"]:
        return output_key, output, audits, errors

    repair_payload = dict(input_payload)
    repair_payload["schema_repair"] = {
        "reason": f"{task_name} response failed local harness validation",
        "failed_gates": [
            str(check["name"])
            for check in verdict["checks"]
            if not bool(check["passed"])
        ],
        "expected_identifiers": _expected_identifiers(task_name, input_payload),
        "allowed_top_level_keys": [_expected_top_level_key(task_name)],
        "invalid_response": parsed,
    }
    repair_payload["repair_instruction"] = (
        "Return a fresh valid JSON object for the original task. Do not explain the repair."
    )
    repair = build_llm_task_request(
        run_id=run_id,
        request_id=f"{run_id}-{index:02d}-{task_name}-repair",
        task_name=task_name,
        provider=provider.provider_name,
        model=provider.model,
        input_payload=repair_payload,
    )
    repair_response = provider.generate(repair)
    if repair_response.status == "success":
        repaired_parsed = dict(repair_response.parsed_payload or {})
        repaired_output = _coerce_output(task_name, repaired_parsed, input_payload)
        repair_verdict = _validate_output(task_name, repaired_parsed, repaired_output, input_payload)
        audits.append(_audit_row(repair, repair_response, attempt="schema_repair", verdict=repair_verdict))
        if repair_verdict["valid"]:
            return output_key, repaired_output, audits, errors
    else:
        audits.append(_audit_row(repair, repair_response, attempt="schema_repair"))
    errors.append(
        _error_row(
            repair.request_id,
            task_name,
            "schema_validation_failure",
            "schema repair did not produce a valid local output",
        )
    )
    return output_key, output, audits, errors


def _audit_row(
    request: object,
    response: object,
    *,
    attempt: str,
    verdict: Mapping[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "attempt": attempt,
        "request": request.to_audit_dict(),
        "response": response.to_audit_dict(),
    }
    if verdict is not None:
        row["harness_verdict"] = dict(verdict)
    return row


def _validate_output(
    task_name: str,
    parsed: Mapping[str, object],
    output: object,
    input_payload: Mapping[str, object],
) -> dict[str, object]:
    checks = [
        _check(
            "allowed_top_level_keys_only",
            set(parsed) <= {_expected_top_level_key(task_name)} and bool(parsed),
            actual=sorted(parsed),
            expected=[_expected_top_level_key(task_name)],
        ),
        _check(
            "required_output_key_present",
            _expected_top_level_key(task_name) in parsed,
            actual=sorted(parsed),
            expected=[_expected_top_level_key(task_name)],
        ),
    ]
    if task_name == "field_semantics":
        checks.extend(_validate_field_semantics(output, input_payload))
    elif task_name == "weak_label_review":
        checks.extend(_validate_weak_label_review(output, input_payload))
    elif task_name == "semantic_query_hints":
        checks.extend(_validate_semantic_query_hints(output, input_payload))
    elif task_name == "schema_gap_policy":
        checks.extend(_validate_schema_gap_policy(output, input_payload))
    elif task_name == "runtime_explanations":
        checks.extend(_validate_runtime_explanations(output, input_payload))
    return {
        "valid": all(bool(check["passed"]) for check in checks),
        "task_name": task_name,
        "checks": checks,
        "output_count": _output_count(task_name, output),
    }


def _validate_field_semantics(
    output: object,
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = output if isinstance(output, list) else []
    expected = _input_values(input_payload.get("fields", []), "feature_name")
    actual = _row_values(rows, "feature_name")
    roles = {str(row.get("llm_semantic_role")) for row in rows if isinstance(row, Mapping)}
    streams = {str(row.get("stream_kind")) for row in rows if isinstance(row, Mapping)}
    return [
        _coverage_check("feature_name_coverage_exact", actual, expected),
        _check("one_row_per_input_field", len(rows) == len(expected), actual=len(rows), expected=len(expected)),
        _check("field_roles_whitelisted", roles <= ALLOWED_FIELD_ROLES, actual=sorted(roles)),
        _check("stream_kind_whitelisted", streams <= {"physiology", "vehicle"}, actual=sorted(streams)),
    ]


def _validate_weak_label_review(
    output: object,
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = output if isinstance(output, list) else []
    expected = _input_values(input_payload.get("task_summaries", []), "task_name")
    actual = _row_values(rows, "task_name")
    decisions = {str(row.get("review_decision")) for row in rows if isinstance(row, Mapping)}
    return [
        _coverage_check("task_name_coverage_exact", actual, expected),
        _check("one_row_per_task_summary", len(rows) == len(expected), actual=len(rows), expected=len(expected)),
        _check("review_decisions_whitelisted", decisions <= ALLOWED_REVIEW_DECISIONS, actual=sorted(decisions)),
    ]


def _validate_semantic_query_hints(
    output: object,
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = output if isinstance(output, list) else []
    names = _row_values(rows, "name")
    recipes = {str(row.get("recipe")) for row in rows if isinstance(row, Mapping)}
    allowed_recipes = set(str(recipe) for recipe in input_payload.get("allowed_recipes", []))
    allowed_recipes = allowed_recipes or ALLOWED_SEMANTIC_RECIPES
    existing_names = set(str(name) for name in input_payload.get("existing_query_names", []))
    return [
        _check("at_least_one_hint", len(rows) >= 1, actual=len(rows), expected=">=1"),
        _check("unique_hint_names", len(names) == len(set(names)), actual=sorted(names)),
        _check("recipe_whitelist", recipes <= allowed_recipes, actual=sorted(recipes), expected=sorted(allowed_recipes)),
        _check("no_existing_query_name_collision", not (set(names) & existing_names), actual=sorted(set(names) & existing_names)),
    ]


def _validate_schema_gap_policy(
    output: object,
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    policy = output if isinstance(output, Mapping) else {}
    rows = policy.get("policies", []) if isinstance(policy.get("policies", []), list) else []
    groups = input_payload.get("missing_measurement_groups", {})
    expected = set(str(group) for group in groups) if isinstance(groups, Mapping) else set()
    actual = _row_values(rows, "measurement_group")
    policy_types = {str(row.get("policy_type")) for row in rows if isinstance(row, Mapping)}
    fabricates = [row for row in rows if isinstance(row, Mapping) and row.get("allows_value_fabrication")]
    return [
        _coverage_check("missing_measurement_group_coverage_exact", actual, expected),
        _check("one_policy_per_missing_group", len(rows) == len(expected), actual=len(rows), expected=len(expected)),
        _check("schema_policy_whitelist", policy_types <= ALLOWED_SCHEMA_POLICIES, actual=sorted(policy_types)),
        _check("no_value_fabrication", not fabricates, actual=len(fabricates), expected=0),
        _check("native_exact_claim_disallowed", not bool(policy.get("native_exact_claim_allowed")), actual=policy.get("native_exact_claim_allowed")),
    ]


def _validate_runtime_explanations(
    output: object,
    input_payload: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = output if isinstance(output, list) else []
    expected = _input_values(input_payload.get("runtime_cases", []), "sample_id")
    actual = _row_values(rows, "sample_id")
    return [
        _coverage_check("sample_id_coverage_exact", actual, expected),
        _check("one_row_per_runtime_case", len(rows) == len(expected), actual=len(rows), expected=len(expected)),
        _check(
            "weak_label_boundary_present",
            all(bool(row.get("weak_label_boundary")) for row in rows if isinstance(row, Mapping)),
            actual=len([row for row in rows if isinstance(row, Mapping) and row.get("weak_label_boundary")]),
            expected=len(rows),
        ),
    ]


def _coverage_check(name: str, actual: set[str], expected: set[str]) -> dict[str, object]:
    return _check(
        name,
        actual == expected,
        actual=sorted(actual),
        expected=sorted(expected),
        missing=sorted(expected - actual),
        extra=sorted(actual - expected),
    )


def _check(name: str, passed: bool, **metadata: object) -> dict[str, object]:
    row = {"name": name, "passed": bool(passed)}
    row.update(metadata)
    return row


def _input_values(rows: object, key: str) -> set[str]:
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get(key))
        for row in rows
        if isinstance(row, Mapping) and row.get(key) is not None
    }


def _row_values(rows: object, key: str) -> set[str]:
    if not isinstance(rows, list):
        return set()
    return {
        str(row.get(key))
        for row in rows
        if isinstance(row, Mapping) and row.get(key) is not None
    }


def _expected_identifiers(task_name: str, input_payload: Mapping[str, object]) -> list[str]:
    if task_name == "field_semantics":
        return sorted(_input_values(input_payload.get("fields", []), "feature_name"))
    if task_name == "weak_label_review":
        return sorted(_input_values(input_payload.get("task_summaries", []), "task_name"))
    if task_name == "schema_gap_policy":
        groups = input_payload.get("missing_measurement_groups", {})
        return sorted(str(group) for group in groups) if isinstance(groups, Mapping) else []
    if task_name == "runtime_explanations":
        return sorted(_input_values(input_payload.get("runtime_cases", []), "sample_id"))
    return []


def _output_count(task_name: str, output: object) -> int:
    if task_name == "schema_gap_policy" and isinstance(output, Mapping):
        policies = output.get("policies", [])
        return len(policies) if isinstance(policies, list) else 0
    if isinstance(output, list):
        return len(output)
    return 0


def _expected_top_level_key(task_name: str) -> str:
    return _output_key(task_name)


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


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
