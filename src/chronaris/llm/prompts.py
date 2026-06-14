"""Prompt builders for Stage I LLM preprocessing."""

from __future__ import annotations

import json
from typing import Mapping

from chronaris.llm.schemas import (
    LLMMessage,
    LLMTaskRequest,
    PROMPT_VERSION,
    SCHEMA_VERSION,
)

SYSTEM_PROMPT = """You are an audited, single-step preprocessing agent for Chronaris Stage I.
You do not browse, call tools, infer from private knowledge, or create new data.
Use only the user-provided JSON payload as evidence.

Hard output rules:
- Return exactly one JSON object and no markdown, prose, comments, or code fences.
- Use only the top-level key specified by response_contract.allowed_top_level_keys.
- Use only keys, enum values, and row shapes specified by response_contract.
- Preserve input identifiers exactly: feature_name, task_name, sample_id, measurement_group.
- If evidence is ambiguous, keep the row but set low confidence and needs_human_review=true.

Hard safety rules:
- Do not fabricate missing BUS values, units, labels, timestamps, source paths, or measurements.
- Do not turn weak labels into manual ground truth.
- Do not claim native exact schema when only canonical exact is available.
- Do not introduce free-form prompt recipes, model actions, hidden chain-of-thought, or policy text outside JSON."""

AGENT_PROTOCOL = {
    "execution_mode": "single_audited_preprocessing_step",
    "allowed_actions": [
        "normalize_field_semantics",
        "review_weak_label_rules",
        "suggest_whitelisted_semantic_query_hints",
        "propose_schema_gap_policy",
        "summarize_runtime_case_explanations",
    ],
    "forbidden_actions": [
        "fabricate_sensor_values",
        "overwrite_labels",
        "claim_manual_ground_truth",
        "claim_native_exact_from_canonical_payload",
        "add_unrequested_top_level_keys",
        "return_markdown_or_free_text",
        "invent_source_paths",
    ],
    "fallback_policy": (
        "When uncertain, keep the required row, lower confidence, and set "
        "needs_human_review=true instead of omitting the item."
    ),
}

TASK_INSTRUCTIONS = {
    "field_semantics": (
        "Build field_semantics rows. Use roles from the allowed set and mark low "
        "confidence or ambiguous fields as needs_human_review."
    ),
    "weak_label_review": (
        "Review risk_proxy, workload_proxy, and event_replay_tag rules. Return "
        "weak_label_rule_review rows. Do not treat the rules as manual truth."
    ),
    "semantic_query_hints": (
        "Return semantic_query_hints rows using only whitelisted recipes: "
        "gap_plus_event, physiology_plus_gap, vehicle_plus_event, coordination_gap."
    ),
    "schema_gap_policy": (
        "Return schema_gap_policy with policies using only drop_or_mask, "
        "canonical_fill_nan, source_requery_required, or human_review_required."
    ),
    "runtime_explanations": (
        "Return runtime_case_explanations rows. Each row must separate "
        "model_prediction, semantic_attribution, schema_gap_note, and weak_label_boundary."
    ),
}

TASK_RESPONSE_CONTRACTS = {
    "field_semantics": {
        "allowed_top_level_keys": ["field_semantics"],
        "required_json": {
            "field_semantics": [
                {
                    "stream_kind": "physiology|vehicle",
                    "measurement_group": "eeg",
                    "feature_name": "eeg.af3",
                    "llm_semantic_role": "physiological_load",
                    "llm_unit_guess": None,
                    "evidence": "schema_card field name and measurement group",
                    "confidence": 0.72,
                    "needs_human_review": True,
                }
            ]
        },
        "rules": [
            "Return one field_semantics row for each input_payload.fields item.",
            "The set of output feature_name values must exactly match input_payload.fields[*].feature_name.",
            "Do not use keys like role, unit, or measurement_role instead of the required keys.",
        ],
    },
    "weak_label_review": {
        "allowed_top_level_keys": ["weak_label_rule_review"],
        "required_json": {
            "weak_label_rule_review": [
                {
                    "task_name": "risk_proxy",
                    "review_decision": "keep_current_rule|revise_current_rule|needs_human_review",
                    "confidence": 0.68,
                    "agreement_basis": "why current weak-label rule is acceptable",
                    "conflict_basis": None,
                    "needs_human_review": True,
                    "recommended_action": "audit_before_training",
                }
            ]
        },
        "rules": [
            "Return one row for each input_payload.task_summaries item.",
            "The set of output task_name values must exactly match input_payload.task_summaries[*].task_name.",
        ],
    },
    "semantic_query_hints": {
        "allowed_top_level_keys": ["semantic_query_hints"],
        "required_json": {
            "semantic_query_hints": [
                {
                    "name": "llm_coordination_gap",
                    "recipe": "coordination_gap",
                    "confidence": 0.66,
                    "source": "semantic_support_card",
                    "needs_human_review": True,
                }
            ]
        },
        "rules": [
            "Use only recipes from input_payload.allowed_recipes.",
            "Every row must include a unique name and recipe.",
            "Do not return existing query names as new hints.",
        ],
    },
    "schema_gap_policy": {
        "allowed_top_level_keys": ["schema_gap_policy"],
        "required_json": {
            "schema_gap_policy": {
                "status": "draft_policy",
                "canonical_exact_boundary": "canonical exact is a service payload contract, not native exact evidence",
                "native_exact_claim_allowed": False,
                "policies": [
                    {
                        "measurement_group": "BUS6000019110021",
                        "policy_type": "canonical_fill_nan",
                        "reason": "missing BUS group remains explicit",
                        "allows_value_fabrication": False,
                        "needs_human_review": False,
                    }
                ],
            }
        },
        "rules": [
            "Return one policy row for each input_payload.missing_measurement_groups key.",
            "The set of output measurement_group values must exactly match input_payload.missing_measurement_groups keys.",
            "Do not return a bare object mapping BUS id to policy.",
        ],
    },
    "runtime_explanations": {
        "allowed_top_level_keys": ["runtime_case_explanations"],
        "required_json": {
            "runtime_case_explanations": [
                {
                    "sample_id": "sample-id",
                    "model_prediction": "risk/workload/event prediction",
                    "semantic_attribution": "query attribution summary",
                    "schema_gap_note": "native aligned, canonical exact, missing groups",
                    "weak_label_boundary": "weak_label_proxy_not_manual_ground_truth",
                    "source_paths": [],
                    "confidence": 0.64,
                    "needs_human_review": True,
                }
            ]
        },
        "rules": [
            "Return one explanation row for each input_payload.runtime_cases item.",
            "The set of output sample_id values must exactly match input_payload.runtime_cases[*].sample_id.",
        ],
    },
}


def build_llm_task_request(
    *,
    run_id: str,
    request_id: str,
    task_name: str,
    provider: str,
    model: str,
    input_payload: Mapping[str, object],
) -> LLMTaskRequest:
    """Build a stable audited request for one preprocessing task."""

    instruction = TASK_INSTRUCTIONS[task_name]
    user_payload = {
        "task_name": task_name,
        "request_id": request_id,
        "prompt_version": PROMPT_VERSION,
        "output_schema_version": SCHEMA_VERSION,
        "agent_protocol": AGENT_PROTOCOL,
        "instruction": instruction,
        "response_contract": TASK_RESPONSE_CONTRACTS[task_name],
        "local_harness_gates": _local_harness_gates(task_name),
        "input_payload": input_payload,
    }
    return LLMTaskRequest(
        request_id=request_id,
        run_id=run_id,
        task_name=task_name,
        provider=provider,
        model=model,
        prompt_version=PROMPT_VERSION,
        output_schema_version=SCHEMA_VERSION,
        input_payload=input_payload,
        messages=(
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=json.dumps(user_payload, ensure_ascii=False, indent=2),
            ),
        ),
    ).with_hash()


def _local_harness_gates(task_name: str) -> list[str]:
    """Describe local gates so the provider sees the same constraints as the harness."""

    common = [
        "valid_json_object",
        "allowed_top_level_keys_only",
        "required_output_key_present",
        "enum_values_whitelisted",
    ]
    task_gates = {
        "field_semantics": ["feature_name_coverage_exact", "one_row_per_input_field"],
        "weak_label_review": ["task_name_coverage_exact", "one_row_per_task_summary"],
        "semantic_query_hints": ["recipe_whitelist", "unique_hint_names", "no_existing_query_name_collision"],
        "schema_gap_policy": ["missing_measurement_group_coverage_exact", "no_value_fabrication"],
        "runtime_explanations": ["sample_id_coverage_exact", "source_path_preservation"],
    }
    return common + task_gates[task_name]
