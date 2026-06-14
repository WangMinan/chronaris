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

SYSTEM_PROMPT = """You are an audited preprocessing assistant for Chronaris Stage I.
Return JSON only. Treat all inputs as structured summaries, not raw truth.
Do not fabricate missing BUS values, do not turn weak labels into ground truth,
and do not claim native exact schema when only canonical exact is available."""

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
            "Do not use keys like role, unit, or measurement_role instead of the required keys.",
        ],
    },
    "weak_label_review": {
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
        }
    },
    "semantic_query_hints": {
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
        ],
    },
    "schema_gap_policy": {
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
            "Do not return a bare object mapping BUS id to policy.",
        ],
    },
    "runtime_explanations": {
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
        "instruction": instruction,
        "response_contract": TASK_RESPONSE_CONTRACTS[task_name],
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
