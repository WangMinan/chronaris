"""A0-A4 builders for the task evaluation LLM comparison."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from chronaris.dataset import attach_llm_preprocessing_context_to_task_entries
from chronaris.models.fusion.semantic_event import (
    CausalEventFusionConfig,
    LLM_SEMANTIC_QUERY_RECIPE_WHITELIST,
    semantic_query_specs_from_llm_hints,
)

COMPLETENESS_FIELDS = (
    "model_prediction",
    "semantic_attribution",
    "schema_gap_note",
    "weak_label_boundary",
)


def build_task_context_comparison(
    sources: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    entries = tuple(sources["task_entries"])
    context = as_mapping(sources["llm_context"])
    attached = attach_llm_preprocessing_context_to_task_entries(entries, context)
    changed_by_task: Counter[str] = Counter()
    source_changed_by_task: Counter[str] = Counter()
    attached_by_task: Counter[str] = Counter()
    for baseline, candidate in zip(entries, attached, strict=True):
        if baseline.label_value != candidate.label_value or baseline.label_name != candidate.label_name:
            changed_by_task[baseline.task_name] += 1
        if baseline.label_source != candidate.label_source:
            source_changed_by_task[baseline.task_name] += 1
        if as_mapping(candidate.context_payload).get("llm_preprocessing_context"):
            attached_by_task[baseline.task_name] += 1

    review_by_task = {
        str(row.get("task_name")): dict(row)
        for row in context.get("weak_label_rule_review", [])
        if isinstance(row, Mapping) and row.get("task_name")
    }
    task_counts = Counter(entry.task_name for entry in entries)
    rows: list[dict[str, object]] = []
    for task_name in sorted(task_counts):
        entry_count = int(task_counts[task_name])
        review = review_by_task.get(task_name, {})
        attached_count = int(attached_by_task[task_name])
        rows.append(
            {
                "condition": "A1_llm_context",
                "baseline_condition": "A0_baseline",
                "task_name": task_name,
                "baseline_entry_count": entry_count,
                "llm_context_attached_entry_count": attached_count,
                "llm_context_coverage_rate": ratio(attached_count, entry_count),
                "weak_label_review_present": bool_text(bool(review)),
                "weak_label_review_decision": review.get("review_decision") or "not_reviewed",
                "weak_label_review_needs_human_review": bool_text(
                    bool(review.get("needs_human_review", False))
                ),
                "label_unchanged": bool_text(changed_by_task[task_name] == 0),
                "label_changed_count": int(changed_by_task[task_name]),
                "label_source_unchanged": bool_text(source_changed_by_task[task_name] == 0),
                "label_source_changed_count": int(source_changed_by_task[task_name]),
                "boundary": "llm_context_attached_without_label_overwrite",
            }
        )
    changed_count = sum(changed_by_task.values())
    attached_count = sum(attached_by_task.values())
    return rows, {
        "condition": "A1_llm_context",
        "baseline_condition": "A0_baseline",
        "task_entry_count": len(entries),
        "unique_sample_count": len({entry.sample_id for entry in entries}),
        "task_counts": dict(task_counts),
        "attached_entry_count": int(attached_count),
        "context_coverage_rate": ratio(attached_count, len(entries)),
        "weak_label_review_task_count": len(review_by_task),
        "label_changed_count": int(changed_count),
        "label_unchanged": changed_count == 0,
        "label_unchanged_text": bool_text(changed_count == 0),
    }


def build_semantic_hint_comparison(
    sources: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    context = as_mapping(sources["llm_context"])
    hints = context.get("semantic_query_hints", [])
    baseline_specs = tuple(CausalEventFusionConfig().query_specs)
    combined_specs = semantic_query_specs_from_llm_hints(hints, include_defaults=True)
    baseline_names = {spec.name for spec in baseline_specs}
    added_specs = tuple(spec for spec in combined_specs if spec.name not in baseline_names)
    rejected_hints = _rejected_semantic_hints(hints, baseline_names)
    semantic_event = _semantic_event_summary(sources["support_summary"])
    baseline_query_count = len(baseline_specs)
    combined_query_count = len(combined_specs)
    rows: list[dict[str, object]] = []
    for spec in combined_specs:
        is_added = spec.name not in baseline_names
        rows.append(
            {
                "condition": "A2_llm_semantic_hints" if is_added else "A0_baseline",
                "query_name": spec.name,
                "recipe": spec.recipe,
                "source": "P20_llm_semantic_hints" if is_added else "built_in_query_bank",
                "recipe_whitelisted": bool_text(spec.recipe in LLM_SEMANTIC_QUERY_RECIPE_WHITELIST),
                "baseline_query_count": baseline_query_count,
                "combined_query_count": combined_query_count,
                "query_coverage_delta": combined_query_count - baseline_query_count,
                "support_view_count": int(semantic_event.get("view_count", 0) or 0),
                "baseline_top_view_id": semantic_event.get("top_view_id"),
                "view_ranking_recomputed": bool_text(False),
                "view_ranking_change_status": "not_recomputed_from_summary_only",
                "top_attribution_change_status": "not_recomputed_from_summary_only",
                "boundary": "whitelisted_recipe_only_not_free_text_prompt",
            }
        )
    return rows, {
        "condition": "A2_llm_semantic_hints",
        "baseline_query_count": baseline_query_count,
        "combined_query_count": combined_query_count,
        "added_query_count": len(added_specs),
        "added_query_names": [spec.name for spec in added_specs],
        "rejected_hint_count": len(rejected_hints),
        "rejected_hints": rejected_hints,
        "recipe_whitelist": sorted(LLM_SEMANTIC_QUERY_RECIPE_WHITELIST),
        "view_count": int(semantic_event.get("view_count", 0) or 0),
        "baseline_top_view_id": semantic_event.get("top_view_id"),
        "view_ranking_recomputed": False,
        "view_ranking_change_status": "not_recomputed_from_summary_only",
        "boundary": "LLM hints extend deterministic query coverage only through whitelisted recipes.",
    }


def build_runtime_explanation_comparison(
    sources: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    runtime_cases = list(sources["runtime_cases"])
    explanations = {
        str(row.get("sample_id")): dict(row)
        for row in sources["runtime_explanations"]
        if isinstance(row, Mapping) and row.get("sample_id")
    }
    rows: list[dict[str, object]] = []
    explained_deltas: list[float] = []
    baseline_scores: list[float] = []
    with_llm_scores: list[float] = []
    complete_with_llm = 0
    for row in runtime_cases:
        sample_id = str(row.get("sample_id") or "")
        baseline_flags = _runtime_baseline_flags(row)
        explanation = explanations.get(sample_id)
        with_llm_flags = _runtime_llm_flags(explanation) if explanation else dict.fromkeys(COMPLETENESS_FIELDS, False)
        baseline_score = completeness_score(baseline_flags)
        with_llm_score = completeness_score(with_llm_flags)
        baseline_scores.append(baseline_score)
        if explanation:
            with_llm_scores.append(with_llm_score)
            explained_deltas.append(with_llm_score - baseline_score)
            if with_llm_score == 1.0:
                complete_with_llm += 1
        rows.append(
            {
                "sample_id": sample_id,
                "condition": "A3_llm_runtime_explanation" if explanation else "A0_baseline_only",
                "has_llm_explanation": bool_text(bool(explanation)),
                "model_prediction_without_llm": bool_text(baseline_flags["model_prediction"]),
                "semantic_attribution_without_llm": bool_text(baseline_flags["semantic_attribution"]),
                "schema_gap_note_without_llm": bool_text(baseline_flags["schema_gap_note"]),
                "weak_label_boundary_without_llm": bool_text(baseline_flags["weak_label_boundary"]),
                "completeness_score_without_llm": baseline_score,
                "model_prediction_with_llm": bool_text(with_llm_flags["model_prediction"]),
                "semantic_attribution_with_llm": bool_text(with_llm_flags["semantic_attribution"]),
                "schema_gap_note_with_llm": bool_text(with_llm_flags["schema_gap_note"]),
                "weak_label_boundary_with_llm": bool_text(with_llm_flags["weak_label_boundary"]),
                "completeness_score_with_llm": with_llm_score,
                "completeness_delta_for_explained_case": with_llm_score - baseline_score if explanation else 0.0,
                "native_feature_schema_status": row.get("native_feature_schema_status"),
                "canonical_feature_schema_status": row.get("canonical_feature_schema_status"),
                "boundary": "runtime_explanation_not_expert_truth",
            }
        )
    case_count = len(runtime_cases)
    explained_count = len([row for row in rows if row["has_llm_explanation"] == "true"])
    return rows, {
        "condition": "A3_llm_runtime_explanation",
        "runtime_case_count": case_count,
        "llm_explained_case_count": explained_count,
        "llm_explained_case_rate": ratio(explained_count, case_count),
        "baseline_average_completeness": mean(baseline_scores),
        "with_llm_average_completeness_for_explained_cases": mean(with_llm_scores),
        "average_completeness_delta_for_explained_cases": mean(explained_deltas),
        "complete_with_llm_case_count": complete_with_llm,
        "required_fields": list(COMPLETENESS_FIELDS),
        "status": "success_bounded_explained_subset",
    }


def build_human_review_packet(
    config: object,
    sources: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    context = as_mapping(sources["llm_context"])
    max_fields = int(getattr(config, "max_human_review_fields"))
    max_schema_gaps = int(getattr(config, "max_human_review_schema_gaps"))
    rows: list[dict[str, object]] = []
    for field in _priority_rows(context.get("field_semantic_dictionary", []))[:max_fields]:
        rows.append(
            _review_row(
                item_type="field_semantic",
                source_id=str(field.get("feature_name") or ""),
                llm_suggestion=(
                    f"role={field.get('llm_semantic_role')}; unit={field.get('llm_unit_guess')}; "
                    f"evidence={field.get('evidence')}"
                ),
                llm_decision="needs_review" if field.get("needs_human_review") else "candidate",
                confidence=field.get("confidence"),
                source_path=str(context.get("field_semantic_dictionary_path") or ""),
            )
        )
    for review in _priority_rows(context.get("weak_label_rule_review", [])):
        decision = str(review.get("review_decision") or "not_reviewed")
        rows.append(
            _review_row(
                item_type="weak_label_rule",
                source_id=str(review.get("task_name") or ""),
                llm_suggestion=(
                    f"decision={decision}; action={review.get('recommended_action')}; "
                    f"basis={review.get('agreement_basis') or review.get('conflict_basis')}"
                ),
                llm_decision=_review_decision_class(review),
                confidence=review.get("confidence"),
                source_path=str(sources["source_paths"]["llm_context_path"]),
            )
        )
    policies = as_mapping(context.get("schema_gap_policy")).get("policies", [])
    for policy in _priority_rows(policies)[:max_schema_gaps]:
        rows.append(
            _review_row(
                item_type="schema_gap_policy",
                source_id=str(policy.get("measurement_group") or ""),
                llm_suggestion=f"policy={policy.get('policy_type')}; reason={policy.get('reason')}",
                llm_decision="needs_human_review" if policy.get("needs_human_review") else "candidate",
                confidence=policy.get("confidence"),
                source_path=str(sources["source_paths"]["runtime_schema_contract_path"]),
            )
        )
    for index, row in enumerate(rows, start=1):
        row["review_item_id"] = f"P21-HR-{index:03d}"
    counts = Counter(str(row["item_type"]) for row in rows)
    return rows, {
        "condition": "A4_human_review_packet",
        "item_count": len(rows),
        "item_counts": dict(counts),
        "human_review_completed": False,
        "validation_status": "pending_human_review",
        "boundary": "packet_generated_only_not_manual_validation_complete",
    }


def build_condition_manifest(
    *,
    config: object,
    task_summary: Mapping[str, object],
    semantic_summary: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    review_summary: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": getattr(config, "run_id"),
        "conditions": [
            {
                "condition_id": "A0",
                "name": "baseline",
                "status": "completed",
                "control": "No LLM context attached; built-in semantic query bank only.",
            },
            {
                "condition_id": "A1",
                "name": "llm_context",
                "status": "completed" if task_summary["label_unchanged"] else "failed_label_changed",
                "label_unchanged": task_summary["label_unchanged"],
                "attached_entry_count": task_summary["attached_entry_count"],
            },
            {
                "condition_id": "A2",
                "name": "llm_semantic_hints",
                "status": "completed_coverage_only",
                "whitelist": semantic_summary["recipe_whitelist"],
                "view_ranking_recomputed": semantic_summary["view_ranking_recomputed"],
            },
            {
                "condition_id": "A3",
                "name": "llm_runtime_explanation",
                "status": runtime_summary["status"],
                "required_fields": runtime_summary["required_fields"],
            },
            {
                "condition_id": "A4",
                "name": "human_review_packet",
                "status": review_summary["validation_status"],
                "human_review_completed": review_summary["human_review_completed"],
            },
        ],
        "boundary": "LLM outputs remain preprocessing context and review material.",
    }


def build_midterm_claims_payload(
    *,
    task_summary: Mapping[str, object],
    semantic_summary: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    review_summary: Mapping[str, object],
    paths: Mapping[str, object],
) -> dict[str, object]:
    return {
        "claim_strength": "中强但限域",
        "evidence_layer": "llm_preprocessing_comparison",
        "allowed_claims": [
            (
                "P21 completed a bounded A0-A4 comparison showing LLM preprocessing "
                "can attach auditable context without changing weak-label values."
            ),
            (
                f"A1 label_unchanged={task_summary['label_unchanged_text']} across "
                f"{task_summary['task_entry_count']} task evaluation weak-label task entries."
            ),
            (
                f"A2 expanded deterministic semantic query coverage from "
                f"{semantic_summary['baseline_query_count']} to "
                f"{semantic_summary['combined_query_count']} through whitelisted recipes."
            ),
            (
                f"A3 LLM explanations fully covered required runtime fields for "
                f"{runtime_summary['complete_with_llm_case_count']} of "
                f"{runtime_summary['llm_explained_case_count']} explained cases."
            ),
            (
                f"A4 generated {review_summary['item_count']} human-review packet items; "
                "manual validation is still pending."
            ),
        ],
        "disallowed_claims": [
            "LLM output is manual ground truth.",
            "LLM semantic hints prove the core causal fusion module.",
            "P21 completes human validation before the review packet is filled.",
            "LLM explanations convert native aligned runtime input into native exact evidence.",
        ],
        "paths": dict(paths),
        "manual_validation_completed": False,
    }


def as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def completeness_score(flags: Mapping[str, bool]) -> float:
    return ratio(sum(1 for field in COMPLETENESS_FIELDS if flags.get(field)), len(COMPLETENESS_FIELDS))


def _semantic_event_summary(support_summary: object) -> Mapping[str, object]:
    return as_mapping(as_mapping(as_mapping(support_summary).get("causal_support")).get("semantic_event"))


def _runtime_baseline_flags(row: Mapping[str, object]) -> dict[str, bool]:
    return {
        "model_prediction": bool(
            row.get("risk_proxy_prediction")
            or row.get("workload_proxy_prediction")
            or row.get("event_replay_tag_prediction")
        ),
        "semantic_attribution": bool(
            row.get("semantic_top_query_name") and row.get("semantic_top_event_attribution")
        ),
        "schema_gap_note": bool(
            row.get("native_feature_schema_status")
            or row.get("canonical_feature_schema_status")
            or row.get("missing_vehicle_feature_count")
        ),
        "weak_label_boundary": bool(row.get("weak_label_boundary")),
    }


def _runtime_llm_flags(row: Mapping[str, object] | None) -> dict[str, bool]:
    if not row:
        return dict.fromkeys(COMPLETENESS_FIELDS, False)
    return {field: bool(row.get(field)) for field in COMPLETENESS_FIELDS}


def _review_row(
    *,
    item_type: str,
    source_id: str,
    llm_suggestion: str,
    llm_decision: str,
    confidence: object,
    source_path: str,
) -> dict[str, object]:
    return {
        "review_item_id": "",
        "item_type": item_type,
        "source_id": source_id,
        "llm_suggestion": llm_suggestion,
        "llm_decision": llm_decision,
        "confidence": confidence,
        "needs_human_review": "true",
        "source_path": source_path,
        "human_reviewer": "",
        "human_decision": "",
        "human_notes": "",
        "validation_status": "pending_human_review",
    }


def _review_decision_class(row: Mapping[str, object]) -> str:
    decision = str(row.get("review_decision") or "")
    if decision == "revise_current_rule":
        return "conflict_requires_review"
    if decision == "keep_current_rule" and row.get("needs_human_review"):
        return "adoptable_after_review"
    if decision == "keep_current_rule":
        return "candidate"
    return "needs_human_review"


def _priority_rows(rows: object) -> list[Mapping[str, object]]:
    if not isinstance(rows, (list, tuple)):
        return []
    normalized = [row for row in rows if isinstance(row, Mapping)]
    return sorted(normalized, key=lambda row: (not bool(row.get("needs_human_review")), str(row)))


def _rejected_semantic_hints(hints: object, baseline_names: set[str]) -> list[dict[str, object]]:
    rejected: list[dict[str, object]] = []
    if not isinstance(hints, (list, tuple)):
        return rejected
    for row in hints:
        if not isinstance(row, Mapping):
            rejected.append({"reason": "not_mapping", "hint": str(row)})
            continue
        name = str(row.get("name") or row.get("query_name") or "").strip()
        recipe = str(row.get("recipe") or "").strip()
        if not name:
            rejected.append({"reason": "missing_name", "recipe": recipe})
        elif name in baseline_names:
            rejected.append({"reason": "duplicate_baseline_name", "name": name, "recipe": recipe})
        elif recipe not in LLM_SEMANTIC_QUERY_RECIPE_WHITELIST:
            rejected.append({"reason": "recipe_not_whitelisted", "name": name, "recipe": recipe})
    return rejected
