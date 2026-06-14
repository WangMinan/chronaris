"""Reporting helpers for Stage I LLM preprocessing."""

from __future__ import annotations

from typing import Mapping, Sequence


def render_stage_i_llm_preprocessing_report(
    *,
    summary: Mapping[str, object],
    comparison_rows: Sequence[Mapping[str, object]],
) -> str:
    """Render a compact Markdown report for the P20 run."""

    harness = dict(summary.get("harness_summary", {}))
    slicing = dict(summary.get("slicing_summary", {}))
    lines = [
        f"# Stage I LLM Preprocessing - {summary['run_id']}",
        "",
        f"- status: `{summary['status']}`",
        f"- mode: `{summary['mode']}`",
        f"- prompt_version: `{summary.get('prompt_version')}`",
        f"- schema_version: `{summary.get('schema_version')}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- context_path: `{summary['context_path']}`",
        f"- harness_summary_path: `{summary.get('harness_summary_path')}`",
        f"- request_count: `{summary['request_count']}`",
        f"- error_count: `{summary['error_count']}`",
        "",
        "## Boundary",
        "",
        "LLM output is preprocessing context, semantic hints, rule review, and runtime explanation. It is not manual ground truth, does not fabricate missing BUS values, and does not convert canonical exact into native exact evidence.",
        "",
        "## Output Counts",
        "",
        f"- field_semantic_count: `{summary['field_semantic_count']}`",
        f"- weak_label_review_count: `{summary['weak_label_review_count']}`",
        f"- semantic_query_hint_count: `{summary['semantic_query_hint_count']}`",
        f"- runtime_explanation_count: `{summary['runtime_explanation_count']}`",
        "",
        "## Harness Gates",
        "",
        f"- attempt_count: `{harness.get('attempt_count')}`",
        f"- provider_failure_attempt_count: `{harness.get('provider_failure_attempt_count')}`",
        f"- schema_repair_attempt_count: `{harness.get('schema_repair_attempt_count')}`",
        f"- failed_initial_attempt_count: `{harness.get('failed_initial_attempt_count')}`",
        f"- final_invalid_task_count: `{harness.get('final_invalid_task_count')}`",
        "",
        "| task_name | attempt | valid | output_count | failed_gates |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in list(harness.get("task_verdicts", [])):
        failed = ", ".join(str(gate) for gate in row.get("failed_gates", []))
        lines.append(
            "| "
            f"`{row.get('task_name')}` | "
            f"`{row.get('attempt')}` | "
            f"`{row.get('valid')}` | "
            f"{row.get('output_count')} | "
            f"`{failed}` |"
        )
    lines.extend(
        [
            "",
            "## Payload Slicing",
            "",
            f"- initial_call_count: `{slicing.get('initial_call_count')}`",
            f"- sliced_task_count: `{slicing.get('sliced_task_count')}`",
            f"- sliced_tasks: `{', '.join(str(task) for task in slicing.get('sliced_tasks', []))}`",
            "",
            "| task_name | item_key | slice | item_count | total_item_count | sliced |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in list(slicing.get("slice_rows", [])):
        lines.append(
            "| "
            f"`{row.get('task_name')}` | "
            f"`{row.get('item_key')}` | "
            f"{row.get('slice_index')}/{row.get('slice_count')} | "
            f"{row.get('item_count')} | "
            f"{row.get('total_item_count')} | "
            f"`{row.get('sliced')}` |"
        )
    lines.extend(
        [
            "",
            "## Pipeline Integration",
            "",
            "- `field_semantic_dictionary` provides audited field-role hints for schema review before Stage I task building.",
            "- `weak_label_rule_review` is attached to thesis task entries as optional context; it does not overwrite labels.",
            "- `semantic_query_hints` are converted only through whitelisted semantic recipes before entering event-level fusion support.",
            "- `schema_gap_policy` keeps native-aligned versus canonical-exact runtime boundaries explicit.",
            "- `runtime_case_explanations` summarize predictions, schema gaps, semantic attribution, and weak-label boundaries for replay cases.",
            "",
            "## Next Comparison Work",
            "",
            "- Compare baseline Stage I task entries against LLM-context-attached entries with label values held fixed.",
            "- Compare semantic support with built-in query bank only versus built-in plus whitelisted LLM semantic hints.",
            "- Compare runtime explanation/report completeness with and without LLM preprocessing context.",
            "- Add human review on a small field/rule sample to measure whether LLM review reduces manual audit effort.",
            "",
        ]
    )
    lines.extend(
        [
        "## Weak-Label Comparison",
        "",
        "| task_name | decision | sample_count | agreement | conflict | human_review | agreement_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            f"`{row['task_name']}` | "
            f"`{row['review_decision']}` | "
            f"{row['sample_count']} | "
            f"{row['agreement_count']} | "
            f"{row['conflict_count']} | "
            f"{row['needs_human_review_count']} | "
            f"{row['label_agreement_rate']:.6f} |"
        )
    lines.extend(["", "## Source Paths", ""])
    for key, value in dict(summary.get("source_paths", {})).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)
