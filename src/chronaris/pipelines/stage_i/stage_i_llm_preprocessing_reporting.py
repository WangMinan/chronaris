"""Reporting helpers for Stage I LLM preprocessing."""

from __future__ import annotations

from typing import Mapping, Sequence


def render_stage_i_llm_preprocessing_report(
    *,
    summary: Mapping[str, object],
    comparison_rows: Sequence[Mapping[str, object]],
) -> str:
    """Render a compact Markdown report for the P20 run."""

    lines = [
        f"# Stage I LLM Preprocessing - {summary['run_id']}",
        "",
        f"- status: `{summary['status']}`",
        f"- mode: `{summary['mode']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- context_path: `{summary['context_path']}`",
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
        "## Weak-Label Comparison",
        "",
        "| task_name | decision | sample_count | agreement | conflict | human_review | agreement_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
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
