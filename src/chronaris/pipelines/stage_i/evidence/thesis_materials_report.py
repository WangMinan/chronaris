"""Markdown report rendering for Stage I thesis materials."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.evidence.thesis_materials import _build_weak_label_rows


def render_stage_i_thesis_materials_report(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[Mapping[str, object]],
    figure_entries: list[Mapping[str, object]],
) -> str:
    weak_label_rows = pd.DataFrame(_build_weak_label_rows(sources))
    best_rows = (
        weak_label_rows.sort_values("test_total")
        .groupby("sample_source", as_index=False)
        .first()[["sample_source", "sample_count", "task_entry_count", "best_test_total", "best_child_run_id"]]
    )
    summary_lines = [
        f"# Stage I Thesis Materials - {run_id}",
        "",
        "## Key Findings",
        "",
    ]
    for row in best_rows.to_dict(orient="records"):
        summary_lines.append(
            f"- `{row['sample_source']}`: sample_count=`{int(row['sample_count'])}`, "
            f"task_entry_count=`{int(row['task_entry_count'])}`, best_test_total=`{row['best_test_total']:.6f}`, "
            f"best_child_run_id=`{row['best_child_run_id']}`"
        )
    summary_lines.extend(
        [
            f"- rigid_body rotation audit: `{sources['rotation_audit']['payload']['rotation_status']}`",
            "",
            "## Tables",
            "",
        ]
    )
    summary_lines.extend(f"- `{entry['table_id']}`: `{entry['path']}`" for entry in table_entries)
    summary_lines.extend(["", "## Figures", ""])
    summary_lines.extend(f"- `{entry['figure_id']}`: `{entry['path']}`" for entry in figure_entries)
    return "\n".join(summary_lines)
