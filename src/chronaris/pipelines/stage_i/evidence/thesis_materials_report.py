"""Markdown report rendering for Stage I thesis materials."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from chronaris.pipelines.stage_i.evidence.thesis_materials_data import (
    build_llm_comparison_rows,
    build_runtime_payload_schema_rows,
    build_runtime_semantic_case_rows,
    build_semantic_event_rows,
    build_weak_label_rows,
)


def render_stage_i_thesis_materials_report(
    *,
    run_id: str,
    sources: Mapping[str, Mapping[str, object]],
    table_entries: list[Mapping[str, object]],
    figure_entries: list[Mapping[str, object]],
    font_note: str | None = None,
) -> str:
    weak = pd.DataFrame(build_weak_label_rows(sources))
    completed = weak.loc[weak["row_type"] == "completed_run"].copy() if not weak.empty else pd.DataFrame()
    if not completed.empty:
        completed["test_total_numeric"] = pd.to_numeric(completed["test_total"], errors="coerce")
        best_rows = completed.sort_values("test_total_numeric").groupby("sample_source", as_index=False).first()
    else:
        best_rows = pd.DataFrame()
    runtime_rows = build_runtime_payload_schema_rows(sources)
    runtime_case_rows = build_runtime_semantic_case_rows(sources)
    semantic_rows = build_semantic_event_rows(sources)
    llm_rows = build_llm_comparison_rows(sources)
    rotation = sources["rotation_audit"]["payload"]

    lines = [
        f"# Stage I Thesis Materials - {run_id}",
        "",
        "## 概览",
        "",
        f"- 本轮将中期报告/PPT 图表刷新为 `{len(figure_entries)}` 张 PNG 与对应 `{len(table_entries)}` 张 CSV，所有数值来自已有 artifact JSON/CSV 或本轮 rotation metadata audit。",
        "- 证据层级继续分开：thesis weak-label、private proxy、public adapter、runtime/schema、semantic support、rigid-body/rotation、LLM preprocessing/comparison 不合并为同一层结论。",
    ]
    for row in best_rows.to_dict(orient="records"):
        lines.append(
            f"- `{row['sample_source']}` weak-label sweep: sample_count=`{int(row['sample_count'])}`, "
            f"task_entry_count=`{int(row['task_entry_count'])}`, best_test_total=`{float(row['test_total']):.6f}`。"
        )
    native = next((row for row in runtime_rows if row["payload_side"] == "left"), {})
    canonical = next((row for row in runtime_rows if row["payload_side"] == "right"), {})
    if native and canonical:
        lines.append(
            f"- runtime/schema: native=`{native.get('schema_status')}` with vehicle `{native.get('vehicle_feature_count')}`, "
            f"canonical=`{canonical.get('schema_status')}` with vehicle `{canonical.get('vehicle_feature_count')}`。"
        )
    if runtime_case_rows:
        first_case = runtime_case_rows[0]
        query_names = sorted({str(row.get("semantic_top_query_name")) for row in runtime_case_rows})
        lines.append(
            f"- runtime semantic case: view_id=`{first_case.get('view_id')}`，windows=`{len(runtime_case_rows)}`，"
            f"query_types=`{','.join(query_names)}`，schema=`native {first_case.get('native_feature_schema_status')} / canonical {first_case.get('canonical_feature_schema_status')}`。"
        )
    if semantic_rows:
        lines.append(
            f"- semantic support: view_count=`{len(semantic_rows)}`, query_count=`{semantic_rows[0].get('query_count')}`，主图改为 event token / query-to-event attribution。"
        )
    llm_a2 = next((row for row in llm_rows if row["condition"] == "A2_llm_semantic_hints"), {})
    llm_a4 = next((row for row in llm_rows if row["condition"] == "A4_human_review_packet"), {})
    if llm_a2 and llm_a4:
        lines.append(
            f"- LLM comparison: `{llm_a2.get('metric_note')}`；human review packet `{llm_a4.get('metric_value')}` 条，仍为 pending review。"
        )
    lines.extend(
        [
            f"- rotation audit: `{rotation.get('rotation_status')}`；{rotation.get('rotation_reading')}。",
            "",
            "## 图表替换说明",
            "",
            "| figure_id | PNG | CSV | 替代的问题 |",
            "| --- | --- | --- | --- |",
        ]
    )
    table_by_id = {entry["table_id"]: entry for entry in table_entries}
    for figure in figure_entries:
        table_entry = table_by_id.get(str(figure["figure_id"]))
        csv_path = table_entry["path"] if table_entry else figure.get("table_path", "")
        lines.append(
            f"| `{figure['figure_id']}` | `{figure['path']}` | `{csv_path}` | {figure.get('replaces_problem', '')} |"
        )
    lines.extend(["", "## Tables", ""])
    lines.extend(
        f"- `{entry['table_id']}`: `{entry['path']}` (`{entry['row_count']}` rows, `{entry['column_count']}` columns)"
        for entry in table_entries
    )
    lines.extend(["", "## Figures", ""])
    lines.extend(
        f"- `{entry['figure_id']}`: `{entry['path']}` | evidence_layer=`{entry['evidence_layer']}`"
        for entry in figure_entries
    )
    lines.extend(
        [
            "",
            "## 仍受数据限制的边界",
            "",
            "- rotation：本轮已重新检查 MySQL metadata，pitch/roll/yaw angle 有候选，pitch_rate/roll_rate/yaw_rate 仍缺失，因此不复跑 rotation-enabled rigid-body 对照。",
            "- runtime：当前保持 native aligned / canonical exact；未声称生产级在线服务，也未声称原生 replay payload 已 exact。",
            "- weak-label sweep：本轮使用已有 stable/partial artifacts 重绘，不包装成大规模搜索。",
            "- LLM semantic hints：P21 已证明 query coverage `3 -> 7`，但没有从 summary 倒推出 view ranking 或 attribution 改善。",
        ]
    )
    if font_note:
        lines.extend(["", f"- Plot font: `{font_note}`"])
    return "\n".join(lines)
