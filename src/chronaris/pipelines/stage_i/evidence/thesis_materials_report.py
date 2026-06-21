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
        f"# 中期图表材料 - {run_id}",
        "",
        "## 概览",
        "",
        f"- 本轮将中期报告图表刷新为 `{len(figure_entries)}` 张 PNG 与对应 `{len(table_entries)}` 张 CSV，所有数值来自已有 JSON/CSV summary 或本轮旋转字段审计。",
        "- 证据层级继续分开：论文弱标注、私有代理、公开适配、运行字段契约、语义融合支撑、刚体/旋转诊断和大语言模型预处理对比分别解读。",
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
            f"- 运行字段契约：原始输入状态=`{native.get('schema_status')}`，飞机状态字段=`{native.get('vehicle_feature_count')}`；"
            f"统一输入状态=`{canonical.get('schema_status')}`，字段维度=`{canonical.get('vehicle_feature_count')}`。"
        )
    if runtime_case_rows:
        first_case = runtime_case_rows[0]
        query_names = sorted({str(row.get("semantic_top_query_name")) for row in runtime_case_rows})
        lines.append(
            f"- 代表性窗口案例：窗口数=`{len(runtime_case_rows)}`，查询类型=`{','.join(query_names)}`，"
            f"字段检查=`原始 {first_case.get('native_feature_schema_status')} / 统一 {first_case.get('canonical_feature_schema_status')}`。"
        )
    if semantic_rows:
        lines.append(
            f"- 语义融合支撑：视图记录=`{len(semantic_rows)}`，查询类型=`{semantic_rows[0].get('query_count')}`；缺少完整归因矩阵时仅展示覆盖/支撑状态。"
        )
    llm_a2 = next((row for row in llm_rows if row["condition"] == "A2_llm_semantic_hints"), {})
    llm_a4 = next((row for row in llm_rows if row["condition"] == "A4_human_review_packet"), {})
    if llm_a2 and llm_a4:
        lines.append(
            f"- 大语言模型预处理对比：`{llm_a2.get('metric_note')}`；复核材料 `{llm_a4.get('metric_value')}` 条，状态为待人工复核。"
        )
    lines.extend(
        [
            f"- 旋转字段诊断：`{rotation.get('rotation_status')}`；{rotation.get('rotation_reading')}。",
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
            "- 旋转诊断：本轮已重新检查 MySQL metadata，pitch/roll/yaw angle 有候选，pitch_rate/roll_rate/yaw_rate 仍缺失，因此不复跑启用旋转残差的刚体对照。",
            "- 运行字段契约：当前保持原始输入已对齐、统一输入已校验；未声称生产级在线服务，也未声称原始回放输入已经完全补齐。",
            "- 弱标注 sweep：本轮使用已有稳定/部分执行产物重绘，不包装成大规模搜索。",
            "- 大语言模型查询建议：当前对比只证明查询覆盖 `3 -> 7`，没有从 summary 倒推出视图排序或归因改善。",
        ]
    )
    if font_note:
        lines.extend(["", f"- Plot font: `{font_note}`"])
    return "\n".join(lines)
