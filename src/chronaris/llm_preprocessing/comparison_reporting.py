"""Markdown reporting helpers for the P21 LLM comparison."""

from __future__ import annotations

from typing import Mapping, Sequence


def render_task_eval_llm_comparison_report(
    *,
    summary: Mapping[str, object],
    condition_manifest: Mapping[str, object],
    task_rows: Sequence[Mapping[str, object]],
    semantic_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    review_rows: Sequence[Mapping[str, object]],
) -> str:
    """Render the engineering-facing task evaluation report."""

    task = dict(summary["task_context"])
    semantic = dict(summary["semantic_hints"])
    runtime = dict(summary["runtime_explanations"])
    review = dict(summary["human_review_packet"])
    lines = [
        f"# task evaluation LLM Preprocessing Comparison - {summary['run_id']}",
        "",
        f"- status: `{summary['status']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- summary_path: `{summary['summary_path']}`",
        f"- condition_manifest_path: `{summary['condition_manifest_path']}`",
        f"- midterm_summary_path: `{summary['midterm_summary_path']}`",
        "",
        "## Boundary",
        "",
        "LLM outputs are used only as preprocessing context, whitelisted semantic hints, runtime explanations, and human-review material. They do not overwrite weak-label values, do not become manual truth, and do not prove the core causal-fusion claim.",
        "",
        "## Conditions",
        "",
        "| condition | name | status | key gate |",
        "| --- | --- | --- | --- |",
    ]
    for row in condition_manifest.get("conditions", []):
        gate = _condition_gate(row)
        lines.append(
            f"| `{row.get('condition_id')}` | `{row.get('name')}` | "
            f"`{row.get('status')}` | {gate} |"
        )
    lines.extend(
        [
            "",
            "## Result Summary",
            "",
            "| area | result | boundary |",
            "| --- | --- | --- |",
            (
                "| task context | "
                f"`{task['attached_entry_count']}/{task['task_entry_count']}` entries attached, "
                f"`label_unchanged={task['label_unchanged_text']}` | "
                "`no_label_overwrite` |"
            ),
            (
                "| semantic hints | "
                f"query coverage `{semantic['baseline_query_count']} -> {semantic['combined_query_count']}`, "
                f"added `{semantic['added_query_count']}` whitelisted hints | "
                "`ranking_not_recomputed_from_summary_only` |"
            ),
            (
                "| runtime explanations | "
                f"`{runtime['llm_explained_case_count']}/{runtime['runtime_case_count']}` cases have LLM explanations, "
                f"complete explained cases `{runtime['complete_with_llm_case_count']}` | "
                "`explanation_not_expert_truth` |"
            ),
            (
                "| human review packet | "
                f"`{review['item_count']}` items generated, "
                f"`human_review_completed={str(review['human_review_completed']).lower()}` | "
                "`pending_human_review` |"
            ),
            "",
            "## Task Context Comparison",
            "",
            "| task | baseline entries | attached entries | label unchanged | review decision | human review |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in task_rows:
        lines.append(
            f"| `{row['task_name']}` | {row['baseline_entry_count']} | "
            f"{row['llm_context_attached_entry_count']} | `{row['label_unchanged']}` | "
            f"`{row['weak_label_review_decision']}` | "
            f"`{row['weak_label_review_needs_human_review']}` |"
        )
    lines.extend(
        [
            "",
            "## Semantic Hint Comparison",
            "",
            "| query | condition | recipe | source | whitelisted | ranking status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in semantic_rows:
        lines.append(
            f"| `{row['query_name']}` | `{row['condition']}` | `{row['recipe']}` | "
            f"`{row['source']}` | `{row['recipe_whitelisted']}` | "
            f"`{row['view_ranking_change_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Runtime Explanation Comparison",
            "",
            "| sample | has LLM | without LLM | with LLM | delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in list(runtime_rows)[:12]:
        lines.append(
            f"| `{row['sample_id']}` | `{row['has_llm_explanation']}` | "
            f"{float(row['completeness_score_without_llm']):.2f} | "
            f"{float(row['completeness_score_with_llm']):.2f} | "
            f"{float(row['completeness_delta_for_explained_case']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Human Review Packet",
            "",
            "| item type | count | validation status |",
            "| --- | ---: | --- |",
        ]
    )
    for item_type, count in sorted(dict(review.get("item_counts", {})).items()):
        lines.append(f"| `{item_type}` | {count} | `pending_human_review` |")
    lines.extend(
        [
            "",
            "The packet contains empty human reviewer, decision, and notes fields. Until those fields are filled by a reviewer, this artifact is review material only, not completed validation.",
            "",
            "## Output Paths",
            "",
        ]
    )
    for key in (
        "task_context_comparison_path",
        "semantic_hint_comparison_path",
        "runtime_explanation_comparison_path",
        "human_review_packet_path",
        "midterm_claims_payload_path",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.extend(["", "## Source Paths", ""])
    for key, value in dict(summary.get("source_paths", {})).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def render_task_eval_llm_midterm_summary(
    *,
    summary: Mapping[str, object],
    condition_manifest: Mapping[str, object],
) -> str:
    """Render the Chinese midterm-facing summary required by P21."""

    del condition_manifest
    task = dict(summary["task_context"])
    semantic = dict(summary["semantic_hints"])
    runtime = dict(summary["runtime_explanations"])
    review = dict(summary["human_review_packet"])
    lines = [
        "# P21 LLM preprocessing 对比实验结果摘要",
        "",
        "更新时间：2026-06-14",
        "",
        "## 一句话结论",
        "",
        (
            "P21 已在现有 task evaluation weak-label 与 runtime 证据链上完成 A0-A4 本地对比："
            f"A1 验证 `{task['task_entry_count']}` 条 task entry 接入 LLM context 后 "
            f"`label_unchanged={task['label_unchanged_text']}`；A2 仅通过 whitelist 将 semantic query "
            f"coverage 从 `{semantic['baseline_query_count']}` 扩到 `{semantic['combined_query_count']}`；"
            f"A3 在 `{runtime['llm_explained_case_count']}` 个已有 LLM explanation case 上补齐 "
            "prediction、semantic attribution、schema gap note 和 weak-label boundary；"
            f"A4 生成 `{review['item_count']}` 条人工复核 packet，但人工未填写前不写成验证完成。"
        ),
        "",
        "## 实验设置",
        "",
        "| 条件 | 含义 | 本轮实现口径 |",
        "| --- | --- | --- |",
        "| A0 baseline | 不接 LLM context | 使用原始 task evaluation task entries、内置 3 条 semantic query 和 runtime case table |",
        "| A1 llm_context | 接入 P20 context | 只 attach context 和 rule review，代码审计 label value / label name 不变 |",
        "| A2 llm_semantic_hints | 接入 LLM hints | 仅允许 `coordination_gap / gap_plus_event / physiology_plus_gap / vehicle_plus_event` recipes |",
        "| A3 llm_runtime_explanation | runtime 解释层 | 对比 runtime case table 与 P20 `runtime_llm_explanations.jsonl` 的四项完整性 |",
        "| A4 human_review_packet | 人工复核材料 | 生成字段语义、weak-label 规则和 schema gap policy 小样本复核表 |",
        "",
        "## 结果表",
        "",
        "| 指标 | 结果 | 中期写作边界 |",
        "| --- | ---: | --- |",
        f"| task entries | `{task['task_entry_count']}` | thesis weak-label entries，不是人工真值 |",
        f"| LLM context 覆盖率 | `{task['context_coverage_rate']:.6f}` | 只新增 context，不改 label |",
        f"| label_unchanged | `{task['label_unchanged_text']}` | 由代码逐 entry 检查得出 |",
        f"| semantic query count | `{semantic['baseline_query_count']} -> {semantic['combined_query_count']}` | whitelist 接入；未用 summary 伪造 ranking 重算 |",
        f"| runtime explained cases | `{runtime['llm_explained_case_count']}/{runtime['runtime_case_count']}` | 有界 P20 explanation subset |",
        f"| explained case completeness delta | `{runtime['average_completeness_delta_for_explained_cases']:.6f}` | runtime explanation 完整性，不是专家复盘真值 |",
        f"| human review packet items | `{review['item_count']}` | `human_review_completed=false` |",
        "",
        "## 可以写进中期报告的表述",
        "",
        (
            "在不改写 `risk_proxy / workload_proxy / event_replay_tag` weak-label 值的前提下，"
            "DeepSeek P20 预处理 context 已通过 P21 对比实验接入 task evaluation 证据链："
            "它提供字段语义、规则复核、whitelisted semantic hints、schema gap policy 和 runtime explanation，"
            "主要增量体现在可审计上下文、解释完整性和人工复核材料组织。"
        ),
        "",
        "## 不能写的表述",
        "",
        "- 不能写成 DeepSeek 或 LLM 替代人工标注。",
        "- 不能写成 LLM semantic hints 证明核心因果融合模块。",
        "- 不能写成 human review 已完成；本轮只生成复核表。",
        "- 不能写成 runtime native input 已经 exact；当前仍是 native aligned / canonical exact 边界。",
        "",
        "## 后续计划",
        "",
        "- 由人工填写 `human_review_packet.csv` 后，再统计可采纳、需复核和冲突项。",
        "- 若要证明 semantic ranking/top attribution 变化，需要基于 feature export tensor 重新运行带 LLM query specs 的 support，而不是从现有 summary 倒推。",
        "- 继续保持 LLM 输出为 preprocessing context，不进入人工真值或核心因果证据层。",
        "",
        "## 资产路径",
        "",
        f"- 工程 summary：`{summary['summary_path']}`",
        f"- 条件 manifest：`{summary['condition_manifest_path']}`",
        f"- task comparison：`{summary['task_context_comparison_path']}`",
        f"- semantic comparison：`{summary['semantic_hint_comparison_path']}`",
        f"- runtime comparison：`{summary['runtime_explanation_comparison_path']}`",
        f"- human review packet：`{summary['human_review_packet_path']}`",
        f"- task evaluation 工程报告：`{summary['report_path']}`",
    ]
    return "\n".join(lines)


def _condition_gate(row: Mapping[str, object]) -> str:
    if row.get("condition_id") == "A1":
        return f"`label_unchanged={str(row.get('label_unchanged')).lower()}`"
    if row.get("condition_id") == "A2":
        return "`recipe_whitelist`"
    if row.get("condition_id") == "A3":
        return "`model_prediction / semantic_attribution / schema_gap_note / weak_label_boundary`"
    if row.get("condition_id") == "A4":
        return f"`human_review_completed={str(row.get('human_review_completed')).lower()}`"
    return "`control`"
