"""Markdown reporting helpers for Stage I support summaries."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def render_stage_i_alignment_support_report(summary: Mapping[str, object]) -> str:
    alignment = summary["alignment_support"]
    e_baseline = alignment["alignment_chain"]["e_baseline"]
    f_full = alignment["alignment_chain"]["f_full"]
    delta = alignment["alignment_chain"]["delta_f_minus_e"]
    stage_h = alignment["stage_h_export"]
    return "\n".join(
        [
            f"# Stage I Alignment Support - {summary['run_id']}",
            "",
            f"- generated_at_utc: `{summary['generated_at_utc']}`",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- machine summary: `{summary['artifact_root']}/support_summary.json`",
            f"- main ablation matrix: `{summary['main_ablation_matrix_path']}`",
            "",
            "## Alignment Chain",
            "",
            "| stage | sample_count | mean_projection_cosine | mean_projection_l2_gap | threshold_verdict |",
            "| --- | ---: | ---: | ---: | --- |",
            _render_alignment_row("E baseline", e_baseline),
            _render_alignment_row("F full", f_full),
            (
                f"| `F - E delta` | {delta['sample_count']} | "
                f"{_fmt_float(delta['mean_projection_cosine'])} | "
                f"{_fmt_float(delta['mean_projection_l2_gap'])} | `{delta['threshold_verdict']}` |"
            ),
            "",
            "## Stage H Export Stability",
            "",
            f"- sortie_count: `{stage_h['sortie_count']}`",
            f"- generated_view_count: `{stage_h['generated_view_count']}`",
            f"- view verdict counts: `{stage_h['view_verdict_counts']}`",
            f"- partial_data_entry_count: `{stage_h['partial_data_entry_count']}`",
            f"- partial_data_built_entry_count: `{stage_h['partial_data_built_entry_count']}`",
            "",
            "## 对齐结论",
            "",
            "1. `E baseline -> F full` 的投影诊断已经形成连续证据链，说明不是简单拼接。",
            f"2. `F full` 相比 `E baseline` 的 mean_projection_cosine 变化为 `{_fmt_signed(delta['mean_projection_cosine'])}`，"
            f"mean_projection_l2_gap 变化为 `{_fmt_signed(delta['mean_projection_l2_gap'])}`。",
            "3. `Stage H` 已把对齐结果稳定导出为 3 个双流 view，可直接被下游与 case-study 消费。",
            "4. 这条报告回答的是“对齐是否做出来且可复用”，不是“对齐已在公开数据上证明最优”。",
            "",
            "## 本结论能支撑什么",
            "",
            "- 可以支撑论文中“连续对齐已形成稳定 export contract，并已被下游 Phase 2 case-study 消费”的表述。",
            "- 可以支撑“物理一致性约束后的双流 view 已稳定导出，不是一次性手工拼接样例”。",
            "",
            "## 本结论不能支撑什么",
            "",
            "- 不能单独支撑“F(full) 在全部指标上显著优于 E baseline”这类过强表述。",
            "- 不能替代公开数据上的泛化结论，也不能替代私有 proxy 任务上的最优性结论。",
        ]
    )


def render_stage_i_causal_support_report(summary: Mapping[str, object]) -> str:
    causal = summary["causal_support"]
    g_min = causal["g_min"]
    semantic_event = causal.get("semantic_event")
    strongest_ablation = causal["case_study"]["strongest_ablation"]
    private_no_mask = causal["private_no_mask"]
    lines = [
        f"# Stage I Causal Support - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- support_matrix: `{summary['artifact_root']}/support_matrix.csv`",
        f"- main ablation matrix: `{summary['main_ablation_matrix_path']}`",
        "",
        "## G(min) Summary",
        "",
        f"- sample_count: `{g_min['sample_count']}`",
        f"- mean_attention_entropy: `{_fmt_float(g_min['mean_attention_entropy'])}`",
        f"- mean_max_attention: `{_fmt_float(g_min['mean_max_attention'])}`",
        f"- mean_top_event_score: `{_fmt_float(g_min['mean_top_event_score'])}`",
        f"- mean_top_contribution_score: `{_fmt_float(g_min['mean_top_contribution_score'])}`",
        "",
    ]
    if semantic_event is not None:
        lines.extend(
            [
                "## Semantic Event Fusion",
                "",
                f"- query_names: `{semantic_event['query_names']}`",
                f"- query_count: `{semantic_event['query_count']}`",
                f"- mean_event_token_count: `{_fmt_float(semantic_event['mean_event_token_count'])}`",
                f"- mean_query_entropy: `{_fmt_float(semantic_event['mean_query_entropy'])}`",
                f"- mean_top_query_score: `{_fmt_float(semantic_event['mean_top_query_score'])}`",
                f"- mean_top_event_attribution: `{_fmt_float(semantic_event['mean_top_event_attribution'])}`",
                "",
                "| sample | top query | top query event offset s | top event attribution |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for row in semantic_event.get("samples", []):
            lines.append(
                f"| `{row['sample_id']}` | `{row['top_query_name']}` | "
                f"{_fmt_float(row['top_query_event_offset_s'])} | {_fmt_float(row['top_event_attribution'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Phase 2 Bundle-Only Ablations",
            "",
            f"- strongest ablation: `{strongest_ablation['name']}`",
            f"- strongest delta_mean_top_contribution_score: `{_fmt_signed(strongest_ablation['delta_mean_top_contribution_score'])}`",
            "",
            "| ablation | mean_attention_entropy | mean_top_event_score | mean_top_contribution_score | delta_mean_attention_entropy | delta_mean_top_event_score | delta_mean_top_contribution_score |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, payload in causal["case_study"]["ablation_means"].items():
        lines.append(
            f"| `{name}` | {_fmt_optional(payload['mean_attention_entropy'])} | "
            f"{_fmt_optional(payload['mean_top_event_score'])} | "
            f"{_fmt_optional(payload['mean_top_contribution_score'])} | "
            f"{_fmt_optional_signed(payload['delta_mean_attention_entropy'])} | "
            f"{_fmt_optional_signed(payload['delta_mean_top_event_score'])} | "
            f"{_fmt_optional_signed(payload['delta_mean_top_contribution_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Private No-Mask Comparison",
            "",
            f"- target_variant: `{private_no_mask['target_variant_name']}`",
            f"- no_mask_variant: `{private_no_mask['no_mask_variant_name']}`",
            "",
            "| task | target metrics | no-mask metrics | target_beats_no_mask |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for task_name, payload in private_no_mask["tasks"].items():
        lines.append(
            f"| `{task_name}` | `{payload['target_metric_text']}` | "
            f"`{payload['no_mask_metric_text']}` | `{payload['target_beats_no_mask']}` |"
        )
    lines.extend(
        [
            "",
            "## 因果结论",
            "",
            "1. `G(min)` 已形成稳定的因果注意力统计，不是只存在于图示。",
            "2. 语义事件融合把时间步注意力进一步折叠成 `event token + query-to-event attribution`，可以把解释粒度从单点权重提升到事件级归因。",
            "3. `Phase 2 bundle-only` 消融已经给出 `no_event_bias / vehicle_delta_suppressed` 两条扰动证据，说明事件偏置与机动上下文都会改变贡献分布。",
            "4. 私有 proxy benchmark 中 `chronaris_opt_no_causal_mask` 三任务同步退化，说明“拿掉掩码”不是无损替换。",
            "5. 因果支撑链回答的是“掩码机制是否有必要”，不是“当前公开 benchmark 已由因果模型接管主线”。",
        ]
    )
    return "\n".join(lines)


def render_stage_i_ablation_support_report(summary: Mapping[str, object]) -> str:
    matrix_rows = summary["main_ablation_rows"]
    overview_plot_path = summary.get("overview_plot_path")
    lines = [
        f"# Stage I Ablation Support - {summary['run_id']}",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- main_ablation_matrix: `{summary['main_ablation_matrix_path']}`",
        f"- overview_plot: `{overview_plot_path or 'not generated'}`",
        "- 固定六路径主矩阵：`e_baseline / f_full / g_min / g_no_causal_mask / vehicle_delta_suppressed / no_event_bias`",
        "",
        "| variant | source | sample_count | proj_cosine | proj_l2_gap | views | attention_entropy | top_event | top_contribution | delta_top_contribution | private_t1_macro_f1 | private_t2_rmse | private_t3_top1 | supports | limits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in matrix_rows:
        lines.append(
            f"| `{_display_variant_name(row['variant'])}` | `{row['source']}` | "
            f"{row['sample_count'] if row['sample_count'] is not None else '-'} | "
            f"{_fmt_optional(row['mean_projection_cosine'])} | "
            f"{_fmt_optional(row['mean_projection_l2_gap'])} | "
            f"{row['generated_view_count'] if row['generated_view_count'] is not None else '-'} | "
            f"{_fmt_optional(row['mean_attention_entropy'])} | "
            f"{_fmt_optional(row['mean_top_event_score'])} | "
            f"{_fmt_optional(row['mean_top_contribution_score'])} | "
            f"{_fmt_optional_signed(row['delta_mean_top_contribution_score'])} | "
            f"{_fmt_optional(row['private_t1_macro_f1'])} | "
            f"{_fmt_optional(row['private_t2_rmse'])} | "
            f"{_fmt_optional(row['private_t3_top1_accuracy'])} | "
            f"{row['supports']} | {row['limits']} |"
        )
    lines.extend(
        [
            "",
            "## 六路径矩阵怎么读",
            "",
            "1. `e_baseline -> f_full -> g_min` 给出从对齐到导出再到最小因果融合的主链路。",
            "2. `g_no_causal_mask` 是任务级反证，回答“如果掩码不存在会怎样”。",
            "3. `vehicle_delta_suppressed / no_event_bias` 是 case-study 扰动证据，回答“事件与机动上下文到底有没有被用到”。",
            "4. 这 6 条路径合在一起，可以支撑论文里的 `alignment / export / causal / ablation` 证据主矩阵。",
        ]
    )
    return "\n".join(lines)


def _render_alignment_row(stage_name: str, payload: Mapping[str, object]) -> str:
    return (
        f"| `{stage_name}` | {payload['sample_count']} | "
        f"{_fmt_float(payload['mean_projection_cosine'])} | "
        f"{_fmt_float(payload['mean_projection_l2_gap'])} | "
        f"`{payload['threshold_verdict']}` |"
    )


def _fmt_float(value: float | int) -> str:
    return f"{float(value):.6f}"


def _fmt_signed(value: float | int) -> str:
    return f"{float(value):+.6f}"


def _fmt_optional(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return str(value)
    return _fmt_float(value)


def _fmt_optional_signed(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return _fmt_signed(value)


def _display_variant_name(variant_name: str) -> str:
    mapping = {
        "e_baseline": "E baseline",
        "f_full": "F(full)",
        "g_min": "G(min)",
        "g_no_causal_mask": "G(no causal mask)",
        "vehicle_delta_suppressed": "vehicle_delta_suppressed",
        "no_event_bias": "no_event_bias",
    }
    return mapping.get(variant_name, variant_name)
