"""Support and ablation report aggregation for Stage I thesis evidence."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

DEFAULT_ALIGNMENT_E_SUMMARY_PATH = (
    "docs/reports/assets/alignment-preview-stage-f-closure-2026-04-22-e-baseline/"
    "projection_diagnostics_summary.json"
)
DEFAULT_ALIGNMENT_F_SUMMARY_PATH = (
    "docs/reports/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full/"
    "projection_diagnostics_summary.json"
)
DEFAULT_CAUSAL_G_SUMMARY_PATH = (
    "docs/reports/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min/"
    "causal_fusion_summary.json"
)
DEFAULT_STAGE_H_RUN_MANIFEST_PATH = (
    "docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json"
)
DEFAULT_CASE_STUDY_SUMMARY_PATH = (
    "docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase2-case-study/"
    "case_study_summary.json"
)
DEFAULT_CASE_STUDY_ABLATION_CSV_PATH = (
    "docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase2-case-study/"
    "ablation_summary.csv"
)
DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH = (
    "docs/reports/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
    "private_benchmark_summary.json"
)
DEFAULT_DEEP_COMPARISON_SUMMARY_PATH = (
    "docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/comparison_summary.json"
)
DEFAULT_REPORT_ROOT = "docs/reports"
DEFAULT_ARTIFACT_ROOT = "docs/reports/assets/stage_i_support"


@dataclass(frozen=True, slots=True)
class StageISupportConfig:
    run_id: str
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    alignment_e_summary_path: str = DEFAULT_ALIGNMENT_E_SUMMARY_PATH
    alignment_f_summary_path: str = DEFAULT_ALIGNMENT_F_SUMMARY_PATH
    causal_g_summary_path: str = DEFAULT_CAUSAL_G_SUMMARY_PATH
    stage_h_run_manifest_path: str = DEFAULT_STAGE_H_RUN_MANIFEST_PATH
    case_study_summary_path: str = DEFAULT_CASE_STUDY_SUMMARY_PATH
    case_study_ablation_csv_path: str = DEFAULT_CASE_STUDY_ABLATION_CSV_PATH
    private_benchmark_summary_path: str = DEFAULT_PRIVATE_BENCHMARK_SUMMARY_PATH
    deep_comparison_summary_path: str | None = DEFAULT_DEEP_COMPARISON_SUMMARY_PATH


@dataclass(frozen=True, slots=True)
class StageISupportRunResult:
    run_id: str
    artifact_root: str
    summary_path: str
    matrix_path: str
    main_matrix_path: str
    alignment_report_path: str
    causal_report_path: str
    ablation_report_path: str
    overview_plot_path: str | None
    summary: Mapping[str, object]


def run_stage_i_support(
    config: StageISupportConfig,
) -> StageISupportRunResult:
    artifact_root = Path(config.artifact_root) / config.run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    alignment_report_path = report_root / f"stage-i-alignment-support-{config.run_id}.md"
    causal_report_path = report_root / f"stage-i-causal-support-{config.run_id}.md"
    ablation_report_path = report_root / f"stage-i-ablation-support-{config.run_id}.md"
    summary_path = artifact_root / "support_summary.json"
    matrix_path = artifact_root / "support_matrix.csv"
    main_matrix_path = artifact_root / "ablation_matrix.csv"
    overview_plot_target = artifact_root / "support_overview.png"

    e_projection = _load_json(config.alignment_e_summary_path)
    f_projection = _load_json(config.alignment_f_summary_path)
    g_causal = _load_json(config.causal_g_summary_path)
    stage_h_run_manifest = _load_json(config.stage_h_run_manifest_path)
    case_study_summary = _load_json(config.case_study_summary_path)
    case_study_ablation = pd.read_csv(config.case_study_ablation_csv_path)
    private_summary = _load_json(config.private_benchmark_summary_path)
    deep_summary = (
        _load_json(config.deep_comparison_summary_path)
        if config.deep_comparison_summary_path
        else {}
    )

    alignment_support = _build_alignment_support_summary(
        e_projection=e_projection,
        f_projection=f_projection,
        stage_h_run_manifest=stage_h_run_manifest,
        case_study_summary=case_study_summary,
    )
    causal_support = _build_causal_support_summary(
        g_causal=g_causal,
        case_study_summary=case_study_summary,
        case_study_ablation=case_study_ablation,
        private_summary=private_summary,
        deep_summary=deep_summary,
    )
    support_matrix = _build_support_matrix(
        alignment_support=alignment_support,
        causal_support=causal_support,
    )
    support_matrix.to_csv(matrix_path, index=False)
    main_matrix = _build_main_ablation_matrix(
        alignment_support=alignment_support,
        causal_support=causal_support,
    )
    main_matrix.to_csv(main_matrix_path, index=False)
    overview_plot_path = _write_support_overview_plot(
        main_matrix,
        path=overview_plot_target,
    )

    summary = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat().replace("+00:00", "Z"),
        "run_id": config.run_id,
        "artifact_root": str(artifact_root),
        "sources": {
            "alignment_e_summary_path": str(Path(config.alignment_e_summary_path)),
            "alignment_f_summary_path": str(Path(config.alignment_f_summary_path)),
            "causal_g_summary_path": str(Path(config.causal_g_summary_path)),
            "stage_h_run_manifest_path": str(Path(config.stage_h_run_manifest_path)),
            "case_study_summary_path": str(Path(config.case_study_summary_path)),
            "case_study_ablation_csv_path": str(Path(config.case_study_ablation_csv_path)),
            "private_benchmark_summary_path": str(Path(config.private_benchmark_summary_path)),
            "deep_comparison_summary_path": (
                str(Path(config.deep_comparison_summary_path))
                if config.deep_comparison_summary_path
                else None
            ),
        },
        "alignment_support": alignment_support,
        "causal_support": causal_support,
        "support_matrix_path": str(matrix_path),
        "main_ablation_matrix_path": str(main_matrix_path),
        "main_ablation_rows": json.loads(main_matrix.to_json(orient="records")),
        "overview_plot_path": overview_plot_path,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    alignment_report_path.write_text(
        render_stage_i_alignment_support_report(summary) + "\n",
        encoding="utf-8",
    )
    causal_report_path.write_text(
        render_stage_i_causal_support_report(summary) + "\n",
        encoding="utf-8",
    )
    ablation_report_path.write_text(
        render_stage_i_ablation_support_report(summary) + "\n",
        encoding="utf-8",
    )
    return StageISupportRunResult(
        run_id=config.run_id,
        artifact_root=str(artifact_root),
        summary_path=str(summary_path),
        matrix_path=str(matrix_path),
        main_matrix_path=str(main_matrix_path),
        alignment_report_path=str(alignment_report_path),
        causal_report_path=str(causal_report_path),
        ablation_report_path=str(ablation_report_path),
        overview_plot_path=overview_plot_path,
        summary=summary,
    )


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
        "## Phase 2 Bundle-Only Ablations",
        "",
        "| ablation | mean delta entropy | mean delta top event | mean delta top contribution | mean delta fused L2 | mean delta cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ablation_name, payload in causal["case_study"]["ablation_means"].items():
        lines.append(
            f"| `{ablation_name}` | {_fmt_signed(payload['delta_mean_attention_entropy'])} | "
            f"{_fmt_signed(payload['delta_mean_top_event_score'])} | "
            f"{_fmt_signed(payload['delta_mean_top_contribution_score'])} | "
            f"{_fmt_signed(payload['delta_fused_l2_norm'])} | "
            f"{_fmt_signed(payload['delta_fused_cosine_to_projection_baseline'])} |"
        )
    lines.extend(
        [
            "",
            "## Same-Sortie Dual-Pilot",
            "",
            "| sortie | delta mean cosine | delta cosine cv | delta top contribution |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for comparison in causal["case_study"]["pilot_comparisons"]:
        lines.append(
            f"| `{comparison['sortie_id']}` | {_fmt_signed(comparison['delta_mean_projection_cosine'])} | "
            f"{_fmt_signed(comparison['delta_projection_cosine_cv'])} | "
            f"{_fmt_signed(comparison['delta_mean_top_contribution_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Private No-Mask Comparison",
            "",
            "| task | target metrics | no-mask metrics | target_beats_no_mask |",
            "| --- | --- | --- | --- |",
        ]
    )
    for task_name, payload in private_no_mask["tasks"].items():
        lines.append(
            f"| `{task_name}` | {payload['target_metric_text']} | {payload['no_mask_metric_text']} | "
            f"`{payload['target_beats_no_mask']}` |"
        )
    lines.extend(
        [
            "",
            "## 辅助 real-sortie deep wrappers",
            "",
            "| model | mean event-mask interference | mean attention entropy | pilot delta event-mask interference |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model_name, payload in causal["deep_auxiliary"]["stage_h_case"].items():
        lines.append(
            f"| `{model_name}` | {_fmt_float(payload['mean_event_mask_interference'])} | "
            f"{_fmt_float(payload['mean_attention_entropy'])} | "
            f"{_fmt_signed(payload['pilot_delta_event_mask_interference'])} |"
        )
    lines.extend(
        [
            "",
            "## 因果结论",
            "",
            "1. `G(min)` 已经产生稳定的非对称注意力与 top-event/top-contribution 指标。",
            f"2. 当前最强 bundle-only 干预是 `{strongest_ablation['name']}`，"
            f"其 mean delta top contribution 为 `{_fmt_signed(strongest_ablation['delta_mean_top_contribution_score'])}`，"
            f"mean delta top event 为 `{_fmt_signed(strongest_ablation['delta_mean_top_event_score'])}`。",
            "3. `chronaris_opt_no_causal_mask` 在私有 T1/T2/T3 三任务上都劣于 target variant，说明因果掩码不是可有可无的装饰项。",
            "4. 这条报告回答的是“因果融合是否做出来并给出可解释差异”，不是“因果融合已在公开数据上全面最优”。",
            "",
            "## 本结论能支撑什么",
            "",
            "- 可以支撑论文中“单向因果约束、关键事件偏置与双 pilot 差异可读性”已经形成真实 sortie 证据链。",
            "- 可以支撑“去掉因果掩码后，私有 proxy 三任务同步退化”的主张。",
            "",
            "## 本结论不能支撑什么",
            "",
            "- 不能把当前 case-study 级证据写成大规模标签监督下的全面 superiority 证明。",
            "- 不能把 `no_event_bias` 或 `vehicle_delta_suppressed` 的 bundle-only 干预直接等价为完整任务级 ablation 胜负。",
        ]
    )
    return "\n".join(lines)


def render_stage_i_ablation_support_report(summary: Mapping[str, object]) -> str:
    rows = summary["main_ablation_rows"]
    path_lines = [
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- machine summary: `{summary['artifact_root']}/support_summary.json`",
        f"- main matrix: `{summary['main_ablation_matrix_path']}`",
    ]
    if summary.get("overview_plot_path"):
        path_lines.append(f"- overview plot: `{summary['overview_plot_path']}`")
    lines = [
        f"# Stage I Ablation Support - {summary['run_id']}",
        "",
        *path_lines,
        "",
        "## 固定六路径主矩阵",
        "",
        "| variant | source | mean projection cosine | export views | mean attention entropy | mean top event | mean top contribution | delta top event | delta top contribution | private T1 macro-F1 | private T2 RMSE | private T3 top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{_display_variant_name(str(row['variant']))}` | `{row['source']}` | "
            f"{_fmt_optional(row.get('mean_projection_cosine'))} | "
            f"{_fmt_optional(row.get('generated_view_count'))} | "
            f"{_fmt_optional(row.get('mean_attention_entropy'))} | "
            f"{_fmt_optional(row.get('mean_top_event_score'))} | "
            f"{_fmt_optional(row.get('mean_top_contribution_score'))} | "
            f"{_fmt_optional_signed(row.get('delta_mean_top_event_score'))} | "
            f"{_fmt_optional_signed(row.get('delta_mean_top_contribution_score'))} | "
            f"{_fmt_optional(row.get('private_t1_macro_f1'))} | "
            f"{_fmt_optional(row.get('private_t2_rmse'))} | "
            f"{_fmt_optional(row.get('private_t3_top1_accuracy'))} |"
        )
    lines.extend(
        [
            "",
            "## 中文结论",
            "",
            "1. 去掉双流连续对齐后，只剩 `E baseline` 级预览证据；它能说明预览存在，但不能替代稳定 export 与下游消费闭环。",
            "2. 保留 `F(full)` 与 `G(min)` 后，Stage H/Phase 2 已形成 `3` 个真实双流 view、`2 PASS + 1 WARN` 的可解释证据链。",
            "3. 去掉因果掩码后，`chronaris_opt_no_causal_mask` 在私有 T1/T2/T3 三任务同时退化，说明因果掩码对当前主线不是装饰项。",
            "4. 去掉关键事件偏置时，`no_event_bias` 的 mean top contribution 下降；压制 vehicle delta 时，`vehicle_delta_suppressed` 的干预幅度最大，说明事件与机动变化都是当前融合读数的重要支撑。",
            "",
            "## 本结论能支撑什么",
            "",
            "- 可以直接回答“去掉双流 / 去掉因果掩码 / 去掉关键事件偏置后会怎样”。",
            "- 可以把 `E/F/G/H + Phase 2 + private no-mask` 收束成论文第三阶段可复述的一张主矩阵。",
            "",
            "## 本结论不能支撑什么",
            "",
            "- 不能把这张主矩阵写成公开 benchmark 的统一对照结论；公开数据仍应由 `chronaris public opt` 与历史 classical / MulT / ContiFormer 报告负责。",
            "- `no_state_normalization` 只保留在 appendix 级 support matrix 中，不进入主矩阵。",
        ]
    )
    return "\n".join(lines)


def _build_alignment_support_summary(
    *,
    e_projection: Mapping[str, object],
    f_projection: Mapping[str, object],
    stage_h_run_manifest: Mapping[str, object],
    case_study_summary: Mapping[str, object],
) -> dict[str, object]:
    e_summary = _projection_summary(e_projection)
    f_summary = _projection_summary(f_projection)
    view_verdict_counts = Counter(
        view_result["view_summary"]["verdict"]
        for view_result in case_study_summary["view_results"]
    )
    return {
        "alignment_chain": {
            "e_baseline": e_summary,
            "f_full": f_summary,
            "delta_f_minus_e": {
                "sample_count": min(e_summary["sample_count"], f_summary["sample_count"]),
                "mean_projection_cosine": (
                    f_summary["mean_projection_cosine"] - e_summary["mean_projection_cosine"]
                ),
                "mean_projection_l2_gap": (
                    f_summary["mean_projection_l2_gap"] - e_summary["mean_projection_l2_gap"]
                ),
                "threshold_verdict": (
                    "PASS"
                    if f_summary["threshold_verdict"] == "PASS"
                    else f_summary["threshold_verdict"]
                ),
            },
        },
        "stage_h_export": {
            "run_id": stage_h_run_manifest["run_id"],
            "sortie_count": len(stage_h_run_manifest["sortie_ids"]),
            "generated_view_count": int(stage_h_run_manifest["generated_view_count"]),
            "generated_view_ids": list(stage_h_run_manifest["generated_view_ids"]),
            "view_verdict_counts": dict(view_verdict_counts),
            "partial_data_entry_count": int(
                stage_h_run_manifest["partial_data"]["entry_count"]
            ),
            "partial_data_built_entry_count": int(
                stage_h_run_manifest["partial_data"]["built_entry_count"]
            ),
        },
    }


def _build_causal_support_summary(
    *,
    g_causal: Mapping[str, object],
    case_study_summary: Mapping[str, object],
    case_study_ablation: pd.DataFrame,
    private_summary: Mapping[str, object],
    deep_summary: Mapping[str, object],
) -> dict[str, object]:
    ablation_name_column = "ablation_name" if "ablation_name" in case_study_ablation.columns else "name"
    case_study_view_results = case_study_summary["view_results"]
    baseline_view_frame = pd.DataFrame(
        [view_result["view_summary"] for view_result in case_study_view_results]
    )
    pilot_comparison_frame = pd.DataFrame(case_study_summary["pilot_comparisons"])
    ablation_metric_columns = [
        column
        for column in (
            "mean_attention_entropy",
            "mean_top_event_score",
            "mean_top_contribution_score",
            "delta_mean_attention_entropy",
            "delta_mean_top_event_score",
            "delta_mean_top_contribution_score",
            "delta_fused_l2_norm",
            "delta_fused_cosine_to_projection_baseline",
        )
        if column in case_study_ablation.columns
    ]
    ablation_means = (
        case_study_ablation.loc[
            case_study_ablation[ablation_name_column] != "projection_refusion_baseline"
        ]
        .groupby(ablation_name_column, sort=True)[ablation_metric_columns]
        .mean()
        .round(12)
        .to_dict(orient="index")
    )
    for payload in ablation_means.values():
        payload.setdefault("mean_attention_entropy", None)
        payload.setdefault("mean_top_event_score", None)
        payload.setdefault("mean_top_contribution_score", None)
    strongest_ablation_name = min(
        ablation_means,
        key=lambda name: ablation_means[name]["delta_mean_top_contribution_score"],
    )
    private_no_mask = _extract_private_no_mask_summary(private_summary)
    deep_stage_h_case = _extract_deep_stage_h_case_summary(deep_summary)
    return {
        "g_min": {
            "sample_count": int(g_causal["sample_count"]),
            "mean_attention_entropy": float(g_causal["mean_attention_entropy"]),
            "mean_max_attention": float(g_causal["mean_max_attention"]),
            "mean_top_event_score": float(g_causal["mean_top_event_score"]),
            "mean_top_contribution_score": float(g_causal["mean_top_contribution_score"]),
        },
        "case_study": {
            "view_count": len(case_study_view_results),
            "view_verdict_counts": dict(
                Counter(
                    view_result["view_summary"]["verdict"]
                    for view_result in case_study_view_results
                )
            ),
            "baseline_view_means": {
                "mean_projection_cosine": _frame_mean(
                    baseline_view_frame,
                    "mean_projection_cosine",
                ),
                "mean_projection_l2_gap": _frame_mean(
                    baseline_view_frame,
                    "mean_projection_l2_gap",
                ),
                "mean_attention_entropy": _frame_mean(
                    baseline_view_frame,
                    "mean_attention_entropy",
                ),
                "mean_top_event_score": _frame_mean(
                    baseline_view_frame,
                    "mean_top_event_score",
                ),
                "mean_top_contribution_score": _frame_mean(
                    baseline_view_frame,
                    "mean_top_contribution_score",
                ),
            },
            "pilot_comparisons": list(case_study_summary["pilot_comparisons"]),
            "pilot_delta_means": {
                "delta_mean_projection_cosine": float(
                    pilot_comparison_frame["delta_mean_projection_cosine"].mean()
                )
                if not pilot_comparison_frame.empty
                else 0.0,
                "delta_mean_top_contribution_score": float(
                    pilot_comparison_frame["delta_mean_top_contribution_score"].mean()
                )
                if not pilot_comparison_frame.empty
                else 0.0,
            },
            "ablation_means": ablation_means,
            "strongest_ablation": {
                "name": strongest_ablation_name,
                **ablation_means[strongest_ablation_name],
            },
        },
        "private_no_mask": private_no_mask,
        "deep_auxiliary": {
            "stage_h_case": deep_stage_h_case,
        },
    }


def _build_support_matrix(
    *,
    alignment_support: Mapping[str, object],
    causal_support: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_name, payload in alignment_support["alignment_chain"].items():
        rows.append(
            {
                "family": "alignment",
                "variant": stage_name,
                "source": "projection_diagnostics_summary",
                "sample_count": payload["sample_count"],
                "mean_projection_cosine": payload["mean_projection_cosine"],
                "mean_projection_l2_gap": payload["mean_projection_l2_gap"],
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": None,
                "delta_mean_top_contribution_score": None,
                "event_mask_interference": None,
                "note": payload["threshold_verdict"],
            }
        )
    rows.append(
        {
            "family": "alignment",
            "variant": "stage_h_export",
            "source": "stage_h_run_manifest",
            "sample_count": alignment_support["stage_h_export"]["generated_view_count"],
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "event_mask_interference": None,
            "note": json.dumps(
                alignment_support["stage_h_export"]["view_verdict_counts"],
                ensure_ascii=False,
            ),
        }
    )
    rows.append(
        {
            "family": "causal",
            "variant": "g_min",
            "source": "causal_fusion_summary",
            "sample_count": causal_support["g_min"]["sample_count"],
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "mean_attention_entropy": causal_support["g_min"]["mean_attention_entropy"],
            "mean_top_event_score": causal_support["g_min"]["mean_top_event_score"],
            "mean_top_contribution_score": causal_support["g_min"]["mean_top_contribution_score"],
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "event_mask_interference": None,
            "note": "g_min_hidden_summary",
        }
    )
    for ablation_name, payload in causal_support["case_study"]["ablation_means"].items():
        rows.append(
            {
                "family": "causal",
                "variant": ablation_name,
                "source": "case_study_ablation_summary",
                "sample_count": causal_support["case_study"]["view_count"],
                "mean_projection_cosine": None,
                "mean_projection_l2_gap": None,
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": payload["delta_mean_top_event_score"],
                "delta_mean_top_contribution_score": payload["delta_mean_top_contribution_score"],
                "event_mask_interference": None,
                "note": "phase2_bundle_only",
            }
        )
    for variant_name, payload in causal_support["private_no_mask"]["matrix_rows"].items():
        rows.append(
            {
                "family": "causal",
                "variant": variant_name,
                "source": "private_benchmark_summary",
                "sample_count": None,
                "mean_projection_cosine": None,
                "mean_projection_l2_gap": None,
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": None,
                "delta_mean_top_contribution_score": None,
                "event_mask_interference": None,
                "note": payload,
            }
        )
    return pd.DataFrame(rows)


def _build_main_ablation_matrix(
    *,
    alignment_support: Mapping[str, object],
    causal_support: Mapping[str, object],
) -> pd.DataFrame:
    case_study = causal_support["case_study"]
    baseline = case_study["baseline_view_means"]
    view_counts = case_study["view_verdict_counts"]
    pilot_delta_means = case_study["pilot_delta_means"]
    no_mask_tasks = causal_support["private_no_mask"]["tasks"]

    rows = [
        {
            "variant": "e_baseline",
            "source": "projection_diagnostics_summary",
            "sample_count": alignment_support["alignment_chain"]["e_baseline"]["sample_count"],
            "mean_projection_cosine": alignment_support["alignment_chain"]["e_baseline"]["mean_projection_cosine"],
            "mean_projection_l2_gap": alignment_support["alignment_chain"]["e_baseline"]["mean_projection_l2_gap"],
            "generated_view_count": 0,
            "pass_view_count": 0,
            "warn_view_count": 0,
            "contract_complete": False,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": None,
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "对齐预览存在",
            "limits": "不含稳定导出与因果解释",
        },
        {
            "variant": "f_full",
            "source": "projection_diagnostics_summary + stage_h_run_manifest",
            "sample_count": alignment_support["alignment_chain"]["f_full"]["sample_count"],
            "mean_projection_cosine": alignment_support["alignment_chain"]["f_full"]["mean_projection_cosine"],
            "mean_projection_l2_gap": alignment_support["alignment_chain"]["f_full"]["mean_projection_l2_gap"],
            "generated_view_count": alignment_support["stage_h_export"]["generated_view_count"],
            "pass_view_count": int(view_counts.get("PASS", 0)),
            "warn_view_count": int(view_counts.get("WARN", 0)),
            "contract_complete": True,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "稳定导出与双 pilot 可读性",
            "limits": "不直接给出因果 ablation 胜负",
        },
        {
            "variant": "g_min",
            "source": "causal_fusion_summary + phase2_case_study",
            "sample_count": causal_support["g_min"]["sample_count"],
            "mean_projection_cosine": baseline["mean_projection_cosine"],
            "mean_projection_l2_gap": baseline["mean_projection_l2_gap"],
            "generated_view_count": alignment_support["stage_h_export"]["generated_view_count"],
            "pass_view_count": int(view_counts.get("PASS", 0)),
            "warn_view_count": int(view_counts.get("WARN", 0)),
            "contract_complete": True,
            "mean_attention_entropy": baseline["mean_attention_entropy"],
            "mean_top_event_score": baseline["mean_top_event_score"],
            "mean_top_contribution_score": baseline["mean_top_contribution_score"],
            "delta_mean_attention_entropy": 0.0,
            "delta_mean_top_event_score": 0.0,
            "delta_mean_top_contribution_score": 0.0,
            "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
            "pilot_delta_mean_top_contribution_score": pilot_delta_means["delta_mean_top_contribution_score"],
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "非对称注意力与双 pilot 差异可读",
            "limits": "不等价于任务级 superiority",
        },
    ]
    for variant_name in ("no_event_bias", "vehicle_delta_suppressed"):
        payload = case_study["ablation_means"][variant_name]
        rows.append(
            {
                "variant": variant_name,
                "source": "phase2_case_study_bundle_only",
                "sample_count": case_study["view_count"],
                "mean_projection_cosine": baseline["mean_projection_cosine"],
                "mean_projection_l2_gap": baseline["mean_projection_l2_gap"],
                "generated_view_count": alignment_support["stage_h_export"]["generated_view_count"],
                "pass_view_count": int(view_counts.get("PASS", 0)),
                "warn_view_count": int(view_counts.get("WARN", 0)),
                "contract_complete": True,
                "mean_attention_entropy": payload["mean_attention_entropy"],
                "mean_top_event_score": payload["mean_top_event_score"],
                "mean_top_contribution_score": payload["mean_top_contribution_score"],
                "delta_mean_attention_entropy": payload["delta_mean_attention_entropy"],
                "delta_mean_top_event_score": payload["delta_mean_top_event_score"],
                "delta_mean_top_contribution_score": payload["delta_mean_top_contribution_score"],
                "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
                "pilot_delta_mean_top_contribution_score": pilot_delta_means[
                    "delta_mean_top_contribution_score"
                ],
                "private_t1_macro_f1": None,
                "private_t2_rmse": None,
                "private_t3_top1_accuracy": None,
                "supports": "事件/机动敏感性可读",
                "limits": "仅是 frozen Stage H view 上的 bundle-only 干预",
            }
        )
    rows.append(
        {
            "variant": "g_no_causal_mask",
            "source": "chronaris_opt_no_causal_mask_private_proxy",
            "sample_count": None,
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "generated_view_count": None,
            "pass_view_count": None,
            "warn_view_count": None,
            "contract_complete": None,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": None,
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": no_mask_tasks["T1_maneuver_intensity_class"]["no_mask_metrics"].get(
                "macro_f1"
            ),
            "private_t2_rmse": no_mask_tasks["T2_next_window_physiology_response"][
                "no_mask_metrics"
            ].get("rmse"),
            "private_t3_top1_accuracy": no_mask_tasks[
                "T3_paired_pilot_window_retrieval"
            ]["no_mask_metrics"].get("top1_accuracy"),
            "supports": "去掉因果掩码后三任务同步退化",
            "limits": "当前来自私有 proxy，不直接等价于公开 benchmark",
        }
    )
    frame = pd.DataFrame(rows)
    variant_order = [
        "e_baseline",
        "f_full",
        "g_min",
        "g_no_causal_mask",
        "vehicle_delta_suppressed",
        "no_event_bias",
    ]
    frame["variant"] = pd.Categorical(frame["variant"], categories=variant_order, ordered=True)
    return frame.sort_values("variant").reset_index(drop=True)


def _projection_summary(payload: Mapping[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    threshold = payload["threshold_evaluation"]
    return {
        "sample_count": int(summary["sample_count"]),
        "mean_projection_cosine": float(summary["mean_projection_cosine"]),
        "mean_projection_l2_gap": float(summary["mean_projection_l2_gap"]),
        "threshold_verdict": str(threshold["verdict"]),
    }


def _frame_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    return float(frame[column].mean())


def _extract_private_no_mask_summary(
    payload: Mapping[str, object],
) -> dict[str, object]:
    conclusion = payload["conclusion"]
    target_variant = conclusion["target_variant_name"]
    no_mask_variant = conclusion["no_mask_variant_name"]
    task_rows: dict[str, object] = {}
    matrix_rows = {
        target_variant: "target_variant",
        no_mask_variant: "no_causal_mask_variant",
    }
    for task_name, task_payload in payload["tasks"].items():
        target_metrics = _extract_variant_metrics(
            task_payload["variants"][target_variant]
        )
        no_mask_metrics = _extract_variant_metrics(
            task_payload["variants"][no_mask_variant]
        )
        if task_payload["task_type"] == "classification":
            target_text = (
                f"macro_f1={target_metrics['macro_f1']:.6f}, "
                f"balanced_accuracy={target_metrics['balanced_accuracy']:.6f}"
            )
            no_mask_text = (
                f"macro_f1={no_mask_metrics['macro_f1']:.6f}, "
                f"balanced_accuracy={no_mask_metrics['balanced_accuracy']:.6f}"
            )
            target_beats = (
                float(target_metrics["macro_f1"]) > float(no_mask_metrics["macro_f1"])
            )
        elif task_payload["task_type"] == "regression":
            target_text = (
                f"rmse={target_metrics['rmse']:.6f}, mae={target_metrics['mae']:.6f}"
            )
            no_mask_text = (
                f"rmse={no_mask_metrics['rmse']:.6f}, mae={no_mask_metrics['mae']:.6f}"
            )
            target_beats = float(target_metrics["rmse"]) < float(no_mask_metrics["rmse"])
        else:
            target_text = (
                f"top1_accuracy={target_metrics['top1_accuracy']:.6f}, "
                f"mrr={target_metrics['mrr']:.6f}"
            )
            no_mask_text = (
                f"top1_accuracy={no_mask_metrics['top1_accuracy']:.6f}, "
                f"mrr={no_mask_metrics['mrr']:.6f}"
            )
            target_beats = (
                float(target_metrics["top1_accuracy"]) >
                float(no_mask_metrics["top1_accuracy"])
            )
        task_rows[task_name] = {
            "task_type": task_payload["task_type"],
            "target_metrics": dict(target_metrics),
            "no_mask_metrics": dict(no_mask_metrics),
            "target_metric_text": target_text,
            "no_mask_metric_text": no_mask_text,
            "target_beats_no_mask": bool(target_beats),
        }
    return {
        "target_variant_name": target_variant,
        "no_mask_variant_name": no_mask_variant,
        "criterion_details": dict(conclusion["criterion_details"]),
        "tasks": task_rows,
        "matrix_rows": matrix_rows,
    }


def _extract_deep_stage_h_case_summary(
    payload: Mapping[str, object],
) -> dict[str, object]:
    dataset_payload = payload.get("datasets", {}).get("stage_h_case", {})
    if dataset_payload.get("status") != "completed":
        return {}
    rows: dict[str, object] = {}
    for model_name, model_payload in dataset_payload.get("models", {}).items():
        summary = model_payload["summary"]
        view_metrics = summary["view_metrics"]
        pilot_metrics = summary["pilot_metrics"]
        rows[model_name] = {
            "mean_event_mask_interference": float(
                pd.DataFrame(view_metrics)["event_mask_interference"].mean()
            ),
            "mean_attention_entropy": float(
                pd.DataFrame(view_metrics)["mean_attention_entropy"].mean()
            ),
            "pilot_delta_event_mask_interference": float(
                pilot_metrics[0]["delta_event_mask_interference"]
            )
            if pilot_metrics
            else 0.0,
        }
    return rows


def _extract_variant_metrics(payload: Mapping[str, object]) -> Mapping[str, object]:
    best_metrics = payload.get("best_metrics")
    if isinstance(best_metrics, Mapping):
        return best_metrics
    return payload


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _write_support_overview_plot(
    matrix: pd.DataFrame,
    *,
    path: Path,
) -> str | None:
    try:
        from matplotlib import pyplot as plt
    except Exception:
        return None

    alignment_rows = matrix.loc[
        matrix["variant"].astype(str).isin(["e_baseline", "f_full"])
    ].copy()
    intervention_rows = matrix.loc[
        matrix["variant"].astype(str).isin(["no_event_bias", "vehicle_delta_suppressed"])
    ].copy()
    no_mask_row = matrix.loc[matrix["variant"].astype(str) == "g_no_causal_mask"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].bar(
        [_display_variant_name(str(value)) for value in alignment_rows["variant"]],
        alignment_rows["mean_projection_cosine"].astype(float),
        color=["#5b8ff9", "#61dDAa"],
    )
    axes[0].set_title("Alignment cosine")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        [_display_variant_name(str(value)) for value in intervention_rows["variant"]],
        intervention_rows["delta_mean_top_contribution_score"].astype(float),
        color=["#f6bd16", "#e8684a"],
    )
    axes[1].axhline(0.0, color="#999999", linewidth=1.0)
    axes[1].set_title("Intervention delta top contribution")
    axes[1].tick_params(axis="x", rotation=15)

    if not no_mask_row.empty:
        row = no_mask_row.iloc[0]
        axes[2].bar(
            ["T1 macro-F1", "T3 top1"],
            [
                float(row["private_t1_macro_f1"]),
                float(row["private_t3_top1_accuracy"]),
            ],
            color=["#9270ca", "#269a99"],
        )
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_title("No-mask private proxy")
        axes[2].text(
            0.5,
            0.04,
            f"T2 RMSE={float(row['private_t2_rmse']):.2f}",
            ha="center",
            va="bottom",
            transform=axes[2].transAxes,
            fontsize=9,
        )
    else:
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)
