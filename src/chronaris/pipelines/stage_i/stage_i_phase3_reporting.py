"""Rendering helpers for Stage I Phase 3 closure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from chronaris.evaluation import save_bar_plot, save_grouped_bar_plot


def render_stage_i_phase3_report(summary: Mapping[str, object]) -> str:
    """Render the top-level Stage I Phase 3 closure markdown."""

    lines = [
        "# Stage I Phase 3 Closure",
        "",
        f"- 生成时间：`{summary['generated_at_utc']}`",
        f"- 机器产物根目录：`{summary['artifact_root']}`",
        "",
        "## 收口说明",
        "",
        "- 本轮收口把 Stage I 主线从 session-level UAB 切换到 window-level UAB，并补齐 NASA CSM attention-state 第二公开数据集。",
        "- `Phase 0 + Phase 1` 的 session-level UAB 资产继续保留为历史参照；当前主线事实以本次 window-level UAB + NASA CSM closure 为准。",
        "- 两条任务保持独立训练与独立评测，只统一数据 contract、pipeline 接口、artifact contract 和闭环文档。",
        "- window-scale baseline 只保留 `LogisticRegression / LinearSVC / Ridge / LinearSVR` 线性 classical baselines；随机森林与核 SVR 不再进入主收口。",
        "- UAB 与 NASA 都使用 `5s / 5s` 固定窗口和 `Leave-One-Subject-Out` 切分；NASA 背景窗口不进入主分类。",
        "",
        "## UAB Window 主线",
        "",
    ]
    lines.extend(_render_baseline_bundle(summary["uab_window"]))
    lines.extend(["## NASA Attention 主线", ""])
    lines.extend(_render_baseline_bundle(summary["nasa_attention"]))
    comparison = summary.get("uab_session_comparison", {})
    if comparison:
        lines.extend(["## UAB Session vs Window", ""])
        for track_name, groups in comparison.items():
            lines.append(f"### {track_name}")
            lines.append("")
            for group_name, payload in groups.items():
                field = "macro_f1" if track_name == "objective" else "rmse"
                lines.append(
                    f"- {group_name}: session `{payload[f'session_{field}']:.4f}` / window `{payload[f'window_{field}']:.4f}` / delta `{payload['delta']:.4f}`"
                )
            lines.append("")
    if summary.get("plot_paths"):
        lines.extend(["## Closure 图表", ""])
        for name, path in sorted(summary["plot_paths"].items()):
            lines.append(f"- `{name}`: `{path}`")
        lines.append("")
    return "\n".join(lines)


def write_phase3_summary_plots(
    artifact_root: Path,
    summary: Mapping[str, object],
) -> dict[str, str]:
    plot_root = artifact_root / "plots"
    plot_paths: dict[str, str] = {}
    plot_paths["phase3_window_counts"] = save_bar_plot(
        {
            "uab_window": float(summary["uab_window"]["dataset_summary"]["window_count"]),
            "nasa_attention": float(summary["nasa_attention"]["dataset_summary"]["window_count"]),
        },
        path=plot_root / "phase3_window_counts.png",
        title="Phase 3 window counts by dataset",
        ylabel="window count",
    )
    comparison = summary.get("uab_session_comparison", {})
    if comparison.get("objective"):
        plot_paths["uab_session_vs_window_objective"] = save_grouped_bar_plot(
            {
                group_name: {
                    "session": payload["session_macro_f1"],
                    "window": payload["window_macro_f1"],
                }
                for group_name, payload in comparison["objective"].items()
            },
            path=plot_root / "uab_session_vs_window_objective.png",
            title="UAB session vs window objective",
            ylabel="macro-F1",
        )
    if comparison.get("subjective"):
        plot_paths["uab_session_vs_window_subjective"] = save_grouped_bar_plot(
            {
                group_name: {
                    "session": payload["session_rmse"],
                    "window": payload["window_rmse"],
                }
                for group_name, payload in comparison["subjective"].items()
            },
            path=plot_root / "uab_session_vs_window_subjective.png",
            title="UAB session vs window subjective",
            ylabel="RMSE",
        )
    return plot_paths


def _render_baseline_bundle(bundle: Mapping[str, object]) -> list[str]:
    dataset_summary = bundle["dataset_summary"]
    lines = [
        f"- artifact root: `{bundle['artifact_root']}`",
        f"- entry_count: `{dataset_summary['entry_count']}`",
        f"- recording_count: `{dataset_summary['recording_count']}`",
        f"- window_count: `{dataset_summary['window_count']}`",
        f"- subset_counts: `{json.dumps(dataset_summary['subset_counts'], ensure_ascii=False)}`",
        f"- label_distribution: `{json.dumps(dataset_summary['label_distribution'], ensure_ascii=False)}`",
        "",
        "### 主结果",
        "",
    ]
    objective_primary = bundle.get("objective_primary", {})
    for group_name, metrics in objective_primary.items():
        if "macro_f1" in metrics:
            lines.append(
                f"- {group_name}: macro-F1 `{metrics['macro_f1']:.4f}`, balanced accuracy `{metrics['balanced_accuracy']:.4f}`"
            )
    subjective_primary = bundle.get("subjective_primary", {})
    for group_name, metrics in subjective_primary.items():
        if "rmse" in metrics:
            lines.append(
                f"- {group_name}: RMSE `{metrics['rmse']:.4f}`, MAE `{metrics['mae']:.4f}`"
            )
    if bundle.get("objective_ablation_results"):
        lines.extend(["", "### 客观消融摘要", ""])
        for group_name, payload in bundle["objective_ablation_results"].items():
            summary = {
                feature_set_name: float(feature_payload["models"][feature_payload["best_model_name"]]["macro_f1"])
                for feature_set_name, feature_payload in payload.items()
            }
            lines.append(f"- {group_name}: `{json.dumps(summary, ensure_ascii=False)}`")
    if bundle.get("subjective_ablation_results"):
        lines.extend(["", "### 主观消融摘要", ""])
        for group_name, payload in bundle["subjective_ablation_results"].items():
            summary = {
                feature_set_name: float(feature_payload["models"][feature_payload["best_model_name"]]["rmse"])
                for feature_set_name, feature_payload in payload.items()
            }
            lines.append(f"- {group_name}: `{json.dumps(summary, ensure_ascii=False)}`")
    lines.append("")
    return lines
