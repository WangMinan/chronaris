"""Reporting and plotting helpers for Stage I baselines."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.evaluation import (
    save_bar_plot,
    save_confusion_matrix_plot,
    save_grouped_bar_plot,
    save_regression_plot,
)

UAB_DATASET_ID = "uab_workload_dataset"
NASA_DATASET_ID = "nasa_csm"


def render_stage_i_baseline_report(
    *,
    artifact_root: str | Path,
    dataset_summary: Mapping[str, object],
    objective_metrics: Mapping[str, object] | None,
    subjective_metrics: Mapping[str, object] | None,
    plot_paths: Mapping[str, str],
) -> str:
    """Render one markdown report for a Stage I baseline run."""

    dataset_id = str(dataset_summary["dataset_id"])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = "# Stage I NASA CSM Attention Baseline" if dataset_id == NASA_DATASET_ID else "# Stage I UAB Baseline"
    lines = [
        title,
        "",
        f"- 生成时间：{generated_at}",
        f"- 机器产物根目录：`{artifact_root}`",
        "",
        "## 数据摘要",
        "",
        f"- 样本总数：`{dataset_summary['entry_count']}`",
        f"- recording 数：`{dataset_summary['recording_count']}`",
        f"- window 数：`{dataset_summary['window_count']}`",
        f"- split_group 数：`{dataset_summary['split_group_count']}`",
        f"- 特征总数：`{dataset_summary['feature_count']}`",
        f"- 模态特征数：`{json.dumps(dataset_summary['feature_group_counts'], ensure_ascii=False)}`",
        f"- subset 计数：`{json.dumps(dataset_summary['subset_counts'], ensure_ascii=False)}`",
        f"- 标签分布：`{json.dumps(dataset_summary['label_distribution'], ensure_ascii=False)}`",
        "",
    ]

    lines.extend(_render_experiment_setup(dataset_id=dataset_id, dataset_summary=dataset_summary))

    if objective_metrics is not None:
        lines.extend(_render_track_section("客观/分类主结果", objective_metrics, metric_fields=("macro_f1", "balanced_accuracy")))
    if subjective_metrics is not None:
        lines.extend(_render_track_section("主观/回归主结果", subjective_metrics, metric_fields=("rmse", "mae", "spearman")))

    if dataset_id == UAB_DATASET_ID and objective_metrics is not None:
        flight_summary = objective_metrics.get("auxiliary_flight_summary", {})
        lines.extend(
            [
                "## Flight 辅助摘要",
                "",
                f"- flight window 数：`{flight_summary.get('sample_count', 0)}`",
                f"- flight split_group 数：`{flight_summary.get('subject_count', 0)}`",
                f"- 理论难度分布：`{json.dumps(flight_summary.get('theoretical_difficulty_distribution', {}), ensure_ascii=False)}`",
                f"- 感知难度均值：`{flight_summary.get('perceived_difficulty_mean', float('nan')):.4f}`",
                "",
            ]
        )

    if plot_paths:
        lines.extend(["## 图表", ""])
        for name, path in sorted(plot_paths.items()):
            lines.append(f"- `{name}`: `{path}`")
        lines.append("")
    return "\n".join(lines)


def render_uab_baseline_report(
    *,
    artifact_root: str | Path,
    dataset_summary: Mapping[str, object],
    objective_metrics: Mapping[str, object],
    subjective_metrics: Mapping[str, object],
    plot_paths: Mapping[str, str],
) -> str:
    return render_stage_i_baseline_report(
        artifact_root=artifact_root,
        dataset_summary=dataset_summary,
        objective_metrics=objective_metrics,
        subjective_metrics=subjective_metrics,
        plot_paths=plot_paths,
    )


def write_best_model_plots(
    *,
    artifact_root: Path,
    objective_metrics: Mapping[str, object] | None,
    objective_predictions: pd.DataFrame,
    subjective_metrics: Mapping[str, object] | None,
    subjective_predictions: pd.DataFrame,
) -> dict[str, str]:
    plot_root = artifact_root / "plots"
    plot_paths: dict[str, str] = {}
    if objective_metrics is not None:
        for evaluation_group, payload in objective_metrics["primary_results"].items():
            best_model = payload["best_model_name"]
            metrics = payload["models"][best_model]
            plot_key = f"objective_confusion_matrix_{evaluation_group}"
            plot_paths[plot_key] = save_confusion_matrix_plot(
                metrics,
                path=plot_root / f"{plot_key}.png",
                title=f"{evaluation_group} objective ({best_model})",
            )
    if subjective_metrics is not None:
        for evaluation_group, payload in subjective_metrics["primary_results"].items():
            best_model = payload["best_model_name"]
            subset_predictions = subjective_predictions.loc[
                (subjective_predictions["evaluation_group"] == evaluation_group)
                & (subjective_predictions["feature_set"] == subjective_metrics["primary_feature_set"])
                & (subjective_predictions["model_name"] == best_model)
            ].copy()
            plot_key = f"subjective_regression_{evaluation_group}"
            plot_paths[plot_key] = save_regression_plot(
                subset_predictions,
                path=plot_root / f"{plot_key}.png",
                title=f"{evaluation_group} subjective ({best_model})",
            )
    return plot_paths


def write_dataset_diagnostic_plots(
    *,
    artifact_root: Path,
    feature_table: pd.DataFrame,
    objective_metrics: Mapping[str, object] | None,
    subjective_metrics: Mapping[str, object] | None,
) -> dict[str, str]:
    plot_root = artifact_root / "plots"
    plot_paths: dict[str, str] = {}
    subset_counts = feature_table["subset_id"].astype(str).value_counts().sort_index().to_dict()
    plot_paths["dataset_subset_counts"] = save_bar_plot(
        subset_counts,
        path=plot_root / "dataset_subset_counts.png",
        title="Stage I subset counts",
        ylabel="window count",
    )

    label_distribution = (
        feature_table.loc[feature_table["objective_label_value"].notna()]
        .groupby(["subset_id", "objective_label_value"])
        .size()
        .unstack(fill_value=0)
    )
    plot_paths["label_distribution"] = save_grouped_bar_plot(
        {
            str(subset_id): {str(label): int(value) for label, value in row.items()}
            for subset_id, row in label_distribution.iterrows()
        },
        path=plot_root / "label_distribution.png",
        title="Objective label distribution",
        ylabel="window count",
    )

    if objective_metrics is not None:
        plot_paths["objective_ablation_primary_metric"] = save_grouped_bar_plot(
            {
                group_name: render_ablation_metrics(payload, "macro_f1")
                for group_name, payload in objective_metrics["ablation_results"].items()
            },
            path=plot_root / "objective_ablation_macro_f1.png",
            title="Objective ablation comparison",
            ylabel="macro-F1",
        )
    if subjective_metrics is not None:
        plot_paths["subjective_ablation_primary_metric"] = save_grouped_bar_plot(
            {
                group_name: render_ablation_metrics(payload, "rmse")
                for group_name, payload in subjective_metrics["ablation_results"].items()
            },
            path=plot_root / "subjective_ablation_rmse.png",
            title="Subjective ablation comparison",
            ylabel="RMSE",
        )
    return plot_paths


def summarize_auxiliary_flight(flight_frame: pd.DataFrame) -> dict[str, object]:
    theoretical = (
        flight_frame["objective_label_value"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    perceived = flight_frame["subjective_target_value"].dropna().astype(float)
    return {
        "sample_count": int(len(flight_frame)),
        "subject_count": int(flight_frame["subject_id"].nunique()),
        "theoretical_difficulty_distribution": {str(key): int(value) for key, value in theoretical.items()},
        "perceived_difficulty_mean": float(perceived.mean()) if not perceived.empty else float("nan"),
    }


def render_ablation_metrics(feature_set_results: Mapping[str, object], rank_field: str) -> dict[str, float]:
    rendered: dict[str, float] = {}
    for feature_set_name, payload in feature_set_results.items():
        best_model_name = payload["best_model_name"]
        rendered[feature_set_name] = float(payload["models"][best_model_name][rank_field])
    return rendered


def _render_track_section(
    title: str,
    metrics: Mapping[str, object],
    *,
    metric_fields: Sequence[str],
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- 主特征集：`{metrics['primary_feature_set']}`",
        f"- 消融列定义：`{json.dumps(metrics['feature_set_columns'], ensure_ascii=False)}`",
        "",
    ]
    for evaluation_group, payload in metrics["primary_results"].items():
        best_name = payload["best_model_name"]
        best_metrics = payload["models"][best_name]
        ablation_summary = render_ablation_metrics(metrics["ablation_results"][evaluation_group], metric_fields[0])
        lines.extend(
            [
                f"### {evaluation_group}",
                "",
                f"- 最优模型：`{best_name}`",
                *(f"- {field}：`{best_metrics[field]:.4f}`" for field in metric_fields),
                f"- 样本数：`{payload['sample_count']}`",
                f"- ablation：`{json.dumps(ablation_summary, ensure_ascii=False)}`",
                f"- 结果说明：{_summarize_ablation_outcome(ablation_summary, primary_feature_set=str(metrics['primary_feature_set']), metric_name=metric_fields[0])}",
                "",
            ]
        )
    return lines


def _render_experiment_setup(
    *,
    dataset_id: str,
    dataset_summary: Mapping[str, object],
) -> list[str]:
    if dataset_id == UAB_DATASET_ID:
        return [
            "## 实验设置",
            "",
            "- 窗口策略：`5s / 5s`，优先使用 EEG/ECG overlap；完全缺 ECG 的 recording 保留为 EEG-only window。",
            "- 模型策略：window-scale 主线默认收敛到线性 classical baselines；历史 session-level `Phase 1` 结果保留原三模型对照作为参照。",
            "- 任务设置：`n_back / heat_the_chair` 为 workload 主线，`flight_simulator` 仅保留辅助摘要，不参与主收口 gate。",
            "- 评测切分：`Leave-One-Subject-Out`。",
            f"- 缺失 ECG window 计数：`{json.dumps(dataset_summary['missing_ecg_session_counts'], ensure_ascii=False)}`",
            "",
        ]
    return [
        "## 实验设置",
        "",
        "- 窗口策略：`5s / 5s`，仅保留完整落在单一非零事件段中的窗口；背景 `event=0` 只进入盘点，不参与主分类。",
        "- 模型策略：window-scale attention baseline 默认使用线性 classical models，避免在 LOSO 下引入不必要的核方法计算开销。",
        "- 任务设置：`benchmark_only / loft_only / combined` 三套 attention-state 分类主结果。",
        "- 评测切分：`Leave-One-Subject-Out`，combined 中同一 subject 的 benchmark 和 LOFT 窗口共享同一 fold。",
        "",
    ]


def _summarize_ablation_outcome(
    ablation_summary: Mapping[str, float],
    *,
    primary_feature_set: str,
    metric_name: str,
) -> str:
    if not ablation_summary:
        return "无可用消融结果。"
    best_name = max(ablation_summary, key=ablation_summary.get) if metric_name == "macro_f1" else min(ablation_summary, key=ablation_summary.get)
    if best_name == primary_feature_set:
        return f"`{primary_feature_set}` 是该组最优消融。"
    return f"`{best_name}` 优于主特征集 `{primary_feature_set}`，说明该组对当前模态裁剪更敏感。"
