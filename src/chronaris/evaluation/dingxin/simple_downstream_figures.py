"""Reader-facing figures for the simplified Dingxin downstream confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)
METHOD_LABELS = {
    "physiology_only": "生理单流",
    "vehicle_only": "航电单流",
    "naive_time_sync": "朴素时间同步",
    "mult": "MulT",
    "contiformer": "ContiFormer",
    "chronaris": "Chronaris",
}
METHOD_COLORS = {
    "physiology_only": "#8CB6D9",
    "vehicle_only": "#3978A8",
    "naive_time_sync": "#969696",
    "mult": "#D39B55",
    "contiformer": "#7E6BB0",
    "chronaris": "#C94C4C",
}
STRESS_LABELS = {
    "timestamp_jitter_ms": "时间戳抖动",
    "absolute_clock_offset_s": "时钟偏移",
    "random_missing_rate": "随机缺失",
    "contiguous_gap_s": "连续缺失",
    "snr_degradation_db": "观测噪声",
}
ABLATION_LABELS = {
    "chronaris_no_continuous_evolution": "去连续演化",
    "chronaris_no_physics": "去物理约束",
    "chronaris_no_causal_mask": "去因果掩码",
    "chronaris_single_scale_lag": "单尺度时延",
}
ABLATION_TASK_LABELS = {
    "simulated_future_workload_classification": "负荷分类",
    "simulated_future_workload_regression": "负荷回归",
    "simulated_maneuver_state_segmentation": "机动分段",
}


def configure_chinese_matplotlib() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
    ):
        if family in available:
            plt.rcParams["font.family"] = family
            plt.rcParams["axes.unicode_minus"] = False
            return family
    plt.rcParams["axes.unicode_minus"] = False
    return str(plt.rcParams["font.family"])


def plot_future_maneuver(
    unit_metrics: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Plot maneuver metrics with the two sortie means visible."""
    configure_chinese_matplotlib()
    specifications = (
        ("macro_f1", "三分类宏平均 F1", False, None),
        ("spearman", "连续分数 Spearman 相关", False, (-0.25, 0.9)),
        ("rmse_ratio_vs_current", "相对当前状态的 RMSE 比率", True, None),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=True)
    for axis, (metric, title, log_scale, limits) in zip(
        axes, specifications, strict=True
    ):
        _plot_method_metric(
            axis,
            unit_metrics,
            task="future_maneuver",
            metric=metric,
            title=title,
            log_scale=log_scale,
            limits=limits,
        )
        if metric == "rmse_ratio_vs_current":
            axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0)
            axis.text(1.03, -0.55, "当前状态基线", fontsize=9, color="#444444")
    figure.suptitle(
        "鼎新未来机动预测：航电单流形成稳定领先",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "柱体为两架次、三随机种子的描述性均值；圆点和三角分别为两个留一架次折的三种子均值。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save(figure, path)


def plot_future_physiology(
    unit_metrics: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Plot physiology errors and the persistence comparison."""
    configure_chinese_matplotlib()
    specifications = (
        ("standardized_rmse_macro", "字段级标准化 RMSE 宏平均", False),
        ("rmse_ratio_vs_persistence", "相对持久性预测的 RMSE 比率", True),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
    for axis, (metric, title, log_scale) in zip(
        axes, specifications, strict=True
    ):
        _plot_method_metric(
            axis,
            unit_metrics,
            task="future_physiology",
            metric=metric,
            title=title,
            log_scale=log_scale,
            limits=None,
        )
        if metric == "rmse_ratio_vs_persistence":
            axis.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0)
            axis.text(1.03, -0.55, "持久性基线", fontsize=9, color="#444444")
    figure.suptitle(
        "鼎新未来生理字段预测：六种方法均未超过持久性基线",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "折一报告 12 个字段，折二报告 11 个字段；两个折中所有方法的正技能字段比例均为 0。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save(figure, path)


def plot_simulation_mechanism_boundary(
    recovery: pd.DataFrame,
    stress: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Show positive timing recovery beside mixed stress robustness."""
    configure_chinese_matplotlib()
    figure = plt.figure(figsize=(17, 6.2))
    grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 1.55))
    targets = (
        ("relative_clock_offset_magnitude_s", "时钟偏移恢复误差"),
        ("primary_physiology_response_lag_s", "生理响应时延恢复误差"),
    )
    for index, (target, title) in enumerate(targets):
        axis = figure.add_subplot(grid[0, index])
        panel = recovery[recovery["target"].astype(str) == target].copy()
        panel["method_order"] = panel["method"].map(
            {method: idx for idx, method in enumerate(METHOD_ORDER)}
        )
        panel = panel.sort_values("method_order")
        positions = np.arange(len(panel))
        bars = axis.barh(
            positions,
            panel["value"],
            color=[METHOD_COLORS[value] for value in panel["method"]],
        )
        axis.set_yticks(
            positions,
            [METHOD_LABELS[value] for value in panel["method"]],
        )
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("平均绝对误差（秒，越低越好）")
        axis.grid(axis="x", alpha=0.22)
        _label_horizontal_bars(axis, bars)

    heat_axis = figure.add_subplot(grid[0, 2])
    factors = [value for value in STRESS_LABELS if value in stress.columns]
    methods = [value for value in METHOD_ORDER if value in stress.index]
    values = stress.loc[methods, factors].to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 1e-6)
    image = heat_axis.imshow(
        values,
        cmap="RdYlGn",
        vmin=-bound,
        vmax=bound,
        aspect="auto",
    )
    heat_axis.set_xticks(
        np.arange(len(factors)),
        [STRESS_LABELS[value] for value in factors],
        rotation=25,
        ha="right",
    )
    heat_axis.set_yticks(
        np.arange(len(methods)),
        [METHOD_LABELS[value] for value in methods],
    )
    heat_axis.set_title("观测压力退化斜率")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            heat_axis.text(
                column,
                row,
                f"{values[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if abs(values[row, column]) > bound * 0.65 else "black",
            )
    colorbar = figure.colorbar(image, ax=heat_axis, shrink=0.82)
    colorbar.set_label("方向归一化斜率（越大表示退化越慢）")
    figure.suptitle(
        "仿真机制证据：时间恢复领先，但缺失鲁棒性没有同步领先",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "恢复真值来自仿真生成器；压力斜率汇总预声明下游指标，不能替代鼎新真实任务效果。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save(figure, path)


def plot_chronaris_ablation_boundary(
    ablation: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Plot the full-model advantage over each simulation ablation."""
    configure_chinese_matplotlib()
    tasks = [value for value in ABLATION_TASK_LABELS if value in set(ablation["task"])]
    ablations = [
        value for value in ABLATION_LABELS if value in set(ablation["ablation_method"])
    ]
    x = np.arange(len(ablations))
    width = 0.22
    figure, axis = plt.subplots(figsize=(12.5, 6.0))
    colors = ("#3978A8", "#D39B55", "#7E6BB0")
    for index, task in enumerate(tasks):
        panel = (
            ablation[ablation["task"].astype(str) == task]
            .set_index("ablation_method")
            .reindex(ablations)
        )
        axis.bar(
            x + (index - (len(tasks) - 1) / 2) * width,
            panel["mean"],
            width=width,
            label=ABLATION_TASK_LABELS[task],
            color=colors[index],
        )
    axis.axhline(0.0, color="#333333", linewidth=1.0)
    axis.set_xticks(x, [ABLATION_LABELS[value] for value in ablations])
    axis.set_ylabel("完整模型的方向归一化优势")
    axis.set_title(
        "Chronaris 仿真消融：连续演化与因果掩码贡献较稳定，物理约束结果混合",
        fontsize=15,
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    figure.text(
        0.5,
        0.015,
        "正值表示完整 Chronaris 优于对应消融；三个任务保留各自主指标尺度，不跨任务比较绝对大小。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(figure, path)


def _plot_method_metric(
    axis,
    unit_metrics: pd.DataFrame,
    *,
    task: str,
    metric: str,
    title: str,
    log_scale: bool,
    limits: tuple[float, float] | None,
) -> None:
    panel = unit_metrics[
        (unit_metrics["task_name"].astype(str) == task)
        & (unit_metrics["metric_name"].astype(str) == metric)
    ].copy()
    methods = [value for value in METHOD_ORDER if value in set(panel["method_name"])]
    overall = panel.groupby("method_name")["metric_value"].mean().reindex(methods)
    fold = (
        panel.groupby(["method_name", "fold_id"])["metric_value"]
        .mean()
        .unstack("fold_id")
        .reindex(methods)
    )
    positions = np.arange(len(methods))
    axis.barh(
        positions,
        overall.to_numpy(dtype=float),
        color=[METHOD_COLORS[value] for value in methods],
        alpha=0.82,
    )
    markers = ("o", "^")
    for index, fold_id in enumerate(fold.columns):
        axis.scatter(
            fold[fold_id],
            positions,
            marker=markers[index % len(markers)],
            s=42,
            facecolor="white",
            edgecolor="#222222",
            linewidth=0.9,
            zorder=3,
            label=f"架次 {index + 1}",
        )
    axis.set_yticks(positions, [METHOD_LABELS[value] for value in methods])
    axis.invert_yaxis()
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.22)
    if log_scale:
        axis.set_xscale("log")
    if limits is not None:
        axis.set_xlim(*limits)
    # Fold markers intentionally carry no inline values: exact values are in the
    # adjacent report table, while labels at the bar ends collide with the two
    # sortie markers on this compact six-method figure.


def _label_horizontal_bars(axis, bars: Iterable, *, log_scale: bool = False) -> None:
    for bar in bars:
        value = float(bar.get_width())
        if not np.isfinite(value):
            continue
        if log_scale and value > 0:
            x = value * 1.08
        else:
            span = axis.get_xlim()[1] - axis.get_xlim()[0]
            x = value + span * 0.015
        axis.text(x, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8.5)


def _save(figure, path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(resolved, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return resolved
