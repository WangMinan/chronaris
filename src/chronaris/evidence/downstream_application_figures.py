"""Chinese thesis figures for the fixed-data downstream evidence pack."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evidence.downstream_application_data import (
    ABLATION_LABELS,
    METHOD_LABELS,
    METHOD_ORDER,
    STRESS_LABELS,
    TARGET_LABELS,
)


METHOD_COLORS = {
    "physiology_only": "#8CB6D9",
    "vehicle_only": "#5B8DB8",
    "naive_time_sync": "#9A9A9A",
    "mult": "#D39B55",
    "contiformer": "#7E6BB0",
    "chronaris": "#C94C4C",
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


def plot_method_primary_metrics(
    table: pd.DataFrame,
    specifications: Sequence[Mapping[str, str]],
    *,
    title: str,
    path: str | Path,
) -> Path:
    configure_chinese_matplotlib()
    figure, axes = plt.subplots(1, len(specifications), figsize=(16, 5.4))
    axes = np.atleast_1d(axes)
    for axis, specification in zip(axes, specifications, strict=True):
        panel = table[
            (table["task"] == specification["task"])
            & (table["consumer"] == specification["consumer"])
            & (table["metric"] == specification["metric"])
        ]
        methods = [value for value in METHOD_ORDER if value in set(panel["method"])]
        panel = panel.set_index("method").loc[methods].reset_index()
        positions = np.arange(len(panel))
        bars = axis.bar(
            positions,
            panel["mean"],
            yerr=panel["std"],
            capsize=3,
            color=[METHOD_COLORS[value] for value in panel["method"]],
            edgecolor="white",
            linewidth=0.8,
        )
        axis.set_title(specification["title"], fontsize=12, pad=10)
        axis.set_ylabel(specification["metric_label"])
        axis.set_xticks(positions, [METHOD_LABELS[value] for value in panel["method"]])
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.22, linewidth=0.8)
        _label_bars(axis, bars)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "柱高为三随机种子均值，误差线为随机种子间标准差；任务选择和下游算法在六方法间完全一致。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return _save(figure, path)


def plot_stress_slope_heatmap(table: pd.DataFrame, path: str | Path) -> Path:
    configure_chinese_matplotlib()
    values = table.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), 1e-6)
    figure, axis = plt.subplots(figsize=(12.5, 6.2))
    image = axis.imshow(values, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(
        np.arange(len(table.columns)),
        [STRESS_LABELS.get(value, value) for value in table.columns],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(table.index)),
        [METHOD_LABELS.get(value, value) for value in table.index],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if abs(value) < bound * 0.65 else "white",
                )
    axis.set_title("观测压力增强时的下游性能退化斜率", fontsize=16, fontweight="bold")
    axis.set_xlabel("压力因素")
    axis.set_ylabel("融合方法")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("方向归一化斜率（越大表示退化越慢）")
    figure.text(
        0.5,
        0.015,
        "每格汇总负荷分类、负荷回归和机动状态分段三个主指标；压力数据不重训编码器或下游模型。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(figure, path)


def plot_mechanism_recovery(table: pd.DataFrame, path: str | Path) -> Path:
    configure_chinese_matplotlib()
    targets = [value for value in TARGET_LABELS if value in set(table["target"])]
    figure, axes = plt.subplots(1, len(targets), figsize=(12.5, 5.2))
    axes = np.atleast_1d(axes)
    for axis, target in zip(axes, targets, strict=True):
        panel = table[table["target"] == target].set_index("method")
        methods = [value for value in METHOD_ORDER[2:] if value in panel.index]
        values = panel.loc[methods, "value"].to_numpy(dtype=float)
        bars = axis.bar(
            np.arange(len(methods)),
            values,
            color=[METHOD_COLORS[value] for value in methods],
        )
        axis.set_title(TARGET_LABELS[target], fontsize=13)
        axis.set_ylabel("平均绝对误差（秒，越低越好）")
        axis.set_xticks(np.arange(len(methods)), [METHOD_LABELS[value] for value in methods])
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.22)
        _label_bars(axis, bars)
    figure.suptitle("融合表示中的时间偏移与响应时延可恢复性", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "岭回归探针只在 G1 训练并由 G1 验证集选参；图中结果来自 G2 锁定压力场景。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.92))
    return _save(figure, path)


def plot_ablation_advantage(
    table: pd.DataFrame,
    specifications: Sequence[Mapping[str, str]],
    path: str | Path,
) -> Path:
    configure_chinese_matplotlib()
    figure, axes = plt.subplots(1, len(specifications), figsize=(16, 5.4))
    axes = np.atleast_1d(axes)
    for axis, specification in zip(axes, specifications, strict=True):
        panel = table[
            (table["task"] == specification["task"])
            & (table["consumer"] == specification["consumer"])
            & (table["metric"] == specification["metric"])
        ].copy()
        panel["label"] = panel["ablation_method"].map(ABLATION_LABELS).fillna(
            panel["ablation_method"]
        )
        bars = axis.bar(
            np.arange(len(panel)),
            panel["mean"],
            yerr=panel["std"],
            capsize=3,
            color="#C94C4C",
            alpha=0.88,
        )
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_title(specification["title"], fontsize=12)
        axis.set_ylabel("完整 Chronaris 的方向归一化优势")
        axis.set_xticks(np.arange(len(panel)), panel["label"])
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.22)
        _label_bars(axis, bars)
    figure.suptitle("Chronaris 关键机制消融", fontsize=16, fontweight="bold")
    figure.text(
        0.5,
        0.015,
        "正值表示完整模型优于对应消融；分类、回归与分段分别保留原始指标尺度，不跨面板直接比较数值大小。",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.92))
    return _save(figure, path)


def _label_bars(axis, bars) -> None:
    values = [bar.get_height() for bar in bars]
    span = max(max(values, default=0.0) - min(values, default=0.0), 1e-6)
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.025 * span if value >= 0 else -0.045 * span),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )


def _save(figure, path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return destination
