"""Chinese-labeled compact figures for simulation quality review."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def render_simulation_audit_figures(
    *,
    validation: pd.DataFrame,
    output_root: Path,
) -> tuple[Mapping[str, object], ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_chinese_font()
    manifests = []
    workload_path = output_root / "workload_distribution.png"
    _plot_workload_distribution(validation, workload_path)
    manifests.append(_figure_row(workload_path, "仿真负荷等级覆盖", "检查低、中、高负荷覆盖是否满足规格"))
    physics_path = output_root / "physics_and_lag_audit.png"
    _plot_physics_and_lag(validation, physics_path)
    manifests.append(_figure_row(physics_path, "物理残差与生理时延审计", "检查生成族物理边界和响应时延可恢复性"))
    observation_path = output_root / "observation_quality.png"
    _plot_observation_quality(validation, observation_path)
    manifests.append(_figure_row(observation_path, "观测缺失与时钟偏移", "检查不同压力场景的缺失与时钟偏移是否生效"))
    state_path = output_root / "maneuver_state_coverage.png"
    _plot_state_coverage(validation, state_path)
    manifests.append(_figure_row(state_path, "机动状态时间覆盖", "检查五类机动状态均形成可用片段"))
    return tuple(manifests)


def _plot_workload_distribution(frame: pd.DataFrame, path: Path) -> None:
    latent = frame.drop_duplicates(["split_id", "trajectory_id"])
    grouped = latent.groupby("split_id", sort=True)[
        ["workload_low_ratio", "workload_medium_ratio", "workload_high_ratio"]
    ].mean()
    grouped = grouped.reindex(_ordered_splits(grouped.index))
    labels = [_display_split(value) for value in grouped.index]
    x = np.arange(len(labels))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 5.2))
    for offset, (column, label, color) in enumerate((
        ("workload_low_ratio", "低负荷", "#4C78A8"),
        ("workload_medium_ratio", "中负荷", "#F2CF5B"),
        ("workload_high_ratio", "高负荷", "#E45756"),
    )):
        values = grouped[column].to_numpy()
        positions = x + (offset - 1) * width
        axis.bar(positions, values, width, label=label, color=color)
        for position, value in zip(positions, values):
            axis.text(position, value + 0.012, f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    axis.axhline(0.15, color="#555555", linestyle="--", linewidth=1, label="单档最低占比 15%")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(0.65, float(grouped.to_numpy().max()) + 0.10))
    axis.set_ylabel("时间占比")
    axis.set_title("仿真负荷等级覆盖")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_physics_and_lag(frame: pd.DataFrame, path: Path) -> None:
    latent = frame.drop_duplicates(["split_id", "trajectory_id"])
    clean = frame.loc[frame["scenario_id"] == "clean_asynchronous"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    families = list(latent["generator_family"].drop_duplicates())
    residual_data = [
        latent.loc[latent["generator_family"] == family, "physical_residual_abs_q95"]
        for family in families
    ]
    axes[0].boxplot(
        residual_data,
        tick_labels=[_display_family(value) for value in families],
        showfliers=False,
    )
    axes[0].axhline(0.35, color="#E45756", linestyle="--", label="G2 95% 上限")
    axes[0].set_ylabel("标准化残差绝对值 95% 分位")
    axes[0].set_title("物理一致性残差")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.005, 0.5)
    axes[0].set_yticks((0.01, 0.03, 0.10, 0.30), ("0.01", "0.03", "0.10", "0.30"))
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    for family, group in clean.groupby("generator_family", sort=True):
        axes[1].scatter(
            group["true_primary_response_lag_s"],
            group["estimated_primary_response_lag_s"],
            label=_display_family(family),
            alpha=0.75,
        )
    bounds = [0, max(32.0, float(clean["true_primary_response_lag_s"].max()) + 1)]
    axes[1].plot(bounds, bounds, color="#333333", linewidth=1, label="理想恢复")
    axes[1].fill_between(bounds, np.asarray(bounds) - 1, np.asarray(bounds) + 1, color="#72B7B2", alpha=0.15)
    axes[1].set_xlim(bounds)
    axes[1].set_ylim(bounds)
    axes[1].set_xlabel("真实主响应时延（秒）")
    axes[1].set_ylabel("互相关估计时延（秒）")
    axes[1].set_title("干净场景响应时延恢复")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_observation_quality(frame: pd.DataFrame, path: Path) -> None:
    grouped = frame.groupby("scenario_id", sort=True).agg(
        vehicle_missing_ratio=("vehicle_missing_ratio", "mean"),
        physiology_missing_ratio=("physiology_missing_ratio", "mean"),
        vehicle_clock_offset_s=("vehicle_clock_offset_s_config", lambda values: float(np.mean(np.abs(values)))),
        physiology_clock_offset_s=("physiology_clock_offset_s_config", lambda values: float(np.mean(np.abs(values)))),
    )
    grouped = grouped.reindex(_ordered_scenarios(grouped.index))
    labels = [_display_scenario(value) for value in grouped.index]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].bar(x - width / 2, grouped["vehicle_missing_ratio"], width, label="航电流", color="#4C78A8")
    axes[0].bar(x + width / 2, grouped["physiology_missing_ratio"], width, label="生理流", color="#E45756")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("观测点缺失比例")
    axes[0].set_title("实际缺失比例")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x - width / 2, grouped["vehicle_clock_offset_s"], width, label="航电流", color="#4C78A8")
    axes[1].bar(x + width / 2, grouped["physiology_clock_offset_s"], width, label="生理流", color="#E45756")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("绝对时钟偏移（秒）")
    axes[1].set_title("配置时钟偏移")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_state_coverage(frame: pd.DataFrame, path: Path) -> None:
    latent = frame.drop_duplicates(["split_id", "trajectory_id"])
    columns = [
        ("state_ratio_steady", "稳态"),
        ("state_ratio_entry", "机动进入"),
        ("state_ratio_sustained", "持续机动"),
        ("state_ratio_exit", "机动退出"),
        ("state_ratio_recovery", "恢复"),
    ]
    grouped = latent.groupby("split_id", sort=True)[[column for column, _ in columns]].mean()
    grouped = grouped.reindex(_ordered_splits(grouped.index))
    fig, axis = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(grouped.index))
    bottom = np.zeros(len(x))
    colors = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#B279A2")
    for (column, label), color in zip(columns, colors):
        values = grouped[column].to_numpy()
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, [_display_split(value) for value in grouped.index])
    axis.set_ylabel("时间占比")
    axis.set_title("五类机动状态时间覆盖")
    axis.legend(ncol=3, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _configure_chinese_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in ("WenQuanYi Zen Hei", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei"):
        if family in available:
            plt.rcParams["font.sans-serif"] = [family, *plt.rcParams.get("font.sans-serif", [])]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _figure_row(path: Path, title: str, purpose: str) -> Mapping[str, object]:
    return {"path": str(path), "title": title, "purpose": purpose}


def _display_split(value: str) -> str:
    return {
        "smoke_g1": "G1 状态空间（冒烟）",
        "smoke_g2": "G2 事件样条（冒烟）",
        "train": "训练集",
        "validation": "验证集",
        "locked_test": "锁定测试集",
    }.get(value, value)


def _display_family(value: str) -> str:
    return {
        "g1_state_space": "G1 状态空间生成族",
        "g2_event_spline": "G2 事件样条生成族",
    }.get(value, value)


def _display_scenario(value: str) -> str:
    return {
        "clean_asynchronous": "干净异步",
        "sampling_jitter": "采样抖动",
        "clock_offset_and_drift": "时钟偏移与漂移",
        "random_missing": "随机缺失",
        "block_missing_and_long_lag": "连续缺失与长时延",
        "mixed_severe": "混合重度压力",
    }.get(value, value)


def _ordered_splits(values) -> list[str]:
    available = set(values)
    preferred = ("train", "validation", "locked_test", "smoke_g1", "smoke_g2")
    return [value for value in preferred if value in available]


def _ordered_scenarios(values) -> list[str]:
    available = set(values)
    preferred = (
        "clean_asynchronous",
        "sampling_jitter",
        "clock_offset_and_drift",
        "random_missing",
        "block_missing_and_long_lag",
        "mixed_severe",
    )
    return [value for value in preferred if value in available]
