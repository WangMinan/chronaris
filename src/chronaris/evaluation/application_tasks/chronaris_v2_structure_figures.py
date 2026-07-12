"""Reader-facing figures for the Chronaris v2 structure gate."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


DISPLAY_NAMES = {
    "structure_01_v1": "v1 参考",
    "structure_02_semantic_groups": "语义组编码",
    "structure_03_learned_causal": "可学习因果注意力",
    "structure_04_private_shared": "私有—共享子空间",
    "structure_05_lag_objective": "滞后条件目标",
    "structure_06_corrected_physics": "修正物理约束",
    "structure_07_missing_curriculum": "缺失课程",
    "structure_08_complete_v2": "完整 v2",
}


def render_structure_gate_figure(
    ranking_csv: str | Path,
    output_path: str | Path,
) -> Path:
    _configure_chinese_font()
    data = pd.read_csv(ranking_csv)
    labels = [DISPLAY_NAMES.get(value, value) for value in data["candidate_id"]]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        data["vehicle_fidelity_ratio"],
        width,
        label="航电信息保真比",
        color="#3973ac",
    )
    axes[0].bar(
        positions + width / 2,
        data["physiology_fidelity_ratio"],
        width,
        label="生理信息保真比",
        color="#d46a4c",
    )
    axes[0].axhline(0.98, color="#202020", linestyle="--", label="98% 晋级门")
    axes[0].set_ylabel("相对最佳单流恢复能力")
    axes[0].set_title("双模态信息保真：两项必须同时达到 98%")
    axes[0].legend(ncol=3, loc="upper center")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].plot(
        positions,
        data["time_mechanism_ratio"],
        marker="o",
        linewidth=2,
        color="#6a4c93",
        label="时间机制误差 / v1",
    )
    axes[1].axhline(1.10, color="#202020", linestyle="--", label="最多恶化 10%")
    axes[1].set_ylabel("误差比例（越低越好）")
    axes[1].set_title("合成时钟偏移与响应时延恢复门禁")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper left")
    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=18, ha="right")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def _configure_chinese_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in (
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
    ):
        if family in available:
            plt.rcParams["font.sans-serif"] = [
                family,
                *plt.rcParams.get("font.sans-serif", []),
            ]
            break
    plt.rcParams["axes.unicode_minus"] = False
