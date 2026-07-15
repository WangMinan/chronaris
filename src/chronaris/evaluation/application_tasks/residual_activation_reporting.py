"""Reader-facing stage-3A report and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evidence.downstream_application_figures import (
    configure_chinese_matplotlib,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


_LABELS = {
    "fixed015_full_shared": "固定0.15门控\n共享适配器",
    "fixed025_residual_task": "固定0.25门控\n任务独立适配器",
    "global015_residual_task": "全局0.15门控\n任务独立适配器",
    "conditional015_residual_task": "样本门控0.15\n任务独立适配器",
    "conditional025_residual_task": "样本门控0.25\n任务独立适配器",
    "conditional015_residual_task_distill": "样本门控0.15\n选择性教师辅助",
    "conditional025_residual_task_distill": "样本门控0.25\n选择性教师辅助",
    "global025_residual_task_distill": "全局0.25门控\n选择性教师辅助",
}


def write_residual_activation_report(
    path: Path,
    *,
    allowance,
    aggregate,
    selected,
) -> None:
    """Write an answer-first Chinese report without changing outer claims."""

    rows = sorted(aggregate, key=lambda row: row["ranking_score"], reverse=True)
    best = rows[0]
    task_gain_without_activation = any(
        int(row["improved_task_count"]) >= 2
        and not bool(row["activation_gate_passed"])
        for row in rows
    )
    conclusion = (
        "教师辅助残差激活、无损和研究门禁均通过，允许进入三随机种子稳定性确认。"
        if selected
        else "本轮已完成残差激活筛选，但没有候选同时通过激活、无损和双任务改善门禁，因此停止在训练内阶段。"
    )
    lines = [
        "# 鼎新教师辅助残差激活与任务解耦报告",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        (
            f"当前排名最高的配置为“{_LABELS[best['candidate_id']].replace(chr(10), '、')}”："
            f"机动强度分类平均宏平均 F1 为 `{best['maneuver_mean_macro_f1']:.4f}`，"
            f"连续生理响应中位 RMSE 比率为 `{best['response_median_rmse_ratio']:.4f}`，"
            f"高生理响应平均归一化平均精确率为 `{best['high_response_mean_normalized_ap']:.4f}`；"
            f"但连续响应平均技能为 `{best['response_mean_skill']:.4f}`，"
            f"中位残差贡献比只有 `{best['median_residual_contribution_ratio']:.4f}`。"
        ),
        "",
        "本轮只使用 6 个唯一训练内验证支持。外层观测、标签、预测和指标均未打开，既有确认指标未改变。",
        "",
        "## 门禁结果",
        "",
        f"- 残差激活门禁：`{str(bool(allowance['activation_gate_passed'])).lower()}`。",
        f"- 安全无损门禁：`{str(bool(allowance['safety_gate_passed'])).lower()}`。",
        f"- 双任务研究门禁：`{str(bool(allowance['research_gate_passed'])).lower()}`。",
        f"- 允许进入三随机种子稳定性确认：`{str(bool(allowance['allow_stage_3b'])).lower()}`。",
        "- 教师训练预测全部来自训练内交叉拟合；正式推理不需要教师集成。",
        "- 配置尚未锁定，外层测试保持关闭。",
        "",
        "## 候选比较",
        "",
        "下表同时展示任务表现和残差是否真正参与预测，用于判断连续因果支路是否只停留在安全基座附近。",
        "",
        "| 配置 | 机动平均 F1 | 连续响应 RMSE 比率 | 高响应归一化 AP | 中位残差贡献比 | 激活 | 无损 | 研究门禁 |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        label = _LABELS[row["candidate_id"]].replace("\n", "、")
        lines.append(
            f"| {label} | {row['maneuver_mean_macro_f1']:.4f} | "
            f"{row['response_median_rmse_ratio']:.4f} | "
            f"{row['high_response_mean_normalized_ap']:.4f} | "
            f"{row['median_residual_contribution_ratio']:.4f} | "
            f"{_yes(row['activation_gate_passed'])} | "
            f"{_yes(row['safety_gate_passed'])} | "
            f"{_yes(row['research_gate_passed'])} |"
        )
    lines.extend(
        [
            "",
            "## 研究判断",
            "",
            (
                "通过候选同时证明残差输出在多数验证支持上离开初始状态、门控没有集中到上下界，并且至少两项任务形成稳定改善。下一步只对前两名候选执行 seeds 17、29、43 的内部稳定性确认，仍不打开外层测试。"
                if selected
                else (
                    "任务独立检查点选择后已有候选在机动分类和高响应识别上达到改善判据，但跨验证支持的中位残差贡献比没有进入预注册有效区间，因此不能证明连续因果支路形成稳定且具有研究意义的增量。按协议不进入三随机种子配置锁定，也不使用外层结果进行同轮补救。"
                    if task_gain_without_activation
                    else "没有候选满足预注册的双任务改善条件。按协议不进入三随机种子配置锁定，也不使用外层结果进行同轮补救。"
                )
            ),
            "",
            "教师辅助在本轮只承担训练内基座误差修正：机动任务使用航电单流和连续时间注意力方法，连续响应使用跨模态 Transformer、连续时间注意力方法与朴素时间同步，高响应使用航电单流和跨模态 Transformer。所有软目标均通过未见对应训练样本标签的交叉拟合模型生成。",
            "",
            "## 证据入口",
            "",
            "- `candidate_summary.csv`：候选级三重门禁与核心指标。",
            "- `candidate_support_metrics.csv`：逐验证支持的任务指标与安全基座差值。",
            "- `residual_activation_diagnostics.csv`：最优 epoch、残差贡献与预测变化比例。",
            "- `sample_gate_statistics.csv`：各任务门控分布与上下界集中度。",
            "- `teacher_oof_statistics.csv`：交叉拟合教师选择比例与权重。",
            "- `access_audit.csv`：外层访问审计。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_residual_activation_figures(compact_root: Path) -> None:
    """Render and register two compact Chinese diagnostic figures."""

    configure_chinese_matplotlib()
    summary = pd.read_csv(compact_root / "candidate_summary.csv")
    labels = [_LABELS[value] for value in summary["candidate_id"]]
    x = np.arange(len(summary))

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    panels = (
        ("maneuver_mean_macro_f1", "机动强度分类\n宏平均 F1", True),
        ("response_median_rmse_ratio", "连续生理响应\n中位 RMSE 比率", False),
        ("high_response_mean_normalized_ap", "高生理响应\n归一化平均精确率", True),
    )
    for axis, (column, title, higher) in zip(axes, panels, strict=True):
        colors = ["#1f77b4" if value else "#9aa0a6" for value in summary["research_gate_passed"]]
        bars = axis.bar(x, summary[column], color=colors)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        if not higher:
            axis.invert_yaxis()
    figure.suptitle("任务表现用于判断残差修正是否转化为稳定增益", fontsize=14)
    overview = compact_root / "residual_activation_task_metrics.png"
    figure.savefig(overview, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    bars = axes[0].bar(x, summary["median_corrected_sample_fraction"], color="#2a9d8f")
    axes[0].axhline(0.20, color="#c23b22", linestyle="--", label="最低修正比例 0.20")
    axes[0].set_title("获得可测量残差修正的样本比例")
    axes[0].set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
    axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(x, summary["median_residual_contribution_ratio"], color="#e9c46a")
    axes[1].axhspan(0.02, 0.30, color="#2a9d8f", alpha=0.15, label="预注册有效区间")
    axes[1].set_title("连续因果支路的中位残差贡献比")
    axes[1].set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
    axes[1].bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("残差激活强度与安全区间", fontsize=14)
    usage = compact_root / "residual_activation_usage.png"
    figure.savefig(usage, dpi=180, bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "figures": [
            {
                "path": overview.name,
                "sha256": sha256_file(overview),
                "purpose": "比较三个鼎新核心任务的训练内表现",
            },
            {
                "path": usage.name,
                "sha256": sha256_file(usage),
                "purpose": "审计残差修正比例与连续因果支路贡献区间",
            },
        ],
        "reader_visible_internal_tokens": False,
        "outer_test_opened": False,
    }
    (compact_root / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _yes(value) -> str:
    return "是" if bool(value) else "否"
