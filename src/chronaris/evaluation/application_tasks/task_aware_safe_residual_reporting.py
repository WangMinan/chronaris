"""Reader-facing report for Dingxin task-aware safe residual screening."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def write_safe_residual_report(
    path, *, allowance, selected, aggregate, partial_aggregate
) -> None:
    if selected is None:
        decision = "冻结安全残差门禁未通过，按预注册要求不进入部分解冻。"
        selected_text = "无候选同时满足三项无损门禁。"
    else:
        decision = "冻结安全残差门禁通过，已按预注册范围完成 Chronaris 部分解冻。"
        selected_text = (
            f"冻结筛选选中 `{selected['candidate_id']}`：机动平均宏平均 F1 "
            f"{selected['maneuver_mean_macro_f1']:.4f}，连续响应中位 RMSE 比率 "
            f"{selected['response_median_rmse_ratio']:.4f}，高响应平均归一化平均精确率 "
            f"{selected['high_response_mean_normalized_ap']:.4f}。"
        )
    path.write_text(
        "\n".join(
            (
                "# 鼎新任务感知安全残差筛选报告",
                "",
                "## 结论",
                "",
                decision,
                "",
                selected_text,
                "",
                "本轮只使用训练内 6 个唯一验证支持；外层观测、标签、预测和指标均未打开。",
                "",
                "## 门禁",
                "",
                f"- 冻结安全残差通过：`{str(allowance['frozen_safety_passed']).lower()}`。",
                f"- 允许部分解冻：`{str(allowance['allow_partial_unfreeze']).lower()}`。",
                f"- 部分解冻安全门禁：`{str(allowance['partial_safety_passed']).lower()}`。",
                f"- 部分解冻研究门禁：`{str(allowance['research_gate_passed']).lower()}`。",
                f"- 允许教师蒸馏：`{str(allowance['allow_teacher_distillation']).lower()}`。",
                "- 历史外层绝对指标没有参与候选选择。",
                "",
                "## 冻结候选概览",
                "",
                *(
                    f"- `{row['candidate_id']}`：安全门禁 "
                    f"`{str(row['frozen_safety_passed']).lower()}`，机动/响应/高响应安全支持数 "
                    f"{row['maneuver_safe_support_count']}/{row['response_safe_support_count']}/"
                    f"{row['high_response_safe_support_count']}。"
                    for row in aggregate
                ),
                "",
                "## 部分解冻候选概览",
                "",
                *(
                    f"- 主干学习率比例 `{str(row['candidate_id']).removeprefix('partial_ratio_')}`："
                    f"安全门禁 `{str(row['partial_safety_passed']).lower()}`，"
                    f"改善任务数 {row['improved_task_count']}/3，"
                    f"研究门禁 `{str(row['research_gate_passed']).lower()}`。"
                    for row in partial_aggregate
                ),
                "",
            )
        ),
        encoding="utf-8",
    )


def write_safe_residual_figures(root) -> None:
    """Render the two compact evidence figures and their chart contract."""

    _configure_font()
    frozen = pd.read_csv(root / "candidate_inner_metrics.csv")
    partial = pd.read_csv(root / "partial_unfreeze_metrics.csv")
    _plot_frozen_supports(frozen, root / "frozen_safety_supports.png")
    _plot_partial_deltas(partial, root / "partial_unfreeze_deltas.png")
    (root / "figure_manifest.json").write_text(
        json.dumps(
            {
                "figures": [
                    {
                        "path": "frozen_safety_supports.png",
                        "question": "冻结安全残差在多少个唯一验证支持上不伤害任务基座？",
                        "takeaway": "五个候选均在三个任务的 6/6 支持上满足安全门禁。",
                        "family": "Matrix & Cohort",
                        "chart_type": "annotated heatmap",
                        "row_count": int(len(frozen)),
                    },
                    {
                        "path": "partial_unfreeze_deltas.png",
                        "question": "部分解冻相对冻结状态是否形成跨任务增益？",
                        "takeaway": "两个学习率比例都只改善一个任务，未达到研究门禁。",
                        "family": "Comparison & Ranking",
                        "chart_type": "three-panel categorical bar",
                        "row_count": int(len(partial)),
                    },
                ],
                "palette_policy": "hard two-root cap",
                "outer_test_opened": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _plot_frozen_supports(frame, path):
    columns = (
        "maneuver_safe_support_count",
        "response_safe_support_count",
        "high_response_safe_support_count",
    )
    labels = {
        "observed_vehicle_scalar": "直接观测机动 / 航电响应",
        "vehicle_vehicle_scalar": "航电机动 / 航电响应",
        "observed_physiology_scalar": "直接观测机动 / 生理响应",
        "vehicle_observed_scalar": "航电机动 / 直接观测响应",
        "observed_vehicle_channel": "逐通道门控直接观测 / 航电响应",
    }
    values = frame[list(columns)].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(9.6, 4.8))
    image = axis.imshow(values, vmin=0, vmax=6, cmap="Blues", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{int(values[row, column])}/6",
                ha="center",
                va="center",
                color="#14213D" if values[row, column] < 4 else "white",
                fontweight="bold",
            )
    axis.set_xticks(range(3), ("机动分类", "连续响应", "高响应识别"))
    axis.set_yticks(
        range(len(frame)),
        [labels[str(value)] for value in frame["candidate_id"]],
    )
    axis.set_title("冻结安全残差的无损验证支持数", loc="left", fontweight="bold")
    axis.set_xlabel("训练内 6 个唯一验证支持；数值越高表示无损性越稳定")
    figure.colorbar(image, ax=axis, label="满足无损门禁的支持数")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_partial_deltas(frame, path):
    candidates = tuple(frame["candidate_id"].drop_duplicates())
    labels = [value.removeprefix("partial_ratio_") for value in candidates]
    colors = ("#3568A8", "#D59A24")
    specifications = (
        ("maneuver", "macro_f1", "机动宏平均 F1 变化", 1.0),
        ("response", "rmse", "连续响应 RMSE 相对改善", 100.0),
        ("high_response", "auprc", "高响应平均精确率变化", 1.0),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    for axis, (task, metric, title, scale) in zip(axes, specifications, strict=True):
        values = []
        for candidate in candidates:
            selected = frame[
                (frame["candidate_id"] == candidate)
                & (frame["task"] == task)
                & (frame["metric"] == metric)
            ]
            if task == "response":
                delta = np.mean(
                    (selected["frozen_value"] - selected["partial_value"])
                    / selected["frozen_value"]
                )
            else:
                delta = selected["direction_normalized_delta"].mean()
            scaled = float(delta) * scale
            values.append(0.0 if abs(scaled) < 1e-12 else scaled)
        bars = axis.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.7)
        axis.axhline(0, color="#263238", linewidth=0.9)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("主干学习率比例")
        axis.set_ylabel("百分点" if task == "response" else "绝对变化")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.margins(y=0.25)
        if max(abs(value) for value in values) < 1e-9:
            axis.set_ylim(-0.01, 0.01)
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(
                f"{value:+.4f}" if task != "response" else f"{value:+.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        "部分解冻相对冻结状态的训练内变化",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        "正值表示改善；仅使用 6 个训练内唯一验证支持，外层测试未打开。",
        fontsize=9,
        color="#455A64",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _configure_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
    ):
        if family in available:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False
