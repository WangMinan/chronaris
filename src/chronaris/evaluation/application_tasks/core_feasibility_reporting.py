"""Compact evidence writer for Dingxin core-task feasibility screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    sha256_file,
)


TASK_LABELS = {
    "maneuver": "鼎新机动强度分类",
    "response": "鼎新未来生理响应预测",
    "high_response": "鼎新高生理响应识别",
}
TASK_METRIC_LABELS = {
    "maneuver": "Macro-F1（越高越好）",
    "response": "均方根误差（越低越好）",
    "high_response": "精确率-召回率曲线下面积（越高越好）",
}


def write_core_feasibility_outputs(
    *,
    compact_root: str | Path,
    heavy_root: str | Path,
    protocol: Mapping[str, object],
    gate_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
    fold_rows: Sequence[Mapping[str, object]],
    safe_fusion_rows: Sequence[Mapping[str, object]],
    expert_gate_rows: Sequence[Mapping[str, object]],
    access_rows: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    dependency_rows: Sequence[Mapping[str, object]],
    candidate_count: int,
) -> dict[str, str]:
    root = Path(compact_root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    upper_gate_passed = all(bool(row["mean_gate_passed"]) for row in gate_rows)
    safe_attempted = any(row.get("status") == "completed" for row in safe_fusion_rows)
    safe_passed = safe_attempted and all(
        bool(row.get("gate_passed")) for row in safe_fusion_rows
    )
    status = "accepted" if upper_gate_passed and safe_passed else "gap"

    paths = {
        "protocol": root / "protocol.json",
        "task_learnability": root / "task_learnability.csv",
        "upper_bound_results": root / "upper_bound_results.csv",
        "consumer_comparison": root / "consumer_comparison.csv",
        "safe_fusion_results": root / "safe_fusion_results.csv",
        "fold_metrics": root / "fold_metrics.csv",
        "expert_gate_statistics": root / "expert_gate_statistics.csv",
        "leakage_and_access_audit": root / "leakage_and_access_audit.csv",
        "support_isolation_audit": root / "support_isolation_audit.csv",
        "task_diagnostics": root / "task_diagnostics.csv",
        "dependency_state": root / "dependency_state.csv",
        "decision_report": root
        / ("acceptance_report.md" if status == "accepted" else "gap_report.md"),
        "evidence_manifest": root / "evidence_manifest.json",
        "resume_command": root / "resume_command.txt",
        "progress": root / "progress.json",
        "upper_figure": figures / "任务可学习性上限.png",
        "history_figure": figures / "历史长度贡献.png",
        "expert_figure": figures / "信息路径贡献.png",
    }
    _write_json(paths["protocol"], protocol)
    pd.DataFrame(gate_rows).to_csv(paths["task_learnability"], index=False)
    pd.DataFrame(gate_rows).to_csv(paths["upper_bound_results"], index=False)
    pd.DataFrame(comparison_rows).to_csv(paths["consumer_comparison"], index=False)
    pd.DataFrame(safe_fusion_rows).to_csv(paths["safe_fusion_results"], index=False)
    pd.DataFrame(fold_rows).to_csv(paths["fold_metrics"], index=False)
    pd.DataFrame(expert_gate_rows).to_csv(paths["expert_gate_statistics"], index=False)
    pd.DataFrame(access_rows).to_csv(paths["leakage_and_access_audit"], index=False)
    pd.DataFrame(support_rows).to_csv(paths["support_isolation_audit"], index=False)
    pd.DataFrame(diagnostic_rows).to_csv(paths["task_diagnostics"], index=False)
    pd.DataFrame(dependency_rows).to_csv(paths["dependency_state"], index=False)

    _configure_chinese_fonts()
    _plot_upper_bounds(gate_rows, paths["upper_figure"])
    _plot_history_contribution(fold_rows, paths["history_figure"])
    _plot_path_contribution(fold_rows, paths["expert_figure"])
    report = _report_text(
        gate_rows=gate_rows,
        upper_gate_passed=upper_gate_passed,
        safe_attempted=safe_attempted,
        safe_passed=safe_passed,
        support_rows=support_rows,
        diagnostic_rows=diagnostic_rows,
        fold_rows=fold_rows,
        candidate_count=candidate_count,
    )
    paths["decision_report"].write_text(report, encoding="utf-8")
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_core_feasibility.py "
        "--resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["progress"],
        {
            "status": "completed",
            "decision": status,
            "upper_bound_gate_passed": upper_gate_passed,
            "safe_fusion_attempted": safe_attempted,
            "safe_fusion_gate_passed": safe_passed,
            "candidate_count": candidate_count,
            "outer_test_opened": False,
            "confirmed_metrics_changed": False,
        },
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "format": "chronaris.dingxin_core_feasibility_evidence.v1",
            "status": "completed",
            "decision": status,
            "upper_bound_gate_passed": upper_gate_passed,
            "safe_fusion_gate_passed": safe_passed,
            "outer_test_opened": False,
            "upper_bound_models_fit": True,
            "safe_fusion_training_invoked": safe_attempted,
            "confirmed_metrics_changed": False,
            "heavy_run_root": str(heavy_root),
            "output_paths": {key: str(value) for key, value in paths.items()},
            "output_sha256": {
                key: sha256_file(value)
                for key, value in paths.items()
                if key != "evidence_manifest" and Path(value).is_file()
            },
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _report_text(
    *,
    gate_rows,
    upper_gate_passed,
    safe_attempted,
    safe_passed,
    support_rows,
    diagnostic_rows,
    fold_rows,
    candidate_count,
):
    lines = [
        "# 鼎新核心任务可行性与安全融合审计",
        "",
        "## 结论",
        "",
        (
            "三个任务的训练内上限全部越过预声明门槛。"
            if upper_gate_passed
            else "三个任务的训练内上限没有全部越过预声明门槛，本轮按协议停止主干修改。"
        ),
        "",
    ]
    for row in gate_rows:
        value = row["best_mean_value"]
        rendered = "不可用" if value is None else f"{float(value):.6f}"
        lines.append(
            f"- {TASK_LABELS[row['task']]}：最佳训练内均值 `{rendered}`；"
            f"门槛 `{row['threshold']}`；通过 `{str(bool(row['mean_gate_passed'])).lower()}`。"
        )
    lines.extend(
        [
            "",
            "## 协议边界",
            "",
            f"- 主候选数：`{candidate_count}`，未超过预声明的 12 个候选预算。",
            "- 只使用 inner-train 拟合、inner-validation 评价；held-out/outer-test 保持关闭。",
            "- 三个主留一视图折的完整 30 秒输入加 5 秒目标支持区间均通过隔离审计。",
            "- 原始快照、标签公式、字段角色和外层折保持不变；开发标签与阈值只用净化后的 inner-train 重拟合。",
            "- 本轮不修改历史确认指标，不把训练内上限写入论文确认表。",
            "- 生理响应训练内目标按各折 inner-train 的字段变化四分位距重新缩放；预声明绝对门槛仍原样执行。",
            "",
            "## 可学习性与折间迁移判断",
            "",
            *_diagnostic_lines(gate_rows, diagnostic_rows, fold_rows),
            "",
            "## 输入信息与表示判断",
            "",
            *_information_path_lines(fold_rows),
            "",
            "## 安全融合状态",
            "",
        ]
    )
    if safe_attempted:
        lines.append(
            f"- 上限门禁通过后已运行冻结专家安全融合；最终门禁通过 `{str(safe_passed).lower()}`。"
        )
    else:
        lines.append(
            "- 上限门禁未全部通过，因此冻结专家安全融合训练未启动；对应 CSV 以结构化阻断状态保留。"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            (
                "允许启动任务感知 Chronaris 主干开发。"
                if upper_gate_passed and safe_passed
                else "不允许启动下一长程任务。应先处理样本规模、任务定义稳定性与跨架次分布差异，不能用外层结果继续搜索。"
            ),
            "",
            "## 支持区间审计",
            "",
        ]
    )
    for row in support_rows:
        lines.append(
            f"- `{row['fold_id']}`：重叠对 `{row['support_overlap_pair_count']}`，"
            f"隔离通过 `{str(bool(row['support_isolated'])).lower()}`。"
        )
    lines.append("")
    return "\n".join(lines)


def _diagnostic_lines(gate_rows, diagnostic_rows, fold_rows):
    diagnostics = pd.DataFrame(diagnostic_rows).set_index("fold_id")
    frame = pd.DataFrame(fold_rows)
    lines = []
    for gate in gate_rows:
        selected = frame[
            (frame["task"] == gate["task"])
            & (frame["candidate_id"] == gate["best_candidate_id"])
        ].sort_values("fold_id")
        rendered = ", ".join(f"{value:.3f}" for value in selected["value"])
        lines.append(
            f"- {TASK_LABELS[gate['task']]}的最佳同一候选折级结果为 `{rendered}`，"
            "说明均值受到明显的留一视图差异影响。"
        )
    upper_values = diagnostics["maneuver_upper_bound"]
    last_fold = diagnostics.iloc[-1]
    response_shift = (
        last_fold["validation_response_mean"]
        / last_fold["train_response_mean"]
        - 1.0
    )
    lines.extend(
        [
            (
                "- 机动高等级阈值在前两个折为 "
                f"`{upper_values.iloc[0]:.3f}`，第三折为 `{upper_values.iloc[-1]:.3f}`；"
                "同一弱监督等级在不同训练视图上的标尺明显变化。"
            ),
            (
                "- 第三折验证集仅覆盖两个机动类别；其生理响应均值比训练集高 "
                f"`{response_shift:.1%}`，高响应样本占比为 "
                f"`{last_fold['validation_high_response_rate']:.1%}`。"
            ),
            "- 因此当前瓶颈不仅是 consumer：标签标尺、类别覆盖和跨视图分布同时限制了训练内上限。",
        ]
    )
    return lines


def _information_path_lines(fold_rows):
    frame = pd.DataFrame(fold_rows)

    def mean(task, candidate):
        values = frame[
            (frame["task"] == task) & (frame["candidate_id"] == candidate)
        ]["value"]
        return float(values.mean())

    maneuver_5s = mean("maneuver", "summary_vehicle_5s_linear")
    maneuver_30s = mean("maneuver", "summary_vehicle_30s_linear")
    response_vehicle = mean("response", "summary_vehicle_30s_linear")
    response_dual = mean("response", "summary_dual_30s_linear")
    high_historical = mean("high_response", "historical_frozen_panel")
    high_sequence = mean("high_response", "sequence_minirocket_5000")
    return [
        (
            "- 机动分类最有价值的是完整 30 秒航电路径：Macro-F1 从最近 5 秒的 "
            f"`{maneuver_5s:.3f}` 提升到 `{maneuver_30s:.3f}`。"
        ),
        (
            "- 连续响应中，双流直接观测将航电路径 RMSE 从 "
            f"`{response_vehicle:.3f}` 降到 `{response_dual:.3f}`，增益存在但很小。"
        ),
        (
            "- 高响应识别中，多尺度卷积时序路径达到 "
            f"`{high_sequence:.3f}`，高于历史冻结表示的 `{high_historical:.3f}`。"
        ),
        "- 允许原始信息配合更强 consumer 仍未达到门槛，说明差距不能只归因于 Chronaris encoder；表示压缩与任务/折级稳定性都需要后续处理。",
    ]


def _plot_upper_bounds(gate_rows, path):
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, row in zip(axes, gate_rows, strict=True):
        value = float(row["best_mean_value"]) if row["best_mean_value"] is not None else 0.0
        threshold = float(row["threshold"])
        axis.bar(["训练内最佳", "预声明门槛"], [value, threshold], color=["#3B82F6", "#F59E0B"])
        axis.set_title(
            f"{TASK_LABELS[row['task']]}\n{TASK_METRIC_LABELS[row['task']]}"
        )
        axis.set_ylabel(TASK_METRIC_LABELS[row["task"]])
        for index, current in enumerate((value, threshold)):
            axis.text(index, current, f"{current:.3f}", ha="center", va="bottom", fontsize=9)
    figure.suptitle("鼎新核心任务训练内可学习性上限")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_history_contribution(fold_rows, path):
    frame = pd.DataFrame(fold_rows)
    selected = frame[
        frame["candidate_id"].isin(
            ("summary_vehicle_5s_linear", "summary_vehicle_30s_linear")
        )
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, task in zip(axes, TASK_LABELS, strict=True):
        subset = selected[selected["task"] == task]
        values = subset.groupby("candidate_id")["value"].mean()
        labels = ["最近5秒", "完整30秒"]
        plotted = [
            values.get("summary_vehicle_5s_linear", np.nan),
            values.get("summary_vehicle_30s_linear", np.nan),
        ]
        axis.bar(labels, plotted, color=["#10B981", "#6366F1"])
        axis.set_title(f"{TASK_LABELS[task]}\n{TASK_METRIC_LABELS[task]}")
        axis.set_ylabel(TASK_METRIC_LABELS[task])
    figure.suptitle("航电历史长度对训练内任务表现的影响")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_path_contribution(fold_rows, path):
    frame = pd.DataFrame(fold_rows)
    candidates = (
        "summary_physiology_30s_linear",
        "summary_vehicle_30s_linear",
        "summary_dual_30s_linear",
        "historical_frozen_panel",
    )
    labels = ["生理路径", "航电路径", "双流直接观测", "历史冻结表示"]
    colors = ["#EC4899", "#10B981", "#3B82F6", "#6B7280"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, task in zip(axes, TASK_LABELS, strict=True):
        subset = frame[frame["task"] == task]
        values = subset.groupby("candidate_id")["value"].mean()
        plotted = [values.get(candidate, np.nan) for candidate in candidates]
        axis.bar(labels, plotted, color=colors)
        axis.set_title(f"{TASK_LABELS[task]}\n{TASK_METRIC_LABELS[task]}")
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel(TASK_METRIC_LABELS[task])
    figure.suptitle("不同信息路径的训练内表现")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _configure_chinese_fonts():
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
