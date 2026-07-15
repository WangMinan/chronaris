"""Compact evidence, Chinese figures, and gate report for Dingxin task 1C."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


METHOD_LABELS = {
    "physiology_only": "生理单流",
    "vehicle_only": "航电单流",
    "naive_time_sync": "朴素时间同步",
    "mult": "MulT",
    "contiformer": "ContiFormer",
    "chronaris": "Chronaris",
    "time_only": "飞行进程诊断",
}


def write_target_reconstruction_outputs(
    *,
    compact_root: str | Path,
    protocol: dict[str, object],
    preregistration: dict[str, object],
    allowance: dict[str, object],
    best: dict[str, object],
    frames: dict[str, pd.DataFrame],
) -> dict[str, str]:
    root = Path(compact_root)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "pre_registration.json", preregistration)
    _write_json(root / "allowance.json", allowance)
    mapping = {
        "training": "matched_clean_training.csv",
        "exports": "representation_export.csv",
        "access": "access_audit.csv",
        "maneuver_targets": "future_maneuver_targets.csv",
        "maneuver_thresholds": "future_maneuver_thresholds.csv",
        "physiology_states": "robust_physiology_states.csv",
        "residual_targets": "physiology_residual_targets.csv",
        "residual_descriptors": "residual_descriptor_contract.csv",
        "inertia_audit": "physiology_inertia_audit.csv",
        "metrics": "matched_clean_metrics.csv",
        "summary": "matched_clean_summary.csv",
        "predictions": "matched_clean_predictions.csv",
        "time_shortcut": "time_shortcut_audit.csv",
        "third_pool": "third_pool_pressure.csv",
    }
    for key, filename in mapping.items():
        frames[key].to_csv(root / filename, index=False)
    _configure_chinese_fonts()
    _plot_maneuver(frames["summary"], figures / "未来机动分数与趋势比较.png")
    _plot_physiology(frames["summary"], figures / "生理惯性残差与高残差风险比较.png")
    _plot_time_shortcut(frames["time_shortcut"], figures / "抑制飞行进程捷径后的相对增益.png")
    _plot_descriptors(frames["residual_descriptors"], figures / "生理残差描述量可靠性.png")
    _plot_third_pool(frames["third_pool"], figures / "第三训练池连续目标压力评价.png")
    report_name = "acceptance_report.md" if allowance["allow_safe_fusion"] else "gap_report.md"
    report_path = root / report_name
    report_path.write_text(
        _report_text(protocol=protocol, allowance=allowance, best=best, frames=frames),
        encoding="utf-8",
    )
    evidence = {
        "format": "chronaris.dingxin_target_reconstruction_evidence.v1",
        "status": "completed",
        "decision": allowance["decision"],
        "allow_safe_fusion": allowance["allow_safe_fusion"],
        "allow_task_aware_research": allowance["allow_task_aware_research"],
        "outer_test_opened": False,
        "chronaris_backbone_modified": False,
        "chronaris_backbone_trained": False,
        "teacher_distillation_started": False,
        "confirmed_metrics_changed": False,
        "legacy_metrics_overwritten": False,
        "matched_clean_method_count": 6,
        "main_split_count": 6,
        "pressure_split_count": 1,
        "figure_count": 5,
        "compact_files": sorted([*mapping.values(), "protocol.json", "pre_registration.json", "allowance.json", report_name]),
        "source_sha256": protocol["source_sha256"],
    }
    _write_json(root / "evidence_manifest.json", evidence)
    _write_json(
        root / "progress.json",
        {
            "status": "completed",
            "decision": allowance["decision"],
            "allow_safe_fusion": allowance["allow_safe_fusion"],
            "outer_test_opened": False,
            "completed_method_split_units": int(
                frames["training"][["method", "split_id"]].drop_duplicates().shape[0]
            ),
        },
    )
    (root / "resume_command.txt").write_text(
        "PYTHONPATH=src /home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_target_reconstruction.py --resume\n",
        encoding="utf-8",
    )
    return {
        "report_path": str(report_path),
        "evidence_manifest_path": str(root / "evidence_manifest.json"),
    }


def _report_text(*, protocol, allowance, best, frames):
    score = best["future_maneuver_score"]
    trend = best["future_maneuver_trend"]
    residual = best["physiology_residual"]
    high = best["high_residual_response"]
    failed = allowance["failed_checks"]
    check_labels = {
        "future_maneuver_score_passed": "未来机动连续分数",
        "future_maneuver_trend_passed": "未来机动变化趋势",
        "physiology_residual_passed": "生理剩余响应",
        "high_residual_response_passed": "高剩余响应风险",
    }
    failed_labels = [check_labels.get(value, value) for value in failed]
    primary_blocker = check_labels.get(
        allowance["primary_blocker"], allowance["primary_blocker"]
    )
    main_split_ids = set(
        frames["metrics"].loc[
            frames["metrics"]["main_selection"].astype(bool), "split_id"
        ]
    )
    high_validation = frames["residual_targets"][
        frames["residual_targets"]["split_id"].isin(main_split_ids)
        & (frames["residual_targets"]["role"] == "validation")
    ]
    high_prevalence = high_validation.groupby("split_id")[
        "high_residual_response"
    ].mean()
    zero_positive_split_count = int((high_prevalence == 0).sum())
    conclusion = (
        "本轮全部训练内门禁通过，可以进入安全基座与 Chronaris 残差训练。"
        if allowance["allow_safe_fusion"]
        else "本轮未通过全部训练内门禁，按协议不启动主干优化或教师蒸馏。"
    )
    return f"""# 鼎新目标重构、时间捷径抑制与统一条件基线报告

## 结论

{conclusion}

本轮在外层测试保持关闭的前提下，使用相同的时间类通道删除合同、相同无标签预训练预算和相同任务头，完成生理单流、航电单流、朴素时间同步、MulT、ContiFormer 与 Chronaris 的训练内比较。既有鼎新三项确认指标没有覆盖、重算或参与候选选择。

## 目标重构结果

- 未来机动连续分数：最佳方法为 `{METHOD_LABELS[score['method']]}`，6 个主选模单元的斯皮尔曼秩相关系数（Spearman）中位数为 `{score['median']:.4f}`。
- 未来机动趋势：最佳方法为 `{METHOD_LABELS[trend['method']]}`，外层训练池平衡平均宏平均 F1（Macro-F1）为 `{trend['mean']:.4f}`，最差单元为 `{trend['worst']:.4f}`。
- 生理惯性残差：最佳方法为 `{METHOD_LABELS[residual['method']]}`，均方根误差比率（RMSE ratio）中位数为 `{residual.get('median_rmse_ratio', float('nan')):.4f}`，平均技能为 `{residual.get('mean_skill', float('nan')):.4f}`，正技能单元数为 `{residual.get('positive_skill_split_count', 0)}/6`。
- 高残差生理响应：最佳方法为 `{METHOD_LABELS[high['method']]}`，平均/中位归一化平均精确率为 `{high.get('mean_normalized_ap', float('nan')):.4f}/{high.get('median_normalized_ap', float('nan')):.4f}`，正值单元数为 `{high.get('positive_normalized_ap_split_count', 0)}/6`。
- 高残差标签的训练内阈值没有稳定迁移：`{zero_positive_split_count}/6` 个主验证单元没有正样本；该事实本身构成目标标尺不稳定证据，而不是模型领先证据。
- 时间捷径：未来机动分数与趋势中较弱一项相对飞行进程诊断的标准化增益为 `{allowance['observed']['time_shortcut_standardized_gain']:.4f}`。
- 第三训练池：完成 `{allowance['observed']['third_pool_continuous_task_count']}` 项连续目标压力评价，不参与主候选排序。

## 门禁

- 决策：未放行（机器状态 `{allowance['decision']}`）。
- 安全融合放行：`{str(allowance['allow_safe_fusion']).lower()}`。
- 任务感知研究放行：`{str(allowance['allow_task_aware_research']).lower()}`。
- 未通过项：`{'、'.join(failed_labels) if failed_labels else '无'}`。
- 首要阻断：`{primary_blocker}`。

## 证据边界

- 本轮新任务属于鼎新真实双流上的弱监督任务重构，用于检查连续动力学与生理惯性之外的可预测增量，不是新增人工专家真值。
- 历史对照面板只保留研究过程连续性；本报告的统一条件比较面板采用新的输入和目标合同，两者不混算。
- 第三训练池只承担连续目标的分布压力诊断，未因缺少完整机动类别而降低划分约束。
- Chronaris 主干未修改，也没有使用任务标签训练编码器；教师蒸馏、仿真、公开数据和外层确认均未启动。
"""


def _plot_maneuver(summary, path):
    frame = summary[summary["task"].isin(("future_maneuver_score", "future_maneuver_trend"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, (task, title, ylabel) in zip(
        axes,
        (
            ("future_maneuver_score", "未来机动连续分数", "Spearman 秩相关"),
            ("future_maneuver_trend", "未来机动变化趋势", "Macro-F1"),
        ),
        strict=True,
    ):
        subset = frame[frame["task"] == task]
        axis.bar([METHOD_LABELS[value] for value in subset["method"]], subset["mean"], color="#4C78A8")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_physiology(summary, path):
    frame = summary[summary["task"].isin(("physiology_residual", "high_residual_response"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    specifications = (
        ("physiology_residual", "生理惯性之外的连续残差", "RMSE（越低越好）", "#F58518"),
        ("high_residual_response", "高残差生理响应风险", "归一化平均精确率", "#54A24B"),
    )
    for axis, (task, title, ylabel, color) in zip(axes, specifications, strict=True):
        subset = frame[frame["task"] == task]
        axis.bar([METHOD_LABELS[value] for value in subset["method"]], subset["mean"], color=color)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_time_shortcut(frame, path):
    pivot = frame.pivot(index="method", columns="task", values="median_standardized_gain")
    methods = list(pivot.index)
    x = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10.5, 5.0))
    axis.bar(x - width / 2, pivot["future_maneuver_score"], width, label="未来机动分数")
    axis.bar(x + width / 2, pivot["future_maneuver_trend"], width, label="未来机动趋势")
    axis.axhline(0.15, color="#E45756", linestyle="--", label="放行门槛 0.15")
    axis.set_xticks(x, [METHOD_LABELS[value] for value in methods], rotation=30)
    axis.set_ylabel("相对飞行进程诊断的标准化增益")
    axis.set_title("抑制飞行进程捷径后的任务增益")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_descriptors(frame, path):
    selected = frame[frame["selected"].astype(bool)].copy()
    selected = selected.groupby("descriptor_key", as_index=False)["reliability_weight"].median().nlargest(15, "reliability_weight")
    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    labels = [_descriptor_label(value) for value in selected["descriptor_key"]]
    axis.barh(labels, selected["reliability_weight"], color="#72B7B2")
    axis.invert_yaxis()
    axis.set_xlabel("训练内生理惯性基线可靠性权重")
    axis.set_title("稳定进入残差目标的生理描述量")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_third_pool(frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    specifications = (
        ("future_maneuver_score", "未来机动连续分数", "Spearman 秩相关"),
        ("physiology_residual", "生理惯性之外的连续残差", "RMSE（越低越好）"),
    )
    for axis, (task, title, ylabel) in zip(axes, specifications, strict=True):
        subset = frame[frame["task"] == task]
        axis.bar(
            [METHOD_LABELS[value] for value in subset["method"]],
            subset["value"],
            color="#B279A2",
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("第三训练池连续目标压力评价")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _descriptor_label(value):
    field, statistic = str(value).split("::", maxsplit=1)
    field_label = {
        "eeg.fp1": "Fp1 脑电",
        "eeg.f3": "F3 脑电",
        "eeg.fz": "Fz 脑电",
        "spo2.toi1": "TOI-1 组织氧",
        "spo2.toi2": "TOI-2 组织氧",
    }.get(field, field)
    statistic_label = {
        "centered_rms": "去均值均方根",
        "mad": "绝对中位差",
        "line_length": "线长度",
        "spectral_entropy": "谱熵",
        "median": "中位数",
        "slope": "局部斜率",
    }.get(statistic, statistic)
    return f"{field_label} · {statistic_label}"


def _configure_chinese_fonts():
    preferred = (
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in preferred:
        if family in available:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
