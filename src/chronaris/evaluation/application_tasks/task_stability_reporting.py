"""Compact evidence and Chinese figures for Dingxin task stability."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


CANDIDATE_LABELS = {
    "vehicle_5s_global": "航电 5 秒全局稳健归一化",
    "vehicle_30s_global": "航电 30 秒全局稳健归一化",
    "dual_30s_global": "双流 30 秒全局稳健归一化",
    "physiology_30s_global": "生理 30 秒全局稳健归一化",
    "vehicle_30s_no_adaptation": "航电 30 秒无输入适配",
    "vehicle_30s_window": "航电 30 秒窗口内因果稳健归一化",
    "vehicle_30s_baseline_relative": "航电 30 秒会话基线相对归一化",
    "vehicle_30s_rank": "航电 30 秒训练域分位数归一化",
    "dual_30s_window": "双流 30 秒窗口内因果稳健归一化",
    "dual_30s_baseline_relative": "双流 30 秒会话基线相对归一化",
    "dual_30s_rank": "双流 30 秒训练域分位数归一化",
    "vehicle_30s_ordinal": "航电 30 秒有序分类头",
    "vehicle_30s_score_bucket": "航电 30 秒连续分数分桶头",
    "vehicle_30s_histgb": "航电 30 秒梯度提升头",
    "dual_30s_histgb": "双流 30 秒梯度提升头",
    "dual_minirocket_1000": "双流 MiniRocket（1000 核）",
    "dual_minirocket_5000": "双流 MiniRocket（5000 核）",
    "dual_multirocket": "双流 MultiRocket",
    "dual_hydra": "双流 HYDRA",
    "dual_30s_joint_risk": "双流连续响应—风险联合头",
    "fieldwise_physiology_30s": "生理字段级响应重组",
}

STABILIZATION_LABELS = {
    "none": "无适配",
    "train_global_robust": "训练域稳健归一化",
    "window_causal_robust": "窗口内因果归一化",
    "pilot_session_baseline_relative": "会话基线相对归一化",
    "train_rank_quantile": "训练域分位数归一化",
}

BLOCKER_LABELS = {
    "protocol_invalid": "开发协议无效",
    "cross_support_generalization_instability": "跨验证支持单元的泛化不稳定",
    "maneuver_relative_gate": "机动强度分类相对门禁",
    "high_response_relative_gate": "高生理响应相对门禁",
    "response_relative_and_near_miss_gates": "连续生理响应正式与近失门禁",
}

def write_task_stability_outputs(
    *,
    compact_root: str | Path,
    protocol: dict[str, object],
    metric_contract: dict[str, object],
    split_manifest: dict[str, object],
    frames: dict[str, pd.DataFrame],
    candidate_manifest: list[dict[str, object]],
    allowance: dict[str, object],
    best_summary: dict[str, object],
) -> dict[str, str]:
    root = Path(compact_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "metric_contract.json", metric_contract)
    _write_json(root / "development_split_manifest.json", split_manifest)
    _write_json(root / "candidate_manifest.json", candidate_manifest)
    _write_json(root / "allowance.json", allowance)
    mapping = {
        "split_validity": "split_validity.csv",
        "support_dedup": "validation_support_dedup.csv",
        "metric_ceiling": "metric_ceiling_audit.csv",
        "target_stability": "target_stability.csv",
        "maneuver_threshold": "maneuver_threshold_stability.csv",
        "response_field": "response_field_stability.csv",
        "distribution_shift": "view_distribution_shift.csv",
        "input_stabilization": "input_stabilization_results.csv",
        "candidate_metrics": "candidate_split_metrics.csv",
        "relative_summary": "relative_feasibility_summary.csv",
        "leakage": "leakage_proxy_audit.csv",
        "access": "access_audit.csv",
    }
    for key, filename in mapping.items():
        frames[key].to_csv(root / filename, index=False)
    _configure_chinese_fonts()
    _plot_original_ceiling(frames["metric_ceiling"], figures / "原训练内门禁理论上限问题.png")
    _plot_split_coverage(frames["metric_ceiling"], figures / "修正后开发划分与类别覆盖.png")
    _plot_maneuver_thresholds(
        frames["maneuver_threshold"], figures / "机动阈值与连续分数跨验证单元变化.png"
    )
    _plot_response_fields(
        frames["response_field"], figures / "连续响应字段与标尺稳定性.png"
    )
    _plot_high_response(
        frames["candidate_metrics"], best_summary, figures / "高响应比例与归一化排序能力.png"
    )
    _plot_stabilization(
        frames["input_stabilization"], figures / "输入稳定化前后跨视图表现.png"
    )
    _plot_gate_overview(allowance, figures / "相对可学习性门禁总览.png")
    report_name = "acceptance_report.md" if allowance["allow_safe_fusion"] else "gap_report.md"
    report_path = root / report_name
    report_path.write_text(
        _report_text(
            protocol=protocol,
            split_manifest=split_manifest,
            allowance=allowance,
            best_summary=best_summary,
            metric_ceiling=frames["metric_ceiling"],
            leakage=frames["leakage"],
        ),
        encoding="utf-8",
    )
    evidence = {
        "status": "completed",
        "decision": allowance["decision"],
        "allow_safe_fusion": allowance["allow_safe_fusion"],
        "allow_task_aware_research": allowance["allow_task_aware_research"],
        "outer_test_opened": False,
        "chronaris_backbone_modified": False,
        "chronaris_backbone_trained": False,
        "safe_fusion_started": False,
        "teacher_distillation_started": False,
        "confirmed_metrics_changed": False,
        "main_split_count": split_manifest["main_split_count"],
        "unique_validation_support_count": split_manifest[
            "unique_main_validation_support_count"
        ],
        "candidate_count": len(candidate_manifest),
        "protocol_repair_rerun_count": protocol["protocol_repair_rerun_count"],
        "primary_blocker": allowance["primary_blocker"],
        "compact_files": sorted(
            [*mapping.values(), "protocol.json", "metric_contract.json", "development_split_manifest.json", "candidate_manifest.json", "allowance.json", report_name]
        ),
        "figure_count": 7,
        "source_sha256": protocol["source_sha256"],
    }
    _write_json(root / "evidence_manifest.json", evidence)
    _write_json(
        root / "progress.json",
        {
            "status": "completed",
            "decision": allowance["decision"],
            "allow_safe_fusion": allowance["allow_safe_fusion"],
            "allow_task_aware_research": allowance["allow_task_aware_research"],
            "outer_test_opened": False,
            "completed_candidate_split_units": int(
                frames["candidate_metrics"][["candidate_id", "split_id"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
    )
    (root / "resume_command.txt").write_text(
        "PYTHONPATH=src /home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_task_stability.py --resume\n",
        encoding="utf-8",
    )
    return {
        "report_path": str(report_path),
        "evidence_manifest_path": str(root / "evidence_manifest.json"),
    }


def _report_text(
    *, protocol, split_manifest, allowance, best_summary, metric_ceiling, leakage
):
    legacy = metric_ceiling[metric_ceiling["split_kind"] == "distribution_pressure_diagnostic"]
    missing = legacy[legacy["fixed_macro_f1_ceiling"] < 1.0]
    leakage_count = int(leakage["direct_label_source_overlap_count"].max())
    derivative_count = int(
        leakage["deterministic_derivative_overlap_count"].max()
    )
    maneuver_name = _candidate_label(best_summary["maneuver"]["candidate_id"])
    response_name = _candidate_label(best_summary["response"]["candidate_id"])
    high_name = _candidate_label(best_summary["high_response"]["candidate_id"])
    blocker = BLOCKER_LABELS.get(
        allowance["primary_blocker"], allowance["primary_blocker"] or "无"
    )
    maneuver_failures = _failed_gate_labels(
        allowance["gate_checks"]["maneuver"],
        {
            "mean_macro_f1": "平均 Macro-F1",
            "median_macro_f1": "中位 Macro-F1",
            "worst_macro_f1": "最差单元 Macro-F1",
            "minimum_class_recall": "最低类别召回率",
            "positive_baseline_lift": "相对基线增益",
            "median_score_spearman": "连续分数秩相关",
        },
    )
    response_failures = _failed_gate_labels(
        allowance["gate_checks"]["response"],
        {
            "median_rmse_ratio": "中位 RMSE 比率",
            "mean_response_skill": "平均响应技能",
            "positive_skill_fraction": "正技能单元比例",
            "median_spearman": "中位秩相关",
            "field_or_dual_stable_gain": "字段级或双流稳定增益",
        },
    )
    high_failures = _failed_gate_labels(
        allowance["gate_checks"]["high_response"],
        {
            "mean_normalized_ap": "平均归一化平均精确率",
            "median_normalized_ap": "中位归一化平均精确率",
            "positive_normalized_ap_fraction": "正增益单元比例",
            "mean_auprc_lift": "平均 AUPRC 增益",
        },
    )
    time_diagnostic = leakage[
        leakage["audit_scope"] == "time_position_only_diagnostic"
    ]
    time_only_f1 = (
        float(time_diagnostic["time_position_only_macro_f1"].max())
        if not time_diagnostic.empty
        else float("nan")
    )
    removed_time_channels = int(
        protocol["input_feature_contract"]["time_like_present_channel_count"]
    )
    pool_lines = "\n".join(
        f"- 外层训练池 {index + 1}：{row['final_selected_count']} 个主选模验证单元，"
        f"{'可用' if row['status'] == 'available' else '不可构造'}。"
        for index, row in enumerate(split_manifest["outer_pool_status"])
    )
    return f"""# 鼎新任务协议修复与跨视图稳定化报告

## 结论

上一轮第一层绝对门禁不能直接作为训练内可学习性结论：缺类验证折的固定三类 Macro-F1 理论上限低于 1，重复验证支持被重复计权，连续响应在不同训练折上使用不同 IQR 标尺，高响应原始 AUPRC 又随正样本比例变化。本轮已用唯一验证支持、相对回归技能和归一化平均精确率替代这些不可比选模口径；正式外层任务和最终门槛没有修改。

本轮决策为 `{allowance['decision']}`：`allow_safe_fusion={str(allowance['allow_safe_fusion']).lower()}`，`allow_task_aware_research={str(allowance['allow_task_aware_research']).lower()}`。按放行规则，唯一首要阻断项为 `{blocker}`；连续响应还存在独立的相对可学习性缺口。

## 协议修复

- 新协议生成 {split_manifest['main_split_count']} 个主选模验证单元，唯一支持哈希数为 {split_manifest['unique_main_validation_support_count']}。
- 所有主选模单元覆盖低、中、高三类。30 秒输入与未来 5 秒目标构成 35 秒完整支持区间；跨角色支持区间重叠为 0，并在完整支持区间之外额外保留至少 35 秒间隔。
- 第三个 outer-train pool 无法在不降低类别覆盖和隔离要求的条件下形成合法主选模单元，已标记为 `split_unavailable`，仅保留自然漂移诊断。
- 原协议中有 {len(missing)} 个诊断折固定三类 Macro-F1 理论上限低于 1；重复 validation support 不再重复计权。

{pool_lines}

## 三项相对可学习性

- 机动强度分类：最佳方案为“{maneuver_name}”，mean/median/worst Macro-F1 为 {best_summary['maneuver']['mean_macro_f1']:.4f}/{best_summary['maneuver']['median_macro_f1']:.4f}/{best_summary['maneuver']['worst_macro_f1']:.4f}，最低类别召回 {best_summary['maneuver']['minimum_class_recall']:.4f}，连续分数 median Spearman 为 {best_summary['maneuver']['median_score_spearman']:.4f}。未通过子门槛：{maneuver_failures}。
- 连续生理响应：最佳方案为“{response_name}”，median RMSE ratio 为 {best_summary['response']['median_rmse_ratio']:.4f}，mean response skill 为 {best_summary['response']['mean_response_skill']:.4f}，正技能验证单元比例为 {best_summary['response']['positive_skill_fraction']:.3f}，median Spearman 为 {best_summary['response']['median_spearman']:.4f}。正式门禁和近失门禁均未通过；正式门禁未通过项：{response_failures}。
- 高生理响应识别：最佳方案为“{high_name}”，mean/median normalized AP 为 {best_summary['high_response']['mean_normalized_ap']:.4f}/{best_summary['high_response']['median_normalized_ap']:.4f}，{best_summary['high_response']['positive_normalized_ap_fraction']:.3f} 的验证单元取得正 normalized AP，mean AUPRC lift 为 {best_summary['high_response']['mean_auprc_lift']:.4f}。正增益来自多个验证单元，但整体门禁仍未通过；未通过项：{high_failures}。

第三个 outer-train pool 在保持三类覆盖、正负样本覆盖和支持隔离时无法形成合法主选模单元，因此第三视图漂移并未被输入稳定化方案证明缓解；本轮只将它隔离为分布压力诊断，未把缺类结果混入主排名。

## 边界与退出动作

- 标签源输入直接重叠数为 {leakage_count}，确定性派生重叠数为 {derivative_count}。轻量候选统一移除了 {removed_time_channels} 个实际存在的时间类航电通道，并且不使用视图、飞行员、架次或显式查询位置作为特征。
- 仅用时间块位置的诊断模型在主验证单元上的最高 Macro-F1 为 {time_only_f1:.4f}；该结果只用于识别时间位置捷径，不参与候选排名。近完美候选还通过了样本谱系和共享航电时间块审计，未发现违反既有字段排除合同的输入。
- 本轮只使用一次允许的协议修复重跑（计数 {protocol['protocol_repair_rerun_count']}），用于修复候选目标批处理错误并落实“完整支持区间之后再保留 35 秒间隔”的实现语义；候选清单没有扩展。
- 外层测试、held-out 预测和历史逐样本外层结果均未读取；既有确认指标不变。
- 本轮没有修改或训练 Chronaris 主干，没有启动冻结专家安全融合、教师蒸馏、仿真或公开数据实验。
- 最终外层门槛仍为 Macro-F1 > {protocol['final_outer_thresholds']['maneuver_macro_f1']['value']}、RMSE < {protocol['final_outer_thresholds']['response_rmse']['value']}、AUPRC > {protocol['final_outer_thresholds']['high_response_auprc']['value']}。
"""


def _plot_original_ceiling(frame, path):
    data = frame[frame["split_kind"] == "distribution_pressure_diagnostic"].copy()
    labels = [f"历史诊断折{i + 1}" for i in range(len(data))]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, data["fixed_macro_f1_ceiling"], color="#3B82F6")
    axis.axhline(0.95, color="#F59E0B", linestyle="--", label="原训练内门槛 0.95")
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("固定三类 Macro-F1 理论上限")
    axis.set_title("原训练内门禁的理论上限问题")
    axis.legend()
    _save(figure, path)


def _plot_split_coverage(frame, path):
    data = frame[frame["included_in_main_ranking"]].copy()
    x = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(data))
    for column, label, color in (
        ("maneuver_low_count", "低机动", "#10B981"),
        ("maneuver_medium_count", "中机动", "#3B82F6"),
        ("maneuver_high_count", "高机动", "#F59E0B"),
    ):
        axis.bar(x, data[column], bottom=bottom, label=label, color=color)
        bottom += data[column].to_numpy()
    axis.set_xticks(x, [f"开发单元{i + 1}" for i in range(len(data))])
    axis.set_ylabel("验证样本数")
    axis.set_title("修正后的开发验证单元与机动类别覆盖")
    axis.legend(ncol=3)
    _save(figure, path)


def _plot_maneuver_thresholds(frame, path):
    data = frame[frame["split_kind"] == "main_selection"].copy()
    x = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(x, data["lower_bound"], marker="o", label="低—中阈值")
    axis.plot(x, data["upper_bound"], marker="o", label="中—高阈值")
    axis.fill_between(
        x,
        data["lower_bound"] - data["bootstrap_lower_iqr"] / 2,
        data["lower_bound"] + data["bootstrap_lower_iqr"] / 2,
        alpha=0.15,
    )
    axis.set_xticks(x, [f"开发单元{i + 1}" for i in range(len(data))])
    axis.set_ylabel("训练折拟合的连续机动分数阈值")
    axis.set_title("机动阈值与连续分数跨验证单元变化")
    axis.legend()
    _save(figure, path)


def _plot_response_fields(frame, path):
    data = frame[frame["split_kind"] == "main_selection"].copy()
    fields = (
        data.groupby("field_name")["validation_to_train_iqr_ratio"]
        .median()
        .sort_values()
        .tail(10)
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    readable = [f"生理字段 {index + 1}" for index in range(len(fields))]
    axis.barh(readable, fields.values, color="#8B5CF6")
    axis.axvline(1.0, color="#374151", linestyle="--")
    axis.set_xlabel("验证/训练字段变化 IQR 比")
    axis.set_title("连续响应字段与标尺稳定性")
    _save(figure, path)


def _plot_high_response(frame, best_summary, path):
    candidate = best_summary["high_response"]["candidate_id"]
    data = frame[(frame["candidate_id"] == candidate) & (frame["task"] == "high_response")]
    pivot = data.pivot(index="split_id", columns="metric", values="value")
    x = np.arange(len(pivot))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(x - 0.2, pivot["prevalence"], width=0.4, label="正样本比例", color="#9CA3AF")
    axis.bar(x + 0.2, pivot["normalized_ap"], width=0.4, label="归一化平均精确率", color="#10B981")
    axis.axhline(0.2, color="#F59E0B", linestyle="--", label="归一化门槛 0.20")
    axis.set_xticks(x, [f"开发单元{i + 1}" for i in range(len(pivot))])
    axis.set_title("高响应正样本比例与归一化排序能力")
    axis.legend()
    _save(figure, path)


def _plot_stabilization(frame, path):
    data = frame.groupby(["stabilization", "task", "metric"], as_index=False)["value"].mean()
    specifications = (
        ("maneuver", "macro_f1", "机动 Macro-F1"),
        ("response", "rmse_ratio", "响应 RMSE 比率"),
        ("high_response", "normalized_ap", "高响应归一化平均精确率"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for axis, (task, metric, title) in zip(axes, specifications, strict=True):
        subset = data[(data["task"] == task) & (data["metric"] == metric)]
        axis.bar(range(len(subset)), subset["value"], color="#3B82F6")
        labels = [
            STABILIZATION_LABELS.get(str(value), str(value))
            for value in subset["stabilization"]
        ]
        axis.set_xticks(range(len(subset)), labels, rotation=25, ha="right")
        axis.set_title(title)
    figure.suptitle("输入稳定化前后跨视图表现")
    _save(figure, path)


def _plot_gate_overview(allowance, path):
    labels = ["协议有效", "机动可学习", "连续响应", "高响应排序"]
    values = [
        allowance["protocol_valid"],
        allowance["maneuver_gate_passed"],
        allowance["response_gate_passed"] or allowance["response_near_miss_passed"],
        allowance["high_response_gate_passed"],
    ]
    colors = ["#10B981" if value else "#EF4444" for value in values]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, [1 if value else 0 for value in values], color=colors)
    axis.set_ylim(0, 1.2)
    axis.set_yticks((0, 1), ("未通过", "通过"))
    axis.set_title("鼎新相对可学习性门禁总览")
    _save(figure, path)


def _configure_chinese_fonts():
    candidates = [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "SimHei",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _candidate_label(candidate_id: str) -> str:
    return CANDIDATE_LABELS.get(str(candidate_id), str(candidate_id))


def _failed_gate_labels(checks, labels) -> str:
    failed = [labels.get(key, key) for key, passed in checks.items() if not passed]
    return "、".join(failed) if failed else "无"
