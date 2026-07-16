"""Canonical Data Analytics artifact for the simplified downstream report."""

from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from .simple_downstream_figures import (
    ABLATION_LABELS,
    ABLATION_TASK_LABELS,
    METHOD_LABELS,
    STRESS_LABELS,
)


TITLE = "Chronaris 鼎新未来下游评价复核报告"
PRIMARY_SQL = """SELECT method_name, method_label,
       maneuver_macro_f1, maneuver_balanced_accuracy, maneuver_spearman,
       maneuver_normalized_mae, maneuver_skill_vs_current_maneuver,
       maneuver_rmse_ratio_vs_current,
       physiology_standardized_rmse_macro, physiology_standardized_mae_macro,
       physiology_skill_vs_persistence, physiology_rmse_ratio_vs_persistence,
       physiology_positive_skill_field_ratio
FROM primary_results_source
ORDER BY method_order"""
RECOVERY_SQL = """SELECT method, method_label, clock_offset_mae_s, response_lag_mae_s
FROM mechanism_recovery_source
ORDER BY method_order"""
STRESS_SQL = """SELECT method_name, method_label, stress_factor, stress_label,
       degradation_slope
FROM stress_boundary_source
ORDER BY method_order, stress_order"""
ABLATION_SQL = """SELECT task_name, task_label, ablation_method, ablation_label,
       full_model_advantage
FROM ablation_boundary_source
ORDER BY task_order, ablation_order"""


def build_report_artifact(
    *,
    primary: pd.DataFrame,
    fold_summary: pd.DataFrame,
    recovery: pd.DataFrame,
    stress: pd.DataFrame,
    ablation: pd.DataFrame,
    generated_at: str,
) -> dict[str, Any]:
    primary_source = _primary_frame(primary)
    recovery_source = _recovery_frame(recovery)
    stress_source = pd.DataFrame(_stress_rows(stress))
    ablation_source = pd.DataFrame(_ablation_rows(ablation))
    primary_rows = _query_records(primary_source, "primary_results_source", PRIMARY_SQL)
    recovery_rows = _query_records(
        recovery_source, "mechanism_recovery_source", RECOVERY_SQL
    )
    stress_rows = _query_records(stress_source, "stress_boundary_source", STRESS_SQL)
    ablation_rows = _query_records(
        ablation_source, "ablation_boundary_source", ABLATION_SQL
    )
    sources = _sources(generated_at)
    manifest = {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "冻结协议下的鼎新未来机动、生理字段与既有仿真机制证据复核。",
        "generatedAt": generated_at,
        "sources": sources,
        "blocks": _blocks(),
        "charts": _charts(),
        "tables": _tables(),
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "primary_results": primary_rows,
            "fold_primary_metrics": _records(fold_summary),
            "mechanism_recovery": recovery_rows,
            "stress_boundary": stress_rows,
            "ablation_boundary": ablation_rows,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def _blocks() -> list[dict[str, Any]]:
    return [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "formal_metrics",
            "body": (
                "## 技术摘要\n\n"
                "**协议冻结后的确认评价没有支持 Chronaris 在鼎新真实未来任务上的整体优势。** "
                "六种方法使用相同 64 维冻结表示出口、相同逻辑回归与岭回归，完成两个留一架次折和三个随机种子。"
                "航电单流在未来机动上明显领先；未来生理字段中，六种方法的正技能字段比例均为 0，均未超过持久性基线。"
            ),
        },
        {
            "id": "technical_summary_mechanism",
            "type": "markdown",
            "sourceId": "simulation_evidence",
            "body": (
                "**仿真仍支持有限的时间机制结论，但不能替代真实任务效果。** Chronaris 的时钟偏移与生理响应时延恢复误差最低；"
                "同时，随机缺失和连续缺失退化不利，物理约束消融结果混合。研究应停止结果驱动调参，转入有边界写作。"
            ),
        },
        {
            "id": "maneuver_finding",
            "type": "markdown",
            "source": _formal_chart_source(PRIMARY_SQL),
            "body": (
                "## 航电历史足以完成未来机动预测\n\n"
                "航电单流的三分类宏平均 F1 为 **0.808**，连续分数 Spearman 相关为 **0.617**，相对当前机动状态技能为 **0.668**。"
                "Chronaris 对应值为 **0.195**、**0.069** 和 **-123.586**。下图按方法比较同一任务，说明融合表示没有带来额外可用信息。"
            ),
        },
        {"id": "maneuver_chart_block", "type": "chart", "chartId": "maneuver_macro_f1"},
        {
            "id": "physiology_finding",
            "type": "markdown",
            "source": _formal_chart_source(PRIMARY_SQL),
            "body": (
                "## 生理字段预测没有越过持久性基线\n\n"
                "字段级标准化 RMSE 宏平均的最佳方法是航电单流（**7.084**），ContiFormer 为 **7.227**，Chronaris 为 **8.890**。"
                "两个折中所有方法的正技能字段比例都为 0，因此模型间排名不能替代“是否超过当前状态延续”的基准判断。"
            ),
        },
        {"id": "physiology_chart_block", "type": "chart", "chartId": "physiology_rmse"},
        {
            "id": "mechanism_finding",
            "type": "markdown",
            "source": _simulation_chart_source(
                "mechanism_recovery_mae.csv", RECOVERY_SQL
            ),
            "body": (
                "## 仿真只支持时间机制的局部有效性\n\n"
                "Chronaris 的时钟偏移恢复误差为 **0.888 秒**，生理响应时延恢复误差为 **7.486 秒**，均为四种融合方法中的最低值。"
                "但其随机缺失和连续缺失方向归一化斜率分别为 **-0.073** 和 **-0.126**，没有形成整体鲁棒性领先。"
            ),
        },
        {"id": "recovery_chart_block", "type": "chart", "chartId": "clock_offset_recovery"},
        {
            "id": "mechanism_tables_note",
            "type": "markdown",
            "body": (
                "完整压力与消融表保留了不利结果。正斜率表示压力增强时退化较慢；消融优势为正表示完整 Chronaris 优于对应消融。"
            ),
        },
        {"id": "stress_table_block", "type": "table", "tableId": "stress_boundary_table"},
        {"id": "ablation_table_block", "type": "table", "tableId": "ablation_boundary_table"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "formal_protocol",
            "body": (
                "## 评价范围与指标定义\n\n"
                "数据包含两个飞行架次、三个飞行员视图和 111 个基础窗口；构造后保留 90 个完整未来视图上下文，其中未来机动有 60 个独立飞机上下文。"
                "输入为预测时刻前 30 秒，目标为随后 5 秒。未来机动报告连续相关、归一化误差、相对当前状态技能与辅助三分类；未来生理按训练折标准化后逐字段报告，并以持久性预测为基线。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "formal_protocol",
            "body": (
                "## 冻结表示与固定下游算法保证比较一致\n\n"
                "五种可训练编码器在任务目标和留出架次关闭时完成任务无关预训练，朴素时间同步在训练折拟合；六种方法均输出 64 维窗口表示。"
                "分类统一使用 `C=1.0` 的逻辑回归，连续任务统一使用 `alpha=1.0` 的岭回归。正式结果打开后没有更换任务、字段、模型或下游算法。"
            ),
        },
        {
            "id": "validation_limitations",
            "type": "markdown",
            "sourceId": "validation_report",
            "body": (
                "## 计算链路正确，但只能作描述性确认\n\n"
                "36 个评价单元和 504 条指标全部有限；报告阶段复核 108 个逐样本文件哈希，并从逐样本预测独立复算核心指标，差异均为浮点舍入量级。"
                "但只有两个架次，不能进行有效显著性推断；数据此前已多次查看，本轮不是从未打开的盲测；未来生理在两折分别保留 12 和 11 个有效字段。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## 建议停止调参并按混合证据写作\n\n"
                "1. 把鼎新未来机动和未来生理任务作为真实数据主表，如实报告航电单流领先及生理任务未超过持久性。\n"
                "2. 把仿真时间恢复、压力退化和机制消融作为独立证据，保留缺失压力与物理约束的负面结果。\n"
                "3. 不再根据两个架次的正式结果修改 Chronaris，不重跑端到端训练、教师蒸馏或仿真迁移。\n"
                "4. 论文结论使用‘时间机制局部有效，但真实下游整体优势尚未成立’。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 仍待回答的问题\n\n"
                "- 更多独立架次下，航电单流优势是否稳定，生理响应是否能出现超过持久性的可预测变化？\n"
                "- Chronaris 的缺失退化来自连续演化、掩码处理还是预训练目标？该问题不在本轮论文主线继续扩展。\n"
                "- 留一视图只能诊断同一轨迹下的视图适配，不能改变本轮跨架次主结论；本轮不新增该训练线。"
            ),
        },
    ]


def _charts() -> list[dict[str, Any]]:
    return [
        {
            "id": "maneuver_macro_f1",
            "title": "未来机动三分类宏平均 F1",
            "description": "两个留一架次折、三个随机种子的描述性均值，越高越好。",
            "type": "bar",
            "intent": "comparison",
            "question": "哪种冻结表示最适合未来机动三分类？",
            "rationale": "方法类别与单一同尺度指标适合横向条形比较。",
            "dataset": "primary_results",
            "encodings": {
                "x": {"field": "method_label"},
                "y": {"field": "maneuver_macro_f1"},
            },
            "options": {"orientation": "horizontal"},
            "source": _formal_chart_source(PRIMARY_SQL),
        },
        {
            "id": "physiology_rmse",
            "title": "未来生理字段标准化 RMSE",
            "description": "字段级 RMSE 宏平均，越低越好；所有方法均未超过持久性基线。",
            "type": "bar",
            "intent": "comparison",
            "question": "哪种冻结表示的未来生理字段误差最低？",
            "rationale": "方法类别与单一误差指标适合横向条形比较。",
            "dataset": "primary_results",
            "encodings": {
                "x": {"field": "method_label"},
                "y": {"field": "physiology_standardized_rmse_macro"},
            },
            "options": {"orientation": "horizontal"},
            "source": _formal_chart_source(PRIMARY_SQL),
        },
        {
            "id": "clock_offset_recovery",
            "title": "仿真时钟偏移恢复误差",
            "description": "平均绝对误差，单位为秒，越低越好。",
            "type": "bar",
            "intent": "comparison",
            "question": "哪种融合表示更好地保留相对时钟偏移？",
            "rationale": "四种方法的单一同尺度误差适合横向条形比较。",
            "dataset": "mechanism_recovery",
            "encodings": {
                "x": {"field": "method_label"},
                "y": {"field": "clock_offset_mae_s"},
            },
            "options": {"orientation": "horizontal"},
            "source": _simulation_chart_source(
                "mechanism_recovery_mae.csv", RECOVERY_SQL
            ),
        },
    ]


def _tables() -> list[dict[str, Any]]:
    return [
        {
            "id": "stress_boundary_table",
            "title": "观测压力退化斜率",
            "description": "方向归一化后越大表示退化越慢；包含 Chronaris 的有利与不利场景。",
            "dataset": "stress_boundary",
            "columns": [
                {"field": "method_label", "label": "方法", "type": "text"},
                {"field": "stress_label", "label": "压力因素", "type": "text"},
                {"field": "degradation_slope", "label": "退化斜率", "type": "number", "format": "number"},
            ],
            "defaultSort": {"field": "degradation_slope", "direction": "desc"},
            "source": _simulation_chart_source(
                "stress_degradation_slopes.csv", STRESS_SQL
            ),
        },
        {
            "id": "ablation_boundary_table",
            "title": "Chronaris 仿真组件消融",
            "description": "正值表示完整模型优于对应消融；不同任务不比较绝对大小。",
            "dataset": "ablation_boundary",
            "columns": [
                {"field": "task_label", "label": "仿真任务", "type": "text"},
                {"field": "ablation_label", "label": "消融项", "type": "text"},
                {"field": "full_model_advantage", "label": "完整模型优势", "type": "number", "format": "number"},
            ],
            "defaultSort": {"field": "full_model_advantage", "direction": "desc"},
            "source": _simulation_chart_source(
                "chronaris_ablation_advantage.csv", ABLATION_SQL
            ),
        },
    ]


def _sources(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "formal_metrics",
            "label": "鼎新未来任务正式确认指标",
            "path": "docs/artifacts/runs/2026-07-16_simple-downstream-confirmation/metrics_long.csv",
            "query": {
                "description": "读取两折三随机种子的冻结表示下游指标并按方法汇总。",
                "sql": PRIMARY_SQL,
                "engine": "sqlite",
                "language": "sql",
                "executed_at": generated_at,
                "tables_used": ["docs/artifacts/runs/2026-07-16_simple-downstream-confirmation/metrics_long.csv"],
                "filters": ["留一架次两个折", "随机种子 17、29、43", "六种冻结表示方法"],
                "metric_definitions": ["机动宏平均 F1 为三类等权 F1", "生理 RMSE 为训练折尺度标准化后的字段宏平均"],
            },
        },
        {
            "id": "formal_protocol",
            "label": "鼎新简化下游冻结协议",
            "path": "docs/requirements/simple-downstream-evaluation-v1.md",
            "query": {
                "description": "任务、时间边界、样本单位、划分和消费者配置冻结合同。",
                "language": "markdown",
                "executed_at": generated_at,
                "tables_used": ["docs/requirements/simple-downstream-evaluation-v1.md"],
            },
        },
        {
            "id": "simulation_evidence",
            "label": "既有仿真机制、压力与消融证据表",
            "path": "docs/artifacts/runs/2026-07-12_downstream-evidence-pack/evidence_manifest.json",
            "query": {
                "description": "复用既有锁定仿真的时间恢复、压力退化和组件消融汇总，不重新训练。",
                "language": "python",
                "executed_at": generated_at,
                "tables_used": [
                    "docs/artifacts/runs/2026-07-12_downstream-evidence-pack/tables/mechanism_recovery_mae.csv",
                    "docs/artifacts/runs/2026-07-12_downstream-evidence-pack/tables/stress_degradation_slopes.csv",
                    "docs/artifacts/runs/2026-07-12_downstream-evidence-pack/tables/chronaris_ablation_advantage.csv",
                ],
            },
        },
        {
            "id": "validation_report",
            "label": "正式结果独立复算与数据验证",
            "path": "docs/artifacts/runs/2026-07-16_simple-downstream-thesis-evidence/validation_report.md",
            "query": {
                "description": "逐样本哈希、指标复算、任务时间边界与折级支持检查。",
                "language": "python",
                "executed_at": generated_at,
                "tables_used": [
                    "docs/artifacts/runs/2026-07-16_simple-downstream-confirmation/prediction_inventory.csv",
                    "artifacts/application_evaluation/2026-07-16_simple-downstream-protocol/context_manifest.csv",
                ],
            },
        },
    ]


def _primary_frame(primary: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "method_name",
        "method_label",
        "maneuver_macro_f1",
        "maneuver_balanced_accuracy",
        "maneuver_spearman",
        "maneuver_normalized_mae",
        "maneuver_skill_vs_current_maneuver",
        "maneuver_rmse_ratio_vs_current",
        "physiology_standardized_rmse_macro",
        "physiology_standardized_mae_macro",
        "physiology_skill_vs_persistence",
        "physiology_rmse_ratio_vs_persistence",
        "physiology_positive_skill_field_ratio",
    )
    frame = primary[list(columns)].copy()
    frame["method_order"] = np.arange(len(frame))
    return frame


def _recovery_frame(recovery: pd.DataFrame) -> pd.DataFrame:
    pivot = recovery.pivot(index="method", columns="target", values="value").reset_index()
    pivot["method_label"] = pivot["method"].map(METHOD_LABELS)
    pivot["method_order"] = pivot["method"].map(
        {method: index for index, method in enumerate(METHOD_LABELS)}
    )
    pivot = pivot.rename(
        columns={
            "relative_clock_offset_magnitude_s": "clock_offset_mae_s",
            "primary_physiology_response_lag_s": "response_lag_mae_s",
        }
    )
    return pivot


def _stress_rows(stress: pd.DataFrame) -> list[dict[str, Any]]:
    factors = [value for value in STRESS_LABELS if value in stress.columns]
    rows = []
    for item in stress.itertuples(index=False):
        method = str(item.method)
        for stress_order, factor in enumerate(factors):
            rows.append(
                {
                    "method_name": method,
                    "method_label": METHOD_LABELS[method],
                    "stress_factor": factor,
                    "stress_label": STRESS_LABELS[factor],
                    "degradation_slope": float(getattr(item, factor)),
                    "method_order": list(METHOD_LABELS).index(method),
                    "stress_order": stress_order,
                }
            )
    return rows


def _ablation_rows(ablation: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    task_order = {value: index for index, value in enumerate(ABLATION_TASK_LABELS)}
    ablation_order = {value: index for index, value in enumerate(ABLATION_LABELS)}
    for item in ablation.itertuples(index=False):
        rows.append(
            {
                "task_name": str(item.task),
                "task_label": ABLATION_TASK_LABELS[str(item.task)],
                "ablation_method": str(item.ablation_method),
                "ablation_label": ABLATION_LABELS[str(item.ablation_method)],
                "full_model_advantage": float(item.mean),
                "task_order": task_order[str(item.task)],
                "ablation_order": ablation_order[str(item.ablation_method)],
            }
        )
    return rows


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.replace({np.nan: None})
    rows = []
    for row in normalized.to_dict("records"):
        rows.append({key: _scalar(value) for key, value in row.items()})
    return rows


def _query_records(frame: pd.DataFrame, table: str, sql: str) -> list[dict[str, Any]]:
    with sqlite3.connect(":memory:") as connection:
        frame.to_sql(table, connection, index=False)
        result = pd.read_sql_query(sql, connection)
    return _records(result)


def _formal_chart_source(sql: str) -> dict[str, Any]:
    return {
        "id": "formal_metrics",
        "label": "鼎新未来任务正式确认指标",
        "path": "docs/artifacts/runs/2026-07-16_simple-downstream-confirmation/metrics_long.csv",
        "query": {
            "description": "从报告汇总表中选择图表字段。",
            "sql": sql,
            "engine": "sqlite",
            "language": "sql",
            "tables_used": ["primary_results_source"],
        },
    }


def _simulation_chart_source(filename: str, sql: str) -> dict[str, Any]:
    return {
        "id": f"simulation_{filename}",
        "label": "既有锁定仿真证据表",
        "path": f"docs/artifacts/runs/2026-07-12_downstream-evidence-pack/tables/{filename}",
        "query": {
            "description": "从既有仿真汇总表中选择图表或表格字段。",
            "sql": sql,
            "engine": "sqlite",
            "language": "sql",
            "tables_used": [sql.split("FROM ", 1)[1].splitlines()[0]],
        },
    }


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
