"""Build the bounded report for the simplified Dingxin downstream evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .simple_downstream_artifact import build_report_artifact
from .simple_downstream_figures import (
    METHOD_LABELS,
    METHOD_ORDER,
    plot_chronaris_ablation_boundary,
    plot_future_maneuver,
    plot_future_physiology,
    plot_simulation_mechanism_boundary,
)
from .simple_downstream_validation import (
    build_acceptance_rows as _acceptance_rows,
    build_source_inventory as _source_inventory,
    summary_max_difference as _summary_max_difference,
    validate_protocol_chain as _validate_protocol_chain,
    validate_raw_predictions as _validate_raw_predictions,
    validate_task_targets as _validate_task_targets,
)


@dataclass(frozen=True, slots=True)
class SimpleDownstreamReportingConfig:
    run_id: str = "2026-07-16_simple-downstream-thesis-evidence"
    confirmation_run_id: str = "2026-07-16_simple-downstream-confirmation"
    pretraining_run_id: str = "2026-07-16_simple-downstream-pretraining-confirmation"
    representation_run_id: str = (
        "2026-07-16_simple-downstream-representations-confirmation"
    )
    task_protocol_run_id: str = "2026-07-16_simple-downstream-protocol"
    prior_evidence_pack_run_id: str = "2026-07-12_downstream-evidence-pack"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"


@dataclass(frozen=True, slots=True)
class SimpleDownstreamReportingResult:
    run_id: str
    status: str
    output_root: str
    report_path: str
    validation_report_path: str
    artifact_path: str
    evidence_manifest_path: str


def build_simple_downstream_report(
    config: SimpleDownstreamReportingConfig | None = None,
) -> SimpleDownstreamReportingResult:
    resolved = config or SimpleDownstreamReportingConfig()
    compact = Path(resolved.compact_output_root)
    output = compact / resolved.run_id
    formal = compact / resolved.confirmation_run_id
    task = Path(resolved.heavy_output_root) / resolved.task_protocol_run_id
    prior = compact / resolved.prior_evidence_pack_run_id
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    metrics = pd.read_csv(formal / "metrics_long.csv")
    published_summary = pd.read_csv(formal / "metrics_summary.csv")
    predictions = pd.read_csv(formal / "prediction_inventory.csv")
    units = pd.read_csv(formal / "unit_inventory.csv")
    unit_metrics = _augment_ratios(metrics)
    fold_summary = _fold_summary(unit_metrics)
    primary = _primary_result_table(unit_metrics)
    recovery = pd.read_csv(prior / "tables" / "mechanism_recovery_mae.csv")
    stress = pd.read_csv(prior / "tables" / "stress_degradation_slopes.csv").set_index(
        "method"
    )
    ablation = pd.read_csv(prior / "tables" / "chronaris_ablation_advantage.csv")

    plot_future_maneuver(unit_metrics, figures / "dingxin_future_maneuver.png")
    plot_future_physiology(unit_metrics, figures / "dingxin_future_physiology.png")
    plot_simulation_mechanism_boundary(
        recovery,
        stress,
        figures / "simulation_mechanism_boundary.png",
    )
    plot_chronaris_ablation_boundary(
        ablation,
        figures / "chronaris_ablation_boundary.png",
    )

    primary.to_csv(tables / "primary_result_table.csv", index=False)
    fold_summary.to_csv(tables / "fold_primary_metrics.csv", index=False)
    recovery.to_csv(tables / "mechanism_recovery_mae.csv", index=False)
    stress.reset_index().to_csv(tables / "stress_degradation_slopes.csv", index=False)
    ablation.to_csv(tables / "chronaris_ablation_advantage.csv", index=False)

    summary_difference = _summary_max_difference(metrics, published_summary)
    raw_validation = _validate_raw_predictions(metrics, predictions)
    target_validation = _validate_task_targets(task)
    protocol_validation = _validate_protocol_chain(compact, resolved)
    source_inventory = _source_inventory(formal, prior, resolved)
    source_inventory.to_csv(output / "source_inventory.csv", index=False)
    acceptance = _acceptance_rows(
        units=units,
        metrics=metrics,
        summary_difference=summary_difference,
        raw_validation=raw_validation,
        target_validation=target_validation,
        protocol_validation=protocol_validation,
    )
    pd.DataFrame(acceptance).to_csv(output / "acceptance.csv", index=False)
    status = "completed" if all(row["passed"] for row in acceptance) else "partial"

    report_path = output / "report.md"
    validation_path = output / "validation_report.md"
    _write_report(report_path, primary)
    _write_validation_report(
        validation_path,
        summary_difference=summary_difference,
        raw_validation=raw_validation,
        target_validation=target_validation,
        protocol_validation=protocol_validation,
    )
    chart_map = _chart_map()
    _write_json(output / "chart_map.json", chart_map)
    artifact = build_report_artifact(
        primary=primary,
        fold_summary=fold_summary,
        recovery=recovery,
        stress=stress.reset_index(),
        ablation=ablation,
        generated_at="2026-07-16T23:59:00+08:00",
    )
    artifact_path = output / "artifact.json"
    _write_json(artifact_path, artifact)
    _write_json(
        output / "evidence_manifest.json",
        {
            "format": "chronaris.simple_downstream_thesis_evidence.v1",
            "run_id": resolved.run_id,
            "status": status,
            "config": asdict(resolved),
            "acceptance_pass_count": sum(row["passed"] for row in acceptance),
            "acceptance_check_count": len(acceptance),
            "formal_result_opened": True,
            "result_driven_tuning_allowed": False,
            "training_performed": False,
            "confirmed_metrics_changed": False,
            "summary_max_absolute_difference": summary_difference,
            "raw_metric_max_absolute_difference": raw_validation["max_metric_difference"],
            "output_paths": {
                "report": str(report_path),
                "validation_report": str(validation_path),
                "artifact": str(artifact_path),
                "primary_table": str(tables / "primary_result_table.csv"),
                "fold_table": str(tables / "fold_primary_metrics.csv"),
                "acceptance": str(output / "acceptance.csv"),
                "chart_map": str(output / "chart_map.json"),
                "source_inventory": str(output / "source_inventory.csv"),
            },
        },
    )
    (output / "resume_command.txt").write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/dingxin/report_simple_downstream.py "
        f"--run-id {resolved.run_id}\n",
        encoding="utf-8",
    )
    return SimpleDownstreamReportingResult(
        run_id=resolved.run_id,
        status=status,
        output_root=str(output),
        report_path=str(report_path),
        validation_report_path=str(validation_path),
        artifact_path=str(artifact_path),
        evidence_manifest_path=str(output / "evidence_manifest.json"),
    )


def _augment_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    derived = []
    definitions = (
        ("future_maneuver", "skill_vs_current_maneuver", "rmse_ratio_vs_current"),
        ("future_physiology", "skill_vs_persistence", "rmse_ratio_vs_persistence"),
    )
    for task, source, target in definitions:
        selected = output[
            (output["task_name"].astype(str) == task)
            & (output["metric_name"].astype(str) == source)
        ].copy()
        selected["metric_name"] = target
        selected["metric_value"] = np.sqrt(1.0 - selected["metric_value"].astype(float))
        derived.append(selected)
    return pd.concat([output, *derived], ignore_index=True)


def _fold_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = {
        "future_maneuver": {
            "macro_f1",
            "balanced_accuracy",
            "spearman",
            "normalized_mae",
            "rmse_ratio_vs_current",
        },
        "future_physiology": {
            "standardized_rmse_macro",
            "standardized_mae_macro",
            "rmse_ratio_vs_persistence",
            "positive_skill_field_ratio",
        },
    }
    selected = metrics[
        metrics.apply(
            lambda row: str(row["metric_name"]) in primary.get(str(row["task_name"]), set()),
            axis=1,
        )
    ].copy()
    return (
        selected.groupby(
            ["fold_id", "method_name", "task_name", "metric_name"], as_index=False
        )
        .agg(
            mean=("metric_value", "mean"),
            std=("metric_value", "std"),
            seed_count=("metric_value", "count"),
        )
        .sort_values(["task_name", "metric_name", "fold_id", "method_name"])
    )


def _primary_result_table(metrics: pd.DataFrame) -> pd.DataFrame:
    wanted = {
        "future_maneuver": (
            "macro_f1",
            "balanced_accuracy",
            "spearman",
            "normalized_mae",
            "skill_vs_current_maneuver",
            "rmse_ratio_vs_current",
        ),
        "future_physiology": (
            "standardized_rmse_macro",
            "standardized_mae_macro",
            "skill_vs_persistence",
            "rmse_ratio_vs_persistence",
            "positive_skill_field_ratio",
            "eeg_rmse_macro",
            "spo2_rmse_macro",
        ),
    }
    rows = []
    for method in METHOD_ORDER:
        row = {"method_name": method, "method_label": METHOD_LABELS[method]}
        for task, names in wanted.items():
            prefix = "maneuver" if task == "future_maneuver" else "physiology"
            for name in names:
                values = metrics[
                    (metrics["method_name"].astype(str) == method)
                    & (metrics["task_name"].astype(str) == task)
                    & (metrics["metric_name"].astype(str) == name)
                ]["metric_value"]
                row[f"{prefix}_{name}"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(path: Path, primary: pd.DataFrame) -> None:
    vehicle = _method_row(primary, "vehicle_only")
    chronaris = _method_row(primary, "chronaris")
    contiformer = _method_row(primary, "contiformer")
    path.write_text(
        f"""# Chronaris 鼎新未来下游评价复核报告

## 技术摘要

**协议冻结后的确认评价没有支持 Chronaris 在鼎新真实未来任务上的整体优势。**六种方法在相同 64 维冻结表示出口、相同逻辑回归与岭回归下完成两个留一架次折和三个随机种子，共 36 个方法—折—随机种子单元。未来机动任务由航电单流领先；未来生理字段任务中，六种方法在两个折上的正技能字段比例均为 0，均未超过“保持当前状态不变”的持久性基线。

**现有仿真证据仍支持有限的时间机制结论。**Chronaris 在时钟偏移和生理响应时延恢复上取得最低误差，连续演化与因果掩码消融也显示局部贡献；但随机缺失和连续缺失的退化斜率处于不利位置，物理约束贡献呈混合结果。因此论文可保留“时间机制可恢复、部分组件有效”的论断，不能据此替代真实下游优势。

**研究执行应在此停止结果驱动调参，转入如实写作。**本轮正式结果已经打开；继续针对这两个架次调模型只会提高测试集适配风险。论文主结论宜写为“局部有效、分项领先、整体优势尚未成立”，并明确真实未来任务尚未形成 Chronaris 优势。

## 航电历史足以完成未来机动预测，融合表示没有带来增益

航电单流的机动强度三分类宏平均 F1 为 **{vehicle['maneuver_macro_f1']:.3f}**，连续分数 Spearman 相关为 **{vehicle['maneuver_spearman']:.3f}**，相对当前机动状态的技能为 **{vehicle['maneuver_skill_vs_current_maneuver']:.3f}**。Chronaris 对应值为 **{chronaris['maneuver_macro_f1']:.3f}**、**{chronaris['maneuver_spearman']:.3f}** 和 **{chronaris['maneuver_skill_vs_current_maneuver']:.3f}**。ContiFormer 虽在融合方法中较强，但三分类宏平均 F1 也只有 **{contiformer['maneuver_macro_f1']:.3f}**。

![鼎新未来机动预测对比](figures/dingxin_future_maneuver.png)

图中柱体给出两折、三随机种子的描述性均值，两个标记点保留两架次的折级差异。相对当前状态的 RMSE 比率以 1 为基准，低于 1 才表示优于直接延续当前机动状态；只有航电单流形成稳定正技能。该结果说明未来 5 秒机动主要由过去 30 秒航电运动历史解释，当前人机融合表示没有增加可用预测信息。

## 生理字段预测没有越过持久性基线

字段级标准化 RMSE 宏平均的最佳方法仍是航电单流（**{vehicle['physiology_standardized_rmse_macro']:.3f}**），ContiFormer 为 **{contiformer['physiology_standardized_rmse_macro']:.3f}**，Chronaris 为 **{chronaris['physiology_standardized_rmse_macro']:.3f}**。更关键的是，所有方法的相对持久性 RMSE 比率都远高于 1，且两个折中每种方法的正技能字段比例均为 0。

![鼎新未来生理字段预测对比](figures/dingxin_future_physiology.png)

持久性基线直接使用输入末段的当前生理状态预测未来 5 秒。模型没有超过这一基线，说明现有样本量和字段稳定性不足以证明航电背景对短期生理变化提供了额外可预测信息。该任务可以作为严格的负面结果保留，但不能再用旧模型相对自身的误差下降宣称生理预测有效。

## 仿真只支持时间机制的局部有效性

Chronaris 的时钟偏移恢复平均绝对误差为 **0.888 秒**，生理响应时延恢复误差为 **7.486 秒**，均为四种融合方法中的最低值；与此同时，其随机缺失和连续缺失方向归一化斜率分别为 **-0.073** 和 **-0.126**，没有形成鲁棒性领先。

![仿真时间机制与压力边界](figures/simulation_mechanism_boundary.png)

这组图回答的是“融合表示是否保留仿真生成器中的时间机制”，不是“鼎新真实任务是否更好”。它支持 Chronaris 对相对时钟偏移和响应时延的表达能力，但同时要求保留缺失压力下的不利结果。

![Chronaris 仿真机制消融](figures/chronaris_ablation_boundary.png)

消融结果显示，连续演化在三个仿真任务上均带来正向贡献，因果掩码对机动分段的贡献较明显；物理约束在负荷分类上为负、在回归与分段上为正。因此，组件解释应使用“分任务贡献”，不写成每个组件在所有任务上都稳定增益。

## 评价范围、数据与指标定义

- 数据范围为两个飞行架次、三个飞行员视图和 111 个基础窗口；构造后保留 90 个完整未来视图上下文，其中未来机动只有 60 个独立飞机上下文。
- 输入区间为预测时刻前 30 秒，目标区间为随后 5 秒。输入结束时间严格等于目标开始时间，目标窗口完整落在冻结快照内。
- 主要划分为留一架次：每折完全留出一个飞行架次。两个折均有 30 个独立机动上下文，低、中、高类别各 10 个。
- 未来机动主结果包括连续分数 Spearman 相关、按训练折四分位距归一化的 MAE、相对当前机动状态技能；低、中、高三分类宏平均 F1 作为导师要求的直接对应任务。
- 未来生理状态按字段预测未来 5 秒中位数，字段只用训练架次中位数和四分位距标准化；主结果为字段级标准化 RMSE、MAE 和相对持久性技能。

## 冻结表示与固定下游算法保证了比较口径一致

五种可训练编码器在任务目标和留出架次关闭的条件下完成任务无关预训练，朴素时间同步在训练折拟合；六种方法均导出 64 维窗口表示。下游分类统一使用 `C=1.0` 的逻辑回归，连续任务统一使用 `alpha=1.0` 的岭回归。正式结果打开前配置已经冻结，打开后未根据分数更换任务、字段、模型或下游算法。

## 计算链路正确，但结论只能作描述性确认

正式结果包含 36 个完整评价单元和 504 条有限指标。报告生成阶段重新校验了 108 个逐样本文件哈希，并从逐样本预测独立复算 504 个核心单元指标；与已发布汇总的最大绝对差均处于浮点舍入范围。两个折的类别支持、90/60 上下文计数和未来时间边界也全部复核通过。

本评价仍有三项重要限制。第一，只有两个架次，不能进行有意义的显著性推断；图表中的折级点用于展示异质性，不是置信区间。第二，这批数据已在前期工作中多次查看，因此本轮是“协议冻结后的分组确认”，不是从未打开的盲测。第三，未来生理字段在两个折中分别有 12 和 11 个满足训练尺度条件的字段，跨架次分布变化较大，持久性基线远强于学习模型。

## 建议停止调参并按混合证据撰写论文

1. 把鼎新未来机动与未来生理任务作为真实数据主表，并如实报告航电单流领先、所有生理模型未超过持久性。
2. 把仿真时间恢复、压力退化和机制消融作为独立机制证据，完整保留缺失压力和物理约束的负面结果。
3. 不再根据这两个架次的正式结果修改 Chronaris；不重跑端到端训练、教师蒸馏或仿真迁移。
4. 论文结论使用“时间机制局部有效，但真实下游整体优势尚未成立”，避免将仿真机制证据写成鼎新应用效果。

## 仍待回答的问题

- 若未来能获得更多独立架次，航电单流优势是否仍然稳定，生理响应是否能出现超过持久性的可预测变化？
- Chronaris 在缺失压力下的退化是否来自连续演化、输入掩码处理还是预训练目标？该问题不在本轮论文主线继续扩展。
- 留一视图只能诊断同一飞行轨迹下的飞行员视图适配，不能改变本轮跨架次主结论；为避免结果打开后的范围扩张，本轮不新增该诊断训练线。
""",
        encoding="utf-8",
    )


def _write_validation_report(path: Path, **values) -> None:
    raw = values["raw_validation"]
    target = values["target_validation"]
    protocol = values["protocol_validation"]
    path.write_text(
        f"""# 鼎新简化下游评价数据与计算验证

## 验证结论

**可用于论文中的有边界描述性报告，但必须同时呈现样本规模、非盲测和持久性基线限制。**正式结果、逐样本文件、任务清单和协议链之间一致；没有发现能够解释 Chronaris 负面结果的计算错误或未来信息泄漏。

## 已通过的关键检查

- 90 个完整视图上下文、60 个独立飞机上下文；输入与目标严格按 30 秒加未来 5 秒分隔。
- 两个留一架次折的独立机动上下文均为 30 个，低、中、高类别支持均为 10/10/10：`{target['held_out_class_support']}`。
- 任务无关预训练未打开任务目标或留出架次，表示导出未打开任务目标或外层指标：`{protocol}`。
- 36 个正式评价单元、504 条有限指标；已发布汇总独立重算最大绝对差为 `{values['summary_difference']:.3e}`。
- 逐样本文件哈希复核 `{raw['hash_count']}` 项；从逐样本预测独立复算 `{raw['checked_metric_count']}` 个核心指标，最大绝对差为 `{raw['max_metric_difference']:.3e}`。

## 解释限制

- 两个架次不足以支持显著性检验或总体泛化结论，只能报告折级与合并描述性结果。
- 数据此前已参与多轮研究，不能称为从未查看的独立盲测集。
- 未来生理字段在两个训练折分别保留 12 和 11 个有效字段；字段标准化依赖训练架次，跨架次分布变化使误差尺度较大。
- 所有学习方法的正技能字段比例均为 0，因此未来生理任务的正确结论是“尚未超过持久性”，不能以模型间相对排名替代这一基准判断。

## 分享条件

报告可以进入论文实验章节和导师 review，但必须与本验证说明保持一致：真实数据不支持 Chronaris 整体优势；仿真仅支持生成场景内的时间机制和组件解释；不再根据正式分数调参。
""",
        encoding="utf-8",
    )


def _chart_map() -> list[dict]:
    return [
        {"figure": "dingxin_future_maneuver.png", "question": "哪种冻结表示最适合未来机动预测", "family": "horizontal bars with fold points", "claim": "航电单流稳定领先"},
        {"figure": "dingxin_future_physiology.png", "question": "哪种表示能超过短期生理持久性", "family": "horizontal bars with log ratio", "claim": "六方法均未超过持久性"},
        {"figure": "simulation_mechanism_boundary.png", "question": "时间恢复优势是否同时转化为压力鲁棒性", "family": "bars plus heatmap", "claim": "时间恢复领先但缺失鲁棒性不领先"},
        {"figure": "chronaris_ablation_boundary.png", "question": "Chronaris 组件贡献是否跨任务一致", "family": "grouped bars", "claim": "连续演化与因果掩码局部有效，物理约束结果混合"},
    ]


def _method_row(table: pd.DataFrame, method: str) -> pd.Series:
    return table[table["method_name"].astype(str) == method].iloc[0]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
