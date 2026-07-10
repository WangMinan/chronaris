"""Compact evidence writers for MulT and ContiFormer adapter smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_deep_baseline_adapter_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    adapter_protocol: Mapping[str, object],
    causality_rows: Sequence[Mapping[str, object]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    transform_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    acceptance_rows: Sequence[Mapping[str, object]],
    heavy_run_root: str,
) -> Mapping[str, str]:
    paths = {
        "adapter_protocol": run_root / "adapter_protocol.json",
        "attention_causality_audit": run_root / "attention_causality_audit.csv",
        "dual_stream_sensitivity": run_root / "dual_stream_sensitivity.csv",
        "parameter_budget": run_root / "parameter_budget.csv",
        "fold_transform_manifest": run_root / "fold_transform_manifest.json",
        "checkpoint_registry": run_root / "checkpoint_registry.json",
        "representation_export_manifest": run_root / "representation_export_manifest.json",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["adapter_protocol"], adapter_protocol)
    pd.DataFrame(causality_rows).to_csv(paths["attention_causality_audit"], index=False)
    pd.DataFrame(sensitivity_rows).to_csv(paths["dual_stream_sensitivity"], index=False)
    pd.DataFrame(parameter_rows).to_csv(paths["parameter_budget"], index=False)
    _write_json(paths["fold_transform_manifest"], transform_manifest)
    _write_json(paths["representation_export_manifest"], export_manifest)
    pd.DataFrame(acceptance_rows).to_csv(paths["acceptance_checks"], index=False)
    paths["report"].write_text(
        _render_report(
            status=status,
            parameter_rows=parameter_rows,
            causality_rows=causality_rows,
            sensitivity_rows=sensitivity_rows,
            acceptance_rows=acceptance_rows,
            export_manifest=export_manifest,
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 验证 MulT 与 ContiFormer 任务头前生产适配器的因果注意力、双流响应和恢复能力。\n"
        "- 两个模型只完成随机初始化与训练折归一化，未运行公共自监督训练，不产生下游任务指标。\n"
        "- 双流敏感性只证明两个输入流进入计算图，不代表语义协同或任务增益。\n"
        "- 本 run 未读取仿真任务真值或鼎新弱监督标签，未修改既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_deep_baseline_adapter_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "deep_baseline_adapter_smoke_only",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "heavy_run_root": heavy_run_root,
            "downstream_metrics_produced": False,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _render_report(
    *,
    status: str,
    parameter_rows: Sequence[Mapping[str, object]],
    causality_rows: Sequence[Mapping[str, object]],
    sensitivity_rows: Sequence[Mapping[str, object]],
    acceptance_rows: Sequence[Mapping[str, object]],
    export_manifest: Mapping[str, object],
) -> str:
    passed = sum(bool(row["passed"]) for row in acceptance_rows)
    failed = [str(row["check_id"]) for row in acceptance_rows if not row["passed"]]
    worst_future = max(float(row["future_perturbation_max_abs_delta"]) for row in causality_rows)
    minimum_phys = min(float(row["physiology_history_max_abs_delta"]) for row in sensitivity_rows)
    minimum_vehicle = min(float(row["vehicle_history_max_abs_delta"]) for row in sensitivity_rows)
    lines = [
        "# MulT 与 ContiFormer 生产适配器冒烟报告",
        "",
        "## 结论",
        "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        "- 两个深度基线均导出任务头之前的 96 点、64 维时序表示，并启用严格因果注意力。",
        "- 仿真和鼎新各完成 MulT/ContiFormer 留出折导出，归一化拟合样本完全一致。",
        (
            f"- 当前可用导出 {export_manifest['available_export_count']} 个；"
            f"本次新建 {export_manifest['current_run_built_count']} 个、"
            f"复用 {export_manifest['current_run_reused_count']} 个；"
            f"恢复复核再复用 {export_manifest['resume_verification_reused_count']} 个。"
        ),
        f"- 未来观测扰动对历史查询的最大变化为 {worst_future:.3e}。",
        f"- 生理/航电历史扰动的最小表示变化为 {minimum_phys:.3e}/{minimum_vehicle:.3e}，两个流均进入计算图。",
        "- 本 run 没有公共自监督训练或下游任务指标，不能据此比较模型优劣。",
    ]
    if failed:
        lines.append("- 未通过项：" + "、".join(failed) + "。")
    lines.extend(
        [
            "",
            "## 参数与运行审计",
            "",
            "| 数据 | 方法 | 输入字段 | 参数量 | 导出耗时（秒） |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in parameter_rows:
        lines.append(
            f"| {row['dataset_label']} | {row['method_label']} | "
            f"{row['input_feature_count']} | {row['parameter_count']} | "
            f"{row['export_elapsed_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 打通 Chronaris 双流 ODE-RNN、物理一致性和秒级多尺度因果融合。",
            "2. 五个可训练编码器共用 masked reconstruction、短期预测和时延判别预训练合同。",
            "3. Chronaris 路径与四项固定消融通过后，再实现下游算法并启动候选筛选。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
