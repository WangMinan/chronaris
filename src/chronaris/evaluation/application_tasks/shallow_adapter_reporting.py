"""Compact writers for the shallow production adapter smoke run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_shallow_adapter_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    adapter_protocol: Mapping[str, object],
    causal_rows: Sequence[Mapping[str, object]],
    parameter_rows: Sequence[Mapping[str, object]],
    transform_manifest: Mapping[str, object],
    export_manifest: Mapping[str, object],
    acceptance_rows: Sequence[Mapping[str, object]],
    heavy_run_root: str,
) -> Mapping[str, str]:
    paths = {
        "adapter_protocol": run_root / "adapter_protocol.json",
        "causal_query_audit": run_root / "causal_query_audit.csv",
        "parameter_budget": run_root / "parameter_budget.csv",
        "fold_transform_manifest": run_root / "fold_transform_manifest.json",
        "representation_export_manifest": run_root / "representation_export_manifest.json",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["adapter_protocol"], adapter_protocol)
    pd.DataFrame(causal_rows).to_csv(paths["causal_query_audit"], index=False)
    pd.DataFrame(parameter_rows).to_csv(paths["parameter_budget"], index=False)
    _write_json(paths["fold_transform_manifest"], transform_manifest)
    _write_json(paths["representation_export_manifest"], export_manifest)
    pd.DataFrame(acceptance_rows).to_csv(paths["acceptance_checks"], index=False)
    paths["report"].write_text(
        _render_report(
            status=status,
            parameter_rows=parameter_rows,
            causal_rows=causal_rows,
            acceptance_rows=acceptance_rows,
            export_manifest=export_manifest,
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 使用生产适配器验证两个单流与朴素时间同步的因果查询、来源隔离和恢复能力。\n"
        "- 编码器只完成随机初始化或无监督训练折拟合，未运行公共自监督训练，不产生下游任务指标。\n"
        "- 参数量、运行时间和查询有效率是工程审计，不代表任务性能排名。\n"
        "- 本 run 未读取仿真任务真值或鼎新弱监督标签，未修改既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_shallow_adapter_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "production_adapter_smoke_only",
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
    causal_rows: Sequence[Mapping[str, object]],
    acceptance_rows: Sequence[Mapping[str, object]],
    export_manifest: Mapping[str, object],
) -> str:
    passed = sum(bool(row["passed"]) for row in acceptance_rows)
    failed = [str(row["check_id"]) for row in acceptance_rows if not row["passed"]]
    lines = [
        "# 两个单流与朴素时间同步生产适配器冒烟报告",
        "",
        "## 结论",
        "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        "- 生理单流与航电单流复用同一个连续时间编码器类；朴素时间同步只使用当前及历史观测。",
        "- 仿真和鼎新各完成三个生产适配器的留出折导出，查询轴均为 96 点、输出均为 64 维。",
        (
            f"- 当前可用导出 {export_manifest['available_export_count']} 个；"
            f"本次新建 {export_manifest['current_run_built_count']} 个、"
            f"复用 {export_manifest['current_run_reused_count']} 个；"
            f"恢复复核再复用 {export_manifest['resume_verification_reused_count']} 个。"
        ),
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
    worst_future = max(float(row["future_perturbation_max_abs_delta"]) for row in causal_rows)
    worst_inactive = max(float(row["inactive_stream_max_abs_delta"]) for row in causal_rows)
    lines.extend(
        [
            "",
            "## 因果边界",
            "",
            f"- 未来观测扰动对当前及历史查询的最大绝对变化：{worst_future:.3e}。",
            f"- 非激活模态扰动对单流输出的最大绝对变化：{worst_inactive:.3e}。",
            "- 训练折归一化与主成分投影的拟合样本哈希均不含验证或留出测试样本。",
            "",
            "## 下一步",
            "",
            "1. 将 MulT 与 ContiFormer 的任务头前时序状态接入同一表示导出器。",
            "2. 为五个可训练编码器实现公共自监督目标和共享训练增强。",
            "3. 深度基线冒烟验证通过后，再进入 Chronaris 连续融合主干改造。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
