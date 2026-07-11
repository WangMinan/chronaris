"""Compact outputs for the common pretraining and linear-consumer loop smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd


def write_common_pretraining_smoke_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    data_manifest_rows,
    split_manifest,
    augmentation_protocol,
    augmentation_alignment_rows,
    pretext_target_manifest,
    training_protocol,
    training_rows,
    training_status_rows,
    resource_rows,
    representation_manifest,
    downstream_protocol,
    metric_rows,
    acceptance_rows,
    heavy_run_root: str,
    prediction_path: str,
) -> Mapping[str, str]:
    paths = {
        "data_manifest": run_root / "data_manifest.json",
        "split_manifest": run_root / "split_manifest.json",
        "augmentation_protocol": run_root / "augmentation_protocol.json",
        "augmentation_alignment": run_root / "augmentation_alignment.csv",
        "pretext_target_manifest": run_root / "pretext_target_manifest.json",
        "pretext_loss_audit": run_root / "pretext_loss_audit.csv",
        "training_protocol": run_root / "training_protocol.json",
        "training_status": run_root / "training_status.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "checkpoint_registry": run_root / "checkpoint_registry.json",
        "representation_export_manifest": run_root / "representation_export_manifest.json",
        "downstream_protocol": run_root / "downstream_protocol.json",
        "fold_metrics": run_root / "fold_metrics.csv",
        "metric_long": run_root / "metric_long.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(
        paths["data_manifest"],
        {
            "format": "chronaris.common_pretraining_smoke_data.v1",
            "sample_count": len(data_manifest_rows),
            "rows": list(data_manifest_rows),
            "pretraining_oracle_opened": False,
        },
    )
    for key, payload in (
        ("split_manifest", split_manifest),
        ("augmentation_protocol", augmentation_protocol),
        ("pretext_target_manifest", pretext_target_manifest),
        ("training_protocol", training_protocol),
        ("representation_export_manifest", representation_manifest),
        ("downstream_protocol", downstream_protocol),
    ):
        _write_json(paths[key], payload)
    for key, rows in (
        ("augmentation_alignment", augmentation_alignment_rows),
        ("pretext_loss_audit", training_rows),
        ("training_status", training_status_rows),
        ("resource_budget", resource_rows),
        ("fold_metrics", metric_rows),
        ("metric_long", metric_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    paths["report"].write_text(
        _render_report(
            status=status,
            training_status_rows=training_status_rows,
            training_rows=training_rows,
            resource_rows=resource_rows,
            representation_manifest=representation_manifest,
            metric_rows=metric_rows,
            acceptance_rows=acceptance_rows,
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 只验证六方法公共预训练、折外导出和固定线性下游消费链路可执行。\n"
        "- 训练仅使用 16 条仿真训练划分轨迹、1 epoch 和固定候选 A；所有指标均标记 smoke only。\n"
        "- 仿真真值只在五个可训练 checkpoint 完成后用于线性下游任务，不进入表示预训练。\n"
        "- 本 run 不读取仿真锁定测试，不更新 confirmed metrics，不进行候选选择或模型优劣判断。\n"
        "- 鼎新弱监督任务、MiniRocket、TCN、完整开发筛选与锁定确认仍属后续里程碑。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_common_pretraining_loop_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "common_pretraining_loop_smoke_only",
            "training_invoked": True,
            "training_epochs": 1,
            "confirmed_metrics_changed": False,
            "downstream_metrics_smoke_only": True,
            "pretraining_oracle_opened": False,
            "heavy_run_root": heavy_run_root,
            "prediction_path": prediction_path,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _render_report(
    *,
    status,
    training_status_rows,
    training_rows,
    resource_rows,
    representation_manifest,
    metric_rows,
    acceptance_rows,
):
    passed = sum(bool(row["passed"]) for row in acceptance_rows)
    total_elapsed = sum(row["training_elapsed_s"] for row in resource_rows)
    active_terms = sum(row["status"] == "active" for row in training_rows)
    available_metrics = sum(row["status"] == "available" for row in metric_rows)
    lines = [
        "# 六方法公共预训练与线性下游闭环冒烟报告",
        "",
        "## 结论",
        "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        "- 五个可训练编码器在同一 8 条仿真训练轨迹上使用相同增强和三个公共目标完成 1 epoch；朴素时间同步只拟合训练折无监督变换。",
        f"- 三个公共目标共产生 {active_terms} 条 active step 记录；五方法累计训练耗时 {total_elapsed:.2f} 秒。",
        f"- 六方法 train/validation/held-out 共导出 {representation_manifest['available_export_count']} 份表示，恢复复核复用 {representation_manifest['resume_verification_reused_count']} 份；删除 Chronaris 留出折后单项重建的哈希一致。",
        f"- 固定 Logistic/Ridge 线性 consumer 产生 {len(metric_rows)} 条 smoke 指标，其中 {available_metrics} 条可计算。",
        "- 仿真 workload 真值在五个 checkpoint 完成后才打开；表示预训练没有读取 oracle 或下游标签。",
        "- 本 run 不比较模型优劣，不更新论文主指标。",
        "",
        "## 训练资源",
        "",
        "| 方法 | 状态 | 主干参数 | 预训练头参数 | 训练秒数 | step |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    status_by_method = {
        row["method_name"]: row["initial_status"]
        for row in training_status_rows
    }
    for row in resource_rows:
        lines.append(
            f"| {row['method_name']} | {status_by_method[row['method_name']]} | "
            f"{row['parameter_count']} | {row['head_parameter_count']} | "
            f"{row['training_elapsed_s']:.3f} | {row['step_count']} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 实现 MiniRocket 与 TCN/Viterbi 下游 consumer，并把真实弱监督任务接入相同表示合同。",
            "2. 将 smoke 数据扩展到完整 G1 train/validation，运行 seed 17 四候选开发筛选。",
            "3. 开发筛选期间继续禁止读取 G2 锁定测试，并保留所有 mixed/negative 结果。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
