"""Compact evidence writers for the representation contract smoke run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_representation_contract_outputs(
    *,
    compact_root: Path,
    run_id: str,
    status: str,
    input_schema: Mapping[str, object],
    representation_schema: Mapping[str, object],
    sample_order_manifest: Mapping[str, object],
    fold_transform_manifest: Mapping[str, object],
    oof_export_manifest: Mapping[str, object],
    acceptance_rows: Sequence[Mapping[str, object]],
    heavy_run_root: str,
) -> Mapping[str, str]:
    compact_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "input_schema": compact_root / "input_schema.json",
        "representation_schema": compact_root / "representation_schema.json",
        "sample_order_manifest": compact_root / "sample_order_manifest.json",
        "fold_transform_manifest": compact_root / "fold_transform_manifest.json",
        "oof_export_manifest": compact_root / "oof_export_manifest.json",
        "acceptance_checks": compact_root / "acceptance_checks.csv",
        "report": compact_root / "report.md",
        "claim_boundary": compact_root / "claim_boundary.md",
        "resume_command": compact_root / "resume_command.txt",
        "evidence_manifest": compact_root / "evidence_manifest.json",
    }
    for key, payload in (
        ("input_schema", input_schema),
        ("representation_schema", representation_schema),
        ("sample_order_manifest", sample_order_manifest),
        ("fold_transform_manifest", fold_transform_manifest),
        ("oof_export_manifest", oof_export_manifest),
    ):
        _write_json(paths[key], payload)
    pd.DataFrame(acceptance_rows).to_csv(paths["acceptance_checks"], index=False)
    paths["report"].write_text(
        _render_report(
            status=status,
            input_schema=input_schema,
            representation_schema=representation_schema,
            oof_export_manifest=oof_export_manifest,
            acceptance_rows=acceptance_rows,
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
            "- 本 run 验证统一输入、表示、训练折隔离、检查点来源和恢复接口。\n"
        "- 六个方法槽位使用合同探针，不是六种正式模型，也不产生可比较任务指标。\n"
        "- 仿真输入与鼎新输入均只验证数据合同；仿真真值和鼎新弱监督标签未进入编码器。\n"
        "- 本 run 未修改任何既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_representation_contract_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "representation_contract_smoke_only",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "heavy_run_root": heavy_run_root,
            "probe_outputs_are_model_results": False,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _render_report(
    *,
    status: str,
    input_schema: Mapping[str, object],
    representation_schema: Mapping[str, object],
    oof_export_manifest: Mapping[str, object],
    acceptance_rows: Sequence[Mapping[str, object]],
) -> str:
    passed = sum(bool(row["passed"]) for row in acceptance_rows)
    failed = [str(row["check_id"]) for row in acceptance_rows if not row["passed"]]
    dingxin = dict(input_schema["dingxin"])
    simulation = dict(input_schema["simulation"])
    lines = [
        "# 统一双流输入与融合表示合同冒烟报告",
        "",
        "## 结论",
        "",
        f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
        (
            f"- 鼎新输入合同包含 {dingxin['physiology_feature_count']} 个生理字段和 "
            f"{dingxin['vehicle_feature_count']} 个航电字段，明确排除 "
            f"{dingxin['excluded_feature_count']} 个机动标签源字段。"
        ),
        (
            f"- 仿真输入合同包含 {simulation['physiology_feature_count']} 个生理字段和 "
            f"{simulation['vehicle_feature_count']} 个航电字段，只读取 observed archive。"
        ),
        (
            f"- 六个方法接口均输出 {representation_schema['output_dim']} 维、"
            f"{representation_schema['query_point_count']} 个查询点的表示，"
            "样本标识、查询时间、有效掩码和留出折检查点来源完全一致。"
        ),
        (
            f"- 当前可用输出 {oof_export_manifest['available_export_count']} 个；"
            f"本次运行新建 {oof_export_manifest['current_run_built_count']} 个、"
            f"复用 {oof_export_manifest['current_run_initial_reused_count']} 个；"
            f"二次恢复校验再复用 {oof_export_manifest['resume_verification_reused_count']} 个。"
        ),
        "- 这里的六方法输出由合同探针生成，只证明基础设施贯通，不代表模型效果。",
    ]
    if failed:
        lines.append("- 未通过项：" + "、".join(failed) + "。")
    lines.extend(
        [
            "",
            "## 已验证边界",
            "",
            "- 鼎新与仿真数据通过同一逻辑批次合同，但保留各自字段结构。",
            "- 训练折中位数/四分位距（IQR）和主成分分析（PCA）的拟合样本哈希不含锁定测试样本。",
            "- 表示文件拒绝标签、输出分数（logits）、预测值、任务名和方法专属诊断量。",
            "- 检查点注册表同时记录训练、验证、留出测试样本集合和文件 SHA-256。",
            "- 删除缺失输出后的重建行为由自动化测试覆盖；完整输出在恢复运行时只校验并复用。",
            "",
            "## 下一步",
            "",
            "1. 用生产适配器替换六个合同探针，先实现两个单流和朴素时间同步。",
            "2. 接入 MulT、ContiFormer 与 Chronaris 连续融合主干，并保持同一 OOF 导出器。",
            "3. 六方法冒烟验证全部通过后，才启动公共自监督目标与正式训练。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
