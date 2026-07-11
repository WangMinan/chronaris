"""Compact report writer for Dingxin five-fold consumer smoke."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def write_dingxin_consumer_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    source_manifest,
    target_manifests,
    consumer_config,
    result_rows,
    resource_rows,
    metric_rows,
    fusion_gain_rows,
    acceptance_rows,
    heavy_run_root: str,
):
    paths = {
        "source_manifest": run_root / "source_manifest.json",
        "target_manifest": run_root / "target_manifest.json",
        "consumer_protocol": run_root / "consumer_protocol.json",
        "consumer_inventory": run_root / "consumer_inventory.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "metric_long": run_root / "metric_long.csv",
        "fusion_gain": run_root / "fusion_gain.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["source_manifest"], source_manifest)
    _write_json(
        paths["target_manifest"],
        {
            "format": "chronaris.dingxin_consumer_targets.v1",
            "folds": target_manifests,
            "threshold_scope": "outer_train_smoke_only",
            "formal_screen_requires_nested_target_refit": True,
        },
    )
    _write_json(
        paths["consumer_protocol"],
        {
            "format": "chronaris.dingxin_consumer_protocol.v1",
            "config": asdict(consumer_config),
            "consumers": ["linear", "minirocket"],
            "fit_role": "train",
            "evaluation_roles": ["validation", "held_out"],
            "method_specific_hyperparameters": False,
            "validation_used_for_selection": False,
            "smoke_only": True,
        },
    )
    for key, rows in (
        ("consumer_inventory", result_rows),
        ("resource_budget", resource_rows),
        ("metric_long", metric_rows),
        ("fusion_gain", fusion_gain_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    available = sum(row["status"] == "available" for row in metric_rows)
    unavailable = len(metric_rows) - available
    elapsed = sum(
        float(row["elapsed_s"])
        for row in resource_rows
        if row["status"] == "completed"
    )
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新五折冻结表示下游消费者工程冒烟报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                "- 五个固定外层折、六种表示方法均使用同一线性模型和 MiniROCKET 配置，形成 30 个方法—折组合、60 个消费者组件。",
                f"- 机动强度分类、生理响应回归和高生理响应识别共生成 {len(metric_rows)} 条 smoke-only 指标；{available} 条可计算，{unavailable} 条按相关系数未定义等原因结构化保留。",
                f"- 方向归一的双流增益接口生成 {len(fusion_gain_rows)} 条记录；消费者首次拟合累计 {elapsed:.2f} 秒，第二遍恢复 60/60。",
                "- 当前标签阈值仍来自 outer-train，只允许工程冒烟；本报告不比较或选择模型，不更新论文确认指标。",
                "",
                "## 下一步",
                "",
                "1. 按每折 inner-train 重建嵌套目标，消除 validation 对标签阈值的间接参与。",
                "2. 用 validation 运行固定候选 screen，并在候选冻结前保持 outer-test 指标关闭。",
                "3. 正式 screen 后再进行多 seed 锁定确认、压力测试和消融。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 使用鼎新现有双流的弱监督任务，不代表人工工作负荷或专家机动真值。\n"
        "- outer-train 阈值仅供固定配置 smoke，不能用于正式候选选择。\n"
        "- 输出指标用于验证消费者、方向合同和恢复链路，不形成方法排名。\n"
        "- 鼎新、仿真与公开数据指标保持独立目录，不计算跨证据平均分。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_consumer_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "dingxin_frozen_consumer_smoke",
            "metric_count": len(metric_rows),
            "fusion_gain_count": len(fusion_gain_rows),
            "threshold_scope": "outer_train_smoke_only",
            "formal_screen_requires_nested_target_refit": True,
            "confirmed_metrics_changed": False,
            "heavy_run_root": heavy_run_root,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
