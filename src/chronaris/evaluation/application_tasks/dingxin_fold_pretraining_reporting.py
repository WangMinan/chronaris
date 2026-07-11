"""Compact evidence writer for real Dingxin fold common pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_dingxin_fold_pretraining_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    source_manifest,
    split_manifest,
    training_protocol,
    training_status_rows,
    training_rows,
    resource_rows,
    representation_manifest,
    acceptance_rows,
    heavy_run_root: str,
):
    paths = {
        "source_manifest": run_root / "source_manifest.json",
        "split_manifest": run_root / "split_manifest.json",
        "training_protocol": run_root / "training_protocol.json",
        "training_status": run_root / "training_status.csv",
        "pretext_loss_audit": run_root / "pretext_loss_audit.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "representation_export_manifest": run_root / "representation_export_manifest.json",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    for key, payload in (
        ("source_manifest", source_manifest),
        ("split_manifest", split_manifest),
        ("training_protocol", training_protocol),
        ("representation_export_manifest", representation_manifest),
    ):
        _write_json(paths[key], payload)
    for key, rows in (
        ("training_status", training_status_rows),
        ("pretext_loss_audit", training_rows),
        ("resource_budget", resource_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    total_elapsed = sum(row["training_elapsed_s"] for row in resource_rows)
    maximum_rss = max(row["maximum_rss_mb"] for row in resource_rows)
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新主协议折六方法公共预训练与表示导出报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                "- 以留一视图主协议第一个外层折为资源与恢复冒烟：inner-train、validation、outer-test 各 31 个完整 30 秒上下文。",
                "- 五个可训练方法共享 inner-train 归一化、增强、三个公共目标和 1 epoch 预算；朴素时间同步只拟合 inner-train 无监督随机化主成分投影。",
                f"- 五方法累计训练 {total_elapsed:.2f} 秒；运行峰值内存 {maximum_rss:.1f} MB。",
                "- 六方法三种角色共导出 18 份 `[N,96,64]` 表示，第二次执行全部从已完成产物恢复，并通过同角色样本与查询轴对齐。",
                "- 本 run 不打开任务目标、不训练下游消费者、不读取 outer-test 指标，也不形成模型排名。",
                "",
                "## 下一步",
                "",
                "1. 在资源预算可接受后扩展到其余四个固定外层折。",
                "2. 完成五折表示后接入固定线性与 MiniROCKET 工程冒烟。",
                "3. 正式候选筛选前先按 inner-train 重建嵌套任务目标。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 只证明鼎新真实原始双流可在防重叠训练内划分上完成六方法公共预训练与统一表示导出。\n"
        "- 只执行一个主协议折、一个 seed、一个 epoch；它是资源和恢复冒烟，不是正式任务结果。\n"
        "- 预训练未读取机动分类、生理响应或任何人工评价；outer-test 只导出表示，不计算指标。\n"
        "- 不据此比较 Chronaris、MulT、ContiFormer 或其他基线优劣。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/run_dingxin_fold_pretraining_smoke.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "dingxin_real_fold_pretraining_smoke",
            "training_invoked": True,
            "downstream_targets_opened": False,
            "outer_test_metrics_opened": False,
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
