"""Compact outputs for Dingxin inner validation split planning."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_inner_split_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    source_manifest,
    plans,
    role_rows,
    coverage_rows,
    acceptance_rows,
):
    paths = {
        "source_manifest": run_root / "source_manifest.json",
        "split_manifest": run_root / "split_manifest.json",
        "context_roles": run_root / "context_roles.csv",
        "task_coverage": run_root / "task_coverage.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["source_manifest"], source_manifest)
    _write_json(
        paths["split_manifest"],
        {
            "format": "chronaris.dingxin_inner_splits.v1",
            "folds": [plan.to_dict() for plan in plans],
            "target_threshold_scope": "outer_train_smoke_only",
            "formal_screen_requires_nested_target_refit": True,
        },
    )
    for key, rows in (
        ("context_roles", role_rows),
        ("task_coverage", coverage_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新外层折训练内验证划分报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                "- 五个外层折均形成 inner-train、validation、overlap embargo 和 outer-test 四种角色；每折 93 个完整输入只出现一次。",
                "- 外层训练组含两个不同架次时，完整留出一个训练架次做 validation；只含同一架次时，按时间块划分并删除所有与 validation 30 秒窗口重叠的中间上下文。",
                "- inner-train 与 validation 在共享航电流上的原始时间区间重叠数为 0；样本集合与 outer-test 也完全分离。",
                "- 每折分类的三种角色都覆盖低、中、高三类；生理响应三种角色均有连续目标和高/非高两类。",
                "- 当前目标阈值由 outer-train 拟合，只允许本阶段固定配置 smoke 使用；正式候选筛选必须按 inner-train 重拟合嵌套目标。",
                "- 本 run 不训练模型、不读取 outer-test 指标，也不形成候选排名。",
                "",
                "## 下一步",
                "",
                "1. 以本 split manifest 为唯一输入拟合五折归一化器和公共预训练 checkpoint。",
                "2. 为六方法导出 inner-train/validation/outer-test 表示并核对样本哈希。",
                "3. 固定 consumer smoke 可使用现有 outer-train 目标；进入正式 screen 前先生成 inner-train 嵌套目标 archive。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 只定义鼎新真实数据的训练内验证和重叠禁区，不产生模型效果证据。\n"
        "- 不同 view 若共享同一架次航电流，不被视为独立 validation group，而改用时间块与重叠禁区。\n"
        "- 当前 outer-train 阈值只服务固定配置 smoke；正式候选选择必须重建 inner-train 嵌套目标。\n"
        "- outer-test 样本只进入表示导出结构验证，不参与超参数或候选选择。\n"
        "- 本 run 未训练模型、未生成指标、未修改既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/build_dingxin_inner_splits.py "
        f"--run-id {run_id}\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "dingxin_inner_validation_splits",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "formal_screen_requires_nested_target_refit": True,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
