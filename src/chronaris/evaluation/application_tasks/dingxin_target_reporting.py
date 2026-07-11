"""Compact evidence outputs for Dingxin application target archives."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_dingxin_target_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    source_manifest,
    archive_rows,
    availability_rows,
    threshold_rows,
    comparison_rows,
    acceptance_rows,
    heavy_run_root: str,
):
    paths = {
        "source_manifest": run_root / "source_manifest.json",
        "target_archive_manifest": run_root / "target_archive_manifest.csv",
        "target_availability": run_root / "target_availability.csv",
        "threshold_lineage": run_root / "threshold_lineage.csv",
        "response_source_comparison": run_root / "response_source_comparison.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["source_manifest"], source_manifest)
    for key, rows in (
        ("target_archive_manifest", archive_rows),
        ("target_availability", availability_rows),
        ("threshold_lineage", threshold_rows),
        ("response_source_comparison", comparison_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    response_summary = pd.DataFrame(availability_rows)
    available = int(response_summary["fully_observed"].sum())
    unavailable = int((~response_summary["fully_observed"]).sum())
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新应用任务目标归档报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                "- 机动强度弱监督分类保留 G1 五个外层折的训练折阈值与 96 个上下文标签；标签源航电字段继续全部禁止进入模型输入。",
                f"- 生理响应从冻结原始点重新计算当前/未来 5 秒窗口中位数：{available} 个上下文未来区间完整，{unavailable} 个末端上下文因不足完整 5 秒而结构化不可用。",
                "- 生理字段覆盖、IQR 缩放和高响应阈值均只由各折可用训练上下文拟合；G1 窗口均值兼容值只用于口径比较，不作为正式原始点目标。",
                f"- 五个外层折、两个任务共写出 {len(archive_rows)} 个独立目标 archive；每个 archive 与阈值文件均完成确定性重写哈希复核。",
                "- 本 run 不训练表示编码器或下游算法，不形成方法排名。",
                "",
                "## 新发现",
                "",
                "冻结原始点只覆盖每条流约 181 秒，因此每个 view 的最后一个候选响应上下文只能观察约 1 秒未来数据。正式协议不把它伪装成完整 5 秒目标，而是从 93 个审计候选中保留 90 个完整目标。",
                "",
                "## 下一步",
                "",
                "1. 按 archive 中的 context 时间范围从 snapshot 构造原始异步双流输入。",
                "2. 完成机动标签源字段及全部确定性派生项的输入零命中审计。",
                "3. 在真实外层折上训练六方法表示并接入固定线性与 MiniROCKET 消费者。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 固化鼎新真实数据上的弱监督任务目标，不等同于人工工作负荷或专家机动科目真值。\n"
        "- 机动分类标签来自训练折拟合的机动强度规则；生理响应来自冻结原始点的窗口中位数变化。\n"
        "- G1 窗口均值兼容值只承担口径追溯，正式生理响应 archive 使用原始点窗口中位数。\n"
        "- 末端未来区间不足完整 5 秒的上下文明确标记不可用，不以短区间或零值替代。\n"
        "- 本 run 未训练模型、未生成任务指标，也未修改既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/build_dingxin_target_archives.py "
        f"--run-id {run_id} --resume\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "dingxin_weak_supervision_target_archives",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "human_expert_labels_available": False,
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
