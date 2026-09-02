#!/usr/bin/env python3
"""Write compact, reproducible evidence for the frozen thesis candidate screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.thesis_candidate_screen_metrics import (
    audit_thesis_candidate_screen,
)


NAMES = {
    "base": "修复后的安全滞后感知融合",
    "explicit_shift": "加入显式时移目标",
    "semantic_pair": "加入可学习事件语义与配对目标",
    "both_objectives": "同时加入时移与语义配对目标",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.output_root or args.input.parent
    state = json.loads(args.input.read_text(encoding="utf-8"))
    audit = audit_thesis_candidate_screen(state)
    audit["source_results_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "gate_audit.json", audit)
    _write_csv(root / "candidate_summary.csv", audit["candidate_rows"])
    _write_csv(root / "application_metrics.csv", audit["application_metric_rows"])
    (root / "report.md").write_text(_report(audit), encoding="utf-8")
    return 0 if audit["selected_candidate"] else 1


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report(audit):
    shift = audit["explicit_shift_gate"]
    pair = audit["event_pair_gate"]
    ranking = sorted(
        audit["candidate_rows"],
        key=lambda row: -row["new_mechanism_gates_passed"],
    )
    lines = [
        "# 论文模型训练内候选冻结报告",
        "",
        (
            f"结论：冻结“{NAMES[audit['selected_candidate']]}”进入受控机制复核；"
            "公开数据与鼎新外层结果在本轮始终关闭。"
        ),
        "",
        "## 训练内门禁",
        "",
        (
            "- 完整性与协议隔离："
            f"{'通过' if audit['protocol_gate_passed'] else '未通过'}，"
            f"共 {audit['row_count']} 个不重复训练单元。"
        ),
        f"- 五分类时移：{'通过' if shift['passed'] else '未通过'}；三个随机种子准确率为 "
        + "/".join(f"{row['accuracy']:.4f}" for row in shift["rows"])
        + "。",
        f"- 事件—响应配对：{'通过' if pair['passed'] else '未通过'}；正确配对相似度差为 "
        + "/".join(f"{row['similarity_gap']:.4f}" for row in pair["rows"])
        + "。",
        (
            "- 运动学目标字段范围与数值尺度："
            f"{'通过' if audit['physical_scope_gate_passed'] else '未通过'}；"
            "CLARE 无适用航电运动学字段时明确记录不可用。"
        ),
        "",
        "## 候选排序",
        "",
        "| 候选 | 新机制门通过数 | 参数量中位数 | 单元训练时长中位数（秒） | 结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ranking:
        lines.append(
            f"| {NAMES[row['candidate']]} | {row['new_mechanism_gates_passed']} | "
            f"{row['parameter_count_median']:.0f} | "
            f"{row['training_elapsed_s_median']:.1f} | "
            f"{'冻结' if row['selected'] else '不选'} |"
        )
    lines.extend(
        [
            "",
            (
                "训练内应用指标只用于同机制门数量候选的后续排序；本次双目标候选"
                "独占两个新机制门，因此不需要构造事后综合分数。CogPilot 难度任务"
                "四个候选均表现为强单流主导下的低增量，CLARE 分组验证方差较大，"
                "这些现象保留为外层实验需要复核的应用边界。"
            ),
            "",
            "## 进入外层前仍需完成",
            "",
            (
                "受控仿真须先完成连续演化、运动学约束和安全单流旁路三项消融硬门；"
                "未来信息扰动合同已经由聚焦测试复核。硬门通过后才允许生成公开"
                " outer-fold 与鼎新分组确认指标。"
            ),
            "",
            f"训练源码提交：`{audit['source_commit']}`。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
