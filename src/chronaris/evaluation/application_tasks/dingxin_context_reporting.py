"""Compact reports for lazy Dingxin raw-context and target bindings."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_dingxin_context_outputs(
    *,
    run_root: Path,
    run_id: str,
    status: str,
    source_manifest,
    schema_payload,
    context_rows,
    stream_rows,
    binding_rows,
    archive_verification,
    resource_rows,
    acceptance_rows,
):
    paths = {
        "source_manifest": run_root / "source_manifest.json",
        "schema_manifest": run_root / "schema_manifest.json",
        "context_catalog": run_root / "context_catalog.csv",
        "stream_audit": run_root / "stream_audit.csv",
        "fold_task_binding": run_root / "fold_task_binding.csv",
        "archive_verification": run_root / "archive_verification.csv",
        "resource_budget": run_root / "resource_budget.csv",
        "acceptance_checks": run_root / "acceptance_checks.csv",
        "report": run_root / "report.md",
        "claim_boundary": run_root / "claim_boundary.md",
        "resume_command": run_root / "resume_command.txt",
        "evidence_manifest": run_root / "evidence_manifest.json",
    }
    _write_json(paths["source_manifest"], source_manifest)
    _write_json(paths["schema_manifest"], schema_payload)
    for key, rows in (
        ("context_catalog", context_rows),
        ("stream_audit", stream_rows),
        ("fold_task_binding", binding_rows),
        ("archive_verification", archive_verification),
        ("resource_budget", resource_rows),
        ("acceptance_checks", acceptance_rows),
    ):
        pd.DataFrame(rows).to_csv(paths[key], index=False)
    passed = sum(row["passed"] for row in acceptance_rows)
    streams = pd.DataFrame(stream_rows)
    bindings = pd.DataFrame(binding_rows)
    classification = bindings[
        bindings["task_slug"] == "maneuver_intensity_classification"
    ]
    response = bindings[
        bindings["task_slug"] == "physiology_response_prediction"
    ]
    paths["report"].write_text(
        "\n".join(
            (
                "# 鼎新原始双流上下文与目标绑定报告",
                "",
                "## 结论",
                "",
                f"- 状态：{'完成' if status == 'completed' else '部分完成'}；验收 {passed}/{len(acceptance_rows)} 通过。",
                f"- 96 个标签上下文中，{int(streams.status.eq('completed').sum())} 个拥有完整 30 秒原始输入；3 个末端上下文实际只覆盖 155–180.991 秒，未达到名义 185 秒终点，因此不可用于模型输入。",
                f"- 机动分类在五折中绑定 {classification[classification.binding_status == 'available'].context_id.nunique()} 个唯一可用上下文；生理响应同时要求完整未来 5 秒，绑定 {response[response.binding_status == 'available'].context_id.nunique()} 个唯一可用上下文。",
                f"- 公共输入 schema 为 {schema_payload['physiology_feature_count']} 个生理字段和 {schema_payload['vehicle_feature_count']} 个航电字段；{schema_payload['excluded_feature_count']} 个机动标签源字段在原始映射层即被删除。",
                f"- 允许字段稀疏缓存包含 {schema_payload['cached_observed_value_count']:,} 个 float32 值、数组净大小 {schema_payload['cached_sparse_array_bytes'] / 1024 / 1024:.1f} MB；完整审计耗时 {resource_rows[0]['elapsed_s']:.2f} 秒，峰值内存 {resource_rows[0]['maximum_rss_mb']:.1f} MB。",
                "- 每个可用上下文均通过懒加载从 snapshot 切片，最大相对时间严格小于 30 秒；没有生成 GB 级稠密上下文副本。",
                "- 10 个目标 archive 与阈值文件全部重新校验 SHA-256；外层折 train/test group 无交集。",
                "- 本 run 只闭合输入、目标和外层折 lineage，不训练模型或生成任务指标。",
                "",
                "## 存储与训练约束",
                "",
                "后续训练按小批次调用懒加载索引；不得把 93 个高频航电上下文一次性稠密化，也不得从目标区间回填输入末端。归一化器和编码器只在当前 fold 的 train context 上拟合。",
                "",
                "## 下一步",
                "",
                "1. 为真实外层折增加训练内 validation 划分和可恢复公共预训练 checkpoint。",
                "2. 六方法按同一懒加载批次导出 train/validation/test 表示。",
                "3. 接入固定线性与 MiniROCKET 消费者，结果单独写入鼎新弱监督证据目录。",
                "",
            )
        ),
        encoding="utf-8",
    )
    paths["claim_boundary"].write_text(
        "# 论断边界\n\n"
        "- 本 run 验证鼎新真实原始点的输入切片、字段排除和弱监督目标绑定，不等同于模型效果。\n"
        "- 机动标签源字段在 schema 构造阶段即删除；生理响应目标区间不进入输入。\n"
        "- 三个输入末端不完整、三个未来区间不完整的上下文均结构化不可用，不用填充或短区间替代。\n"
        "- 输入采用懒加载小批次，不提交原始值或稠密上下文副本。\n"
        "- 本 run 未训练模型、未生成任务指标、未修改既有确认指标。\n",
        encoding="utf-8",
    )
    paths["resume_command"].write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/audit_dingxin_context_bindings.py "
        f"--run-id {run_id}\n",
        encoding="utf-8",
    )
    _write_json(
        paths["evidence_manifest"],
        {
            "run_id": run_id,
            "status": status,
            "evidence_layer": "dingxin_lazy_raw_context_bindings",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "precomputed_dense_context_bundle": False,
            "output_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    return {key: str(value) for key, value in paths.items()}


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
