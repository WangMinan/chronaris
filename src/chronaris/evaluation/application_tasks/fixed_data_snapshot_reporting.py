"""Compact, reviewable outputs for the ignored G2a raw snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def write_snapshot_compact_outputs(
    *,
    config,
    compact_run_root: Path,
    snapshot_manifest: Mapping[str, object],
    consistency_rows: Sequence[Mapping[str, object]],
    exclusion_rows: Sequence[Mapping[str, object]],
    status: str,
    resumed: bool,
) -> dict[str, str]:
    compact_run_root.mkdir(parents=True, exist_ok=True)
    manifest_path = compact_run_root / "raw_snapshot_manifest.json"
    _write_json(manifest_path, snapshot_manifest)
    consistency_path = compact_run_root / "raw_snapshot_consistency.csv"
    pd.DataFrame(consistency_rows).to_csv(consistency_path, index=False)
    exclusion_path = compact_run_root / "raw_input_field_exclusion.csv"
    pd.DataFrame(exclusion_rows).to_csv(exclusion_path, index=False)

    report_path = compact_run_root / "report.md"
    report_path.write_text(
        _render_report(
            snapshot_manifest=snapshot_manifest,
            consistency_rows=consistency_rows,
            exclusion_rows=exclusion_rows,
            status=status,
            resumed=resumed,
        ),
        encoding="utf-8",
    )
    claim_boundary_path = compact_run_root / "claim_boundary.md"
    claim_boundary_path.write_text(
        "# 论断边界\n\n"
        "- 本 run 证明现有两个鼎新架次的原始异步点已按既有时间范围完成本机冻结。\n"
        "- snapshot 只包含既有数据库副本，不是新增现场数据。\n"
        "- 原始高频值位于 Git 忽略目录；仓库只保留计数、哈希和字段排除清单。\n"
        "- 本 run 未训练模型、未生成下游指标。\n"
        "- 机动标签源字段保留在审计副本中，但明确禁止进入机动分类编码器输入。\n",
        encoding="utf-8",
    )
    resume_path = compact_run_root / "resume_command.txt"
    resume_path.write_text(
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/freeze_fixed_data.py "
        f"--run-id {config.run_id} --resume\n",
        encoding="utf-8",
    )
    output_paths = {
        "raw_snapshot_manifest": str(manifest_path),
        "raw_snapshot_consistency": str(consistency_path),
        "raw_input_field_exclusion": str(exclusion_path),
        "report": str(report_path),
        "claim_boundary": str(claim_boundary_path),
        "resume_command": str(resume_path),
        "progress": str(compact_run_root / "progress.json"),
        "run_log": str(compact_run_root / "run.log"),
    }
    evidence_path = compact_run_root / "evidence_manifest.json"
    _write_json(
        evidence_path,
        {
            "run_id": config.run_id,
            "status": status,
            "evidence_layer": "dingxin_fixed_raw_input_snapshot",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "raw_values_committed": False,
            "snapshot_root": snapshot_manifest["snapshot_root"],
            "source_manifest_hashes": snapshot_manifest["source_manifest_hashes"],
            "output_paths": output_paths,
        },
    )
    return {
        "compact_manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "evidence_manifest_path": str(evidence_path),
    }


def write_snapshot_unavailable(
    *,
    config,
    error: BaseException,
) -> str:
    """Write a compact, non-fabricated unavailable record for live read failure."""

    compact_run_root = Path(config.compact_output_root) / config.run_id
    compact_run_root.mkdir(parents=True, exist_ok=True)
    path = compact_run_root / "raw_snapshot_unavailable.json"
    _write_json(
        path,
        {
            "run_id": config.run_id,
            "status": "unavailable",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "fallback_boundary": (
                "do not fabricate raw Dingxin streams; continue simulation work and keep "
                "post-alignment evidence separate"
            ),
        },
    )
    return str(path)


def _render_report(
    *,
    snapshot_manifest: Mapping[str, object],
    consistency_rows: Sequence[Mapping[str, object]],
    exclusion_rows: Sequence[Mapping[str, object]],
    status: str,
    resumed: bool,
) -> str:
    file_records = list(snapshot_manifest["files"])
    physiology_points = sum(
        int(row["point_count"]) for row in file_records if row["stream_kind"] == "physiology"
    )
    vehicle_points = sum(
        int(row["point_count"]) for row in file_records if row["stream_kind"] == "vehicle"
    )
    consistency_pass = sum(bool(row["count_matches_reference"]) for row in consistency_rows)
    exclusions_observed = sum(bool(row["observed_in_snapshot"]) for row in exclusion_rows)
    return "\n".join(
        [
            "# 鼎新原始异步点冻结报告",
            "",
            "## 结论",
            "",
            f"- run 状态：`{status}`；本次是否复用已校验 snapshot：`{str(resumed).lower()}`。",
            f"- 共写入 {len(file_records)} 个文件：生理点 {physiology_points}，共享航电点 {vehicle_points}。",
            f"- 点数一致性：{consistency_pass}/{len(consistency_rows)} 项通过。",
            f"- 机动标签源字段在审计副本中的可见性：{exclusions_observed}/{len(exclusion_rows)}。",
            "- 原始高频值保存在 Git 忽略目录；仓库产物只包含哈希、范围、计数和排除清单。",
            "",
            "## 存储布局",
            "",
            "- 每个 sortie 只保存一份共享航电点。",
            "- 每个 view 保存一份按 pilot 过滤的生理点。",
            "- 文件采用确定性 gzip JSONL，可流式读取并用 SHA-256 校验。",
            "",
            "## 输入边界",
            "",
            "机动标签源字段存在于冻结审计副本中，但六方法构造机动分类输入时必须同时删除原字段、窗口统计和确定性派生项。",
            "",
            "## 下一步",
            "",
            "1. 从原始点按训练折重新生成窗口中位数生理响应目标。",
            "2. 使用同一 snapshot 构造六方法原始输入和统一查询轴。",
            "3. 并行进入方法无关的 G1/G2 仿真生成器实现。",
            "",
        ]
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
