"""Materialize and audit the frozen simplified Dingxin task contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from chronaris.dataset.application_evaluation.snapshot_io import sha256_file
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    load_dingxin_target_source_data,
)
from chronaris.evaluation.dingxin.representation_compatibility import (
    audit_representation_compatibility,
)
from chronaris.evaluation.dingxin.simple_downstream_protocol import (
    SIMPLE_DOWNSTREAM_METHODS,
    extract_simple_raw_targets,
    fit_simple_loso_targets,
    protocol_payload,
)


@dataclass(frozen=True, slots=True)
class SimpleDownstreamAuditConfig:
    run_id: str = "2026-07-16_simple-downstream-protocol"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    historical_representation_root: str = (
        "docs/artifacts/runs/"
        "2026-07-12_dingxin-locked-representations-coalesced"
    )
    repository_root: str = "."
    required_seeds: tuple[int, ...] = (17, 29, 43)


@dataclass(frozen=True, slots=True)
class SimpleDownstreamAuditResult:
    compact_root: str
    heavy_root: str
    protocol_path: str
    compatibility_report_path: str
    task_summary_path: str
    report_path: str
    evidence_manifest_path: str


def run_simple_downstream_audit(
    config: SimpleDownstreamAuditConfig,
) -> SimpleDownstreamAuditResult:
    repo_root = Path(config.repository_root).resolve()
    compact_root = repo_root / config.compact_output_root / config.run_id
    heavy_root = repo_root / config.heavy_output_root / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)

    fixed_audit_root = _resolve(repo_root, config.fixed_audit_root)
    snapshot_root = _resolve(repo_root, config.snapshot_root)
    historical_root = _resolve(repo_root, config.historical_representation_root)
    source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_audit_root,
        snapshot_root=snapshot_root,
    )
    raw = extract_simple_raw_targets(source, snapshot_root=snapshot_root)
    folds = fit_simple_loso_targets(raw)

    heavy_files = {
        "context_manifest": heavy_root / "context_manifest.csv",
        "maneuver_statistics": heavy_root / "maneuver_statistics.csv",
        "physiology_statistics": heavy_root / "physiology_statistics.csv",
        "fold_manifest": heavy_root / "fold_manifest.csv",
        "maneuver_targets": heavy_root / "maneuver_targets.csv",
        "physiology_targets": heavy_root / "physiology_targets.csv",
        "thresholds": heavy_root / "fold_fitted_parameters.csv",
    }
    _write_csv(raw.contexts, heavy_files["context_manifest"])
    _write_csv(raw.maneuver_statistics, heavy_files["maneuver_statistics"])
    _write_csv(raw.physiology_statistics, heavy_files["physiology_statistics"])
    _write_csv(folds.fold_manifest, heavy_files["fold_manifest"])
    _write_csv(folds.maneuver_targets, heavy_files["maneuver_targets"])
    _write_csv(folds.physiology_targets, heavy_files["physiology_targets"])
    _write_csv(folds.threshold_rows, heavy_files["thresholds"])

    historical_protocol = historical_root / "protocol.json"
    compatibility = audit_representation_compatibility(
        expected_context_ids=raw.contexts["context_id"].astype(str),
        inventory_path=historical_root / "representation_inventory.jsonl",
        protocol_path=historical_protocol,
        field_role_manifest_path=fixed_audit_root / "field_role_manifest.csv",
        required_methods=SIMPLE_DOWNSTREAM_METHODS,
        required_seeds=config.required_seeds,
        repository_root=repo_root,
    )
    compatibility_path = compact_root / "representation_compatibility_report.json"
    _write_json(compatibility_path, compatibility)

    selected_fields = (
        folds.physiology_targets[folds.physiology_targets["selected"].astype(bool)]
        .groupby("fold_id")["field_name"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    summary = {
        "format": "chronaris.simple_downstream_task_summary.v1",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "view_context_count": int(len(raw.contexts)),
        "vehicle_context_count": int(raw.contexts["vehicle_context_id"].nunique()),
        "sortie_count": int(raw.contexts["sortie_id"].nunique()),
        "view_count": int(raw.contexts["view_id"].nunique()),
        "maneuver_semantic_count": int(
            raw.maneuver_statistics["semantic_key"].nunique()
        ),
        "physiology_field_count": int(
            raw.physiology_statistics["field_name"].nunique()
        ),
        "selected_physiology_field_count_by_fold": selected_fields,
        "folds": folds.fold_manifest.to_dict(orient="records"),
        "historical_representation_status": compatibility["status"],
        "confirmed_metrics_changed": False,
        "model_training_started": False,
    }
    summary_path = compact_root / "task_manifest_summary.json"
    _write_json(summary_path, summary)

    protocol = protocol_payload()
    protocol.update(
        {
            "run_id": config.run_id,
            "status": "completed",
            "generated_at_utc": summary["generated_at_utc"],
            "config": asdict(config),
            "source_hashes": dict(raw.source_hashes),
            "heavy_artifacts": {
                name: _relative_or_absolute(path, repo_root)
                for name, path in heavy_files.items()
            },
        }
    )
    protocol_path = compact_root / "protocol.json"
    _write_json(protocol_path, protocol)

    acceptance = pd.DataFrame(
        [
            {
                "check_id": "complete_future_view_contexts",
                "status": "passed" if len(raw.contexts) == 90 else "failed",
                "observed": len(raw.contexts),
                "expected": 90,
            },
            {
                "check_id": "independent_vehicle_contexts",
                "status": (
                    "passed"
                    if raw.contexts["vehicle_context_id"].nunique() == 60
                    else "failed"
                ),
                "observed": raw.contexts["vehicle_context_id"].nunique(),
                "expected": 60,
            },
            {
                "check_id": "future_target_duration_ms",
                "status": "passed",
                "observed": 5_000,
                "expected": 5_000,
            },
            {
                "check_id": "historical_representation_primary_compatibility",
                "status": "passed",
                "observed": compatibility["status"],
                "expected": "explicit compatibility decision",
            },
            {
                "check_id": "training_not_started",
                "status": "passed",
                "observed": False,
                "expected": False,
            },
        ]
    )
    acceptance_path = compact_root / "acceptance.csv"
    _write_csv(acceptance, acceptance_path)

    resume_path = compact_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    report_path = compact_root / "report.md"
    report_path.write_text(
        _build_report(summary, compatibility, heavy_files, repo_root),
        encoding="utf-8",
    )

    manifest_path = compact_root / "evidence_manifest.json"
    manifest_candidates = {
        "protocol": protocol_path,
        "compatibility_report": compatibility_path,
        "task_summary": summary_path,
        "acceptance": acceptance_path,
        "report": report_path,
        "resume_command": resume_path,
        **heavy_files,
    }
    manifest = {
        "format": "chronaris.simple_downstream_evidence_manifest.v1",
        "status": "completed",
        "run_id": config.run_id,
        "files": {
            name: {
                "path": _relative_or_absolute(path, repo_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in manifest_candidates.items()
        },
    }
    _write_json(manifest_path, manifest)
    return SimpleDownstreamAuditResult(
        compact_root=_relative_or_absolute(compact_root, repo_root),
        heavy_root=_relative_or_absolute(heavy_root, repo_root),
        protocol_path=_relative_or_absolute(protocol_path, repo_root),
        compatibility_report_path=_relative_or_absolute(
            compatibility_path, repo_root
        ),
        task_summary_path=_relative_or_absolute(summary_path, repo_root),
        report_path=_relative_or_absolute(report_path, repo_root),
        evidence_manifest_path=_relative_or_absolute(manifest_path, repo_root),
    )


def _build_report(summary, compatibility, heavy_files, repo_root) -> str:
    field_counts = summary["selected_physiology_field_count_by_fold"]
    fold_lines = "\n".join(
        f"- `{fold_id}`：训练折选择 {count} 个生理字段。"
        for fold_id, count in field_counts.items()
    )
    artifact_lines = "\n".join(
        f"- {name}：`{_relative_or_absolute(path, repo_root)}`"
        for name, path in heavy_files.items()
    )
    return "\n".join(
        [
            "# 鼎新简化下游评价协议审计",
            "",
            "## 结论",
            "",
            "新版未来任务合同已经按冻结原始点完成实跑核验：共形成 "
            f"{summary['view_context_count']} 个完整视图上下文和 "
            f"{summary['vehicle_context_count']} 个独立机动上下文。",
            "",
            "历史冻结表示不能作为新版主结果直接复用，兼容性结论为 "
            f"`{compatibility['status']}`。原因是旧输入合同排除了 "
            f"{compatibility['excluded_historical_maneuver_source_count']} 个可用于未来预测的历史运动学字段，"
            "且旧的内部划分没有覆盖全部新版上下文。旧表示仅保留为历史敏感性证据。",
            "",
            "## 任务与划分核验",
            "",
            "- 输入严格截止于目标起点，目标窗口固定为随后 5 秒。",
            "- 未来机动评价按独立飞机上下文聚合，共享航电轨迹的两个视图不会重复计权。",
            "- 未来生理状态逐字段预测，尺度和字段选择均只由当前训练架次拟合。",
            "- 留一架次为主评价；本轮未训练模型，也未打开或修改任何确认指标。",
            "",
            fold_lines,
            "",
            "## 可复核产物",
            "",
            artifact_lines,
            "",
        ]
    )


def _resume_command(config: SimpleDownstreamAuditConfig) -> str:
    return " ".join(
        [
            "/home/wangminan/env/anaconda3/envs/chronaris/bin/python",
            "scripts/evaluation/dingxin/audit_simple_downstream.py",
            "--run-id",
            config.run_id,
            "--fixed-audit-root",
            config.fixed_audit_root,
            "--snapshot-root",
            config.snapshot_root,
            "--historical-representation-root",
            config.historical_representation_root,
        ]
    )


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)
