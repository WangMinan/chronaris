"""Rotation-field audit for task evaluation rigid-body diagnostics."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chronaris.access import (
    MySQLCliRunner,
    MySQLFlightTaskReader,
    MySQLRealBusContextReader,
    MySQLSettings,
)
from chronaris.models.alignment.physics_state_mapping import (
    build_rigid_body_mapping_diagnostics,
)
from chronaris.modeling.common.run_observer import (
    StageIRunProgress,
    open_task_eval_run_observer,
)
from chronaris.schema.models import SortieLocator

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

DEFAULT_ARTIFACT_ROOT = "docs/artifacts/runs"
DEFAULT_REPORT_ROOT = "docs/artifacts/runs"
DEFAULT_RIGID_BODY_SUMMARY_PATH = (
    "docs/artifacts/runs/2026-06-07_rigid-body-diagnostics/"
    "rigid_body_ablation_summary.json"
)


@dataclass(frozen=True, slots=True)
class StageIRigidBodyRotationAuditConfig:
    """Configuration for one rotation audit run."""

    run_id: str
    rigid_body_summary_path: str = DEFAULT_RIGID_BODY_SUMMARY_PATH
    output_root: str = DEFAULT_ARTIFACT_ROOT
    report_root: str = DEFAULT_REPORT_ROOT
    sortie_id: str = "20251005_四01_ACT-4_云_J20_22#01"
    mysql_database: str = "rjgx_backend"
    vehicle_measurement: str = "BUS6000019110020"
    bus_access_rule_id: int = 6000019510066
    bus_analysis_id: int = 6000019110020
    strict_mysql_field_labels: bool = False


@dataclass(frozen=True, slots=True)
class StageIRigidBodyRotationAuditRunResult:
    """Artifacts written by one rotation audit run."""

    run_id: str
    artifact_root: str
    summary_path: str
    report_path: str
    summary: Mapping[str, object]


def run_task_eval_rigid_body_rotation_audit(
    config: StageIRigidBodyRotationAuditConfig,
) -> StageIRigidBodyRotationAuditRunResult:
    run_root = Path(config.output_root) / config.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    with open_task_eval_run_observer(
        run_root=run_root,
        run_id=config.run_id,
        stage_name="task_eval_rigid_body_rotation_audit",
        logger=LOGGER,
        initial_progress={"artifact_root": str(run_root), "sortie_id": config.sortie_id},
    ) as progress:
        return _run_task_eval_rigid_body_rotation_audit_observed(
            config=config,
            run_root=run_root,
            progress=progress,
        )


def _run_task_eval_rigid_body_rotation_audit_observed(
    *,
    config: StageIRigidBodyRotationAuditConfig,
    run_root: Path,
    progress: StageIRunProgress,
) -> StageIRigidBodyRotationAuditRunResult:
    rigid_body_summary = _load_json(config.rigid_body_summary_path)
    rigid_body_family = rigid_body_summary.get("families", {}).get("rigid_body", {})
    current_diagnostics = rigid_body_family.get("rigid_body_mapping_diagnostics", {}) or {}
    feature_labels = current_diagnostics.get("feature_labels", {}) or {}
    feature_names = tuple(sorted(feature_labels))
    refreshed_diagnostics = build_rigid_body_mapping_diagnostics(
        feature_names,
        field_labels=feature_labels,
    )
    mysql_field_labels, metadata_summary = _resolve_vehicle_field_labels(config)
    progress.update(
        "feature_labels_loaded",
        feature_count=len(feature_names),
        metadata_status=metadata_summary["status"],
    )

    metadata_matches = _group_rotation_candidates(mysql_field_labels)
    feature_matches = _group_rotation_candidates(feature_labels)
    can_enable_rotation = bool(refreshed_diagnostics.enabled_residuals and "rotation" in refreshed_diagnostics.enabled_residuals)
    summary = {
        "run_id": config.run_id,
        "artifact_root": str(run_root),
        "evidence_layer": "rotation_diagnostics",
        "source_rigid_body_summary_path": config.rigid_body_summary_path,
        "source_sortie_id": config.sortie_id,
        "vehicle_field_metadata": metadata_summary,
        "feature_export_feature_rotation_groups": _serialize_mapping_groups(refreshed_diagnostics.groups),
        "feature_export_missing_requirements": {
            key: list(value)
            for key, value in refreshed_diagnostics.missing_requirements.items()
        },
        "mysql_rotation_candidates": metadata_matches,
        "feature_rotation_candidates": feature_matches,
        "rotation_enabled": can_enable_rotation,
        "rotation_status": "enabled" if can_enable_rotation else "disabled",
        "rotation_reading": (
            "current sortie still lacks paired rate fields for pitch/roll/yaw"
            if not can_enable_rotation
            else "rotation can be enabled on the current sortie"
        ),
    }
    summary_path = run_root / "rigid_body_rotation_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_root = Path(config.report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"task-eval-rigid-body-rotation-audit-{config.run_id}.md"
    report_path.write_text(
        render_task_eval_rigid_body_rotation_audit_report(summary) + "\n",
        encoding="utf-8",
    )
    progress.finish(summary_path=str(summary_path), report_path=str(report_path))
    return StageIRigidBodyRotationAuditRunResult(
        run_id=config.run_id,
        artifact_root=str(run_root),
        summary_path=str(summary_path),
        report_path=str(report_path),
        summary=summary,
    )


def render_task_eval_rigid_body_rotation_audit_report(summary: Mapping[str, object]) -> str:
    lines = [
        f"# task evaluation Rigid-Body Rotation Audit - {summary['run_id']}",
        "",
        f"- evidence_layer: `{summary['evidence_layer']}`",
        f"- source_sortie_id: `{summary['source_sortie_id']}`",
        f"- vehicle_field_metadata: `{summary['vehicle_field_metadata']}`",
        f"- rotation_status: `{summary['rotation_status']}`",
        "",
        "## feature export Feature Matches",
        "",
        "| group | matched_fields |",
        "| --- | --- |",
    ]
    for group_name, rows in summary.get("feature_export_feature_rotation_groups", {}).items():
        matched = ", ".join(
            f"{row['feature_name']}::{row['label']}"
            for row in rows
        ) or "-"
        lines.append(f"| `{group_name}` | `{matched}` |")
    lines.extend(
        [
            "",
            "## MySQL Metadata Candidates",
            "",
            "| group | matched_fields |",
            "| --- | --- |",
        ]
    )
    for group_name, rows in summary.get("mysql_rotation_candidates", {}).items():
        matched = ", ".join(
            f"{row['field_name']}::{row['label']}"
            for row in rows
        ) or "-"
        lines.append(f"| `{group_name}` | `{matched}` |")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "1. 本审计只判断 `rotation` 是否具备真实启用条件，不把缺失字段包装成已验证物理约束。",
            f"2. 当前结论：`{summary['rotation_reading']}`。",
            f"3. 当前 feature export 缺口：`{summary['feature_export_missing_requirements']}`。",
        ]
    )
    return "\n".join(lines)


def _group_rotation_candidates(field_labels: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    groups = {
        "pitch": ("俯仰", "pitch"),
        "pitch_rate": ("俯仰角速度", "俯仰速率", "pitch_rate", "pitch rate"),
        "roll": ("横滚", "滚转", "roll"),
        "roll_rate": ("横滚角速度", "滚转角速度", "roll_rate", "roll rate"),
        "yaw": ("真航向", "航向", "偏航", "heading", "yaw"),
        "yaw_rate": ("航向角速度", "偏航角速度", "yaw_rate", "yaw rate", "航向速率"),
    }
    matches: dict[str, list[dict[str, str]]] = {name: [] for name in groups}
    for field_name, label in field_labels.items():
        normalized = str(label).lower()
        for group_name, tokens in groups.items():
            if any(token.lower() in normalized for token in tokens):
                matches[group_name].append({"field_name": str(field_name), "label": str(label)})
                break
    return matches


def _serialize_mapping_groups(
    groups: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    return {
        str(group_name): [
            {
                "feature_name": str(row["feature_name"]),
                "label": str(row["label"]),
            }
            for row in rows
        ]
        for group_name, rows in groups.items()
    }


def _load_json(path_like: str) -> dict[str, object]:
    return json.loads(Path(path_like).read_text(encoding="utf-8"))


def _extract_secret(md_text: str, key: str) -> str:
    pattern = re.compile(rf"^\+?\s*{re.escape(key)}:\s*(.+)$", re.MULTILINE)
    matched = pattern.search(md_text)
    if not matched:
        raise RuntimeError(f"Missing secret key in docs/SECRETS.md: {key}")
    return matched.group(1).strip()


def _resolve_mysql_settings(config: StageIRigidBodyRotationAuditConfig) -> MySQLSettings:
    host = os.environ.get("CHRONARIS_MYSQL_HOST")
    port = os.environ.get("CHRONARIS_MYSQL_PORT")
    user = os.environ.get("CHRONARIS_MYSQL_USER")
    password = os.environ.get("CHRONARIS_MYSQL_PASSWORD")
    if not (user and password):
        secrets_text = (Path.cwd() / "docs" / "SECRETS.md").read_text(encoding="utf-8")
        user = user or _extract_secret(secrets_text, "username")
        password = password or _extract_secret(secrets_text, "password")
    host = host or "127.0.0.1"
    port = port or "3306"
    return MySQLSettings(
        host=host,
        port=int(port),
        database=config.mysql_database,
        user=user,
        password_env=None,
        password_value=password,
    )


def _resolve_vehicle_field_labels(
    config: StageIRigidBodyRotationAuditConfig,
) -> tuple[dict[str, str], dict[str, object]]:
    try:
        runner = MySQLCliRunner(_resolve_mysql_settings(config))
        context_reader = MySQLRealBusContextReader(
            runner=runner,
            flight_task_reader=MySQLFlightTaskReader(runner),
        )
        context = context_reader.fetch_context(
            locator=SortieLocator(sortie_id=config.sortie_id),
            access_rule_id=config.bus_access_rule_id,
            analysis_id=config.bus_analysis_id,
        )
    except Exception as exc:  # pragma: no cover - exercised by live audit runs.
        if config.strict_mysql_field_labels:
            raise
        return {}, {"status": "unavailable", "field_count": 0, "error": str(exc)}

    measurement = context.analysis.measurement or config.vehicle_measurement
    labels: dict[str, str] = {}
    for detail in context.detail_list:
        labels[detail.col_field] = detail.col_name
        labels[f"{measurement}.{detail.col_field}"] = detail.col_name
    for structure in context.structure_list:
        labels.setdefault(structure.col_field, structure.col_name)
        labels.setdefault(f"{measurement}.{structure.col_field}", structure.col_name)
    for access_detail in context.access_rule_details:
        if access_detail.col_name:
            labels.setdefault(access_detail.col_field, access_detail.col_name)
            labels.setdefault(f"{measurement}.{access_detail.col_field}", access_detail.col_name)
    return labels, {
        "status": "loaded",
        "field_count": len(labels),
        "measurement": measurement,
        "analysis_id": context.analysis.analysis_id,
        "access_rule_id": config.bus_access_rule_id,
        "error": None,
    }
