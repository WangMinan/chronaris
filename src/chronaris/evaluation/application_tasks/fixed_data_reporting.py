"""Artifact and report writers for the G1 fixed-data audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    FieldRoleRecord,
    FoldTaskLabelResult,
    OuterFoldDefinition,
)


def write_fixed_data_audit_outputs(
    *,
    config,
    run_root: Path,
    records: pd.DataFrame,
    contexts: Sequence[ApplicationContextRecord],
    field_roles: Sequence[FieldRoleRecord],
    folds: Sequence[OuterFoldDefinition],
    fold_results: Sequence[FoldTaskLabelResult],
    metadata_rows: Sequence[Mapping[str, object]],
    metadata_errors: Sequence[Mapping[str, object]],
    source_hashes: Mapping[str, str],
    status: str,
) -> dict[str, str]:
    """Write the compact, reviewable G1 artifact root."""

    run_root.mkdir(parents=True, exist_ok=True)
    context_path = run_root / "context_sample_manifest.jsonl"
    _write_jsonl(context_path, (context.to_dict() for context in contexts))

    field_role_path = run_root / "field_role_manifest.csv"
    pd.DataFrame(role.to_dict() for role in field_roles).to_csv(field_role_path, index=False)

    sampling_rows = _sampling_rows(records)
    sampling_path = run_root / "sampling_interval_summary.csv"
    pd.DataFrame(sampling_rows).to_csv(sampling_path, index=False)
    missingness_path = run_root / "missingness_summary.csv"
    pd.DataFrame(_missingness_rows(sampling_rows)).to_csv(missingness_path, index=False)

    metadata_path = run_root / "vehicle_metadata_source.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    metadata_errors_path = run_root / "vehicle_metadata_errors.json"
    _write_json(metadata_errors_path, {"errors": list(metadata_errors)})

    label_field_path = run_root / "label_field_manifest.json"
    maneuver_roles = [role for role in field_roles if role.selected_for_maneuver_label]
    response_roles = [role for role in field_roles if role.selected_for_response_target]
    _write_json(
        label_field_path,
        {
            "maneuver_label_fields": [role.to_dict() for role in maneuver_roles],
            "physiology_response_fields": [role.to_dict() for role in response_roles],
            "selection_boundary": (
                "vehicle labels require own-aircraft semantics and exclude target/quality fields; "
                "physiology targets are EEG/SpO2 name-derived candidates"
            ),
        },
    )

    split_path = run_root / "split_manifest.json"
    _write_json(
        split_path,
        {
            "split_protocols": [fold.to_dict() for fold in folds],
            "fit_boundary": "all label thresholds and scales are fit on outer-train context ids only",
        },
    )
    label_rows = [dict(row) for result in fold_results for row in result.label_rows]
    threshold_rows = [dict(row) for result in fold_results for row in result.threshold_rows]
    labels_path = run_root / "fold_task_labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    thresholds_path = run_root / "fold_label_thresholds.csv"
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)

    overlap_rows = _overlap_rows(field_roles)
    overlap_path = run_root / "label_feature_overlap_audit.csv"
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)

    classification_count = sum(context.classification_eligible for context in contexts)
    response_count = sum(context.response_eligible for context in contexts)
    response_field_names = sorted({role.feature_name for role in response_roles})
    fold_status = {result.fold_id: result.status for result in fold_results}
    warnings = [warning for result in fold_results for warning in result.warnings]
    data_manifest = {
        "run_id": config.run_id,
        "status": status,
        "training_invoked": False,
        "confirmed_metrics_changed": False,
        "data_role": "dingxin_existing_real_dual_stream_weak_supervision",
        "data_mode": "feature_export_window_summary",
        "source_paths": {
            "e_run_manifest": config.e_run_manifest_path,
            "f_run_manifest": config.f_run_manifest_path,
        },
        "source_hashes": dict(source_hashes),
        "record_count": int(len(records)),
        "sortie_count": int(records["sortie_id"].nunique()),
        "view_count": int(records["view_id"].nunique()),
        "pilot_count": int(records["pilot_id"].nunique()),
        "records_by_sortie": records["sortie_id"].astype(str).value_counts().sort_index().to_dict(),
        "records_by_view": records["view_id"].astype(str).value_counts().sort_index().to_dict(),
        "context_candidate_count": len(contexts),
        "classification_context_count": int(classification_count),
        "response_context_count": int(response_count),
        "expected_classification_context_count": 96,
        "expected_response_context_count": 93,
        "selected_maneuver_field_count": len(maneuver_roles),
        "selected_response_candidate_role_count": len(response_roles),
        "selected_response_candidate_field_count": len(response_field_names),
        "metadata_error_count": len(metadata_errors),
        "fold_status": fold_status,
        "warnings": warnings,
        "response_representative_statistic": "window_mean_fallback",
        "raw_snapshot_required_for_main_median_target": True,
        "post_alignment_maneuver_representation_leakage_safe": False,
        "post_alignment_boundary": (
            "existing reference projections may encode maneuver label-source fields; "
            "the final leakage-safe maneuver benchmark requires raw snapshot retraining with exclusions"
        ),
    }
    data_manifest_path = run_root / "data_manifest.json"
    _write_json(data_manifest_path, data_manifest)

    claim_boundary_path = run_root / "claim_boundary.md"
    claim_boundary_path.write_text(
        "# 论断边界\n\n"
        "- 本 run 只证明固定数据、字段语义、训练折标签和划分合同可执行。\n"
        "- 机动强度标签是弱监督构造，不是专家机动科目标注。\n"
        "- 生理响应目标当前使用窗口均值 fallback；原始点冻结后主实验改用窗口中位数。\n"
        "- 现有对齐后投影可能已编码机动标签源字段，不能直接作为防泄漏分类主结果。\n"
        "- 本 run 未训练模型、未修改既有确认指标。\n",
        encoding="utf-8",
    )

    resume_path = run_root / "resume_command.txt"
    resume_path.write_text(_resume_command(config) + "\n", encoding="utf-8")
    report_path = run_root / "report.md"
    report_path.write_text(
        _render_report(
            data_manifest=data_manifest,
            maneuver_roles=maneuver_roles,
            response_roles=response_roles,
            label_rows=label_rows,
            fold_results=fold_results,
        ),
        encoding="utf-8",
    )

    output_paths = {
        "data_manifest": str(data_manifest_path),
        "field_role_manifest": str(field_role_path),
        "sampling_interval_summary": str(sampling_path),
        "missingness_summary": str(missingness_path),
        "context_sample_manifest": str(context_path),
        "label_field_manifest": str(label_field_path),
        "fold_label_thresholds": str(thresholds_path),
        "fold_task_labels": str(labels_path),
        "label_feature_overlap_audit": str(overlap_path),
        "split_manifest": str(split_path),
        "vehicle_metadata_source": str(metadata_path),
        "vehicle_metadata_errors": str(metadata_errors_path),
        "claim_boundary": str(claim_boundary_path),
        "resume_command": str(resume_path),
        "report": str(report_path),
        "progress": str(run_root / "progress.json"),
        "run_log": str(run_root / "run.log"),
    }
    evidence_manifest_path = run_root / "evidence_manifest.json"
    _write_json(
        evidence_manifest_path,
        {
            "run_id": config.run_id,
            "status": status,
            "evidence_layer": "dingxin_fixed_data_task_contract_audit",
            "training_invoked": False,
            "confirmed_metrics_changed": False,
            "source_paths": [config.e_run_manifest_path, config.f_run_manifest_path],
            "output_paths": output_paths,
            "claim_boundary_path": str(claim_boundary_path),
        },
    )
    return {
        "report_path": str(report_path),
        "evidence_manifest_path": str(evidence_manifest_path),
    }


def _sampling_rows(records: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for view_id, view_frame in records.groupby("view_id", sort=True):
        for stream_kind, stats_column in (
            ("physiology", "raw_physiology_stats"),
            ("vehicle", "raw_vehicle_stats"),
        ):
            by_feature: dict[str, dict[str, int]] = {}
            for record in view_frame.itertuples(index=False):
                stats = getattr(record, stats_column)
                feature_map = stats.get("features", {}) if isinstance(stats, Mapping) else {}
                for feature_name, payload in feature_map.items():
                    aggregate = by_feature.setdefault(
                        str(feature_name),
                        {"observed_window_count": 0, "observed_point_count": 0},
                    )
                    if not isinstance(payload, Mapping):
                        continue
                    count = int(payload.get("count") or 0)
                    if count > 0:
                        aggregate["observed_window_count"] += 1
                        aggregate["observed_point_count"] += count
            for feature_name, aggregate in sorted(by_feature.items()):
                observed_windows = aggregate["observed_window_count"]
                observed_points = aggregate["observed_point_count"]
                rows.append(
                    {
                        "sortie_id": str(view_frame.iloc[0]["sortie_id"]),
                        "view_id": str(view_id),
                        "stream_kind": stream_kind,
                        "feature_name": feature_name,
                        "total_window_count": int(len(view_frame)),
                        "observed_window_count": observed_windows,
                        "observed_point_count": observed_points,
                        "mean_points_per_observed_window": (
                            observed_points / observed_windows if observed_windows else None
                        ),
                        "approx_sampling_interval_s": (
                            5.0 * observed_windows / observed_points if observed_points else None
                        ),
                    }
                )
    return rows


def _missingness_rows(sampling_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "sortie_id": row["sortie_id"],
            "view_id": row["view_id"],
            "stream_kind": row["stream_kind"],
            "feature_name": row["feature_name"],
            "total_window_count": row["total_window_count"],
            "missing_window_count": int(row["total_window_count"]) - int(row["observed_window_count"]),
            "missing_window_ratio": (
                1.0 - int(row["observed_window_count"]) / int(row["total_window_count"])
                if int(row["total_window_count"]) else None
            ),
        }
        for row in sampling_rows
    ]


def _overlap_rows(field_roles: Sequence[FieldRoleRecord]) -> list[dict[str, object]]:
    return [
        {
            "sortie_id": role.sortie_id,
            "feature_name": role.feature_name,
            "display_label": role.display_label,
            "semantic_key": role.semantic_key,
            "is_maneuver_label_source": role.selected_for_maneuver_label,
            "allowed_in_maneuver_raw_input": role.allowed_in_maneuver_input,
            "raw_overlap_detected": (
                role.selected_for_maneuver_label and role.allowed_in_maneuver_input
            ),
            "existing_projection_contamination_risk": role.selected_for_maneuver_label,
            "exclusion_reason": role.exclusion_reason,
        }
        for role in field_roles
        if role.stream_kind == "vehicle"
    ]


def _render_report(
    *,
    data_manifest: Mapping[str, object],
    maneuver_roles: Sequence[FieldRoleRecord],
    response_roles: Sequence[FieldRoleRecord],
    label_rows: Sequence[Mapping[str, object]],
    fold_results: Sequence[FoldTaskLabelResult],
) -> str:
    response_field_names = sorted({role.feature_name for role in response_roles})
    lines = [
        "# 固定数据与下游任务审计报告",
        "",
        "## 结论",
        "",
        f"- run 状态：`{data_manifest['status']}`。",
        f"- 现有窗口：`{data_manifest['record_count']}`；sortie：`{data_manifest['sortie_count']}`；view：`{data_manifest['view_count']}`。",
        f"- 30 秒机动分类上下文：`{data_manifest['classification_context_count']}`；未来生理响应上下文：`{data_manifest['response_context_count']}`。",
        f"- 机动标签源字段：`{len(maneuver_roles)}`；生理响应候选字段：`{len(response_field_names)}`。",
        "- 标签阈值、稳健尺度和高响应界限均按外层训练折拟合。",
        "- 现有对齐后投影可能包含机动标签源字段，因此不能直接升级为防泄漏分类主结果。",
        "",
        "## 机动标签字段",
        "",
        "| sortie | 语义键 | 字段 | 中文标签 | 覆盖窗口 |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for role in maneuver_roles:
        lines.append(
            f"| `{role.sortie_id}` | `{role.semantic_key}` | `{role.feature_name}` | "
            f"{role.display_label or '未解析'} | {role.observed_window_count}/{role.total_window_count} |"
        )
    lines.extend(
        [
            "",
            "## 生理响应字段",
            "",
            "当前审计使用窗口均值构造相邻窗口变化；原始点冻结后，正式主实验按规格切换为窗口中位数。",
            "",
            "| 字段 | 类别 | 覆盖窗口（跨 sortie 角色行） |",
            "| --- | --- | ---: |",
        ]
    )
    for feature_name in sorted({role.feature_name for role in response_roles}):
        matching = [role for role in response_roles if role.feature_name == feature_name]
        lines.append(
            f"| `{feature_name}` | `{matching[0].semantic_category}` | "
            f"{sum(role.observed_window_count for role in matching)} |"
        )
    lines.extend(["", "## Fold 状态", "", "| fold | split | 状态 | warnings |", "| --- | --- | --- | --- |"])
    for result in fold_results:
        lines.append(
            f"| `{result.fold_id}` | `{result.split_strategy}` | `{result.status}` | "
            f"{'；'.join(result.warnings) if result.warnings else '-'} |"
        )
    lines.extend(["", "## 训练折分类分布", "", "| fold | low | medium | high |", "| --- | ---: | ---: | ---: |"])
    for result in fold_results:
        distribution = Counter(
            row.get("class_label")
            for row in result.label_rows
            if row.get("task_id") == "dingxin_maneuver_intensity_weak_classification_v2"
            and row.get("split_role") == "train"
            and row.get("class_label") is not None
        )
        lines.append(
            f"| `{result.fold_id}` | {distribution.get('low', 0)} | "
            f"{distribution.get('medium', 0)} | {distribution.get('high', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 按 G2a 冻结同两个架次的原始异步点。",
            "2. 使用本 run 的字段排除清单重新构造六方法原始输入。",
            "3. 保持本 run 的 split 和训练折参数拟合边界不变。",
            "",
        ]
    )
    return "\n".join(lines)


def _resume_command(config) -> str:
    return (
        "/home/wangminan/env/anaconda3/envs/chronaris/bin/python "
        "scripts/evaluation/application_tasks/audit_fixed_data.py "
        f"--run-id {config.run_id} --output-root {config.output_root} "
        f"--e-run-manifest {config.e_run_manifest_path} "
        f"--f-run-manifest {config.f_run_manifest_path} "
        f"--bus-access-rule-id {config.bus_access_rule_id}"
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
