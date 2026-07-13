"""Three-fold, outer-closed development screen for core-task recovery."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from chronaris.evaluation.application_tasks.core_recovery_protocol import (
    CoreRecoveryCandidate,
    build_locked_candidate_registry,
)
from chronaris.evaluation.application_tasks.core_recovery_consumer_screen import (
    evaluate_fitted_residual_minirocket,
    evaluate_frozen_minirocket_sequences,
    evaluate_task_routed_consumer_grid,
)
from chronaris.evaluation.application_tasks.core_recovery_run import (
    _validate_source_checkpoint,
)
from chronaris.evaluation.application_tasks.core_recovery_target_data import (
    build_task_aware_target_bundle,
    fit_core_recovery_split_targets,
    load_core_recovery_target_source,
)
from chronaris.evaluation.application_tasks.core_recovery_training import (
    CoreRecoveryTrainingConfig,
    build_core_recovery_method_model,
    train_core_recovery_method,
)
from chronaris.evaluation.application_tasks.dingxin_fold_pretraining_data import (
    load_dingxin_fold_pretraining_data,
)


MAIN_FOLD_IDS = (
    "leave_one_view_out__fold01",
    "leave_one_view_out__fold02",
    "leave_one_view_out__fold03",
)


@dataclass(frozen=True, slots=True)
class CoreRecoveryDevelopmentConfig:
    run_id: str = "2026-07-13_chronaris-core-task-recovery-development"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    source_checkpoint_root: str = (
        "artifacts/application_evaluation/2026-07-12_dingxin-locked-pretraining-coalesced/"
        "checkpoints"
    )
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    e_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    )
    f_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    )
    task_decision_path: str = (
        "docs/artifacts/runs/2026-07-13_dingxin-core-task-audit/task_decision.json"
    )
    method_names: tuple[str, ...] = ("chronaris",)
    candidate_ids: tuple[str, ...] = (
        "chronaris-c01",
        "chronaris-c02",
        "chronaris-c03",
        "chronaris-c04",
    )
    fold_ids: tuple[str, ...] = MAIN_FOLD_IDS
    frozen_backbone_steps: int = 60
    partial_unfreeze_steps: int = 60
    validation_interval_steps: int = 10
    patience_evaluations: int = 4
    seed: int = 17
    device: str = "cuda"
    consumer_kernel_grid: tuple[int, ...] = (1_000, 2_500, 5_000, 10_000)


@dataclass(frozen=True, slots=True)
class CoreRecoveryDevelopmentResult:
    run_id: str
    status: str
    completed_run_count: int
    unavailable_run_count: int
    failed_run_count: int
    selected_candidate_id: str | None
    development_gate_passed: bool
    report_path: str
    result_path: str


def run_core_recovery_development(
    config: CoreRecoveryDevelopmentConfig,
) -> CoreRecoveryDevelopmentResult:
    decision = json.loads(Path(config.task_decision_path).read_text(encoding="utf-8"))
    if decision.get("decision_status") != "locked":
        raise ValueError("core-recovery task definition is not locked")
    target_mode = str(decision["primary_maneuver_target"])
    candidates = _resolve_candidates(config)
    source = load_core_recovery_target_source(
        fixed_audit_root=config.fixed_audit_root,
        snapshot_root=config.snapshot_root,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
    rows: list[dict[str, object]] = []
    for fold_id in config.fold_ids:
        data = load_dingxin_fold_pretraining_data(
            fold_id=fold_id,
            snapshot_root=config.snapshot_root,
            fixed_audit_root=config.fixed_audit_root,
            inner_split_root=config.inner_split_root,
        )
        train_ids = data.fold.train_sample_ids
        validation_ids = data.fold.validation_sample_ids
        batch = data.load_batch(train_ids + validation_ids)
        fitted_targets = fit_core_recovery_split_targets(
            source,
            train_context_ids=train_ids,
            evaluation_context_ids=validation_ids,
            maneuver_target_mode=target_mode,
        )
        targets, field_names = build_task_aware_target_bundle(
            source,
            fitted_targets,
            sample_ids=batch.sample_ids,
        )
        for candidate in candidates:
            if candidate.train_stride_s != 5.0:
                rows.append(
                    _unavailable_row(
                        candidate,
                        fold_id=fold_id,
                        reason="dense_inner_train_contexts_not_materialized",
                    )
                )
                continue
            source_checkpoint = _source_checkpoint_path(
                config,
                fold_id=fold_id,
                method_name=candidate.method_name,
            )
            try:
                model, source_payload = build_core_recovery_method_model(
                    source_checkpoint_path=source_checkpoint,
                    batch=batch,
                    train_sample_ids=train_ids,
                    validation_sample_ids=validation_ids,
                    physiology_target_count=len(field_names),
                    device=config.device,
                )
                _validate_source_checkpoint(
                    source_payload,
                    fold_id=fold_id,
                    expected_method=candidate.method_name,
                )
                model.eval()
                with torch.inference_mode():
                    frozen_encoding = model.encode_frozen(batch)
                consumer_rows = tuple(
                    row
                    for n_kernels in config.consumer_kernel_grid
                    for row in evaluate_frozen_minirocket_sequences(
                        encoding=frozen_encoding,
                        targets=targets,
                        train_sample_ids=train_ids,
                        validation_sample_ids=validation_ids,
                        n_kernels=n_kernels,
                        random_state=config.seed,
                    )
                )
                consumer_grid_rows = evaluate_task_routed_consumer_grid(
                    encoding=frozen_encoding,
                    targets=targets,
                    train_sample_ids=train_ids,
                    validation_sample_ids=validation_ids,
                    kernel_grid=config.consumer_kernel_grid,
                    random_state=config.seed,
                )
                naive_consumer_grid_rows = ()
                if candidate.method_name == "chronaris":
                    naive_encoding = type(frozen_encoding)(
                        sample_ids=frozen_encoding.sample_ids,
                        valid_mask=frozen_encoding.valid_mask,
                        sequence_embedding=frozen_encoding.observed_sequence,
                    )
                    naive_consumer_grid_rows = evaluate_task_routed_consumer_grid(
                        encoding=naive_encoding,
                        targets=targets,
                        train_sample_ids=train_ids,
                        validation_sample_ids=validation_ids,
                        kernel_grid=config.consumer_kernel_grid,
                        random_state=config.seed,
                    )
                training_config = CoreRecoveryTrainingConfig(
                    backbone_learning_rate_ratio=candidate.backbone_lr_ratio,
                    high_response_weight=candidate.high_response_weight,
                    frozen_backbone_steps=config.frozen_backbone_steps,
                    partial_unfreeze_steps=config.partial_unfreeze_steps,
                    validation_interval_steps=config.validation_interval_steps,
                    patience_evaluations=config.patience_evaluations,
                    seed=config.seed,
                    device=config.device,
                )
                heavy_root = (
                    Path(config.heavy_output_root)
                    / config.run_id
                    / "checkpoints"
                    / f"seed_{config.seed}"
                    / fold_id
                )
                result = train_core_recovery_method(
                    model=model,
                    batch=batch,
                    targets=targets,
                    train_sample_ids=train_ids,
                    validation_sample_ids=validation_ids,
                    source_checkpoint_path=source_checkpoint,
                    output_root=heavy_root,
                    candidate_id=candidate.candidate_id,
                    config=training_config,
                    precomputed_frozen_encoding=frozen_encoding,
                )
                adapted_consumer = (
                    tuple(
                        evaluate_fitted_residual_minirocket(
                        model=model,
                        encoding=frozen_encoding,
                        targets=targets,
                        train_sample_ids=train_ids,
                        validation_sample_ids=validation_ids,
                        n_kernels=n_kernels,
                        random_state=config.seed,
                        )
                        for n_kernels in config.consumer_kernel_grid
                    )
                    if candidate.method_name == "chronaris"
                    and config.partial_unfreeze_steps == 0
                    else None
                )
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "method_name": candidate.method_name,
                        "fold_id": fold_id,
                        "status": result.status,
                        "unavailable_reason": None,
                        "train_stride_s": candidate.train_stride_s,
                        "backbone_lr_ratio": candidate.backbone_lr_ratio,
                        "high_response_weight": candidate.high_response_weight,
                        "completed_steps": result.completed_steps,
                        "best_step": result.best_step,
                        "training_elapsed_s": result.training_elapsed_s,
                        "checkpoint_path": result.best_checkpoint_path,
                        **dict(result.best_validation_metrics),
                        "frozen_consumer_rows": consumer_rows,
                        "consumer_grid_rows": consumer_grid_rows,
                        "naive_consumer_grid_rows": naive_consumer_grid_rows,
                        "adapted_consumer_rows": adapted_consumer,
                    }
                )
            except Exception as error:  # noqa: BLE001 - retain a complete candidate ledger
                rows.append(
                    {
                        **_unavailable_row(
                            candidate,
                            fold_id=fold_id,
                            reason=f"{type(error).__name__}: {error}",
                        ),
                        "status": "failed",
                    }
                )
    aggregates = _aggregate_candidates(rows, required_fold_count=len(config.fold_ids))
    consumer_summary = _aggregate_consumer_grid(
        rows,
        required_fold_count=len(config.fold_ids),
    )
    selected = _select_candidate(aggregates)
    selected_route = _select_consumer_route(consumer_summary)
    compact_root = Path(config.compact_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    _write_csv(compact_root / "fold_metrics.csv", rows)
    _write_csv(compact_root / "candidate_summary.csv", aggregates)
    _write_csv(compact_root / "consumer_route_summary.csv", consumer_summary)
    result_path = compact_root / "development_result.json"
    payload = {
        "format": "chronaris.core_task_recovery_development.v1",
        "run_id": config.run_id,
        "status": "completed",
        "config": asdict(config),
        "task_definition": target_mode,
        "fold_rows": rows,
        "candidate_summary": aggregates,
        "consumer_route_summary": consumer_summary,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "selected_consumer_route": selected_route,
        "development_gate_passed": selected_route is not None,
        "outer_test_accessed": False,
        "confirmed_metrics_changed": False,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = compact_root / "report.md"
    _write_report(report_path, payload)
    completed = sum(row["status"] == "completed" for row in rows)
    unavailable = sum(row["status"] == "unavailable" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    return CoreRecoveryDevelopmentResult(
        run_id=config.run_id,
        status="completed",
        completed_run_count=completed,
        unavailable_run_count=unavailable,
        failed_run_count=failed,
        selected_candidate_id=None if selected is None else str(selected["candidate_id"]),
        development_gate_passed=selected_route is not None,
        report_path=str(report_path),
        result_path=str(result_path),
    )


def _resolve_candidates(config: CoreRecoveryDevelopmentConfig) -> tuple[CoreRecoveryCandidate, ...]:
    registry = {row.candidate_id: row for row in build_locked_candidate_registry()}
    unknown = sorted(set(config.candidate_ids) - set(registry))
    if unknown:
        raise ValueError(f"unknown core-recovery candidates: {unknown}")
    candidates = tuple(registry[value] for value in config.candidate_ids)
    if any(row.method_name not in config.method_names for row in candidates):
        raise ValueError("candidate selection and method selection disagree")
    return candidates


def _source_checkpoint_path(config, *, fold_id: str, method_name: str) -> Path:
    method_root = Path(config.source_checkpoint_root) / f"seed_{config.seed}" / fold_id / method_name
    matches = sorted(method_root.rglob("best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one source checkpoint under {method_root}, found {len(matches)}"
        )
    return matches[0]


def _unavailable_row(candidate, *, fold_id: str, reason: str) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "method_name": candidate.method_name,
        "fold_id": fold_id,
        "status": "unavailable",
        "unavailable_reason": reason,
        "train_stride_s": candidate.train_stride_s,
        "backbone_lr_ratio": candidate.backbone_lr_ratio,
        "high_response_weight": candidate.high_response_weight,
        "completed_steps": 0,
        "best_step": 0,
        "training_elapsed_s": 0.0,
        "checkpoint_path": None,
    }


def _aggregate_candidates(rows, *, required_fold_count: int) -> list[dict[str, object]]:
    aggregates = []
    candidate_ids = sorted({str(row["candidate_id"]) for row in rows})
    for candidate_id in candidate_ids:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        completed = [row for row in candidate_rows if row["status"] == "completed"]
        base = {
            "candidate_id": candidate_id,
            "method_name": candidate_rows[0]["method_name"],
            "completed_fold_count": len(completed),
            "required_fold_count": required_fold_count,
            "all_folds_completed": len(completed) == required_fold_count,
        }
        if len(completed) != required_fold_count:
            aggregates.append({**base, "development_gate_passed": False})
            continue
        for metric in (
            "maneuver_macro_f1",
            "response_rmse",
            "high_response_auprc",
            "direct_maneuver_macro_f1",
            "direct_response_rmse",
            "direct_high_response_auprc",
        ):
            values = [float(row[metric]) for row in completed if row.get(metric) is not None]
            base[f"mean_{metric}"] = statistics.fmean(values) if values else None
            base[f"worst_{metric}"] = (
                (min(values) if "f1" in metric or "auprc" in metric else max(values))
                if values
                else None
            )
        base["all_paired_no_harm_passed"] = all(
            bool(row["paired_no_harm_passed"]) for row in completed
        )
        base["development_gate_passed"] = bool(
            base["mean_maneuver_macro_f1"] >= 0.85
            and base["mean_response_rmse"] <= 0.315
            and base["mean_high_response_auprc"] >= 0.875
            and base["all_paired_no_harm_passed"]
        )
        base["selection_score"] = float(
            1
            - base["mean_maneuver_macro_f1"]
            + base["mean_response_rmse"]
            + 1
            - base["mean_high_response_auprc"]
        )
        aggregates.append(base)
    return aggregates


def _select_candidate(aggregates):
    passing = [row for row in aggregates if row.get("development_gate_passed")]
    return min(passing, key=lambda row: (row["selection_score"], row["candidate_id"])) if passing else None


def _aggregate_consumer_grid(rows, *, required_fold_count: int):
    flat = []
    for row in rows:
        if row["status"] != "completed":
            continue
        for item in row.get("consumer_grid_rows", ()):
            flat.append(
                {
                    "method_name": row["method_name"],
                    "candidate_id": row["candidate_id"],
                    "fold_id": row["fold_id"],
                    **item,
                }
            )
        for item in row.get("naive_consumer_grid_rows", ()):
            flat.append(
                {
                    "method_name": "naive_time_sync",
                    "candidate_id": "naive_time_sync-head-only",
                    "fold_id": row["fold_id"],
                    **item,
                }
            )
    summary = []
    keys = sorted(
        {
            (
                row["method_name"],
                row["task"],
                row["sequence_mode"],
                row["n_kernels"],
                row["head_config_id"],
            )
            for row in flat
        }
    )
    for method_name, task, sequence_mode, n_kernels, head_config_id in keys:
        group = [
            row
            for row in flat
            if row["method_name"] == method_name
            and row["task"] == task
            and row["sequence_mode"] == sequence_mode
            and row["n_kernels"] == n_kernels
            and row["head_config_id"] == head_config_id
        ]
        item = {
            "method_name": method_name,
            "task": task,
            "sequence_mode": sequence_mode,
            "n_kernels": n_kernels,
            "head_config_id": head_config_id,
            "completed_fold_count": len(group),
            "required_fold_count": required_fold_count,
        }
        if len(group) != required_fold_count:
            item["development_gate_passed"] = False
            summary.append(item)
            continue
        values = [float(row["value"]) for row in group if row["value"] is not None]
        item["mean_value"] = statistics.fmean(values) if values else None
        item["worst_value"] = (
            (min(values) if group[0]["direction"] == "higher" else max(values))
            if values
            else None
        )
        item["direction"] = group[0]["direction"]
        item["development_gate_passed"] = len(values) == required_fold_count
        summary.append(item)
    return summary


def _select_consumer_route(summary):
    selected = {}
    for task in ("maneuver", "response", "high_response"):
        candidates = [
            row
            for row in summary
            if row["method_name"] == "chronaris"
            and row["task"] == task
            and row.get("development_gate_passed")
        ]
        if not candidates:
            return None
        reverse = candidates[0]["direction"] == "higher"
        selected[task] = sorted(
            candidates,
            key=lambda row: (
                -row["mean_value"] if reverse else row["mean_value"],
                row["n_kernels"],
                row["head_config_id"],
            ),
        )[0]
    passed = bool(
        selected["maneuver"]["mean_value"] >= 0.70
        and selected["response"]["mean_value"] <= 0.88
        and selected["high_response"]["mean_value"] >= 0.70
    )
    return {"passed": passed, "tasks": selected} if passed else None


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, payload: dict[str, object]) -> None:
    route = payload.get("selected_consumer_route")
    selected = route["tasks"] if route else {}
    labels = {
        "maneuver": "鼎新机动强度分类宏平均 F1（Macro-F1）",
        "response": "鼎新未来生理响应预测均方根误差（RMSE）",
        "high_response": "鼎新高生理响应识别精确率—召回率曲线下面积（AUPRC）",
    }
    lines = [
        "# Chronaris 核心任务恢复开发筛选",
        "",
        "本轮只使用鼎新内层训练与验证角色，外层测试保持关闭。",
        "",
        "- 锁定机动任务：当前 5 秒机动强度分类。",
        f"- 开发门槛是否通过：`{str(payload['development_gate_passed']).lower()}`。",
        "- 历史确认指标未修改。",
        "",
    ]
    for task in ("maneuver", "response", "high_response"):
        if task in selected:
            lines.append(
                f"- {labels[task]}：{float(selected[task]['mean_value']):.6f}。"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
