"""Run-level orchestration for Chronaris core-task recovery development."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class CoreRecoverySmokeConfig:
    run_id: str = "2026-07-13_chronaris-core-task-recovery-smoke"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    fold_id: str = "leave_one_view_out__fold01"
    source_checkpoint_path: str = (
        "artifacts/application_evaluation/2026-07-12_dingxin-locked-pretraining-coalesced/"
        "checkpoints/seed_17/leave_one_view_out__fold01/chronaris/best.pt"
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
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class CoreRecoverySmokeResult:
    run_id: str
    status: str
    fold_id: str
    candidate_id: str
    completed_steps: int
    best_validation_metrics: dict[str, float | bool | None]
    report_path: str
    checkpoint_path: str


def run_core_recovery_smoke(config: CoreRecoverySmokeConfig) -> CoreRecoverySmokeResult:
    decision = json.loads(Path(config.task_decision_path).read_text(encoding="utf-8"))
    if decision.get("decision_status") != "locked":
        raise ValueError("core-recovery task definition is not locked")
    target_mode = str(decision["primary_maneuver_target"])
    data = load_dingxin_fold_pretraining_data(
        fold_id=config.fold_id,
        snapshot_root=config.snapshot_root,
        fixed_audit_root=config.fixed_audit_root,
        inner_split_root=config.inner_split_root,
    )
    train_ids = data.fold.train_sample_ids
    validation_ids = data.fold.validation_sample_ids
    batch = data.load_batch(train_ids + validation_ids)
    source = load_core_recovery_target_source(
        fixed_audit_root=config.fixed_audit_root,
        snapshot_root=config.snapshot_root,
        e_run_manifest_path=config.e_run_manifest_path,
        f_run_manifest_path=config.f_run_manifest_path,
    )
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
    model, source_payload = build_core_recovery_method_model(
        source_checkpoint_path=config.source_checkpoint_path,
        batch=batch,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        physiology_target_count=len(field_names),
        device=config.device,
    )
    _validate_source_checkpoint(source_payload, fold_id=config.fold_id)
    training_config = CoreRecoveryTrainingConfig(
        backbone_learning_rate_ratio=0.05,
        high_response_weight=0.5,
        frozen_backbone_steps=1,
        partial_unfreeze_steps=1,
        validation_interval_steps=1,
        patience_evaluations=4,
        seed=17,
        device=config.device,
    )
    heavy_root = Path(config.heavy_output_root) / config.run_id / "checkpoints"
    result = train_core_recovery_method(
        model=model,
        batch=batch,
        targets=targets,
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        source_checkpoint_path=config.source_checkpoint_path,
        output_root=heavy_root,
        candidate_id="chronaris-c01",
        config=training_config,
    )
    compact_root = Path(config.compact_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "chronaris.core_task_recovery_smoke.v1",
        "run_id": config.run_id,
        "status": result.status,
        "fold_id": config.fold_id,
        "candidate_id": "chronaris-c01",
        "task_definition": target_mode,
        "training_config": asdict(training_config),
        "selected_physiology_target_fields": list(field_names),
        "best_validation_metrics": dict(result.best_validation_metrics),
        "completed_steps": result.completed_steps,
        "best_step": result.best_step,
        "checkpoint_path": result.best_checkpoint_path,
        "source_checkpoint_path": config.source_checkpoint_path,
        "source_checkpoint_exposure_audit_passed": True,
        "outer_test_accessed": False,
        "confirmed_metrics_changed": False,
    }
    (compact_root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = compact_root / "report.md"
    report_path.write_text(
        "\n".join(
            (
                "# Chronaris 核心任务恢复训练冒烟",
                "",
                "本次实跑验证观测信息保留残差、任务感知多目标损失、冻结主干和部分解冻两个阶段能够在鼎新训练内数据上完成。",
                "",
                f"- 外层折：`{config.fold_id}`。",
                f"- 训练步数：{result.completed_steps}。",
                f"- 锁定机动任务：`{target_mode}`。",
                f"- 最佳训练内验证指标：`{json.dumps(result.best_validation_metrics, ensure_ascii=False, sort_keys=True)}`。",
                "- 外层测试未访问，历史确认指标未修改。",
                "",
            )
        ),
        encoding="utf-8",
    )
    return CoreRecoverySmokeResult(
        run_id=config.run_id,
        status=result.status,
        fold_id=config.fold_id,
        candidate_id="chronaris-c01",
        completed_steps=result.completed_steps,
        best_validation_metrics=dict(result.best_validation_metrics),
        report_path=str(report_path),
        checkpoint_path=result.best_checkpoint_path,
    )


def _validate_source_checkpoint(payload, *, fold_id, expected_method="chronaris"):
    if str(payload["method_name"]) != expected_method:
        raise ValueError("core-recovery source checkpoint method does not match")
    if bool(payload.get("label_used_for_encoder_training")):
        raise ValueError("source checkpoint already used task labels")
    if not bool(payload.get("selection_uses_public_pretext_only")):
        raise ValueError("source checkpoint selection exposure is not task-independent")
    if str(payload["fold"]["fold_id"]) != fold_id:
        raise ValueError("source checkpoint fold does not match the smoke fold")
