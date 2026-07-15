"""Execute Dingxin task 1C without opening outer-test evidence."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.core_feasibility_features import (
    load_or_build_feature_cache,
)
from chronaris.evaluation.application_tasks.core_feasibility_protocol import sha256_file
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    build_dingxin_lazy_context_index,
)
from chronaris.evaluation.application_tasks.dingxin_target_data import (
    load_dingxin_target_source_data,
)
from chronaris.evaluation.application_tasks.target_reconstruction_contracts import (
    MATCHED_METHODS,
    TARGET_RECONSTRUCTION_GATES,
    decide_target_reconstruction_allowance,
)
from chronaris.evaluation.application_tasks.target_reconstruction_evaluation import (
    evaluate_matched_clean_panel,
)
from chronaris.evaluation.application_tasks.target_reconstruction_maneuver import (
    build_future_maneuver_targets,
)
from chronaris.evaluation.application_tasks.target_reconstruction_physiology import (
    build_physiology_residual_targets,
    build_robust_physiology_states,
)
from chronaris.evaluation.application_tasks.target_reconstruction_reporting import (
    write_target_reconstruction_outputs,
)
from chronaris.evaluation.application_tasks.target_reconstruction_training import (
    MatchedCleanTrainingConfig,
    train_matched_clean_representations,
)


PRESSURE_SPLIT_ID = "legacy_diagnostic__leave_one_view_out__fold03"


@dataclass(frozen=True, slots=True)
class DingxinTargetReconstructionConfig:
    run_id: str = "2026-07-15_dingxin-target-reconstruction"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    source_workspace_root: str = "/home/wangminan/projects/chronaris"
    fixed_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    stability_root: str = "docs/artifacts/runs/2026-07-14_dingxin-task-stability"
    e_manifest: str = "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    f_manifest: str = "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    snapshot_root: str = "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    seed: int = 17
    max_epochs: int = 12
    batch_size: int = 16
    patience: int = 4
    device: str = "cuda"
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinTargetReconstructionResult:
    status: str
    decision: str
    allow_safe_fusion: bool
    allow_task_aware_research: bool
    compact_run_root: str
    heavy_run_root: str
    report_path: str


def run_dingxin_target_reconstruction(
    config: DingxinTargetReconstructionConfig,
) -> DingxinTargetReconstructionResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    source_workspace = Path(config.source_workspace_root)
    fixed_root = Path(config.fixed_root)
    stability_root = Path(config.stability_root)
    snapshot_root = source_workspace / config.snapshot_root
    e_manifest = source_workspace / config.e_manifest
    f_manifest = source_workspace / config.f_manifest
    split_path = stability_root / "development_split_manifest.json"
    source_paths = {
        "fixed_data_manifest": fixed_root / "data_manifest.json",
        "context_catalog": fixed_root / "context_sample_manifest.jsonl",
        "field_roles": fixed_root / "field_role_manifest.csv",
        "task_stability_protocol": stability_root / "protocol.json",
        "task_stability_splits": split_path,
        "task_stability_allowance": stability_root / "allowance.json",
        "snapshot_manifest": snapshot_root / "snapshot_manifest.json",
        "feature_export_e_manifest": e_manifest,
        "feature_export_f_manifest": f_manifest,
    }
    missing = sorted(name for name, path in source_paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"task 1C source files missing: {missing}")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    plans = [
        row
        for row in split_manifest["folds"]
        if bool(row["main_selection"]) or str(row["fold_id"]) == PRESSURE_SPLIT_ID
    ]
    main_plans = [row for row in plans if bool(row["main_selection"])]
    pressure_plans = [row for row in plans if not bool(row["main_selection"])]
    if len(main_plans) != 6 or len(pressure_plans) != 1:
        raise ValueError("task 1C requires six main splits and one third-pool pressure split")
    training_config = MatchedCleanTrainingConfig(
        seed=config.seed,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
        patience=config.patience,
        device=config.device,
        resume=config.resume,
    )
    preregistration = _preregistration(config, training_config, split_manifest)
    _write_json(compact_root / "pre_registration.json", preregistration)
    _write_json(
        compact_root / "progress.json",
        {"status": "running", "stage": "matched_clean_label_closed_training", "outer_test_opened": False},
    )
    training, exports, access, pretraining_protocol = train_matched_clean_representations(
        config=training_config,
        plans=plans,
        heavy_root=heavy_root,
        snapshot_root=snapshot_root,
        fixed_root=fixed_root,
    )
    _write_json(
        compact_root / "progress.json",
        {"status": "running", "stage": "target_reconstruction_after_encoder_lock", "outer_test_opened": False},
    )
    contexts = pd.read_json(fixed_root / "context_sample_manifest.jsonl", lines=True)
    all_ids = tuple(
        sorted(
            {
                str(value)
                for plan in plans
                for role in ("train_sample_ids", "validation_sample_ids")
                for value in plan[role]
            }
        )
    )
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot_root,
        field_role_manifest_path=fixed_root / "field_role_manifest.csv",
        context_manifest_path=fixed_root / "context_sample_manifest.jsonl",
    )
    cache = load_or_build_feature_cache(
        index=index,
        sample_ids=all_ids,
        output_path=heavy_root / "raw_causal_query_cache.npz",
    )
    maneuver_targets, maneuver_thresholds = build_future_maneuver_targets(
        plans=plans,
        fixed_root=fixed_root,
        e_run_manifest_path=str(e_manifest),
        f_run_manifest_path=str(f_manifest),
    )
    target_source = load_dingxin_target_source_data(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
    )
    physiology_states = build_robust_physiology_states(
        target_source,
        snapshot_root=snapshot_root,
    )
    residual_targets, residual_descriptors, inertia_audit = build_physiology_residual_targets(
        plans=plans,
        states=physiology_states,
        cache=cache,
        contexts=contexts,
    )
    metrics, summary, best, predictions = evaluate_matched_clean_panel(
        plans=plans,
        representation_root=heavy_root / "representations",
        seed=config.seed,
        contexts=contexts,
        maneuver_targets=maneuver_targets,
        physiology_targets=residual_targets,
    )
    time_shortcut = _time_shortcut_frame(summary)
    third_pool = _third_pool_frame(metrics)
    panel = _panel_status(training, exports, metrics)
    observed = _observed(best, time_shortcut, third_pool)
    protocol_valid = bool(
        panel["missing_method_split_units"] == 0
        and len(main_plans) == 6
        and len(pressure_plans) == 1
        and int(pretraining_protocol["input_feature_contract"]["time_like_present_channel_count"]) == 30
        and not bool(access["forbidden_request_count"].astype(bool).any())
        and not bool(access["outer_test_opened"].astype(bool).any())
        and bool(inertia_audit["cross_fitted_train_predictions"].astype(bool).all())
        and not bool(inertia_audit["validation_future_used_for_fit"].astype(bool).any())
    )
    allowance = decide_target_reconstruction_allowance(
        panel=panel,
        time_shortcut={"standardized_gain": observed["time_shortcut_standardized_gain"]},
        maneuver_score={"median_spearman": observed["future_maneuver_score_median_spearman"]},
        maneuver_trend={
            "mean_macro_f1": observed["future_maneuver_trend_mean_macro_f1"],
            "worst_macro_f1": observed["future_maneuver_trend_worst_macro_f1"],
        },
        physiology_residual={
            "median_rmse_ratio": observed["physiology_residual_median_rmse_ratio"],
            "mean_skill": observed["physiology_residual_mean_skill"],
            "positive_skill_split_count": observed["physiology_residual_positive_skill_split_count"],
        },
        high_residual={
            "mean_normalized_ap": observed["high_residual_mean_normalized_ap"],
            "median_normalized_ap": observed["high_residual_median_normalized_ap"],
            "positive_split_count": observed["high_residual_positive_split_count"],
        },
        third_pool={"completed_continuous_task_count": observed["third_pool_continuous_task_count"]},
        protocol_valid=protocol_valid,
    )
    allowance["observed"] = observed
    allowance["gates"] = TARGET_RECONSTRUCTION_GATES
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    protocol = {
        "format": "chronaris.dingxin_target_reconstruction_protocol.v1",
        "run_id": config.run_id,
        "git_head": _git_head(),
        "baseline_commit": "c7667de7e89e9cb7c74eba1cecad712d4d0ddc7f",
        "config": asdict(config),
        "pretraining_protocol": pretraining_protocol,
        "source_sha256": source_sha,
        "main_split_count": len(main_plans),
        "pressure_split_count": len(pressure_plans),
        "matched_method_count": len(MATCHED_METHODS),
        "panel_status": panel,
        "protocol_valid": protocol_valid,
        "outer_test_opened": False,
        "held_out_predictions_opened": False,
        "legacy_outer_metrics_used_for_selection": False,
        "legacy_confirmed_metrics_changed": False,
        "chronaris_backbone_modified": False,
        "chronaris_backbone_trained": False,
        "safe_fusion_started": False,
        "teacher_distillation_started": False,
    }
    frames = {
        "training": training,
        "exports": exports,
        "access": access,
        "maneuver_targets": maneuver_targets,
        "maneuver_thresholds": maneuver_thresholds,
        "physiology_states": physiology_states,
        "residual_targets": residual_targets,
        "residual_descriptors": residual_descriptors,
        "inertia_audit": inertia_audit,
        "metrics": metrics,
        "summary": summary,
        "predictions": predictions,
        "time_shortcut": time_shortcut,
        "third_pool": third_pool,
    }
    paths = write_target_reconstruction_outputs(
        compact_root=compact_root,
        protocol=protocol,
        preregistration=preregistration,
        allowance=allowance,
        best=best,
        frames=frames,
    )
    return DingxinTargetReconstructionResult(
        status="completed",
        decision=str(allowance["decision"]),
        allow_safe_fusion=bool(allowance["allow_safe_fusion"]),
        allow_task_aware_research=bool(allowance["allow_task_aware_research"]),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=paths["report_path"],
    )


def _preregistration(config, training_config, split_manifest):
    return {
        "format": "chronaris.dingxin_target_reconstruction_preregistration.v1",
        "registered_before_task_training": True,
        "baseline_commit": "c7667de7e89e9cb7c74eba1cecad712d4d0ddc7f",
        "methods": list(MATCHED_METHODS),
        "main_split_count": 6,
        "pressure_split_id": PRESSURE_SPLIT_ID,
        "development_split_manifest_sha256": sha256_file(Path(config.stability_root) / "development_split_manifest.json"),
        "training_config": asdict(training_config),
        "input_contract": "task_1b_time_like_vehicle_channels_removed",
        "tasks": {
            "legacy_tasks": "reported_only_not_used_for_task_1c_gate",
            "future_maneuver_score": "next_5s_score_using_inner_train_semantic_scales",
            "future_maneuver_trend": "decrease_stable_increase_from_train_abs_delta_q33",
            "physiology_residual": "robust_future_state_minus_cross_fitted_physiology_inertia",
            "high_residual_response": "train_q75_of_residual_response",
        },
        "physiology_descriptors": {
            "eeg": ["centered_rms", "mad", "line_length", "spectral_entropy"],
            "slow_physiology": ["median", "slope", "mad"],
            "aggregation": "train_reliability_weighted_median",
        },
        "time_shortcut_control": "train_fitted_cubic_phase_residualization",
        "gates": TARGET_RECONSTRUCTION_GATES,
        "outer_test_opened": False,
        "teacher_distillation_allowed": False,
        "confirmed_metrics_immutable": True,
    }


def _panel_status(training, exports, metrics):
    completed_training = training[
        training["main_selection"].astype(bool)
        & training["status"].isin(("completed", "resumed"))
    ]
    method_counts = completed_training.groupby("method")["split_id"].nunique()
    split_counts = completed_training.groupby("split_id")["method"].nunique()
    formal = metrics[
        metrics["main_selection"].astype(bool)
        & (metrics["representation_variant"] == "train_phase_residualized_pooled_64d")
        & metrics["method"].isin(MATCHED_METHODS)
        & metrics["metric"].isin(("spearman", "macro_f1", "rmse", "normalized_ap"))
    ]
    primary_units = formal[["method", "split_id", "task"]].drop_duplicates()
    expected_units = 6 * 6 * 4
    return {
        "completed_method_count": int((method_counts == 6).sum()),
        "completed_main_split_count": int((split_counts == 6).sum()),
        "missing_method_split_units": int(36 - completed_training[["method", "split_id"]].drop_duplicates().shape[0]),
        "completed_primary_metric_units": int(len(primary_units)),
        "expected_primary_metric_units": expected_units,
        "representation_export_count": int(len(exports)),
    }


def _time_shortcut_frame(summary):
    frame = summary[summary["task"].isin(("future_maneuver_score", "future_maneuver_trend"))].copy()
    return frame[["method", "task", "median_standardized_gain_over_time"]].rename(
        columns={"median_standardized_gain_over_time": "median_standardized_gain"}
    )


def _third_pool_frame(metrics):
    frame = metrics[
        (~metrics["main_selection"].astype(bool))
        & metrics["method"].isin(MATCHED_METHODS)
        & (metrics["representation_variant"] == "train_phase_residualized_pooled_64d")
        & (
            ((metrics["task"] == "future_maneuver_score") & (metrics["metric"] == "spearman"))
            | ((metrics["task"] == "physiology_residual") & (metrics["metric"] == "rmse"))
        )
    ].copy()
    frame["task_name"] = frame["task"].map(
        {"future_maneuver_score": "未来机动分数", "physiology_residual": "生理残差"}
    )
    return frame


def _observed(best, time_shortcut, third_pool):
    score = best["future_maneuver_score"]
    trend = best["future_maneuver_trend"]
    residual = best["physiology_residual"]
    high = best["high_residual_response"]
    gain_by_task = {
        task: float(frame["median_standardized_gain"].max())
        for task, frame in time_shortcut.groupby("task")
    }
    completed_third = sum(
        third_pool[third_pool["task"] == task]["method"].nunique() == 6
        for task in ("future_maneuver_score", "physiology_residual")
    )
    return {
        "time_shortcut_standardized_gain": min(gain_by_task.values()),
        "future_maneuver_score_median_spearman": float(score["median"]),
        "future_maneuver_trend_mean_macro_f1": float(trend["mean"]),
        "future_maneuver_trend_worst_macro_f1": float(trend["worst"]),
        "physiology_residual_median_rmse_ratio": float(residual["median_rmse_ratio"]),
        "physiology_residual_mean_skill": float(residual["mean_skill"]),
        "physiology_residual_positive_skill_split_count": int(residual["positive_skill_split_count"]),
        "high_residual_mean_normalized_ap": float(high["mean_normalized_ap"]),
        "high_residual_median_normalized_ap": float(high["median_normalized_ap"]),
        "high_residual_positive_split_count": int(high["positive_normalized_ap_split_count"]),
        "third_pool_continuous_task_count": int(completed_third),
    }


def _git_head():
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
