"""Run the bounded Dingxin core-task feasibility and safe-fusion audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from chronaris.evaluation.application_tasks.core_feasibility_candidates import (
    run_fieldwise_candidate,
    run_sequence_candidate,
    run_summary_candidate,
)
from chronaris.evaluation.application_tasks.core_feasibility_data import (
    fold_task_data,
    historical_frozen_rows,
    maneuver_scores_by_fold,
    response_delta_index,
)
from chronaris.evaluation.application_tasks.core_feasibility_features import (
    load_or_build_feature_cache,
)
from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    PRIMARY_FOLDS,
    UPPER_BOUND_GATES,
    InnerRoleAccessGuard,
    aggregate_gate_rows,
    audit_train_validation_support,
    purge_train_for_full_support,
    role_map_for_fold,
)
from chronaris.evaluation.application_tasks.core_feasibility_reporting import (
    write_core_feasibility_outputs,
)
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    build_dingxin_lazy_context_index,
)
from chronaris.evaluation.application_tasks.dingxin_nested_target_data import (
    build_dingxin_nested_targets,
)
from chronaris.evaluation.application_tasks.core_feasibility_runtime import (
    blocked_safe_fusion_rows,
    dependency_rows,
    protocol_payload,
    task_diagnostic_rows,
    write_json,
)


SUMMARY_CANDIDATES = (
    ("summary_vehicle_5s_linear", "vehicle", 5.0, "linear"),
    ("summary_vehicle_30s_linear", "vehicle", 30.0, "linear"),
    ("summary_dual_30s_linear", "dual", 30.0, "linear"),
    ("summary_physiology_30s_linear", "physiology", 30.0, "linear"),
    ("summary_vehicle_30s_histgb", "vehicle", 30.0, "histgb"),
    ("summary_dual_30s_histgb", "dual", 30.0, "histgb"),
)
SEQUENCE_CANDIDATES = (
    ("sequence_minirocket_1000", "minirocket_1000"),
    ("sequence_minirocket_5000", "minirocket_5000"),
    ("sequence_multirocket", "multirocket"),
    ("sequence_hydra", "hydra"),
)
CANDIDATE_COUNT = len(SUMMARY_CANDIDATES) + len(SEQUENCE_CANDIDATES) + 2


def candidate_panel():
    return [
        *[
            {
                "candidate_id": candidate_id,
                "representation": "causal_summary",
                "modality": modality,
                "history_s": history_s,
                "head_family": family,
            }
            for candidate_id, modality, history_s, family in SUMMARY_CANDIDATES
        ],
        *[
            {
                "candidate_id": candidate_id,
                "representation": family,
                "modality": "task_specific",
                "history_s": 30.0,
                "head_family": "linear_panel",
            }
            for candidate_id, family in SEQUENCE_CANDIDATES
        ],
        {
            "candidate_id": "fieldwise_physiology_30s",
            "representation": "causal_summary",
            "modality": "physiology",
            "history_s": 30.0,
            "head_family": "fieldwise_ridge",
        },
        {
            "candidate_id": "historical_frozen_panel",
            "representation": "frozen_64d",
            "modality": "method_panel",
            "history_s": 30.0,
            "head_family": "historical_linear_and_minirocket",
        },
    ]


@dataclass(frozen=True, slots=True)
class DingxinCoreFeasibilityConfig:
    run_id: str = "2026-07-14_dingxin-core-feasibility"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    source_workspace_root: str = "/home/wangminan/projects/chronaris"
    fixed_audit_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    inner_split_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    nested_target_root: str = "docs/artifacts/runs/2026-07-11_dingxin-nested-targets"
    historical_validation_root: str = (
        "docs/artifacts/runs/2026-07-11_dingxin-nested-validation"
    )
    representation_run_id: str = "2026-07-12_dingxin-locked-representations-coalesced"
    e_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    )
    f_run_manifest_path: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    )
    random_state: int = 17
    resume: bool = True


@dataclass(frozen=True, slots=True)
class DingxinCoreFeasibilityResult:
    status: str
    upper_bound_gate_passed: bool
    safe_fusion_gate_passed: bool
    allow_next_goal: bool
    compact_run_root: str
    heavy_run_root: str
    report_path: str
    evidence_manifest_path: str


def run_dingxin_core_feasibility(
    config: DingxinCoreFeasibilityConfig,
) -> DingxinCoreFeasibilityResult:
    if CANDIDATE_COUNT > 12:
        raise ValueError("core-feasibility candidate budget exceeds 12")
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    state_root = heavy_root / "candidate_states"
    state_root.mkdir(exist_ok=True)

    fixed_root = Path(config.fixed_audit_root)
    split_path = Path(config.inner_split_root) / "split_manifest.json"
    target_path = Path(config.nested_target_root) / "nested_targets.csv"
    threshold_path = Path(config.nested_target_root) / "nested_thresholds.csv"
    context_path = fixed_root / "context_sample_manifest.jsonl"
    role_path = fixed_root / "field_role_manifest.csv"
    source_workspace = Path(config.source_workspace_root)
    e_run_manifest_path = source_workspace / config.e_run_manifest_path
    f_run_manifest_path = source_workspace / config.f_run_manifest_path
    snapshot_root = (
        source_workspace
        / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
    sources = {
        "fixed_data_manifest": fixed_root / "data_manifest.json",
        "field_roles": role_path,
        "context_catalog": context_path,
        "inner_splits": split_path,
        "nested_targets": target_path,
        "nested_thresholds": threshold_path,
        "snapshot_manifest": snapshot_manifest_path,
        "historical_validation_metrics": Path(config.historical_validation_root)
        / "metric_long.csv",
        "feature_export_e_manifest": e_run_manifest_path,
        "feature_export_f_manifest": f_run_manifest_path,
    }
    missing = sorted(name for name, path in sources.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"core-feasibility sources missing: {missing}")

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    base_plans = {
        str(plan["fold_id"]): plan
        for plan in split_payload["folds"]
        if str(plan["fold_id"]) in PRIMARY_FOLDS
    }
    if tuple(base_plans) != PRIMARY_FOLDS:
        raise ValueError("primary inner split panel is incomplete or reordered")
    context_catalog = pd.read_json(context_path, lines=True)
    plans = {
        fold_id: purge_train_for_full_support(
            plan=base_plans[fold_id],
            context_catalog=context_catalog,
        )
        for fold_id in PRIMARY_FOLDS
    }
    development_split_root = heavy_root / "development_split"
    development_split_root.mkdir(parents=True, exist_ok=True)
    derived_split_path = development_split_root / "split_manifest.json"
    write_json(
        derived_split_path,
        {
            "format": "chronaris.dingxin_inner_development_splits.v1",
            "source_split_manifest": str(split_path),
            "support_definition": "30s_input_plus_5s_target",
            "outer_test_opened": False,
            "folds": [plans[fold_id] for fold_id in PRIMARY_FOLDS],
        },
    )
    support_rows = []
    for fold_id in PRIMARY_FOLDS:
        support = audit_train_validation_support(
            plan=plans[fold_id],
            context_catalog=context_catalog,
        )
        support.update(
            {
                "source_train_context_count": len(
                    base_plans[fold_id]["train_sample_ids"]
                ),
                "purged_train_context_count": len(
                    plans[fold_id]["purged_for_full_35s_support_sample_ids"]
                ),
                "outer_context_identifiers_in_development_manifest": 0,
            }
        )
        support_rows.append(support)
    if not all(row["support_isolated"] for row in support_rows):
        raise PermissionError("inner train/validation support intervals overlap")

    target_frame, threshold_frame, target_fold_frame = build_dingxin_nested_targets(
        fixed_audit_root=fixed_root,
        snapshot_root=snapshot_root,
        inner_split_root=development_split_root,
        e_run_manifest_path=str(e_run_manifest_path),
        f_run_manifest_path=str(f_run_manifest_path),
    )
    derived_target_path = development_split_root / "nested_targets.csv"
    derived_threshold_path = development_split_root / "nested_thresholds.csv"
    derived_target_fold_path = development_split_root / "nested_target_folds.csv"
    target_frame.to_csv(derived_target_path, index=False)
    threshold_frame.to_csv(derived_threshold_path, index=False)
    target_fold_frame.to_csv(derived_target_fold_path, index=False)
    sources.update(
        {
            "development_splits": derived_split_path,
            "development_targets": derived_target_path,
            "development_thresholds": derived_threshold_path,
            "development_target_folds": derived_target_fold_path,
        }
    )

    protocol = protocol_payload(
        config=config,
        sources=sources,
        support_rows=support_rows,
        candidate_count=CANDIDATE_COUNT,
        candidate_panel=candidate_panel(),
    )
    write_json(compact_root / "protocol.json", protocol)
    write_json(
        compact_root / "progress.json",
        {
            "status": "running",
            "stage": "raw_feature_cache",
            "outer_test_opened": False,
            "candidate_count": CANDIDATE_COUNT,
        },
    )

    all_inner_ids = tuple(
        sorted(
            {
                str(sample_id)
                for plan in plans.values()
                for key in ("train_sample_ids", "validation_sample_ids")
                for sample_id in plan[key]
            }
        )
    )
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot_root,
        field_role_manifest_path=role_path,
        context_manifest_path=context_path,
    )
    cache = load_or_build_feature_cache(
        index=index,
        sample_ids=all_inner_ids,
        output_path=heavy_root / "raw_causal_query_cache.npz",
    )
    maneuver_scores = maneuver_scores_by_fold(
        plans=plans,
        fixed_root=fixed_root,
        e_run_manifest_path=str(e_run_manifest_path),
        f_run_manifest_path=str(f_run_manifest_path),
    )
    field_delta_index = response_delta_index(
        fixed_root=fixed_root,
        snapshot_root=snapshot_root,
    )

    comparison_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []
    for fold_id in PRIMARY_FOLDS:
        plan = plans[fold_id]
        guard = InnerRoleAccessGuard(role_map_for_fold(plan))
        guard.record_blocked_probe(fold_id=fold_id, role="held_out")
        guard.record_blocked_probe(fold_id=fold_id, role="outer_test")
        train_ids = guard.request(
            fold_id=fold_id,
            role="inner_train",
            sample_ids=plan["train_sample_ids"],
            purpose="fit_upper_bound_models",
        )
        validation_ids = guard.request(
            fold_id=fold_id,
            role="validation",
            sample_ids=plan["validation_sample_ids"],
            purpose="evaluate_upper_bound_models",
        )
        access_rows.extend(guard.audit_rows)
        task_data = fold_task_data(
            fold_id=fold_id,
            train_ids=train_ids,
            validation_ids=validation_ids,
            target_frame=target_frame,
            maneuver_scores=maneuver_scores[fold_id],
        )
        for candidate_id, modality, history_s, family in SUMMARY_CANDIDATES:
            rows, selected = run_summary_candidate(
                state_root=state_root,
                resume=config.resume,
                fold_id=fold_id,
                candidate_id=candidate_id,
                modality=modality,
                history_s=history_s,
                family=family,
                cache=cache,
                task_data=task_data,
                random_state=config.random_state,
            )
            comparison_rows.extend(rows)
            fold_rows.extend(selected)
        for candidate_id, family in SEQUENCE_CANDIDATES:
            rows, selected = run_sequence_candidate(
                state_root=state_root,
                resume=config.resume,
                fold_id=fold_id,
                candidate_id=candidate_id,
                family=family,
                cache=cache,
                task_data=task_data,
                random_state=config.random_state,
            )
            comparison_rows.extend(rows)
            fold_rows.extend(selected)
        rows, selected = run_fieldwise_candidate(
            state_root=state_root,
            resume=config.resume,
            fold_id=fold_id,
            cache=cache,
            task_data=task_data,
            threshold_frame=threshold_frame,
            field_delta_index=field_delta_index,
        )
        comparison_rows.extend(rows)
        fold_rows.extend(selected)

    historical_comparisons, historical_selected = historical_frozen_rows(
        Path(config.historical_validation_root) / "metric_long.csv"
    )
    comparison_rows.extend(historical_comparisons)
    fold_rows.extend(historical_selected)
    gate_rows = aggregate_gate_rows(fold_rows, gates=UPPER_BOUND_GATES)
    upper_gate_passed = all(row["mean_gate_passed"] for row in gate_rows)
    if upper_gate_passed:
        from chronaris.evaluation.application_tasks.core_feasibility_safe_fusion import (
            run_frozen_safe_fusion,
        )

        safe_fusion_rows, expert_gate_rows = run_frozen_safe_fusion(
            plans=plans,
            cache=cache,
            target_frame=target_frame,
            maneuver_scores=maneuver_scores,
            representation_root=(
                source_workspace
                / "artifacts/application_evaluation"
                / config.representation_run_id
            ),
            random_state=config.random_state,
        )
    else:
        safe_fusion_rows, expert_gate_rows = blocked_safe_fusion_rows(gate_rows)
    safe_fusion_gate_passed = bool(safe_fusion_rows) and all(
        bool(row.get("gate_passed")) for row in safe_fusion_rows
    )
    dependencies = dependency_rows()
    diagnostic_rows = task_diagnostic_rows(
        target_frame=target_frame,
        threshold_frame=threshold_frame,
        plans=plans,
    )
    paths = write_core_feasibility_outputs(
        compact_root=compact_root,
        heavy_root=heavy_root,
        protocol=protocol,
        gate_rows=gate_rows,
        comparison_rows=comparison_rows,
        fold_rows=fold_rows,
        safe_fusion_rows=safe_fusion_rows,
        expert_gate_rows=expert_gate_rows,
        access_rows=access_rows,
        support_rows=support_rows,
        diagnostic_rows=diagnostic_rows,
        dependency_rows=dependencies,
        candidate_count=CANDIDATE_COUNT,
    )
    allow_next = upper_gate_passed and safe_fusion_gate_passed
    return DingxinCoreFeasibilityResult(
        status="accepted" if allow_next else "gap",
        upper_bound_gate_passed=upper_gate_passed,
        safe_fusion_gate_passed=safe_fusion_gate_passed,
        allow_next_goal=allow_next,
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=paths["decision_report"],
        evidence_manifest_path=paths["evidence_manifest"],
    )
