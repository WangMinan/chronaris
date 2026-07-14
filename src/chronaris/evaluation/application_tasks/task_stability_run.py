"""Execute Dingxin task-protocol repair and cross-view stability screening."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from chronaris.evaluation.application_tasks.core_feasibility_data import (
    maneuver_scores_by_fold,
    response_delta_index,
)
from chronaris.evaluation.application_tasks.core_feasibility_features import (
    load_or_build_feature_cache,
)
from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    InnerRoleAccessGuard,
    role_map_for_fold,
    sha256_file,
)
from chronaris.evaluation.application_tasks.dingxin_context_data import (
    build_dingxin_lazy_context_index,
)
from chronaris.evaluation.application_tasks.task_stability_audits import (
    leakage_proxy_rows,
    metric_ceiling_rows,
    response_field_stability_rows,
    target_stability_rows,
)
from chronaris.evaluation.application_tasks.task_stability_candidates import (
    candidate_manifest,
    run_candidate_split,
)
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    FINAL_OUTER_THRESHOLDS,
    RELATIVE_GATES,
    aggregate_values,
    decide_allowance,
    stable_sha256,
)
from chronaris.evaluation.application_tasks.task_stability_features import (
    distribution_shift_rows,
)
from chronaris.evaluation.application_tasks.task_stability_reporting import (
    write_task_stability_outputs,
)
from chronaris.evaluation.application_tasks.task_stability_splits import (
    build_development_splits,
)


@dataclass(frozen=True, slots=True)
class DingxinTaskStabilityConfig:
    run_id: str = "2026-07-14_dingxin-task-stability"
    compact_output_root: str = "docs/artifacts/runs"
    heavy_output_root: str = "artifacts/application_evaluation"
    source_workspace_root: str = "/home/wangminan/projects/chronaris"
    fixed_root: str = "docs/artifacts/runs/2026-07-10_fixed-data-audit"
    legacy_inner_root: str = "docs/artifacts/runs/2026-07-11_dingxin-inner-splits"
    e_manifest: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json"
    )
    f_manifest: str = (
        "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json"
    )
    snapshot_root: str = (
        "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
    )
    random_state: int = 17
    resume: bool = True
    protocol_repair_rerun_count: int = 1


@dataclass(frozen=True, slots=True)
class DingxinTaskStabilityResult:
    status: str
    decision: str
    allow_safe_fusion: bool
    allow_task_aware_research: bool
    compact_run_root: str
    heavy_run_root: str
    report_path: str


def run_dingxin_task_stability(
    config: DingxinTaskStabilityConfig,
) -> DingxinTaskStabilityResult:
    compact_root = Path(config.compact_output_root) / config.run_id
    heavy_root = Path(config.heavy_output_root) / config.run_id
    compact_root.mkdir(parents=True, exist_ok=True)
    heavy_root.mkdir(parents=True, exist_ok=True)
    source_workspace = Path(config.source_workspace_root)
    fixed_root = Path(config.fixed_root)
    legacy_inner_root = Path(config.legacy_inner_root)
    e_manifest = source_workspace / config.e_manifest
    f_manifest = source_workspace / config.f_manifest
    snapshot_root = source_workspace / config.snapshot_root
    context_path = fixed_root / "context_sample_manifest.jsonl"
    role_path = fixed_root / "field_role_manifest.csv"
    source_paths = {
        "fixed_data_manifest": fixed_root / "data_manifest.json",
        "outer_split_contract": fixed_root / "split_manifest.json",
        "context_catalog": context_path,
        "field_roles": role_path,
        "legacy_inner_split": legacy_inner_root / "split_manifest.json",
        "snapshot_manifest": snapshot_root / "snapshot_manifest.json",
        "feature_export_e_manifest": e_manifest,
        "feature_export_f_manifest": f_manifest,
    }
    missing = sorted(name for name, path in source_paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"task-stability sources missing: {missing}")
    _write_json(
        compact_root / "progress.json",
        {
            "status": "running",
            "stage": "development_split_construction",
            "outer_test_opened": False,
        },
    )
    manifest, split_validity, targets, thresholds, dedup = build_development_splits(
        fixed_root=fixed_root,
        legacy_inner_root=legacy_inner_root,
        heavy_root=heavy_root,
        snapshot_root=snapshot_root,
        e_run_manifest_path=str(e_manifest),
        f_run_manifest_path=str(f_manifest),
    )
    main_plans = {
        str(row["fold_id"]): row for row in manifest["folds"] if row["main_selection"]
    }
    all_plans = {str(row["fold_id"]): row for row in manifest["folds"]}
    all_ids = tuple(
        sorted(
            {
                str(value)
                for plan in main_plans.values()
                for role in ("train_sample_ids", "validation_sample_ids")
                for value in plan[role]
            }
        )
    )
    access_rows = []
    sanitized_plans = {}
    for split_id, plan in main_plans.items():
        guard = InnerRoleAccessGuard(role_map_for_fold(plan))
        train_ids = guard.request(
            fold_id=split_id,
            role="inner_train",
            sample_ids=plan["train_sample_ids"],
            purpose="fit_task_stability_candidate",
        )
        validation_ids = guard.request(
            fold_id=split_id,
            role="validation",
            sample_ids=plan["validation_sample_ids"],
            purpose="evaluate_task_stability_candidate",
        )
        access_rows.extend(guard.audit_rows)
        sanitized_plans[split_id] = {
            **plan,
            "train_sample_ids": list(train_ids),
            "validation_sample_ids": list(validation_ids),
            "held_out_sample_ids": [],
        }
    index = build_dingxin_lazy_context_index(
        snapshot_root=snapshot_root,
        field_role_manifest_path=role_path,
        context_manifest_path=context_path,
    )
    cache = load_or_build_feature_cache(
        index=index,
        sample_ids=all_ids,
        output_path=(
            heavy_root
            / (
                "raw_causal_query_cache_protocol_attempt_"
                f"{config.protocol_repair_rerun_count}.npz"
            )
        ),
    )
    cache, input_contract = _remove_explicit_time_shortcuts(
        cache=cache,
        vehicle_raw_to_index=index.plan.vehicle_raw_to_index,
        role_path=role_path,
    )
    maneuver_scores = maneuver_scores_by_fold(
        plans=all_plans,
        fixed_root=fixed_root,
        e_run_manifest_path=str(e_manifest),
        f_run_manifest_path=str(f_manifest),
    )
    field_delta_index = response_delta_index(
        fixed_root=fixed_root, snapshot_root=snapshot_root
    )
    ceiling = metric_ceiling_rows(
        manifest=manifest,
        targets=targets,
        maneuver_scores=maneuver_scores,
    )
    target_stability, maneuver_thresholds = target_stability_rows(
        manifest=manifest,
        targets=targets,
        thresholds=thresholds,
        maneuver_scores=maneuver_scores,
    )
    response_fields = response_field_stability_rows(
        manifest=manifest,
        targets=targets,
        thresholds=thresholds,
        field_delta_index=field_delta_index,
    )
    shift_rows = []
    for split_id, plan in sanitized_plans.items():
        shift_rows.extend(
            distribution_shift_rows(
                cache,
                split_id=split_id,
                train_sample_ids=plan["train_sample_ids"],
                validation_sample_ids=plan["validation_sample_ids"],
            )
        )
    candidates = candidate_manifest()
    for candidate in candidates:
        candidate["input_feature_contract_sha256"] = input_contract[
            "input_feature_contract_sha256"
        ]
        candidate["config_sha256"] = stable_sha256(
            {key: value for key, value in candidate.items() if key != "config_sha256"}
        )
    _write_json(compact_root / "candidate_manifest.json", candidates)
    state_root = (
        heavy_root
        / f"candidate_states_protocol_attempt_{config.protocol_repair_rerun_count}"
        / str(input_contract["input_feature_contract_sha256"])[:12]
    )
    metric_rows = []
    completed = 0
    for candidate in candidates:
        for split_id, plan in sanitized_plans.items():
            metric_rows.extend(
                run_candidate_split(
                    candidate=candidate,
                    plan=plan,
                    cache=cache,
                    targets=targets,
                    thresholds=thresholds,
                    maneuver_scores=maneuver_scores,
                    field_delta_index=field_delta_index,
                    state_root=state_root,
                    random_state=config.random_state,
                )
            )
            completed += 1
            _write_json(
                compact_root / "progress.json",
                {
                    "status": "running",
                    "stage": "bounded_candidate_panel",
                    "completed_candidate_split_units": completed,
                    "total_candidate_split_units": len(candidates) * len(sanitized_plans),
                    "outer_test_opened": False,
                },
            )
    candidate_metrics = pd.DataFrame(metric_rows)
    relative_summary, best_summary = _relative_summary(candidate_metrics)
    input_stabilization = _input_stabilization_rows(candidate_metrics, candidates)
    leakage = leakage_proxy_rows(
        role_path=role_path,
        manifest=manifest,
        candidate_metrics=candidate_metrics,
        input_contract=input_contract,
        context_path=context_path,
        targets=targets,
    )
    access = pd.DataFrame(access_rows)
    selected_validity = split_validity[
        split_validity["split_id"].isin(main_plans)
    ]
    forbidden_access = access[
        access["requested_role"].isin(("held_out", "outer_test"))
    ]
    protocol_valid = bool(
        manifest["main_split_count"] >= 6
        and manifest["unique_main_validation_support_count"] >= 6
        and len(selected_validity) == manifest["main_split_count"]
        and selected_validity["valid_main_split"].astype(bool).all()
        and all(row["support_overlap_count"] == 0 for row in main_plans.values())
        and all(
            row["minimum_support_embargo_ms"] >= 35_000
            for row in main_plans.values()
        )
        and dedup[dedup["included_in_main_ranking"]]["validation_support_hash"].is_unique
        and not bool(leakage["direct_label_source_overlap_count"].max())
        and not bool(leakage["deterministic_derivative_overlap_count"].max())
        and not bool(access["outer_test_accessed"].astype(bool).any())
        and not bool(forbidden_access["allowed"].astype(bool).any())
    )
    field_or_dual_gain = _field_or_dual_gain(relative_summary, candidates)
    allowance = decide_allowance(
        protocol_valid=protocol_valid,
        maneuver=best_summary["maneuver"],
        response=best_summary["response"],
        high_response=best_summary["high_response"],
        field_or_dual_gain=field_or_dual_gain,
    )
    allowance.update(
        {
            "field_or_dual_stable_gain": field_or_dual_gain,
            "outer_test_opened": False,
            "safe_fusion_started": False,
            "task_aware_training_started": False,
        }
    )
    source_sha = {name: sha256_file(path) for name, path in source_paths.items()}
    protocol = {
        "format": "chronaris.dingxin_task_stability_protocol.v1",
        "run_id": config.run_id,
        "git_head": _git_head(),
        "baseline_commit": "26e9c8ebaa124de2cfc9075421aca77f20e5e700",
        "mainline_commit": "23b968e0e9e2b33080fb87781622f325715df9e7",
        "source_sha256": source_sha,
        "input_feature_contract": input_contract,
        "candidate_count": len(candidates),
        "protocol_repair_rerun_count": config.protocol_repair_rerun_count,
        "protocol_repair_reason": (
            "corrected target-fit batch handling and interpreted the 35-second "
            "embargo after each full 35-second support interval"
        ),
        "main_split_count": manifest["main_split_count"],
        "unique_validation_support_count": manifest[
            "unique_main_validation_support_count"
        ],
        "outer_test_opened": False,
        "held_out_predictions_opened": False,
        "historical_outer_sample_predictions_opened": False,
        "chronaris_backbone_modified": False,
        "chronaris_backbone_trained": False,
        "safe_fusion_started": False,
        "final_outer_thresholds": FINAL_OUTER_THRESHOLDS,
        "relative_gates": RELATIVE_GATES,
    }
    metric_contract = _metric_contract()
    frames = {
        "split_validity": _mark_selected(split_validity, main_plans),
        "support_dedup": dedup,
        "metric_ceiling": ceiling,
        "target_stability": target_stability,
        "maneuver_threshold": maneuver_thresholds,
        "response_field": response_fields,
        "distribution_shift": pd.DataFrame(shift_rows),
        "input_stabilization": input_stabilization,
        "candidate_metrics": candidate_metrics,
        "relative_summary": relative_summary,
        "leakage": leakage,
        "access": access,
    }
    paths = write_task_stability_outputs(
        compact_root=compact_root,
        protocol=protocol,
        metric_contract=metric_contract,
        split_manifest=manifest,
        frames=frames,
        candidate_manifest=candidates,
        allowance=allowance,
        best_summary=best_summary,
    )
    return DingxinTaskStabilityResult(
        status="completed",
        decision=str(allowance["decision"]),
        allow_safe_fusion=bool(allowance["allow_safe_fusion"]),
        allow_task_aware_research=bool(allowance["allow_task_aware_research"]),
        compact_run_root=str(compact_root),
        heavy_run_root=str(heavy_root),
        report_path=paths["report_path"],
    )


def _relative_summary(frame: pd.DataFrame):
    available = frame[(frame["status"] == "completed") & frame["ranking_eligible"]]
    rows = []
    by_candidate = {}
    for candidate_id in available["candidate_id"].unique():
        subset = available[available["candidate_id"] == candidate_id]
        maneuver = _metric_frame(subset, "maneuver")
        response = _metric_frame(subset, "response")
        high = _metric_frame(subset, "high_response")
        maneuver_aggregate = _pool_balanced_aggregate(
            maneuver, "macro_f1", higher_is_better=True
        )
        response_aggregate = _pool_balanced_aggregate(
            response, "rmse_ratio", higher_is_better=False
        )
        high_aggregate = _pool_balanced_aggregate(
            high, "normalized_ap", higher_is_better=True
        )
        payloads = {
            "maneuver": {
                "candidate_id": candidate_id,
                "mean_macro_f1": maneuver_aggregate["mean"],
                "median_macro_f1": maneuver_aggregate["median"],
                "worst_macro_f1": maneuver_aggregate["worst"],
                "macro_f1_q25": maneuver_aggregate["q25"],
                "macro_f1_q75": maneuver_aggregate["q75"],
                "macro_f1_bootstrap_ci_low": maneuver_aggregate[
                    "bootstrap_ci_low"
                ],
                "macro_f1_bootstrap_ci_high": maneuver_aggregate[
                    "bootstrap_ci_high"
                ],
                "minimum_class_recall": float(
                    maneuver["minimum_class_recall"].min()
                ),
                "mean_macro_f1_lift": _pool_balanced_mean(
                    maneuver, "macro_f1_lift"
                ),
                "median_score_spearman": float(
                    maneuver["score_spearman"].median()
                ),
            },
            "response": {
                "candidate_id": candidate_id,
                "mean_rmse_ratio": response_aggregate["mean"],
                "median_rmse_ratio": response_aggregate["median"],
                "worst_rmse_ratio": response_aggregate["worst"],
                "rmse_ratio_q25": response_aggregate["q25"],
                "rmse_ratio_q75": response_aggregate["q75"],
                "rmse_ratio_bootstrap_ci_low": response_aggregate[
                    "bootstrap_ci_low"
                ],
                "rmse_ratio_bootstrap_ci_high": response_aggregate[
                    "bootstrap_ci_high"
                ],
                "mean_response_skill": _pool_balanced_mean(
                    response, "response_skill"
                ),
                "positive_skill_fraction": float(
                    np.mean(response["response_skill"] > 0)
                ),
                "median_spearman": float(response["spearman"].median()),
                "mean_rmse": _pool_balanced_mean(response, "rmse"),
            },
            "high_response": {
                "candidate_id": candidate_id,
                "mean_normalized_ap": high_aggregate["mean"],
                "median_normalized_ap": high_aggregate["median"],
                "worst_normalized_ap": high_aggregate["worst"],
                "normalized_ap_q25": high_aggregate["q25"],
                "normalized_ap_q75": high_aggregate["q75"],
                "normalized_ap_bootstrap_ci_low": high_aggregate[
                    "bootstrap_ci_low"
                ],
                "normalized_ap_bootstrap_ci_high": high_aggregate[
                    "bootstrap_ci_high"
                ],
                "positive_normalized_ap_fraction": float(
                    np.mean(high["normalized_ap"] > 0)
                ),
                "mean_auprc_lift": _pool_balanced_mean(high, "auprc_lift"),
                "mean_auprc": _pool_balanced_mean(high, "auprc"),
            },
        }
        by_candidate[candidate_id] = payloads
        for task, payload in payloads.items():
            rows.append({"task": task, **payload})
    maneuver_best = max(
        (value["maneuver"] for value in by_candidate.values()),
        key=lambda row: (row["mean_macro_f1"], row["median_macro_f1"], row["worst_macro_f1"]),
    )
    response_best = min(
        (value["response"] for value in by_candidate.values()),
        key=lambda row: (row["median_rmse_ratio"], -row["mean_response_skill"]),
    )
    high_best = max(
        (value["high_response"] for value in by_candidate.values()),
        key=lambda row: (row["mean_normalized_ap"], row["median_normalized_ap"]),
    )
    return pd.DataFrame(rows), {
        "maneuver": maneuver_best,
        "response": response_best,
        "high_response": high_best,
    }


def _metric_frame(frame, task):
    subset = frame[frame["task"] == task]
    return (
        subset.pivot(
            index=["split_id", "outer_pool_id"], columns="metric", values="value"
        )
        .reset_index()
        .sort_values(["outer_pool_id", "split_id"])
    )


def _pool_balanced_mean(frame: pd.DataFrame, metric: str) -> float:
    return float(frame.groupby("outer_pool_id", sort=True)[metric].mean().mean())


def _pool_balanced_aggregate(
    frame: pd.DataFrame, metric: str, *, higher_is_better: bool
) -> dict[str, float]:
    values = frame[metric].to_numpy(dtype=np.float64)
    result = aggregate_values(values, higher_is_better=higher_is_better)
    result["mean"] = _pool_balanced_mean(frame, metric)
    pools = tuple(sorted(frame["outer_pool_id"].unique()))
    rng = np.random.default_rng(17)
    samples = []
    for _ in range(2_000):
        sampled_pools = rng.choice(pools, size=len(pools), replace=True)
        pool_means = []
        for pool in sampled_pools:
            pool_values = frame.loc[
                frame["outer_pool_id"] == pool, metric
            ].to_numpy(dtype=np.float64)
            pool_means.append(
                float(np.mean(rng.choice(pool_values, size=len(pool_values), replace=True)))
            )
        samples.append(float(np.mean(pool_means)))
    result["bootstrap_ci_low"] = float(np.quantile(samples, 0.025))
    result["bootstrap_ci_high"] = float(np.quantile(samples, 0.975))
    return result


def _input_stabilization_rows(metrics, candidates):
    identifiers = {
        "vehicle_30s_no_adaptation",
        "vehicle_30s_global",
        "vehicle_30s_window",
        "vehicle_30s_baseline_relative",
        "vehicle_30s_rank",
    }
    modes = {
        str(row["candidate_id"]): str(row.get("stabilization", "not_applicable"))
        for row in candidates
    }
    selected = metrics[
        metrics["candidate_id"].isin(identifiers)
        & metrics.apply(
            lambda row: (row["task"], row["metric"])
            in {
                ("maneuver", "macro_f1"),
                ("response", "rmse_ratio"),
                ("high_response", "normalized_ap"),
            },
            axis=1,
        )
    ].copy()
    selected["stabilization"] = selected["candidate_id"].map(modes)
    return selected


def _field_or_dual_gain(summary, candidates):
    modality = {str(row["candidate_id"]): row.get("modality") for row in candidates}
    response = summary[summary["task"] == "response"].copy()
    response["modality"] = response["candidate_id"].map(modality)
    enhanced = response[
        response["modality"].eq("dual")
        | response["candidate_id"].eq("fieldwise_physiology_30s")
    ]
    single = response[response["modality"].isin(("vehicle", "physiology"))]
    if enhanced.empty or single.empty:
        return False
    best_enhanced = float(enhanced["median_rmse_ratio"].min())
    best_single = float(single["median_rmse_ratio"].min())
    positive = float(
        enhanced.sort_values("median_rmse_ratio").iloc[0]["positive_skill_fraction"]
    )
    return bool(best_enhanced <= best_single * 0.99 and positive >= 0.5)


def _mark_selected(validity, main_plans):
    frame = validity.copy()
    frame["selected_for_main_ranking"] = frame["split_id"].isin(main_plans)
    return frame


def _metric_contract():
    return {
        "format": "chronaris.dingxin_task_stability_metric_contract.v1",
        "maneuver": {
            "primary": "fixed_three_class_macro_f1",
            "diagnostic_missing_class_rule": "exclude_from_main_ranking",
            "additional": [
                "support_aware_macro_f1",
                "per_class_recall",
                "ordinal_mae",
                "continuous_score_mae",
                "continuous_score_spearman",
            ],
        },
        "response": {
            "rmse_ratio": "RMSE_model / RMSE_train_mean_baseline",
            "response_skill": "1 - MSE_model / MSE_train_mean_baseline",
            "additional": ["rmse", "mae", "nrmse_std", "nrmse_iqr", "spearman"],
            "absolute_outer_threshold_not_used_for_inner_selection": True,
        },
        "high_response": {
            "normalized_ap": "(AUPRC - prevalence) / (1 - prevalence)",
            "random_auprc_baseline": "validation prevalence",
            "additional": ["auprc", "auroc", "balanced_accuracy", "auprc_lift"],
        },
        "aggregation": {
            "unit": "unique_validation_support_hash",
            "outer_pool_weighting": "equal_weight_after_within_pool_aggregation",
            "mean": "mean of outer-pool support means",
            "median_quartiles_and_worst": "unique validation support units",
            "bootstrap_ci": "hierarchical resampling of pools then supports",
            "statistics": ["mean", "median", "worst", "q25", "q75", "bootstrap_ci"],
            "window_level_significance": False,
        },
    }


def _remove_explicit_time_shortcuts(*, cache, vehicle_raw_to_index, role_path):
    roles = pd.read_csv(role_path)
    text = (
        roles[
            [
                "feature_name",
                "source_field",
                "display_label",
                "unit_hint",
                "semantic_category",
            ]
        ]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    time_like = set(
        roles.loc[
            roles["allowed_in_maneuver_input"].astype(bool)
            & text.str.contains(r"time|timestamp|rtc|时间", regex=True),
            "feature_name",
        ].astype(str)
    )
    removed_indices = sorted(
        {
            int(index)
            for raw_mapping in vehicle_raw_to_index.values()
            for name, index in raw_mapping.items()
            if str(name) in time_like
        }
    )
    channel_count = int(cache.vehicle_values.shape[-1])
    keep = np.asarray(
        [index for index in range(channel_count) if index not in removed_indices],
        dtype=np.int64,
    )
    if not len(keep):
        raise ValueError("time-shortcut exclusion removed every vehicle channel")
    contract_payload = {
        "explicit_query_or_block_position_used": False,
        "view_pilot_or_sortie_identity_used": False,
        "time_like_allowed_field_count": len(time_like),
        "time_like_present_channel_count": len(removed_indices),
        "time_like_present_channels_sha256": stable_sha256(removed_indices),
        "vehicle_channel_count_before": channel_count,
        "vehicle_channel_count_after": len(keep),
    }
    contract_payload["input_feature_contract_sha256"] = stable_sha256(
        contract_payload
    )
    return (
        replace(
            cache,
            vehicle_values=cache.vehicle_values[:, :, keep],
            vehicle_mask=cache.vehicle_mask[:, :, keep],
            vehicle_age_s=cache.vehicle_age_s[:, :, keep],
        ),
        contract_payload,
    )


def _git_head():
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
