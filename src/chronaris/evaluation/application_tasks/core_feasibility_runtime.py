"""Protocol, dependency, and blocked-state helpers for feasibility runs."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    PRIMARY_FOLDS,
    UPPER_BOUND_GATES,
    sha256_file,
    stable_payload_sha256,
)


def protocol_payload(
    *, config, sources, support_rows, candidate_count, candidate_panel
):
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    snapshot_manifest = json.loads(
        Path(sources["snapshot_manifest"]).read_text(encoding="utf-8")
    )
    source_workspace = Path(config.source_workspace_root)
    snapshot_hashes = {
        item["relative_path"]: sha256_file(
            source_workspace
            / "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"
            / item["relative_path"]
        )
        for item in snapshot_manifest["files"]
    }
    git_head = subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    payload = {
        "format": "chronaris.dingxin_core_feasibility_protocol.v1",
        "run_id": config.run_id,
        "git_head": git_head,
        "base_commit": "23b968e0e9e2b33080fb87781622f325715df9e7",
        "branch_name": "codex/dingxin-core-feasibility-20260714",
        "source_file_sha256": source_hashes,
        "snapshot_file_sha256": snapshot_hashes,
        "folds": list(PRIMARY_FOLDS),
        "methods": [
            "physiology_only",
            "vehicle_only",
            "naive_time_sync",
            "mult",
            "contiformer",
            "chronaris",
        ],
        "random_seeds": [config.random_state],
        "allowed_roles": ["inner_train", "inner_validation"],
        "forbidden_roles": ["held_out", "outer_test"],
        "outer_test_opened": False,
        "candidate_budget": 12,
        "candidate_count": candidate_count,
        "candidate_panel": candidate_panel,
        "consumer_config": {
            "maneuver": [
                "balanced_logistic",
                "linear_svm",
                "ordinal_cumulative",
                "continuous_score_then_bucket",
                "hist_gradient_boosting",
            ],
            "response": [
                "ridge_raw_or_log1p",
                "huber",
                "pls",
                "kernel_ridge",
                "hist_gradient_boosting",
                "fieldwise_ridge",
            ],
            "high_response": [
                "bce",
                "balanced_bce",
                "continuous_regression_threshold",
                "joint_regression_risk",
            ],
            "selection_role": "inner_validation",
            "feature_selection_fit_role": "inner_train",
        },
        "upper_bound_gates": UPPER_BOUND_GATES,
        "historical_reference": {
            "maneuver_macro_f1": 0.9409316261641733,
            "response_rmse": 0.29117097989424773,
            "high_response_auprc": 0.8838786894481233,
        },
        "support_isolation": support_rows,
        "confirmed_metrics_mutable": False,
    }
    payload["protocol_sha256"] = stable_payload_sha256(payload)
    return payload


def blocked_safe_fusion_rows(gate_rows):
    rows = []
    gate_stats = []
    for row in gate_rows:
        rows.append(
            {
                "task": row["task"],
                "metric": row["metric"],
                "value": None,
                "threshold": None,
                "gate_passed": False,
                "status": "blocked_by_upper_bound_gate",
                "outer_test_accessed": False,
            }
        )
        gate_stats.append(
            {
                "task": row["task"],
                "fold_id": None,
                "selected_gate": None,
                "direct_metric": None,
                "continuous_metric": None,
                "fused_metric": None,
                "status": "not_trained_upper_bound_failed",
            }
        )
    return rows, gate_stats


def dependency_rows():
    rows = []
    for name in ("numpy", "pandas", "scikit-learn", "torch", "aeon", "matplotlib"):
        try:
            version = importlib.metadata.version(name)
            status = "available"
        except importlib.metadata.PackageNotFoundError:
            version = None
            status = "unavailable"
        rows.append({"dependency": name, "version": version, "status": status})
    rows.extend(
        (
            {
                "dependency": "python",
                "version": sys.version.split()[0],
                "status": "available",
            },
            {
                "dependency": "platform",
                "version": platform.platform(),
                "status": "available",
            },
        )
    )
    return rows


def task_diagnostic_rows(*, target_frame, threshold_frame, plans):
    rows = []
    for fold_id, plan in plans.items():
        maneuver = target_frame[
            (target_frame["fold_id"] == fold_id)
            & (
                target_frame["task_slug"]
                == "maneuver_intensity_classification"
            )
        ]
        response = target_frame[
            (target_frame["fold_id"] == fold_id)
            & (target_frame["task_slug"] == "physiology_response_prediction")
            & (target_frame["status"] == "completed")
        ]
        class_bounds = threshold_frame[
            (threshold_frame["fold_id"] == fold_id)
            & (threshold_frame["parameter_type"] == "class_bounds")
        ].iloc[0]
        high_bound = threshold_frame[
            (threshold_frame["fold_id"] == fold_id)
            & (threshold_frame["parameter_type"] == "high_response_bound")
        ].iloc[0]
        payload = {
            "fold_id": fold_id,
            "purged_train_context_count": len(
                plan["purged_for_full_35s_support_sample_ids"]
            ),
            "maneuver_lower_bound": float(class_bounds.lower_bound),
            "maneuver_upper_bound": float(class_bounds.upper_bound),
            "high_response_threshold": float(high_bound.high_response_threshold),
        }
        for role in ("train", "validation"):
            maneuver_role = maneuver[maneuver["role"] == role]
            response_role = response[response["role"] == role]
            counts = maneuver_role["class_target"].value_counts().sort_index()
            payload[f"{role}_maneuver_count"] = len(maneuver_role)
            payload[f"{role}_maneuver_class_counts"] = ";".join(
                f"{int(label)}:{int(count)}" for label, count in counts.items()
            )
            payload[f"{role}_maneuver_class_count"] = int(
                maneuver_role["class_target"].nunique()
            )
            payload[f"{role}_response_count"] = len(response_role)
            payload[f"{role}_response_mean"] = float(
                response_role["continuous_target"].mean()
            )
            payload[f"{role}_response_std"] = float(
                response_role["continuous_target"].std(ddof=1)
            )
            payload[f"{role}_high_response_rate"] = float(
                response_role["binary_target"].mean()
            )
        rows.append(payload)
    return rows


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
