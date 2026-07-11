"""Acceptance and trajectory-level paired audits for the G4 consumer smoke."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from chronaris.evaluation.application_tasks.application_metrics import (
    paired_trajectory_statistic,
)


def build_paired_unit_statistic_rows(
    unit_score_rows,
    *,
    sample_manifest_rows,
    reference_method: str = "chronaris",
    seed: int = 17,
    smoke_only: bool = True,
):
    trajectory_by_sample = {
        row["sample_id"]: row["trajectory_id"] for row in sample_manifest_rows
    }
    grouped = defaultdict(list)
    for row in unit_score_rows:
        if row["role"] != "held_out":
            continue
        normalized = float(row["value"])
        if row["direction"] == "lower":
            normalized = -normalized
        key = (
            row["method"],
            row["consumer"],
            row["metric"],
            trajectory_by_sample[row["sample_id"]],
        )
        grouped[key].append(normalized)
    unit_means = {key: float(np.mean(values)) for key, values in grouped.items()}
    methods = sorted({key[0] for key in unit_means if key[0] != reference_method})
    task_keys = sorted({(key[1], key[2]) for key in unit_means})
    rows = []
    for consumer, metric in task_keys:
        for comparison_method in methods:
            reference_by_trajectory = {
                key[3]: value
                for key, value in unit_means.items()
                if key[:3] == (reference_method, consumer, metric)
            }
            comparison_by_trajectory = {
                key[3]: value
                for key, value in unit_means.items()
                if key[:3] == (comparison_method, consumer, metric)
            }
            trajectories = sorted(
                set(reference_by_trajectory) & set(comparison_by_trajectory)
            )
            if len(trajectories) < 2:
                continue
            statistic = paired_trajectory_statistic(
                [reference_by_trajectory[value] for value in trajectories],
                [comparison_by_trajectory[value] for value in trajectories],
                seed=seed,
            )
            rows.append(
                {
                    "role": "held_out",
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "consumer": consumer,
                    "unit_metric": metric,
                    "difference_semantics": "positive_favors_reference",
                    "mean_difference": statistic.mean_difference,
                    "bootstrap_lower": statistic.bootstrap_lower,
                    "bootstrap_upper": statistic.bootstrap_upper,
                    "permutation_p_value": statistic.permutation_p_value,
                    "independent_unit_count": statistic.independent_unit_count,
                    "smoke_only": bool(smoke_only),
                }
            )
    return rows


def build_application_consumer_acceptance_rows(
    *,
    data,
    checkpoint_rows,
    target_manifest,
    initial_export_rows,
    recovery_export_rows,
    resume_export_rows,
    alignment_hashes,
    method_results,
    resumed_method_results,
    metric_rows,
    paired_rows,
    representation_recovery,
    consumer_recovery,
    checkpoint_hashes_unchanged,
):
    role_counts = {
        role: len(sample_ids) for role, sample_ids in data.role_sample_ids.items()
    }
    checks = [
        _check(
            "application_context_grid_complete",
            len(data.batch.sample_ids) == 64
            and role_counts == {"train": 32, "validation": 16, "held_out": 16},
            {"sample_count": len(data.batch.sample_ids), "role_counts": role_counts},
        ),
        _check(
            "pretraining_checkpoint_gate_complete",
            len(checkpoint_rows) == 6
            and sum(row["training_status"] == "completed" for row in checkpoint_rows) == 5
            and not any(row["label_used_for_encoder_training"] for row in checkpoint_rows),
            {"checkpoint_count": len(checkpoint_rows)},
        ),
        _check(
            "oracle_opened_only_after_five_checkpoints",
            target_manifest["oracle_opened_after_checkpoint_count"] == 5
            and set(target_manifest["allowed_oracle_fields"])
            == {"true_time_s", "workload", "maneuver_state"},
            {
                "checkpoint_count": target_manifest[
                    "oracle_opened_after_checkpoint_count"
                ],
                "allowed_fields": target_manifest["allowed_oracle_fields"],
            },
        ),
        _check(
            "six_method_representation_coverage",
            len(initial_export_rows) == 18
            and len(alignment_hashes) == 3
            and all(row["sample_count"] == role_counts[row["role"]] for row in initial_export_rows),
            {"export_count": len(initial_export_rows), "alignment": alignment_hashes},
        ),
        _check(
            "representation_single_item_recovery",
            representation_recovery["hash_match"]
            and sum(row["status"] == "completed" for row in recovery_export_rows) == 1
            and sum(row["status"] == "resumed" for row in recovery_export_rows) == 17,
            representation_recovery,
        ),
        _check(
            "representation_resume_all_items",
            len(resume_export_rows) == 18
            and all(row["status"] == "resumed" for row in resume_export_rows),
            {"resumed_count": sum(row["status"] == "resumed" for row in resume_export_rows)},
        ),
        _check(
            "three_consumers_cover_six_methods",
            len(method_results) == 6
            and len(resumed_method_results) == 6
            and all(result.status == "resumed" for result in resumed_method_results),
            {
                "method_count": len(method_results),
                "resume_count": sum(result.status == "resumed" for result in resumed_method_results),
            },
        ),
        _check(
            "minirocket_train_only_variance_filter_recorded",
            all(
                result.model_manifest[
                    "minirocket_variance_filter_fit_role"
                ]
                == "train"
                and 0
                < result.model_manifest[
                    "minirocket_input_channel_count_after_train_filter"
                ]
                <= result.model_manifest[
                    "minirocket_input_channel_count_before_filter"
                ]
                for result in method_results
            ),
            {
                result.method_name: result.model_manifest[
                    "minirocket_input_channel_count_after_train_filter"
                ]
                for result in method_results
            },
        ),
        _check(
            "consumer_metric_schema_complete",
            len(metric_rows) == 384
            and {row["consumer"] for row in metric_rows}
            == {"linear", "minirocket", "causal_tcn_raw", "causal_tcn_duration"},
            {
                "metric_row_count": len(metric_rows),
                "available_count": sum(row["status"] == "available" for row in metric_rows),
            },
        ),
        _check(
            "trajectory_paired_interface_available",
            bool(paired_rows)
            and all(row["independent_unit_count"] == 4 for row in paired_rows),
            {"paired_row_count": len(paired_rows)},
        ),
        _check(
            "consumer_component_single_item_recovery",
            consumer_recovery["minirocket"]["component_status"]
            == {
                "linear": "resumed",
                "minirocket": "completed",
                "causal_tcn": "resumed",
            }
            and consumer_recovery["tcn"]["component_status"]
            == {
                "linear": "resumed",
                "minirocket": "resumed",
                "causal_tcn": "completed",
            }
            and all(
                value
                for section in ("minirocket", "tcn")
                for key, value in consumer_recovery[section].items()
                if key != "component_status"
            ),
            consumer_recovery,
        ),
        _check(
            "pretraining_checkpoints_immutable",
            checkpoint_hashes_unchanged,
            {"unchanged": checkpoint_hashes_unchanged},
        ),
    ]
    return checks


def _check(name, passed, details):
    return {
        "check_name": name,
        "passed": bool(passed),
        "details": details,
    }
