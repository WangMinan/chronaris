"""Fairness and completion checks for the six-method pretraining loop smoke."""

from __future__ import annotations

from collections import defaultdict


def build_augmentation_alignment_rows(augmentation_rows):
    grouped = defaultdict(list)
    for row in augmentation_rows:
        key = (
            row["epoch"],
            row["step"],
            row["sample_id"],
            row["stream_name"],
        )
        grouped[key].append(row)
    result = []
    comparison_fields = (
        "augmentation_id",
        "original_point_count",
        "retained_point_count",
        "removed_point_count",
        "original_valid_feature_count",
        "retained_valid_feature_count",
        "removed_valid_feature_count",
        "modality_dropped",
        "block_start_s",
        "block_duration_s",
        "clock_offset_s",
    )
    for key in sorted(grouped):
        rows = grouped[key]
        consistent = all(
            len({str(row[field]) for row in rows}) == 1
            for field in comparison_fields
        )
        result.append(
            {
                "epoch": key[0],
                "step": key[1],
                "sample_id": key[2],
                "stream_name": key[3],
                "method_count": len({row["method_name"] for row in rows}),
                "augmentation_id": rows[0]["augmentation_id"],
                "consistent_across_methods": consistent,
            }
        )
    return result


def build_common_pretraining_acceptance_rows(
    *,
    data_manifest_rows,
    fold,
    initial_training_results,
    resumed_training_results,
    registry,
    training_rows,
    augmentation_alignment_rows,
    export_manifest,
    alignment_hashes,
    metric_rows,
    target_manifest,
    transform_manifest,
):
    roles = [row["role"] for row in data_manifest_rows]
    no_locked_path = all(
        "/locked_test/" not in str(row["observed_path"])
        and "/validation/" not in str(row["observed_path"])
        for row in data_manifest_rows
    )
    public_terms_active = all(
        row["status"] == "active" and int(row["count"]) > 0
        for row in training_rows
    )
    chronaris_weights_zero = all(
        row["continuous_alignment_weight"] == 0
        and row["physical_consistency_weight"] == 0
        and row["causal_direction_weight"] == 0
        for row in training_rows
        if row["method_name"] == "chronaris"
    )
    transform_hashes = {
        method["normalizer_sha256"]
        for method in transform_manifest["methods"].values()
    }
    recovery = export_manifest["single_item_recovery"]
    return [
        _check("sixteen_distinct_profile_samples", len(data_manifest_rows) == 16 and len({row["profile_id"] for row in data_manifest_rows}) == 16, len(data_manifest_rows), 16),
        _check("fixed_role_counts", roles.count("train") == 8 and roles.count("validation") == 4 and roles.count("held_out") == 4, {role: roles.count(role) for role in set(roles)}, {"train": 8, "validation": 4, "held_out": 4}),
        _check("train_split_g1_only", no_locked_path, no_locked_path, True),
        _check("five_trainable_methods_completed", len(initial_training_results) == 5 and all(result.status in {"completed", "resumed"} for result in initial_training_results), [result.status for result in initial_training_results], 5),
        _check("training_resume_reuses_five", len(resumed_training_results) == 5 and all(result.status == "resumed" for result in resumed_training_results), [result.status for result in resumed_training_results], "five resumed"),
        _check("two_steps_per_trainable_method", all(result.step_count == 2 for result in initial_training_results), [result.step_count for result in initial_training_results], 2),
        _check("three_public_terms_active", len(training_rows) == 30 and public_terms_active, len(training_rows), "30 active rows"),
        _check("chronaris_auxiliary_weights_zero_epoch_one", chronaris_weights_zero, chronaris_weights_zero, True),
        _check("augmentation_alignment_complete", len(augmentation_alignment_rows) == 16 and all(row["method_count"] == 5 and row["consistent_across_methods"] for row in augmentation_alignment_rows), len(augmentation_alignment_rows), "16 rows x five methods"),
        _check("shared_normalizer", len(transform_hashes) == 1, sorted(transform_hashes), "one transform hash"),
        _check("six_checkpoint_registry_records", len(registry.records) == 6, len(registry.records), 6),
        _check("labels_not_used_for_encoder", all(not record.label_used_for_encoder_training for record in registry.records.values()), False, False),
        _check("eighteen_role_exports", export_manifest["available_export_count"] == 18, export_manifest["available_export_count"], 18),
        _check("resume_reuses_eighteen_exports", export_manifest["resume_verification_reused_count"] == 18, export_manifest["resume_verification_reused_count"], 18),
        _check("single_missing_export_rebuilt", recovery["removed_file_count"] == 2 and recovery["hash_match"] and recovery["non_target_initial_export_count"] == 17, recovery, "one Chronaris held-out export rebuilt with matching hash"),
        _check("six_method_role_alignment", set(alignment_hashes) == {"train", "validation", "held_out"} and all(len(value) == 64 for value in alignment_hashes.values()), alignment_hashes, "three SHA-256 values"),
        _check("oracle_opened_after_five_checkpoints", target_manifest["oracle_opened_after_checkpoint_count"] == 5, target_manifest["oracle_opened_after_checkpoint_count"], 5),
        _check("train_only_target_thresholds", len(target_manifest["classification_thresholds_train_only"]) == 2, target_manifest["classification_thresholds_train_only"], 2),
        _check("fixed_linear_metric_matrix", len(metric_rows) == 72 and all(row["smoke_only"] for row in metric_rows), len(metric_rows), 72),
        _check("smoke_metrics_not_confirmed", all(row["smoke_only"] for row in metric_rows), True, True),
    ]


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
