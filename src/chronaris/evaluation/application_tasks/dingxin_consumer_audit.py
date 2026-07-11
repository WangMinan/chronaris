"""Acceptance checks for five-fold Dingxin frozen consumers."""

from __future__ import annotations


def build_dingxin_consumer_acceptance_rows(
    *,
    result_rows,
    metric_rows,
    fusion_gain_rows,
    target_manifests,
    prediction_hash_match_count,
):
    statuses = {row["status"] for row in metric_rows}
    unavailable = [row for row in metric_rows if row["status"] == "unavailable"]
    configs = {row["consumer_config_sha256"] for row in result_rows}
    folds = {row["fold_id"] for row in result_rows}
    methods = {row["method_name"] for row in result_rows}
    return [
        _check("five_folds_consumed", len(folds) == 5, len(folds), 5),
        _check("six_methods_consumed", len(methods) == 6, sorted(methods), 6),
        _check("thirty_method_fold_bundles", len(result_rows) == 30, len(result_rows), 30),
        _check("sixty_components_built_or_reused", sum(row["initial_component_count"] for row in result_rows) == 60, sum(row["initial_component_count"] for row in result_rows), 60),
        _check("resume_reuses_sixty_components", sum(row["resume_component_count"] for row in result_rows) == 60 and all(row["resume_all_components"] for row in result_rows), sum(row["resume_component_count"] for row in result_rows), 60),
        _check("resume_predictions_are_identical", prediction_hash_match_count == 30, prediction_hash_match_count, 30),
        _check("fixed_consumer_config", len(configs) == 1, sorted(configs), "one config hash"),
        _check("expected_metric_matrix", len(metric_rows) == 1680, len(metric_rows), 1680),
        _check("metric_status_is_structured", statuses <= {"available", "unavailable"} and all(row["reason"] == "metric_not_defined" for row in unavailable), sorted(statuses), "available/unavailable"),
        _check("expected_fusion_gain_matrix", len(fusion_gain_rows) == 1120, len(fusion_gain_rows), 1120),
        _check("outer_train_thresholds_smoke_only", all(row["threshold_scope"] == "outer_train_smoke_only" and row["smoke_only"] for row in metric_rows), False, False),
        _check("target_counts_are_fixed", sum(sum(item["maneuver_counts"].values()) for item in target_manifests) == 440 and sum(sum(item["response_counts"].values()) for item in target_manifests) == 425, {"maneuver": sum(sum(item["maneuver_counts"].values()) for item in target_manifests), "response": sum(sum(item["response_counts"].values()) for item in target_manifests)}, {"maneuver": 440, "response": 425}),
        _check("real_metrics_not_mixed_with_simulation", {row["dataset"] for row in metric_rows} == {"dingxin_existing_dual_stream"}, sorted({row["dataset"] for row in metric_rows}), ["dingxin_existing_dual_stream"]),
        _check("encoder_labels_remain_closed", all(not row["label_used_for_encoder_training"] for row in result_rows), False, False),
        _check("smoke_metrics_not_confirmed", all(row["smoke_only"] for row in metric_rows), True, True),
    ]


def _check(check_id, passed, actual, expected):
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }
