"""Contracts for Dingxin target reconstruction and matched-clean screening."""

from __future__ import annotations

from collections.abc import Mapping


MATCHED_METHODS = (
    "physiology_only",
    "vehicle_only",
    "naive_time_sync",
    "mult",
    "contiformer",
    "chronaris",
)

TARGET_RECONSTRUCTION_GATES = {
    "matched_clean_panel": {"required_method_count": 6, "required_main_split_count": 6},
    "time_shortcut": {"minimum_standardized_gain": 0.15},
    "future_maneuver_score": {"minimum_median_spearman": 0.65},
    "future_maneuver_trend": {
        "minimum_mean_macro_f1": 0.75,
        "minimum_worst_macro_f1": 0.60,
    },
    "physiology_residual": {
        "maximum_median_rmse_ratio": 0.90,
        "minimum_mean_skill": 0.10,
        "minimum_positive_skill_split_count": 4,
    },
    "high_residual_response": {
        "minimum_mean_normalized_ap": 0.25,
        "minimum_median_normalized_ap": 0.15,
        "minimum_positive_split_count": 5,
    },
    "third_pool": {"minimum_completed_continuous_tasks": 2},
}


def decide_target_reconstruction_allowance(
    *,
    panel: Mapping[str, object],
    time_shortcut: Mapping[str, float],
    maneuver_score: Mapping[str, float],
    maneuver_trend: Mapping[str, float],
    physiology_residual: Mapping[str, float],
    high_residual: Mapping[str, float],
    third_pool: Mapping[str, object],
    protocol_valid: bool,
) -> dict[str, object]:
    """Apply the predeclared 1C gates without consulting outer-test evidence."""

    gate = TARGET_RECONSTRUCTION_GATES
    checks = {
        "protocol_valid": bool(protocol_valid),
        "matched_clean_panel_complete": bool(
            panel["completed_method_count"]
            == gate["matched_clean_panel"]["required_method_count"]
            and panel["completed_main_split_count"]
            == gate["matched_clean_panel"]["required_main_split_count"]
            and panel["missing_method_split_units"] == 0
        ),
        "time_shortcut_suppressed": bool(
            time_shortcut["standardized_gain"]
            >= gate["time_shortcut"]["minimum_standardized_gain"]
        ),
        "future_maneuver_score_passed": bool(
            maneuver_score["median_spearman"]
            >= gate["future_maneuver_score"]["minimum_median_spearman"]
        ),
        "future_maneuver_trend_passed": bool(
            maneuver_trend["mean_macro_f1"]
            >= gate["future_maneuver_trend"]["minimum_mean_macro_f1"]
            and maneuver_trend["worst_macro_f1"]
            >= gate["future_maneuver_trend"]["minimum_worst_macro_f1"]
        ),
        "physiology_residual_passed": bool(
            physiology_residual["median_rmse_ratio"]
            <= gate["physiology_residual"]["maximum_median_rmse_ratio"]
            and physiology_residual["mean_skill"]
            >= gate["physiology_residual"]["minimum_mean_skill"]
            and physiology_residual["positive_skill_split_count"]
            >= gate["physiology_residual"]["minimum_positive_skill_split_count"]
        ),
        "high_residual_response_passed": bool(
            high_residual["mean_normalized_ap"]
            >= gate["high_residual_response"]["minimum_mean_normalized_ap"]
            and high_residual["median_normalized_ap"]
            >= gate["high_residual_response"]["minimum_median_normalized_ap"]
            and high_residual["positive_split_count"]
            >= gate["high_residual_response"]["minimum_positive_split_count"]
        ),
        "third_pool_continuous_evaluation_completed": bool(
            third_pool["completed_continuous_task_count"]
            >= gate["third_pool"]["minimum_completed_continuous_tasks"]
        ),
    }
    passed = all(checks.values())
    blockers = [name for name, value in checks.items() if not value]
    return {
        "decision": "accepted" if passed else "gap",
        "allow_safe_fusion": passed,
        "allow_task_aware_research": passed,
        "primary_blocker": None if passed else blockers[0],
        "failed_checks": blockers,
        "checks": checks,
        "outer_test_opened": False,
        "chronaris_backbone_modified": False,
        "chronaris_backbone_trained": False,
        "teacher_distillation_started": False,
    }
