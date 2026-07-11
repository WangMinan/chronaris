"""Acceptance checks for Dingxin inner validation and overlap embargoes."""

from __future__ import annotations


def build_inner_split_acceptance_rows(
    *,
    plans,
    role_rows,
    coverage_rows,
    overlap_count,
    deterministic_rebuild,
):
    coverage = {
        (row["fold_id"], row["task_slug"], row["role"]): row
        for row in coverage_rows
    }
    checks = [
        _check(
            "five_outer_folds_have_inner_validation",
            len(plans) == 5
            and all(plan.fold.validation_sample_ids for plan in plans),
            {"fold_count": len(plans)},
        ),
        _check(
            "all_available_inputs_have_one_fold_role",
            all(
                len(plan.fold.train_sample_ids)
                + len(plan.fold.validation_sample_ids)
                + len(plan.embargo_sample_ids)
                + len(plan.fold.held_out_sample_ids)
                == 93
                for plan in plans
            ),
            {
                plan.fold.fold_id: {
                    "inner_train": len(plan.fold.train_sample_ids),
                    "validation": len(plan.fold.validation_sample_ids),
                    "embargo": len(plan.embargo_sample_ids),
                    "outer_test": len(plan.fold.held_out_sample_ids),
                }
                for plan in plans
            },
        ),
        _check(
            "fold_roles_are_sample_disjoint",
            all(_plan_roles_disjoint(plan) for plan in plans),
            {"fold_count": len(plans)},
        ),
        _check(
            "shared_vehicle_intervals_do_not_overlap",
            overlap_count == 0,
            {"inner_train_validation_overlap_count": overlap_count},
        ),
        _check(
            "same_sortie_splits_use_temporal_embargo",
            all(
                plan.embargo_sample_ids
                for plan in plans
                if plan.inner_split_strategy.startswith("shared_vehicle_temporal")
            ),
            {
                plan.fold.fold_id: len(plan.embargo_sample_ids)
                for plan in plans
                if plan.inner_split_strategy.startswith("shared_vehicle_temporal")
            },
        ),
        _check(
            "cross_sortie_splits_hold_out_complete_sortie",
            all(
                len(plan.validation_group_ids) == 1
                for plan in plans
                if plan.inner_split_strategy
                == "leave_one_outer_train_sortie_out"
            ),
            {
                plan.fold.fold_id: list(plan.validation_group_ids)
                for plan in plans
                if plan.inner_split_strategy
                == "leave_one_outer_train_sortie_out"
            },
        ),
        _check(
            "classification_roles_cover_three_classes",
            all(
                row["class_values"] == [0, 1, 2]
                for row in coverage_rows
                if row["task_slug"] == "maneuver_intensity_classification"
            ),
            {"classification_role_count": sum(row["task_slug"] == "maneuver_intensity_classification" for row in coverage_rows)},
        ),
        _check(
            "response_roles_cover_binary_and_continuous_targets",
            all(
                row["binary_values"] == [0, 1]
                and row["continuous_finite_count"] == row["context_count"]
                and row["context_count"] > 0
                for row in coverage_rows
                if row["task_slug"] == "physiology_response_prediction"
            ),
            {"response_role_count": sum(row["task_slug"] == "physiology_response_prediction" for row in coverage_rows)},
        ),
        _check(
            "outer_thresholds_are_smoke_only",
            all(
                row["target_threshold_scope"] == "outer_train_smoke_only"
                for row in coverage_rows
            ),
            {"formal_screen_requires_nested_target_refit": True},
        ),
        _check(
            "inner_split_rebuild_is_deterministic",
            deterministic_rebuild,
            {"deterministic": deterministic_rebuild},
        ),
        _check(
            "role_manifest_has_no_duplicate_fold_context",
            not role_rows.duplicated(["fold_id", "context_id"]).any(),
            {"role_row_count": len(role_rows)},
        ),
    ]
    return checks


def _plan_roles_disjoint(plan):
    values = (
        set(plan.fold.train_sample_ids),
        set(plan.fold.validation_sample_ids),
        set(plan.embargo_sample_ids),
        set(plan.fold.held_out_sample_ids),
    )
    return all(
        not values[left] & values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _check(name, passed, details):
    return {"check_name": name, "passed": bool(passed), "details": details}
