"""Fold-fitted label and test-group isolation tests."""

from __future__ import annotations

import copy

from chronaris.dataset.application_evaluation import (
    build_application_contexts,
    build_field_role_manifest,
    build_fold_task_labels,
    build_outer_folds,
)

from tests.evaluation.application_tasks.helpers import make_records, vehicle_labels_by_sortie


def test_fold_label_thresholds_are_fitted_without_test_context_values() -> None:
    records = make_records()
    contexts = build_application_contexts(records)
    roles = build_field_role_manifest(
        records,
        vehicle_labels_by_sortie=vehicle_labels_by_sortie(records),
    )
    fold = build_outer_folds(contexts, split_strategy="leave_one_view_out")[0]
    baseline = build_fold_task_labels(records, contexts, fold, roles)

    perturbed = records.copy(deep=True)
    held_out_view = fold.held_out_group
    for row_index in perturbed.index[perturbed["view_id"] == held_out_view]:
        vehicle_stats = copy.deepcopy(perturbed.at[row_index, "raw_vehicle_stats"])
        physiology_stats = copy.deepcopy(perturbed.at[row_index, "raw_physiology_stats"])
        for payload in vehicle_stats["features"].values():
            payload["std"] = float(payload["std"]) * 10_000.0
            payload["delta"] = float(payload["delta"]) * 10_000.0
        for payload in physiology_stats["features"].values():
            payload["mean"] = float(payload["mean"]) * 10_000.0
        perturbed.at[row_index, "raw_vehicle_stats"] = vehicle_stats
        perturbed.at[row_index, "raw_physiology_stats"] = physiology_stats
    rerun = build_fold_task_labels(perturbed, contexts, fold, roles)

    assert baseline.status == "completed"
    assert rerun.status == "completed"
    assert baseline.threshold_rows == rerun.threshold_rows
    assert baseline.fit_sample_hashes == rerun.fit_sample_hashes


def test_fold_labels_record_both_application_tasks_and_split_roles() -> None:
    records = make_records()
    contexts = build_application_contexts(records)
    roles = build_field_role_manifest(
        records,
        vehicle_labels_by_sortie=vehicle_labels_by_sortie(records),
    )
    fold = build_outer_folds(contexts, split_strategy="leave_one_sortie_out")[0]
    result = build_fold_task_labels(records, contexts, fold, roles)

    assert result.status == "completed"
    assert {row["task_name"] for row in result.label_rows} == {
        "当前机动强度弱监督分类",
        "机动诱发生理响应预测",
    }
    assert {row["split_role"] for row in result.label_rows} == {"train", "test"}
    assert {row["fit_sample_hash"] for row in result.label_rows} == set(
        result.fit_sample_hashes.values()
    )


def test_zero_iqr_maneuver_semantics_are_excluded_from_fold_scalers() -> None:
    records = make_records()
    for row_index in records.index:
        stats = copy.deepcopy(records.at[row_index, "raw_vehicle_stats"])
        stats["features"]["BUS.self_roll"]["std"] = 0.0
        stats["features"]["BUS.self_roll"]["delta"] = 0.0
        records.at[row_index, "raw_vehicle_stats"] = stats
    contexts = build_application_contexts(records)
    roles = build_field_role_manifest(
        records,
        vehicle_labels_by_sortie=vehicle_labels_by_sortie(records),
    )
    fold = build_outer_folds(contexts, split_strategy="leave_one_view_out")[0]
    result = build_fold_task_labels(
        records,
        contexts,
        fold,
        roles,
        minimum_maneuver_semantic_count=3,
    )

    excluded = [
        row
        for row in result.threshold_rows
        if row["parameter_type"] == "semantic_scaler_excluded"
    ]
    fitted = [
        row
        for row in result.threshold_rows
        if row["parameter_type"] == "semantic_scaler"
    ]
    assert any(row["parameter_name"] == "roll" for row in excluded)
    assert not any(row["parameter_name"] == "roll" for row in fitted)
