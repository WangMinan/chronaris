from __future__ import annotations

import pandas as pd
import pytest

from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    InnerRoleAccessGuard,
)
from chronaris.evaluation.application_tasks.task_stability_audits import (
    leakage_proxy_rows,
)
from chronaris.evaluation.application_tasks.task_stability_candidates import (
    candidate_manifest,
    run_candidate_split,
)
from chronaris.evaluation.application_tasks.task_stability_contracts import (
    decide_allowance,
    high_response_metrics,
    maneuver_metric_ceiling,
    response_metrics,
)
from chronaris.evaluation.application_tasks.task_stability_splits import (
    _dedup_rows,
    support_audit,
)


def test_missing_class_fixed_macro_ceiling_is_two_thirds():
    audit = maneuver_metric_ceiling([1, 1, 2, 2])
    assert audit["fixed_macro_f1_ceiling"] == pytest.approx(2 / 3)
    assert audit["support_aware_macro_f1_ceiling"] == 1.0
    assert not audit["complete_class_support"]


def test_complete_three_class_split_has_unit_ceiling():
    audit = maneuver_metric_ceiling([0, 0, 1, 1, 2, 2])
    assert audit["fixed_macro_f1_ceiling"] == 1.0
    assert audit["minimum_class_count"] == 2
    assert audit["complete_class_support"]


def test_35_second_support_embargo_is_measured_after_full_support():
    context = pd.DataFrame(
        [
            {"context_id": "train", "sortie_id": "s", "start_offset_ms": 0},
            {"context_id": "validation", "sortie_id": "s", "start_offset_ms": 70_000},
        ]
    )
    result = support_audit(
        {
            "train_sample_ids": ["train"],
            "validation_sample_ids": ["validation"],
        },
        context,
    )
    assert result["support_overlap_count"] == 0
    assert result["minimum_anchor_separation_ms"] == 70_000
    assert result["minimum_support_embargo_ms"] == 35_000


def test_shared_vehicle_time_block_cannot_cross_roles():
    context = pd.DataFrame(
        [
            {"context_id": "pilot_a", "sortie_id": "s", "start_offset_ms": 0},
            {"context_id": "pilot_b", "sortie_id": "s", "start_offset_ms": 0},
        ]
    )
    with pytest.raises(PermissionError, match="shared vehicle support unit"):
        support_audit(
            {
                "train_sample_ids": ["pilot_a"],
                "validation_sample_ids": ["pilot_b"],
            },
            context,
        )


def test_duplicate_validation_support_gets_zero_weight():
    frame = _dedup_rows(
        [
            {
                "fold_id": "a",
                "outer_pool_id": "p1",
                "split_kind": "main_selection",
                "main_selection": True,
                "validation_support_hash": "same",
            },
            {
                "fold_id": "b",
                "outer_pool_id": "p2",
                "split_kind": "diagnostic",
                "main_selection": False,
                "validation_support_hash": "same",
            },
        ]
    )
    assert frame["effective_weight"].tolist() == [1.0, 0.0]
    assert frame.loc[1, "duplicate_of_split_id"] == "a"


def test_response_skill_and_normalized_ap_formulas():
    response = response_metrics(
        train_target=[0.0, 2.0],
        validation_target=[0.0, 2.0],
        prediction=[0.0, 2.0],
    )
    assert response["rmse_ratio"] == 0.0
    assert response["response_skill"] == 1.0
    high = high_response_metrics(
        validation_target=[0, 0, 1, 1], probability=[0.1, 0.2, 0.8, 0.9]
    )
    assert high["prevalence"] == 0.5
    assert high["normalized_ap"] == 1.0


def test_outer_test_access_fails_closed():
    guard = InnerRoleAccessGuard({"train": "train", "validation": "validation"})
    with pytest.raises(PermissionError):
        guard.request(
            fold_id="split",
            role="outer_test",
            sample_ids=("unknown_outer",),
            purpose="forbidden",
        )
    assert not guard.audit_rows[-1]["allowed"]
    assert guard.audit_rows[-1]["outer_test_accessed"] is False


def test_label_source_and_deterministic_derivative_exclusion(tmp_path):
    role_path = tmp_path / "roles.csv"
    pd.DataFrame(
        [
            {
                "feature_name": "label_source",
                "selected_for_maneuver_label": True,
                "allowed_in_maneuver_input": False,
            },
            {
                "feature_name": "allowed_signal",
                "selected_for_maneuver_label": False,
                "allowed_in_maneuver_input": True,
            },
        ]
    ).to_csv(role_path, index=False)
    audit = leakage_proxy_rows(
        role_path=role_path,
        manifest={
            "folds": [
                {
                    "shared_vehicle_unit_cross_role_count": 0,
                    "support_overlap_count": 0,
                }
            ]
        },
        candidate_metrics=pd.DataFrame(),
    )
    assert audit.loc[0, "direct_label_source_overlap_count"] == 0
    assert audit.loc[0, "deterministic_derivative_overlap_count"] == 0
    assert audit.loc[0, "valid"]


def test_diagnostic_candidate_resume_is_deterministic(tmp_path):
    candidate = candidate_manifest()[-1]
    plan = {
        "fold_id": "split",
        "outer_pool_id": "pool",
        "validation_support_hash": "support",
        "train_sample_ids": ["train"],
        "validation_sample_ids": ["validation"],
        "held_out_sample_ids": [],
    }
    kwargs = {
        "candidate": candidate,
        "plan": plan,
        "cache": None,
        "targets": pd.DataFrame(),
        "thresholds": pd.DataFrame(),
        "maneuver_scores": {},
        "field_delta_index": {},
        "state_root": tmp_path,
    }
    first = run_candidate_split(**kwargs)
    second = run_candidate_split(**kwargs)
    assert first == second
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_allowance_decision_supports_response_near_miss_path():
    allowance = decide_allowance(
        protocol_valid=True,
        maneuver={
            "mean_macro_f1": 0.9,
            "median_macro_f1": 0.9,
            "worst_macro_f1": 0.7,
            "minimum_class_recall": 0.6,
            "mean_macro_f1_lift": 0.3,
            "median_score_spearman": 0.6,
        },
        response={
            "median_rmse_ratio": 0.98,
            "mean_response_skill": 0.01,
            "positive_skill_fraction": 0.5,
            "median_spearman": 0.1,
        },
        high_response={
            "mean_normalized_ap": 0.3,
            "median_normalized_ap": 0.3,
            "positive_normalized_ap_fraction": 0.8,
            "mean_auprc_lift": 0.2,
        },
        field_or_dual_gain=True,
    )
    assert allowance["decision"] == "passed_with_response_risk"
    assert allowance["allow_safe_fusion"]
    assert allowance["response_task_risk"] == "high"
