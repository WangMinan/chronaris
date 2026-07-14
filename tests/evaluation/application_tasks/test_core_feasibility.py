from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronaris.evaluation.application_tasks.core_feasibility_features import (
    DingxinFeatureCache,
    summary_features,
)
from chronaris.evaluation.application_tasks.core_feasibility_models import (
    FoldTargets,
    evaluate_feature_family,
)
from chronaris.evaluation.application_tasks.core_feasibility_protocol import (
    UPPER_BOUND_GATES,
    InnerRoleAccessGuard,
    aggregate_gate_rows,
    audit_train_validation_support,
    purge_train_for_full_support,
)
from chronaris.evaluation.application_tasks.core_feasibility_run import (
    CANDIDATE_COUNT,
    candidate_panel,
)
from chronaris.evaluation.application_tasks.core_feasibility_safe_fusion import (
    GATE_VALUES,
    _evaluate_task_gate,
)


def test_outer_roles_fail_closed_and_are_audited():
    guard = InnerRoleAccessGuard({"train": "train", "validation": "validation", "held": "held_out"})
    assert guard.request(
        fold_id="fold",
        role="inner_train",
        sample_ids=("train",),
        purpose="fit",
    ) == ("train",)
    with pytest.raises(PermissionError):
        guard.request(
            fold_id="fold",
            role="outer_test",
            sample_ids=("held",),
            purpose="probe",
        )
    assert guard.audit_rows[-1]["allowed"] is False
    assert guard.audit_rows[-1]["outer_test_accessed"] is False


def test_support_audit_detects_overlap_and_accepts_isolation():
    catalog = pd.DataFrame(
        [
            {"context_id": "a", "sortie_id": "s", "start_offset_ms": 0, "end_offset_ms": 30_000},
            {"context_id": "b", "sortie_id": "s", "start_offset_ms": 35_000, "end_offset_ms": 65_000},
            {"context_id": "c", "sortie_id": "s", "start_offset_ms": 20_000, "end_offset_ms": 50_000},
        ]
    )
    isolated = audit_train_validation_support(
        plan={
            "fold_id": "isolated",
            "train_sample_ids": ["a"],
            "validation_sample_ids": ["b"],
        },
        context_catalog=catalog,
    )
    overlapping = audit_train_validation_support(
        plan={
            "fold_id": "overlap",
            "train_sample_ids": ["a"],
            "validation_sample_ids": ["c"],
        },
        context_catalog=catalog,
    )
    assert isolated["support_isolated"] is True
    assert overlapping["support_isolated"] is False


def test_full_support_purge_removes_overlap_and_outer_identifiers():
    catalog = pd.DataFrame(
        [
            {
                "context_id": "keep",
                "sortie_id": "s",
                "start_offset_ms": 0,
                "end_offset_ms": 30_000,
            },
            {
                "context_id": "purge",
                "sortie_id": "s",
                "start_offset_ms": 20_000,
                "end_offset_ms": 50_000,
            },
            {
                "context_id": "validation",
                "sortie_id": "s",
                "start_offset_ms": 50_000,
                "end_offset_ms": 80_000,
            },
        ]
    )
    derived = purge_train_for_full_support(
        plan={
            "fold_id": "fold",
            "outer_split_strategy": "outer",
            "inner_split_strategy": "inner",
            "train_sample_ids": ["keep", "purge"],
            "validation_sample_ids": ["validation"],
            "held_out_sample_ids": ["secret_outer_id"],
        },
        context_catalog=catalog,
    )
    assert derived["train_sample_ids"] == ["keep"]
    assert derived["purged_for_full_35s_support_sample_ids"] == ["purge"]
    assert derived["held_out_sample_ids"] == []
    assert "secret_outer_id" not in str(derived)
    assert audit_train_validation_support(
        plan=derived,
        context_catalog=catalog,
    )["support_isolated"] is True


def test_gate_aggregation_uses_one_candidate_across_folds():
    rows = [
        {
            "fold_id": f"f{fold}",
            "candidate_id": candidate,
            "task": "maneuver",
            "metric": "macro_f1",
            "value": value,
        }
        for fold, candidate, value in (
            (1, "a", 0.96),
            (2, "a", 0.95),
            (3, "a", 0.94),
            (1, "b", 0.99),
            (2, "b", 0.50),
            (3, "b", 0.50),
        )
    ]
    gates = aggregate_gate_rows(
        rows,
        gates={"maneuver": UPPER_BOUND_GATES["maneuver"]},
    )
    assert gates[0]["best_candidate_id"] == "a"
    assert gates[0]["best_mean_value"] == pytest.approx(0.95)


def test_summary_features_are_deterministic_and_history_sensitive():
    values = np.arange(2 * 4 * 2, dtype=np.float32).reshape(2, 4, 2)
    mask = np.ones_like(values, dtype=bool)
    cache = DingxinFeatureCache(
        sample_ids=("a", "b"),
        timestamps_s=np.asarray([[0, 10, 20, 30], [0, 10, 20, 30]], dtype=float),
        physiology_values=values,
        physiology_mask=mask,
        physiology_age_s=np.zeros_like(values),
        vehicle_values=values + 1,
        vehicle_mask=mask,
        vehicle_age_s=np.zeros_like(values),
    )
    full = summary_features(
        cache, sample_ids=("a", "b"), modality="vehicle", history_s=30
    )
    recent = summary_features(
        cache, sample_ids=("a", "b"), modality="vehicle", history_s=5
    )
    repeated = summary_features(
        cache, sample_ids=("a", "b"), modality="vehicle", history_s=30
    )
    np.testing.assert_array_equal(full, repeated)
    assert not np.array_equal(full, recent)


def test_linear_upper_bound_head_is_deterministic():
    rng = np.random.default_rng(17)
    train = rng.normal(size=(24, 12))
    validation = rng.normal(size=(12, 12))
    targets = FoldTargets(
        train_maneuver=np.tile(np.arange(3), 8),
        validation_maneuver=np.tile(np.arange(3), 4),
        train_maneuver_score=np.linspace(0, 1, 24),
        validation_maneuver_score=np.linspace(0, 1, 12),
        train_response=np.linspace(0.1, 1.2, 24),
        validation_response=np.linspace(0.1, 1.2, 12),
        train_high_response=np.tile((0, 0, 0, 1), 6),
        validation_high_response=np.tile((0, 0, 0, 1), 3),
    )
    first = evaluate_feature_family(
        fold_id="fold",
        candidate_id="candidate",
        train_features=train,
        validation_features=validation,
        targets=targets,
        family="linear",
        tasks=("maneuver",),
    )
    second = evaluate_feature_family(
        fold_id="fold",
        candidate_id="candidate",
        train_features=train,
        validation_features=validation,
        targets=targets,
        family="linear",
        tasks=("maneuver",),
    )
    assert first == second


def test_label_source_fields_remain_excluded_from_maneuver_input():
    frame = pd.read_csv(
        "docs/artifacts/runs/2026-07-10_fixed-data-audit/field_role_manifest.csv"
    )
    selected = frame[frame["selected_for_maneuver_label"].astype(bool)]
    assert not selected.empty
    assert not selected["allowed_in_maneuver_input"].astype(bool).any()


def test_candidate_budget_and_safe_gate_initialization_are_locked():
    assert CANDIDATE_COUNT == 12
    assert len(candidate_panel()) == 12
    assert len({row["candidate_id"] for row in candidate_panel()}) == 12
    assert GATE_VALUES[0] == 0.0


def test_safe_gate_records_zero_initialization_and_frozen_experts():
    rng = np.random.default_rng(17)
    direct_train = rng.normal(size=(30, 8))
    direct_validation = rng.normal(size=(12, 8))
    continuous_train = rng.normal(size=(30, 8))
    continuous_validation = rng.normal(size=(12, 8))
    targets = FoldTargets(
        train_maneuver=np.tile(np.arange(3), 10),
        validation_maneuver=np.tile(np.arange(3), 4),
        train_maneuver_score=np.zeros(30),
        validation_maneuver_score=np.zeros(12),
        train_response=np.zeros(30),
        validation_response=np.zeros(12),
        train_high_response=np.zeros(30, dtype=np.int64),
        validation_high_response=np.zeros(12, dtype=np.int64),
    )
    result = _evaluate_task_gate(
        fold_id="fold",
        task="maneuver",
        direct_train=direct_train,
        direct_validation=direct_validation,
        continuous_train=continuous_train,
        continuous_validation=continuous_validation,
        targets=targets,
        random_state=17,
    )
    assert result["initial_gate"] == 0.0
    assert result["experts_frozen"] is True
