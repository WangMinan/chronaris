from __future__ import annotations

import pytest

from chronaris.modeling.training.chronaris_v2_selection import (
    TaskIndependentCandidateEvidence,
    chronaris_v2_structure_candidates,
    rank_task_independent_candidates,
)
from chronaris.modeling.training.chronaris_v2_training import (
    chronaris_v2_structure_training_grid,
)


def _evidence(candidate_id: str, *, loss: float = 0.2, **overrides):
    values = {
        "candidate_id": candidate_id,
        "public_self_supervised_validation_loss": loss,
        "vehicle_fidelity_ratio": 0.99,
        "physiology_fidelity_ratio": 0.99,
        "worst_fold_fidelity_ratio": 0.985,
        "effective_rank": 20.0,
        "near_zero_variance_fraction": 0.01,
        "clock_offset_mae_s": 1.05,
        "response_lag_mae_s": 2.1,
        "v1_clock_offset_mae_s": 1.0,
        "v1_response_lag_mae_s": 2.0,
        "causal_future_invariance_passed": True,
        "invalid_query_pooling_passed": True,
        "lag_mask_passed": True,
        "parameter_count": 1000,
    }
    values.update(overrides)
    return TaskIndependentCandidateEvidence(**values)


def test_structure_screen_is_exactly_eight_cumulative_candidates() -> None:
    candidates = chronaris_v2_structure_candidates()
    assert len(candidates) == 8
    assert candidates[0].architecture_version == "v1"
    assert candidates[-1].candidate_id == "structure_08_complete_v2"
    assert candidates[-1].corrected_physics
    assert candidates[-1].missingness_curriculum
    training_candidates = chronaris_v2_structure_training_grid()
    assert len(training_candidates) == 7
    assert {value.structure_candidate_id for value in training_candidates} == {
        value.candidate_id for value in candidates[1:]
    }


def test_task_independent_ranking_gates_before_lexicographic_sort() -> None:
    rows = rank_task_independent_candidates(
        (
            _evidence("good_b", loss=0.20),
            _evidence("good_a", loss=0.19),
            _evidence("failed", loss=0.01, vehicle_fidelity_ratio=0.97),
        ),
        top_k=1,
    )
    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["good_a"]["rank"] == 1
    assert by_id["good_a"]["selected_for_inner_validation"]
    assert by_id["failed"]["rank"] is None
    assert not by_id["failed"]["gate_passed"]


def test_selection_rejects_any_forbidden_evidence_source() -> None:
    with pytest.raises(ValueError, match="contaminated"):
        _evidence("bad", outer_test_opened=True)
