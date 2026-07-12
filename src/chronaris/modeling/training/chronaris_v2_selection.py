"""Fail-closed task-independent selection contracts for Chronaris v2.

This module deliberately contains no downstream labels or metrics. It is the
only ranking surface used before a Chronaris v2 configuration is locked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ChronarisV2StructureCandidate:
    """One cumulative diagnostic structure candidate from the preregistration."""

    candidate_id: str
    architecture_version: str
    semantic_group_encoder: bool
    learned_causal_attention: bool
    private_shared_subspaces: bool
    lag_conditioned_objective: bool
    corrected_physics: bool
    missingness_curriculum: bool


def chronaris_v2_structure_candidates() -> tuple[ChronarisV2StructureCandidate, ...]:
    """Return the exact eight cumulative structure-screen candidates."""

    return (
        ChronarisV2StructureCandidate("structure_01_v1", "v1", False, False, False, False, False, False),
        ChronarisV2StructureCandidate("structure_02_semantic_groups", "v2", True, False, False, False, False, False),
        ChronarisV2StructureCandidate("structure_03_learned_causal", "v2", True, True, False, False, False, False),
        ChronarisV2StructureCandidate("structure_04_private_shared", "v2", True, True, True, False, False, False),
        ChronarisV2StructureCandidate("structure_05_lag_objective", "v2", True, True, True, True, False, False),
        ChronarisV2StructureCandidate("structure_06_corrected_physics", "v2", True, True, True, True, True, False),
        ChronarisV2StructureCandidate("structure_07_missing_curriculum", "v2", True, True, True, True, False, True),
        ChronarisV2StructureCandidate("structure_08_complete_v2", "v2", True, True, True, True, True, True),
    )


def chronaris_v2_structure_candidate(
    candidate_id: str,
) -> ChronarisV2StructureCandidate:
    matches = tuple(
        candidate
        for candidate in chronaris_v2_structure_candidates()
        if candidate.candidate_id == candidate_id
    )
    if len(matches) != 1:
        raise ValueError(f"unknown Chronaris v2 structure candidate: {candidate_id}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class TaskIndependentCandidateEvidence:
    """Development evidence allowed to influence candidate selection."""

    candidate_id: str
    public_self_supervised_validation_loss: float
    vehicle_fidelity_ratio: float
    physiology_fidelity_ratio: float
    worst_fold_fidelity_ratio: float
    effective_rank: float
    near_zero_variance_fraction: float
    clock_offset_mae_s: float
    response_lag_mae_s: float
    v1_clock_offset_mae_s: float
    v1_response_lag_mae_s: float
    causal_future_invariance_passed: bool
    invalid_query_pooling_passed: bool
    lag_mask_passed: bool
    parameter_count: int
    task_labels_opened: bool = False
    outer_test_opened: bool = False
    sealed_confirmation_opened: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        numeric = (
            self.public_self_supervised_validation_loss,
            self.vehicle_fidelity_ratio,
            self.physiology_fidelity_ratio,
            self.worst_fold_fidelity_ratio,
            self.effective_rank,
            self.near_zero_variance_fraction,
            self.clock_offset_mae_s,
            self.response_lag_mae_s,
            self.v1_clock_offset_mae_s,
            self.v1_response_lag_mae_s,
        )
        if min(numeric) < 0:
            raise ValueError("task-independent evidence values must be non-negative")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if self.task_labels_opened or self.outer_test_opened or self.sealed_confirmation_opened:
            raise ValueError(
                "candidate evidence is contaminated by a forbidden selection source"
            )

    @property
    def time_mechanism_ratio(self) -> float:
        clock = _safe_ratio(self.clock_offset_mae_s, self.v1_clock_offset_mae_s)
        lag = _safe_ratio(self.response_lag_mae_s, self.v1_response_lag_mae_s)
        return max(clock, lag)

    @property
    def gate_passed(self) -> bool:
        return (
            self.vehicle_fidelity_ratio >= 0.98
            and self.physiology_fidelity_ratio >= 0.98
            and self.worst_fold_fidelity_ratio >= 0.98
            and self.effective_rank >= 2.0
            and self.near_zero_variance_fraction <= 0.10
            and self.time_mechanism_ratio <= 1.10
            and self.causal_future_invariance_passed
            and self.invalid_query_pooling_passed
            and self.lag_mask_passed
        )

    def to_dict(self) -> Mapping[str, object]:
        return {
            **asdict(self),
            "time_mechanism_ratio": self.time_mechanism_ratio,
            "gate_passed": self.gate_passed,
        }


def rank_task_independent_candidates(
    evidence: Sequence[TaskIndependentCandidateEvidence],
    *,
    top_k: int = 3,
) -> tuple[Mapping[str, object], ...]:
    """Gate then rank by the preregistered task-independent lexicographic key."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ids = [row.candidate_id for row in evidence]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate evidence must contain unique candidate ids")
    eligible = [row for row in evidence if row.gate_passed]
    eligible.sort(
        key=lambda row: (
            row.public_self_supervised_validation_loss,
            -row.worst_fold_fidelity_ratio,
            row.time_mechanism_ratio,
            row.parameter_count,
            row.candidate_id,
        )
    )
    rank_by_id = {row.candidate_id: rank for rank, row in enumerate(eligible, 1)}
    selected_ids = {row.candidate_id for row in eligible[:top_k]}
    return tuple(
        {
            **row.to_dict(),
            "rank": rank_by_id.get(row.candidate_id),
            "selected_for_inner_validation": row.candidate_id in selected_ids,
        }
        for row in sorted(evidence, key=lambda value: value.candidate_id)
    )


def _safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if value == 0 else float("inf")
    return value / baseline
