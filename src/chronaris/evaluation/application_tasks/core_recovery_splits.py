"""Nested task-decision and dense-training split utilities for Dingxin recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd


INPUT_HISTORY_MS = 30_000
TARGET_HORIZON_MS = 5_000
MINIMUM_ANCHOR_SEPARATION_MS = INPUT_HISTORY_MS + TARGET_HORIZON_MS


@dataclass(frozen=True, slots=True)
class ContextSupportInterval:
    context_id: str
    sortie_id: str
    view_id: str
    anchor_offset_ms: int
    support_start_ms: int
    support_end_ms: int

    def __post_init__(self) -> None:
        if self.support_end_ms <= self.support_start_ms:
            raise ValueError("context support interval must be non-empty")
        if not self.support_start_ms <= self.anchor_offset_ms <= self.support_end_ms:
            raise ValueError("context anchor must lie inside its support interval")

    def overlaps(self, other: "ContextSupportInterval") -> bool:
        return (
            self.sortie_id == other.sortie_id
            and max(self.support_start_ms, other.support_start_ms)
            < min(self.support_end_ms, other.support_end_ms)
        )


@dataclass(frozen=True, slots=True)
class TaskDecisionSplit:
    split_id: str
    train_context_ids: tuple[str, ...]
    evaluation_context_ids: tuple[str, ...]
    purged_context_ids: tuple[str, ...]
    split_strategy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def context_support_intervals(
    context_catalog: pd.DataFrame,
    *,
    target_horizon_ms: int = TARGET_HORIZON_MS,
) -> tuple[ContextSupportInterval, ...]:
    required = {
        "context_id",
        "sortie_id",
        "view_id",
        "start_offset_ms",
        "end_offset_ms",
    }
    missing = sorted(required - set(context_catalog.columns))
    if missing:
        raise ValueError(f"context catalog is missing columns: {missing}")
    if target_horizon_ms <= 0:
        raise ValueError("target horizon must be positive")
    rows = []
    for row in context_catalog.itertuples(index=False):
        rows.append(
            ContextSupportInterval(
                context_id=str(row.context_id),
                sortie_id=str(row.sortie_id),
                view_id=str(row.view_id),
                anchor_offset_ms=int(row.end_offset_ms),
                support_start_ms=int(row.start_offset_ms),
                support_end_ms=int(row.end_offset_ms) + target_horizon_ms,
            )
        )
    identifiers = [row.context_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("context catalog contains duplicate context identifiers")
    return tuple(rows)


def purge_overlapping_training_contexts(
    train_intervals: Sequence[ContextSupportInterval],
    evaluation_intervals: Sequence[ContextSupportInterval],
) -> tuple[tuple[ContextSupportInterval, ...], tuple[ContextSupportInterval, ...]]:
    kept = []
    purged = []
    for train in train_intervals:
        if any(train.overlaps(evaluation) for evaluation in evaluation_intervals):
            purged.append(train)
        else:
            kept.append(train)
    return tuple(kept), tuple(purged)


def build_task_decision_splits(
    *,
    context_catalog: pd.DataFrame,
    eligible_context_ids: Sequence[str],
    minimum_train_count: int = 12,
    minimum_evaluation_count: int = 4,
    temporal_block_count: int = 3,
) -> tuple[TaskDecisionSplit, ...]:
    """Build CV wholly inside an outer fold's inner-train sample set."""

    if min(minimum_train_count, minimum_evaluation_count, temporal_block_count) <= 0:
        raise ValueError("task-decision split sizes must be positive")
    intervals = {
        row.context_id: row for row in context_support_intervals(context_catalog)
    }
    eligible = tuple(dict.fromkeys(str(value) for value in eligible_context_ids))
    missing = sorted(set(eligible) - set(intervals))
    if missing:
        raise ValueError(f"eligible task-decision contexts are absent: {missing[:5]}")
    selected = [intervals[value] for value in eligible]
    groups = sorted({row.sortie_id for row in selected})
    candidate_splits: list[tuple[str, list[ContextSupportInterval], str]] = []
    if len(groups) >= 2:
        for sortie_id in groups:
            evaluation = [row for row in selected if row.sortie_id == sortie_id]
            candidate_splits.append(
                (f"task-decision-sortie-{sortie_id}", evaluation, "leave_one_inner_train_sortie_out")
            )
    else:
        anchors = sorted({row.anchor_offset_ms for row in selected})
        blocks = _contiguous_blocks(anchors, temporal_block_count)
        for index, block in enumerate(blocks, start=1):
            evaluation = [row for row in selected if row.anchor_offset_ms in block]
            candidate_splits.append(
                (f"task-decision-time-{index:02d}", evaluation, "purged_contiguous_time_block")
            )
    outputs = []
    for split_id, evaluation, strategy in candidate_splits:
        evaluation_ids = {row.context_id for row in evaluation}
        initial_train = [row for row in selected if row.context_id not in evaluation_ids]
        train, purged = purge_overlapping_training_contexts(initial_train, evaluation)
        if len(train) < minimum_train_count or len(evaluation) < minimum_evaluation_count:
            continue
        _assert_support_isolation(train, evaluation)
        outputs.append(
            TaskDecisionSplit(
                split_id=split_id,
                train_context_ids=tuple(row.context_id for row in train),
                evaluation_context_ids=tuple(row.context_id for row in evaluation),
                purged_context_ids=tuple(row.context_id for row in purged),
                split_strategy=strategy,
            )
        )
    return tuple(outputs)


def select_dense_training_anchors(
    *,
    candidate_anchors_ms: Sequence[int],
    training_support_bounds: Sequence[tuple[int, int]],
    protected_intervals: Sequence[ContextSupportInterval],
    sortie_id: str,
) -> tuple[int, ...]:
    """Keep only anchors whose complete 30s+5s support is train-owned."""

    kept = []
    for anchor in sorted(set(int(value) for value in candidate_anchors_ms)):
        support_start = anchor - INPUT_HISTORY_MS
        support_end = anchor + TARGET_HORIZON_MS
        inside_train = any(
            lower <= support_start and support_end <= upper
            for lower, upper in training_support_bounds
        )
        if not inside_train:
            continue
        candidate = ContextSupportInterval(
            context_id=f"dense::{sortie_id}::{anchor}",
            sortie_id=sortie_id,
            view_id="dense_train",
            anchor_offset_ms=anchor,
            support_start_ms=support_start,
            support_end_ms=support_end,
        )
        if any(candidate.overlaps(protected) for protected in protected_intervals):
            continue
        kept.append(anchor)
    return tuple(kept)


def _contiguous_blocks(values: Sequence[int], block_count: int) -> tuple[frozenset[int], ...]:
    if len(values) < block_count:
        return ()
    size, remainder = divmod(len(values), block_count)
    blocks = []
    offset = 0
    for index in range(block_count):
        count = size + int(index < remainder)
        blocks.append(frozenset(values[offset : offset + count]))
        offset += count
    return tuple(block for block in blocks if block)


def _assert_support_isolation(
    train: Sequence[ContextSupportInterval],
    evaluation: Sequence[ContextSupportInterval],
) -> None:
    for left in train:
        for right in evaluation:
            if left.overlaps(right):
                raise ValueError(
                    f"task-decision support overlap: {left.context_id} vs {right.context_id}"
                )
