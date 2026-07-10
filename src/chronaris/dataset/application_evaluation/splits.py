"""Outer split construction for fixed-data application contexts."""

from __future__ import annotations

from typing import Sequence

from chronaris.dataset.application_evaluation.contracts import (
    ApplicationContextRecord,
    OuterFoldDefinition,
)


SUPPORTED_SPLIT_STRATEGIES = ("leave_one_view_out", "leave_one_sortie_out")


def build_outer_folds(
    contexts: Sequence[ApplicationContextRecord],
    *,
    split_strategy: str,
) -> tuple[OuterFoldDefinition, ...]:
    """Build deterministic outer folds without fitting task parameters."""

    if split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(f"unsupported split strategy: {split_strategy}")
    group_attr = "view_id" if split_strategy == "leave_one_view_out" else "sortie_id"
    groups = tuple(sorted({str(getattr(context, group_attr)) for context in contexts}))
    folds: list[OuterFoldDefinition] = []
    for fold_index, held_out_group in enumerate(groups, start=1):
        train_groups = tuple(group for group in groups if group != held_out_group)
        test_groups = (held_out_group,)
        train_contexts = [
            context for context in contexts
            if str(getattr(context, group_attr)) in train_groups
        ]
        test_contexts = [
            context for context in contexts
            if str(getattr(context, group_attr)) == held_out_group
        ]
        folds.append(
            OuterFoldDefinition(
                fold_id=f"{split_strategy}__fold{fold_index:02d}",
                split_strategy=split_strategy,
                held_out_group=held_out_group,
                train_group_ids=train_groups,
                test_group_ids=test_groups,
                classification_train_context_ids=tuple(
                    context.context_id for context in train_contexts if context.classification_eligible
                ),
                classification_test_context_ids=tuple(
                    context.context_id for context in test_contexts if context.classification_eligible
                ),
                response_train_context_ids=tuple(
                    context.context_id for context in train_contexts if context.response_eligible
                ),
                response_test_context_ids=tuple(
                    context.context_id for context in test_contexts if context.response_eligible
                ),
            )
        )
    return tuple(folds)
