"""Context and grouped-split contract tests."""

from __future__ import annotations

from chronaris.dataset.application_evaluation import (
    build_application_contexts,
    build_outer_folds,
)

from tests.evaluation.application_tasks.helpers import make_records


def test_context_counts_match_locked_fixed_data_shape() -> None:
    contexts = build_application_contexts(make_records())

    assert len(contexts) == 96
    assert sum(context.classification_eligible for context in contexts) == 96
    assert sum(context.response_eligible for context in contexts) == 93
    assert len({context.context_id for context in contexts}) == 96


def test_outer_folds_keep_train_and_test_contexts_disjoint() -> None:
    contexts = build_application_contexts(make_records())

    view_folds = build_outer_folds(contexts, split_strategy="leave_one_view_out")
    sortie_folds = build_outer_folds(contexts, split_strategy="leave_one_sortie_out")

    assert len(view_folds) == 3
    assert len(sortie_folds) == 2
    for fold in (*view_folds, *sortie_folds):
        assert set(fold.classification_train_context_ids).isdisjoint(
            fold.classification_test_context_ids
        )
        assert set(fold.response_train_context_ids).isdisjoint(fold.response_test_context_ids)
        assert set(fold.train_group_ids).isdisjoint(fold.test_group_ids)
