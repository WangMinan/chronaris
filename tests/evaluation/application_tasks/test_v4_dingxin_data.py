import pytest

from chronaris.evaluation.application_tasks.v4_dingxin_data import dingxin_v4_inner_folds
from tests.evaluation.dingxin.test_simple_downstream_protocol import _synthetic_raw_bundle


def test_dingxin_inner_roles_keep_shared_views_and_entire_target_support_together():
    contexts = _synthetic_raw_bundle().contexts
    folds, embargo = dingxin_v4_inner_folds(contexts)
    for fold in folds:
        roles = (fold.train_sample_ids, fold.validation_sample_ids, embargo[fold.fold_id], fold.held_out_sample_ids)
        assert set().union(*map(set, roles)) == set(contexts.context_id)
        assert sum(map(len, roles)) == len(contexts)
        train, validation, excluded, held = (contexts[contexts.context_id.isin(ids)] for ids in roles)
        assert tuple(frame.vehicle_context_id.nunique() for frame in (train, validation, excluded)) == (18, 6, 6)
        assert train.target_end_exclusive_ms.max() <= validation.input_start_offset_ms.min()
        for frame in (train, validation, excluded, held):
            assert set(contexts[contexts.vehicle_context_id.isin(frame.vehicle_context_id)].context_id) == set(frame.context_id)
    changed = contexts.copy()
    changed.loc[changed.target_start_offset_ms == 115000, "target_end_exclusive_ms"] += 1
    with pytest.raises(ValueError, match="18/6/6"):
        dingxin_v4_inner_folds(changed)
