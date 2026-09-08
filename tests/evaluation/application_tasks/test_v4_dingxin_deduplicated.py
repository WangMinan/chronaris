from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from chronaris.evaluation.application_tasks import v4_dingxin_data as original
from chronaris.evaluation.application_tasks.v4_dingxin_deduplicated import (
    PROTOCOL, deduplicated_dingxin_data, fit_deduplicated_targets)
from tests.evaluation.dingxin.test_simple_downstream_protocol import _synthetic_raw_bundle


def test_retained_record_keeps_views_purges_supports_and_fits_only_allowed_targets(monkeypatch):
    raw = _synthetic_raw_bundle()
    folds, embargo = original.dingxin_v4_inner_folds(raw.contexts)
    data = original.V4DingxinData(None, folds, embargo, raw, None, {}, {}, {}, {}, (), 'a'*64)
    rows = [dict(sortie_id=row.sortie_id, target_start_offset_ms=row.target_start_offset_ms,
                 content_sha256=str(row.target_start_offset_ms))
            for row in raw.contexts.drop_duplicates('vehicle_context_id').itertuples(index=False)]
    monkeypatch.setattr(original, 'audit_dingxin_vehicle_reuse', lambda data: {'context_rows':rows})
    dedup, manifest = deduplicated_dingxin_data(data)
    fold = dedup.folds[0]
    assert manifest['retained_sortie'] == sorted(raw.contexts.sortie_id.unique())[0]
    assert len(dedup.folds) == 1 and dedup.evaluation_protocol == PROTOCOL
    contexts = dedup.raw_targets.contexts
    frames = [contexts[contexts.context_id.isin(ids)] for ids in
              (fold.train_sample_ids, fold.validation_sample_ids, fold.held_out_sample_ids, dedup.embargo[fold.fold_id])]
    assert [frame.vehicle_context_id.nunique() for frame in frames] == [12, 3, 3, 12]
    assert sum(map(len, frames)) == len(contexts)
    assert frames[0].target_end_exclusive_ms.max() <= frames[1].input_start_offset_ms.min()
    assert frames[1].target_end_exclusive_ms.max() <= frames[2].input_start_offset_ms.min()
    for frame in frames:
        assert set(contexts[contexts.vehicle_context_id.isin(frame.vehicle_context_id)].context_id) == set(frame.context_id)
    altered = replace(dedup.raw_targets, maneuver_statistics=dedup.raw_targets.maneuver_statistics.copy(),
                      physiology_statistics=dedup.raw_targets.physiology_statistics.copy())
    nontrain_vehicle = contexts.loc[~contexts.context_id.isin(fold.train_sample_ids), 'vehicle_context_id']
    altered.maneuver_statistics.loc[altered.maneuver_statistics.vehicle_context_id.isin(nontrain_vehicle), 'future_std'] += 999
    altered.physiology_statistics.loc[~altered.physiology_statistics.context_id.isin(fold.train_sample_ids), 'future_median'] += 999
    refit = fit_deduplicated_targets(altered, fold, fit_ids=fold.train_sample_ids)
    pd.testing.assert_frame_equal(refit.threshold_rows, dedup.fitted_targets.threshold_rows)
    with pytest.raises(ValueError, match='allowed training'):
        fit_deduplicated_targets(altered, fold, fit_ids=fold.held_out_sample_ids)
    bad = replace(fold, validation_sample_ids=dedup.embargo[fold.fold_id][:1])
    with pytest.raises(ValueError, match='complete input plus target support'):
        fit_deduplicated_targets(altered, bad, fit_ids=fold.train_sample_ids)
    result = original.build_dingxin_outer_consumer_inputs(dedup, fold.fold_id)
    outer = result['fold']
    outer_rows = contexts[contexts.context_id.isin(outer.train_sample_ids)]
    assert outer_rows.vehicle_context_id.nunique() == 21
    assert outer_rows.target_end_exclusive_ms.max() <= frames[2].input_start_offset_ms.min()
    assert tuple(result['targets'].manifest['fit_sample_ids']) == outer.train_sample_ids
    weights = result['targets'].sample_weights.numpy()
    assert np.isclose(weights.sum(), 24)
    assert result['context']['evaluation_protocol'] == PROTOCOL
    # A second record with genuinely different content must not be discarded.
    rows[-1]['content_sha256'] = 'changed'
    with pytest.raises(ValueError, match='verified two-record'):
        deduplicated_dingxin_data(data)
