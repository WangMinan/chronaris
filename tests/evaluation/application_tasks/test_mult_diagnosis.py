import numpy as np
import torch
from chronaris.evaluation.application_tasks.mult_diagnosis import different_record_donors, intervene
from chronaris.modeling.fusion_encoders.causal_query import CausalQueryStream


def test_donors_remove_pairing_without_crossing_subject():
    rows = [dict(subject=s, record_id=s+r, cutoff_s=t) for s in ('a', 'b') for r in ('1', '2') for t in (30, 35)]
    ids = np.arange(len(rows))
    donor = different_record_donors(rows, ids)
    for i, j in zip(ids, donor):
        assert rows[i]['subject'] == rows[j]['subject']
        assert rows[i]['record_id'] != rows[j]['record_id']
        assert rows[i]['cutoff_s'] == rows[j]['cutoff_s']


def test_interventions_preserve_original_and_query_clock():
    x = torch.arange(8.).reshape(1, 4, 2)
    mask = torch.ones_like(x, dtype=torch.bool); mask[:, 0] = False
    x[~mask] = 0
    q = CausalQueryStream(x, mask, mask.any(-1), torch.where(mask, 1., torch.inf), torch.arange(4.)[None],
                          torch.where(mask, 0, -1))
    original = x.clone()
    reversed_q = intervene(q, 'reverse_valid')
    torch.testing.assert_close(reversed_q.values[:, 1:], x[:, 1:].flip(1))
    assert torch.equal(reversed_q.timestamps_s, q.timestamps_s)
    assert not intervene(q, 'missing').feature_mask.any()
    assert not intervene(q, 'zero_values').values.any()
    assert torch.equal(q.values, original)


def test_native_audit_replays_missing_final_query_and_detects_flattening(tmp_path, monkeypatch):
    import pandas as pd
    from chronaris.dataset.cogpilot_history_features import native_dynamic_features, QUERY_SECONDS, PHYSIOLOGY_BANDS
    from chronaris.dataset.cogpilot_performance import file_digest
    from chronaris.evaluation.application_tasks import mult_diagnosis as audit
    monkeypatch.setattr(audit, 'STREAM_COLUMNS', {'test': ('a', 'b', 'c', 'd')})
    t = np.arange(0, 29.7, 1/256)
    x = np.sin(2*np.pi*t); x[t > 10] += 100
    path = tmp_path/'test_stream-test_dat.csv'
    pd.DataFrame(dict(time_dn=t/86400, a=x, b=x, c=x, d=x)).to_csv(path, index=False)
    loaded = pd.read_csv(path); times = loaded.time_dn.to_numpy()*86400
    outputs = [native_dynamic_features(times, loaded[c].to_numpy(), 0., QUERY_SECONDS, b)
               for c, b in zip(('a', 'b', 'c', 'd'), PHYSIOLOGY_BANDS)]
    row = dict(record_id=str(tmp_path), source_origin_native=0., cutoff_s=30., sample_id='synthetic', subject='a',
               final_features=np.concatenate([r[0][-1] for r in outputs]).tolist(),
               final_mask=np.concatenate([r[1][-1] for r in outputs]).tolist())
    out = audit.native_clip_audit(dict(windows=[row], sources={str(path): file_digest(path)}))
    assert len(out['rows']) == 4
    assert all(not r['final_feature_valid'] and r['feature_replay_delta'] == 0 for r in out['rows'])
    assert all(r['final_flattened'] and r['std_retained'] == 0 for r in out['rows'])
