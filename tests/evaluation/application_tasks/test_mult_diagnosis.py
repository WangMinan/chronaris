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
