import pytest
import torch

from chronaris.modeling.fusion_encoders.sensorllm_adapter import alignment_support


def test_alignment_support_preserves_order_and_records_missing_trend_supervision():
    values = torch.arange(24).reshape(4, 3, 2).float()
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[0, :2, 1] = True
    mask[2, 0, 0] = True
    mask[3, :, 0] = True
    eligible, channels, support = alignment_support(values, mask, ('a', 'b'), ('w0', 'empty', 'single', 'w3'))
    assert eligible == [0, 3] and channels == [1, None, None, 0]
    assert support['eligible_sample_ids'] == ['w0', 'w3']
    assert [(r['sample_id'], r['max_channel_points']) for r in support['excluded']] == [('empty', 0), ('single', 1)]
    assert [eligible[u % len(eligible)] for u in range(4)] == [0, 3, 0, 3]
    with pytest.raises(ValueError, match='no training window'):
        alignment_support(values, torch.zeros_like(mask), ('a', 'b'), ('w0', 'empty', 'single', 'w3'))
    values[0, 0, 1] = torch.nan
    with pytest.raises(ValueError, match='invalid SensorLLM'):
        alignment_support(values, mask, ('a', 'b'), ('w0', 'empty', 'single', 'w3'))
