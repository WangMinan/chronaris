from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_correctness as audit


@pytest.mark.parametrize('delta,passed', [(1.073e-6, True), (2e-5, False), (float('nan'), False), (float('inf'), False)])
def test_roundoff_policy_preserves_failure_for_leaks_and_nonfinite_values(tmp_path, monkeypatch, delta, passed):
    checkpoint = tmp_path/'checkpoint.pt'
    payload = dict(fold={'validation_sample_ids': ['a'], 'fold_id': 'f'}, seed=17, format='test')
    torch.save(payload, checkpoint)
    monkeypatch.setattr(audit, 'load_common_pretraining_checkpoint', lambda *a, **k: (None, None, None, payload))
    monkeypatch.setattr(audit, 'perturb_future_observations', lambda b, **k: b)
    monkeypatch.setattr(audit, 'remove_future_observations', lambda b, **k: b)
    counter = iter([0., 0., 0., delta])
    def predict(batch):
        return SimpleNamespace(timestamps_s=torch.zeros(1, 1), valid_mask=torch.ones(1, 1, dtype=torch.bool),
                               sequence_embedding=torch.full((1, 1, 1), next(counter), dtype=torch.float64))
    monkeypatch.setattr(audit, 'TrainedFusionAdapter', lambda **k: predict)
    result = audit.audit_checkpoint_causality(checkpoint, SimpleNamespace(sample_ids=('a',)), device='cpu')
    assert result['passed'] is passed
    assert not result['legacy_1e6_passed']
    assert result['historical_atol'] == 1e-5 and result['checkpoint_unchanged']


def test_availability_changes_are_rejected_even_with_zero_numeric_delta(tmp_path, monkeypatch):
    checkpoint = tmp_path/'checkpoint.pt'
    payload = dict(fold={'validation_sample_ids': ['a'], 'fold_id': 'f'}, seed=17, format='test')
    torch.save(payload, checkpoint)
    monkeypatch.setattr(audit, 'load_common_pretraining_checkpoint', lambda *a, **k: (None, None, None, payload))
    monkeypatch.setattr(audit, 'perturb_future_observations', lambda b, **k: b)
    monkeypatch.setattr(audit, 'remove_future_observations', lambda b, **k: b)
    available = iter([True, False])
    monkeypatch.setattr(audit, 'TrainedFusionAdapter', lambda **k: lambda b: SimpleNamespace(
        timestamps_s=torch.zeros(1, 1), valid_mask=torch.full((1, 1), next(available)), sequence_embedding=torch.zeros(1, 1, 1)))
    with pytest.raises(AssertionError, match='availability'):
        audit.audit_checkpoint_causality(checkpoint, SimpleNamespace(sample_ids=('a',)), device='cpu')
