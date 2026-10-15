import json

import pytest

from chronaris.evaluation.application_tasks.causality_recovery import install_policy
from chronaris.evaluation.application_tasks.chapter5_assets import sha
from chronaris.evaluation.application_tasks import v4_correctness, chapter5_core


def test_explicit_policy_binds_new_auditor_and_rejects_changed_evidence(tmp_path, monkeypatch):
    source = tmp_path/'audit.py'
    source.write_text('HISTORICAL_ATOL = 1e-5\ndef audit_checkpoint_causality(*a, **k):\n    return dict(passed=True, legacy_1e6_passed=False)\n')
    precision, sweep = tmp_path/'precision.json', tmp_path/'sweep.json'
    precision.write_text(json.dumps([dict(negative_control='causal_mask_disabled', delta=2.)]*2+
        [dict(dtype='torch.float64', deltas={'sequence':0.}, mask_equal=True)]))
    sweep.write_text(json.dumps([dict(revised_passed=True, revised_atol=1e-5, historical_max_delta={'future_missing':1.1e-6})]*48))
    files = {str(p):sha(p) for p in (source, precision, sweep)}
    policy = tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(files=files, auditor_path=str(source), precision_path=str(precision), sweep_path=str(sweep),
        original_auditor_sha256=[sha(v4_correctness.__file__)], historical_atol=1e-5)))
    monkeypatch.setattr(v4_correctness, 'audit_checkpoint_causality', v4_correctness.audit_checkpoint_causality)
    monkeypatch.setattr(chapter5_core, '_train', lambda: dict(status='completed', files={}))
    install_policy(policy)
    result = v4_correctness.audit_checkpoint_causality()
    assert result['passed'] and not result['legacy_1e6_passed']
    assert result['validation_policy']['sha256'] == sha(policy)
    assert chapter5_core._train()['files'] == files | {str(policy):sha(policy)}
    source.write_text('tampered')
    with pytest.raises(ValueError, match='source changed'):
        install_policy(policy)


def test_policy_cannot_apply_to_an_unreviewed_original_auditor(tmp_path):
    policy = tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(files={}, original_auditor_sha256=['different-source'])))
    with pytest.raises(ValueError, match='reviewed original auditor'):
        install_policy(policy)
