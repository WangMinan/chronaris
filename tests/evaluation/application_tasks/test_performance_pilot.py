import json

import numpy as np
import pytest

from chronaris.evaluation.application_tasks.performance_pilot_audit import (
    FIT_SUBJECTS, SELECT_SUBJECTS, SCORE_SUBJECTS, qualify_performance_task,
)
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_data import PerformancePilotData


def test_qualification_checks_coverage_and_trivial_persistence(tmp_path):
    subjects = FIT_SUBJECTS + SELECT_SUBJECTS + SCORE_SUBJECTS
    windows, target_points = [], []
    for subject in subjects:
        for i in range(100):
            sample = f'{subject}:{i}'
            y = 1. + i / 100
            windows.append(dict(sample_id=sample, subject=subject, record_id=f'{subject}:{i%12}',
                                accepted=True, reason='', target=y, persistence=y))
            t = np.arange(.25, 5.01, .25).tolist()
            target_points.append(dict(sample_id=sample, future_t=t, future_y=[y]*20,
                                      past_t=[v-5.125 for v in t], past_y=[y]*20))
    shape = (len(windows), 96, 12)
    path = tmp_path/'snapshot.npz'
    np.savez_compressed(path, values=np.ones(shape, np.float32), masks=np.ones(shape, bool),
                        observation_times=np.broadcast_to(np.linspace(0, 30, 96, endpoint=False)[None, :, None], shape),
                        metadata=np.array(json.dumps(dict(windows=windows, target_points=target_points,
                                                           sources={str(i):'0'*64 for i in range(420)},
                                                           records=[{} for _ in range(84)]))))
    result = qualify_performance_task(path, tmp_path/'out')
    assert result['status'] == 'failed'
    assert result['gates']['per_subject_support']
    assert not result['gates']['persistence_not_trivial']
    assert result['encoder_updates'] == 0


def test_data_binding_fit_only_normalization_and_sealed_scoring(tmp_path):
    subjects = FIT_SUBJECTS + SELECT_SUBJECTS + SCORE_SUBJECTS
    rows = [dict(sample_id=s, subject=s, accepted=True, target=float(i+1), persistence=0.)
            for i, s in enumerate(subjects)]
    shape = (len(rows), 96, 12)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)/1000
    path = tmp_path/'snapshot.npz'; receipt = tmp_path/'qualification.json'
    def build(values):
        np.savez_compressed(path, values=values, masks=np.ones(shape, bool),
                            observation_times=np.broadcast_to(np.linspace(0, 30, 96, endpoint=False)[None, :, None], shape),
                            metadata=np.array(json.dumps(dict(windows=rows))))
        receipt.write_text(json.dumps(dict(status='passed', gates={'test': True}, snapshot_sha256=file_digest(path))))
        return PerformancePilotData(path, receipt, 'cpu')
    first = build(x)
    with pytest.raises(PermissionError):
        first.indices('score')
    with pytest.raises(PermissionError):
        first.queries(first.roles['score'])
    x[3:] = 1e8
    second = build(x)
    assert first.normalizer == second.normalizer
    receipt.write_text(json.dumps(dict(status='failed', gates={}, snapshot_sha256=file_digest(path))))
    with pytest.raises(ValueError, match='did not pass'):
        PerformancePilotData(path, receipt, 'cpu')
