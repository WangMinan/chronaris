from copy import deepcopy
import json

import numpy as np
import torch
import pytest

from chronaris.representation.collation import _observation_age


def test_vectorized_age_matches_last_observation_with_missing_and_repeated_times():
    rng = np.random.default_rng(17)
    for count in (0, 1, 97):
        times = np.sort(rng.integers(0, 30, count)).astype(np.float64)
        mask = rng.random((count, 5)) > .5
        mask[:, 0] = False
        expected = np.full(mask.shape, np.inf, np.float32)
        last = np.full(5, -np.inf)
        for row, time in enumerate(times):
            last[mask[row]] = time
            valid = np.isfinite(last)
            expected[row, valid] = np.maximum(time-last[valid], 0).astype(np.float32)
        np.testing.assert_array_equal(_observation_age(times, mask), expected)


def test_shared_physiology_preserves_dropout_rng_and_accumulated_gradients():
    from chronaris.modeling.training import build_trainable_fusion_encoder
    from chronaris.representation import collate_observation_samples, build_lag_discrimination_inputs
    from tests.modeling.training.test_pretraining_encoders import _sample
    torch.set_num_threads(1)
    torch.manual_seed(17)
    raw = collate_observation_samples([_sample('a'), _sample('b')])
    shifted = build_lag_discrimination_inputs(raw, ('0'*64, '1'*64)).negative_batch
    reference = build_trainable_fusion_encoder('chronaris', physiology_feature_names=('physiology.a',),
        vehicle_feature_names=('vehicle.a',))
    revised = deepcopy(reference)
    records = []
    for model, reuse in ((reference, False), (revised, True)):
        torch.manual_seed(29)
        first = model(raw)
        second = model(shifted, **({'physiology_encoding':first.auxiliary['alignment_output'].physiology} if reuse else {}))
        (first.sequence_embedding.square().mean()+second.sequence_embedding.square().mean()).backward()
        records.append((second.sequence_embedding.detach(), torch.get_rng_state(),
            [p.grad for p in model.parameters()]))
    assert torch.equal(records[0][0], records[1][0])
    assert torch.equal(records[0][1], records[1][1])
    for a, b in zip(records[0][2], records[1][2], strict=True):
        if a is None: assert b is None
        else: torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_performance_reuse_verifies_unit_files_and_keeps_chained_cost(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks import stage5_formal_recovery as recovery
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    original = tmp_path/'completed.json'
    model = tmp_path/'model.pt'; model.write_bytes(b'preserved model')
    unit = {'method':'sensorllm_deepseek'}
    original.write_text(json.dumps(dict(unit=unit, source_code_sha256='original', plan_sha256='old',
        historical_total_seconds=321., total_seconds=0.)))
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps(dict(source_code_sha256='current', failed_sensor_unit_restarts=False,
        files={str(original):sha256_file(original),str(model):sha256_file(model)},
        unit_files={'0':[str(original),str(model)]},
        completed={'stage5_formal_unit__0':{'path':str(original),'sha256':sha256_file(original)}})))
    frozen = dict(recovery={'path':str(inventory),'sha256':sha256_file(inventory)},units=[unit],plan_sha256='new')
    monkeypatch.setattr(recovery,'v4_workflow_source_sha256',lambda:'current')
    result = recovery.reuse_completed({'root':str(tmp_path/'new')},frozen,0)
    assert result['source_code_sha256']=='original' and result['historical_total_seconds']==321.
    assert recovery.reuse_completed({'root':str(tmp_path/'new')},frozen,1) is None
    model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):
        recovery.reuse_completed({'root':str(tmp_path/'new')},frozen,0)
