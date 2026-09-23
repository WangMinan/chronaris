import json

import pytest

from chronaris.evaluation.application_tasks import stage5
from chronaris.evaluation.application_tasks.v4_pipeline import pipeline_groups
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def test_fixed_matrix_and_receipts_preserve_history_and_stop_before_confirmation(tmp_path, monkeypatch):
    units = stage5.units()
    assert len(units) == 36 and sum(u['inherited'] for u in units) == 4
    assert len({tuple(u.items()) for u in units}) == 36
    assert {u['seed'] for u in units} == {17, 29, 43}
    assert {u['fold_index'] for u in units} == {0, 1, 2}
    stages = [s for group in pipeline_groups('stage5') for s, _ in group]
    assert stages[-1] == 'stage5_report'
    assert not {'freeze', 'native_neural', 'simulation_generation'}.intersection(stages)
    assert stages.index('stage5_execution__cogpilot__check') < stages.index('stage5_unit__0')
    frozen = seal_development_plan(dict(units=units, contracts={}, confirmation_opened=False, source_code_sha256='source'))
    path, receipt = tmp_path/'stage5_plan.json', tmp_path/'receipt.json'
    stage5.write_result(path, frozen)
    stage5.write_result(receipt, frozen)
    stage5.write_result(tmp_path/'pipeline_state.json', {'completed': {'stage5_plan': {
        'path': str(receipt), 'sha256': sha256_file(receipt)}}})
    monkeypatch.setattr(stage5, 'v4_workflow_source_sha256', lambda: 'source')
    assert stage5.verified_plan({'root': str(tmp_path)}) == frozen
    changed = dict(frozen); changed['confirmation_opened'] = True
    stage5.write_result(path, changed)
    with pytest.raises(ValueError, match='completion receipt'):
        stage5.verified_plan({'root': str(tmp_path)})
    stage5.write_result(path, frozen)
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='evidence changed'):
        stage5.verified_plan({'root': str(tmp_path)})


def test_new_unit_passes_exact_fold_seed_and_requires_execution_qualification(tmp_path, monkeypatch):
    units = stage5.units()
    index = next(i for i, u in enumerate(units) if u['domain'] == 'clare' and u['method'] == 'chronaris'
                 and u['fold_index'] == 1 and u['seed'] == 29)
    frozen = dict(units=units, plan_sha256='plan', contracts={'clare/1/29': {'contract_sha256': 'contract'}})
    monkeypatch.setattr(stage5, 'verified_plan', lambda config: frozen)
    check = tmp_path/'performance/clare/stage4_reference/check.json'
    receipt = tmp_path/'qualification.json'
    for path in (check, receipt):
        stage5.write_result(path, {'passed': True})
    stage5.write_result(tmp_path/'pipeline_state.json', {'completed': {'stage5_execution__clare__check': {
        'path': str(receipt), 'sha256': sha256_file(receipt)}}})
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return dict(status='waiting_gpu')
    monkeypatch.setattr(stage5, 'run_common_contract_smoke', run)
    config = dict(root=str(tmp_path), data_root='data', registry_path='registry')
    assert stage5.run_stage5(f'stage5_unit__{index}', config)['status'] == 'waiting_gpu'
    assert calls[0]['seed'] == 29 and calls[0]['fold_index'] == 1
    assert calls[0]['expected_contract_sha256'] == 'contract' and calls[0]['cuda_graph_recurrence']
    assert not (tmp_path/'units'/f'{index}.json').exists()
    stage5.write_result(check, {'passed': False})
    with pytest.raises(ValueError, match='qualification changed'):
        stage5.run_stage5(f'stage5_unit__{index}', config)
    assert len(calls) == 1
