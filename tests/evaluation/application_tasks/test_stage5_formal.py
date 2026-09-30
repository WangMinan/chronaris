from dataclasses import replace
import json

import pytest

from chronaris.evaluation.application_tasks import stage5_formal
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract
from chronaris.evaluation.application_tasks.v4_pipeline import pipeline_groups
from tests.evaluation.application_tasks.test_common_downstream_contract import _case


def test_all_method_matrix_preserves_real_routes_and_stops_before_outer_evaluation():
    units=stage5_formal.units()
    assert len(units)==279 and len({tuple(u.items()) for u in units})==279
    assert sum(len(stage5_formal.METHODS[u['method']]['routes']) for u in units)==477
    assert {d:sum(u['domain']==d for u in units) for d in ('clare','cogpilot','dingxin')}=={'clare':135,'cogpilot':120,'dingxin':24}
    assert all(u['domain']=='clare' for u in units if u['method']=='sensorllm_deepseek')
    assert all(u['fold_index']==0 for u in units if u['domain']=='dingxin')
    stages=[s for group in pipeline_groups('stage5_formal_train') for s,_ in group]
    assert stages[-1]=='stage5_formal_models'
    assert not {'native_neural','simulation_generation','core_evaluate','report'} & set(stages)


def test_formal_contract_requires_sealed_cohort_and_never_accepts_outer_observations(tmp_path):
    kwargs,_,_,_=_case(tmp_path)
    with pytest.raises(ValueError,match='sealed confirmation'):
        build_common_contract(domain='clare',scope='confirmation_training',**kwargs)
    kwargs['fold']=replace(kwargs['fold'],held_out_sample_ids=('sealed-outer',),development_only=False)
    kwargs['context']=kwargs['context']|{'cohort_role':'confirmation'}
    kwargs['targets']=replace(kwargs['targets'],manifest={'source_role':'fixed_confirmation_subjects'})
    result=build_common_contract(domain='clare',scope='confirmation_training',seed=29,**kwargs)
    assert result['seed']==29 and not result['confirmation_opened']
    assert set(result['roles'])=={'train','validation'}
    kwargs['observations']=kwargs['observations']|{'held_out':kwargs['observations']['validation']}
    with pytest.raises(ValueError,match='exactly train and validation'):
        build_common_contract(domain='clare',scope='confirmation_training',**kwargs)


@pytest.mark.parametrize('method',['chronaris','naive_time_sync','chronos2'])
def test_formal_dispatch_uses_planned_fold_seed_cohort_and_recipe(tmp_path,monkeypatch,method):
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    unit=dict(domain='clare',method=method,fold_index=3,seed=29,recipe={'frozen':True},budget={'fixed':True})
    monkeypatch.setattr(stage5_formal,'verified_plan',lambda c:{'units':[unit]})
    receipts={}
    for domain in ('clare','cogpilot'):
        p=tmp_path/f'{domain}.json';p.write_text('{"passed": true}')
        receipts[f'stage5_formal_execution__{domain}__check']={'path':str(p),'sha256':sha256_file(p)}
    stage5_formal.write_result(tmp_path/'pipeline_state.json',{'completed':receipts})
    calls=[]
    def run(**kwargs):calls.append(kwargs);return {'status':'waiting_gpu'}
    monkeypatch.setattr(stage5_formal,'run_common_contract_smoke',run)
    monkeypatch.setattr(stage5_formal,'run_recent_model_smoke',run)
    result=stage5_formal.run_stage5_formal('stage5_formal_unit__0',dict(root=str(tmp_path),confirmation_data_root='sealed',registry_path='registry'))
    assert result['status']=='waiting_gpu'
    assert calls[0]['fold_index']==3 and calls[0]['seed']==29 and calls[0]['subject_role']=='confirmation'
    assert calls[0]['data_root']=='sealed' and calls[0]['full']
    if method=='chronos2':assert calls[0]['expected_budget']==unit['budget']
    else:
        assert calls[0]['expected_recipe']==unit['recipe']
        assert calls[0]['recipe']==(None if method=='naive_time_sync' else 'stage4_reference')


def test_formal_recovery_preserves_original_source_and_rejects_tampering(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks import stage5_formal_recovery as recovery
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    old = tmp_path/'old.json'
    unit = {'method': 'chronaris'}
    old.write_text(json.dumps(dict(unit=unit, source_code_sha256='old-source', plan_sha256='old-plan', total_seconds=123.)))
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps(dict(source_code_sha256='new-source', parent_source_code_sha256='old-source', execution={'qualification':{'path':str(old),'sha256':sha256_file(old)}}, files={str(old):sha256_file(old)}, completed={'stage5_formal_unit__0':{'path':str(old),'sha256':sha256_file(old)}})))
    frozen = dict(recovery={'path':str(inventory),'sha256':sha256_file(inventory)}, units=[unit], plan_sha256='new-plan')
    monkeypatch.setattr(recovery,'v4_workflow_source_sha256',lambda:'new-source')
    result = recovery.reuse_completed({'root':str(tmp_path/'new')}, frozen, 0)
    assert result['source_code_sha256']=='old-source' and result['reuse']['extra_encoder_updates']==0
    assert result['historical_total_seconds']==123. and result['total_seconds']==0.
    assert recovery.reuse_completed({'root':str(tmp_path/'new')},frozen,8) is None
    assert recovery.reuse_execution(frozen, 'qualification')['reuse']['source_code_sha256']=='old-source'
    old.write_text('{}')
    with pytest.raises(ValueError, match='source changed'):
        recovery.reuse_completed({'root':str(tmp_path/'new')},frozen,0)
