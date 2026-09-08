import json

import pytest

from chronaris.evaluation.application_tasks import v4_conditional_review as module
from chronaris.evaluation.application_tasks import v4_configuration_freeze as freeze
from chronaris.evaluation.application_tasks import v4_review_plan, v4_adoption, v4_simulation_review_results
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def test_trigger_requires_two_independent_seeds_for_the_same_state_and_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(freeze, '_completed_scores', lambda root,state,route,update,method,candidate,**kwargs:
                        dict(method=method,candidate=candidate,route=route,seed=kwargs['seed'],missingness_p95=1.))
    monkeypatch.setattr(freeze, '_pressure_p95', lambda *args, **kwargs: 1.)
    records=[]
    for seed in (17,29,43):
        unit=tmp_path/'review/simulation/chronaris/reference/review'/f'seed{seed}'
        unit.mkdir(parents=True);(unit/'run_state.json').write_text('{}')
        root=tmp_path/'pressure/chronaris/reference/review'/f'seed{seed}'/'self_supervised/1500'
        root.mkdir(parents=True);conditions={}
        for condition in ('clean_asynchronous','contiguous_gap_15s','contiguous_gap_30s'):
            result=root/condition/'result.json';result.parent.mkdir()
            value=11. if condition=='contiguous_gap_15s' and seed in (17,29) else 1.
            result.write_text(json.dumps({'encoding_diagnostics':{'distributions':{'vehicle_state_norm':{'status':'completed','p99':value}}}}))
            conditions[condition]={'result_path':str(result),'result_sha256':sha256_file(result),
                'prediction_path':str(result),'prediction_sha256':sha256_file(result),'representation_sha256':'a'*64}
        (root/'run_state.json').write_text(json.dumps({'conditions':conditions}))
        records.append(dict(method='chronaris',candidate='reference',route='self_supervised',seed=seed,missingness_p95=1.))
    args=(tmp_path/'review',tmp_path/'pressure',{'self_supervised':'reference'})
    _,triggered=freeze._simulation_freeze_evidence({'completed':records},*args)
    assert len(triggered)==1 and triggered[0]['seeds']==[17,29]
    _,triggered=freeze._simulation_freeze_evidence({'completed':records[:1]},*args)
    assert triggered==[]


def test_conditional_plan_is_bounded_and_preserves_parent_results(tmp_path, monkeypatch):
    parent=tmp_path/'parent';pressure=tmp_path/'pressure'
    parent.mkdir();pressure.mkdir()
    units=[dict(domain=domain,method='chronaris',candidate_name='reference',seed=17,routes=list(module.ROUTES))
           for domain in ('simulation','cogpilot','clare')]
    plan=dict(format='chronaris.v4_three_seed_review_plan.v1',units=units,plan_sha256='a'*64,
              source_code_sha256='b'*64,screen_root='screen',data_root='data',confirmation_feedback_used=False)
    (parent/'selection_plan.json').write_text(json.dumps(plan))
    for domain in ('simulation','cogpilot','clare'):
        (parent/domain/'chronaris/reference').mkdir(parents=True)
    (pressure/'chronaris/reference').mkdir(parents=True)
    state={'status':'completed','failed_units':[],'completed_units':['simulation/chronaris/reference/fold01/seed17'],'plan_sha256':'a'*64}
    (parent/'run_state.json').write_text(json.dumps(state))
    (pressure/'queue_state.json').write_text(json.dumps(state | {'completed_units':['chronaris/reference/seed17/self_supervised']}))
    monkeypatch.setattr(v4_review_plan,'load_verified_review_plan',lambda *args,**kwargs:plan)
    monkeypatch.setattr(v4_adoption,'collect_adoption_decisions',lambda **kwargs:dict(
        status='single_factor_decisions_ready_not_frozen',decisions={f'chronaris/{route}':{'recommended_candidate':'reference'} for route in module.ROUTES}))
    monkeypatch.setattr(v4_simulation_review_results,'collect_simulation_review_results',lambda **kwargs:{'files':{}})
    triggers=[]
    monkeypatch.setattr(freeze,'_simulation_freeze_evidence',lambda *args:({},triggers))
    kwargs=dict(parent_root=parent,parent_pressure_root=pressure,data_root='data',registry_path='registry')
    assert module.build_conditional_review_plan(**kwargs)['status']=='not_applicable'
    triggers.append({'route':'self_supervised','seeds':[17,29]})
    extra=module.build_conditional_review_plan(**kwargs)
    added=[u for u in extra['units'] if u['candidate_name']=='analytic_decay']
    assert len(added)==21 and all(u['routes']==['self_supervised'] for u in added)
    assert {u['seed'] for u in added}=={17,29,43}
    root=tmp_path/'new_review';new_pressure=tmp_path/'new_pressure'
    before={str(p):sha256_file(p) for p in parent.rglob('*.json')}
    module.prepare_conditional_review(extra,output_root=root,pressure_root=new_pressure)
    module.prepare_conditional_review(extra,output_root=root,pressure_root=new_pressure)
    assert before=={str(p):sha256_file(p) for p in parent.rglob('*.json')}
    assert (root/'simulation/chronaris/reference').resolve()==(parent/'simulation/chronaris/reference').resolve()
    assert json.loads((root/'run_state.json').read_text())['completed_units']==state['completed_units']
    pressure_plan={'inherited_pressure_root':str(pressure),'plan_sha256':'c'*64}
    module.inherit_completed_pressure(pressure_plan,output_root=new_pressure)
    assert json.loads((new_pressure/'queue_state.json').read_text())['plan_sha256']=='c'*64
    plan['format']='chronaris.v4_conditional_review_plan.v1'
    with pytest.raises(ValueError,match='no recursive search'):
        module.build_conditional_review_plan(**kwargs)
