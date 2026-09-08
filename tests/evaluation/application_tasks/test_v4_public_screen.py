import json
import pytest
from chronaris.evaluation.application_tasks import v4_public_screen as module


def _summary(method,route):
    names=(['cosine_temperature','quality_gate','capacity64'] if route=='self_supervised' else
           ['projected_attention','quality_gate','single_stream_fidelity']) if method=='chronaris' else ['reference','capacity64']
    return dict(status='development_screen_complete_not_final_adoption',confirmation_feedback_used=False,seed=17,
        data_manifest_sha256='a'*64,cohort_source_code_sha256='b'*64,
        advance_to_public_development=names,completed=[{'candidate':name} for name in ['reference']+names],
        primary_metrics=(('classification','linear','macro_f1','higher'),))


def test_public_plan_merges_routes_and_keeps_reference_without_extra_candidate_search(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    registry=tmp_path/'registry.json';registry.write_text('{}')
    monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:_summary(kwargs['method'],kwargs['route']))
    kwargs=dict(diagnostic_root=tmp_path,pressure_root=tmp_path,registry_path=registry)
    plan=module.build_public_screen_plan(**kwargs)
    assert plan['status']=='ready_for_public_development' and len(plan['units'])==28
    for domain in ('cogpilot','clare'):
        units={row['candidate_name']:row for row in plan['units'] if row['domain']==domain and row['method']=='chronaris'}
        assert len(units)==6
        assert units['reference']['routes']==list(module.ROUTES)
        assert units['reference']['purposes']['self_supervised']=='repaired_reference_comparator'
        assert units['quality_gate']['routes']==list(module.ROUTES)
        assert units['cosine_temperature']['routes']==['self_supervised']
        assert units['cosine_temperature']['joint_updates']==0
        assert units['projected_attention']['routes']==['task_guided']
    invoked=[]
    monkeypatch.setattr(module,'run_candidate_development',lambda **kw:invoked.append(kw) or {'completed':True})
    out=tmp_path/'run'
    assert module.run_public_screen(**kwargs,output_root=out)['status']=='completed'
    assert len(invoked)==28

    monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:_summary(kwargs['method'],kwargs['route']) | {'status':'blocked_by_execution_failure'})
    blocked=module.run_public_screen(**kwargs,output_root=tmp_path/'blocked')
    assert blocked['status']=='blocked_by_simulation_or_pressure_failure' and blocked['units']==[]
    assert len(invoked)==28
    monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:_summary(kwargs['method'],kwargs['route']))
    assert module.run_public_screen(**kwargs,output_root=out)['status']=='completed'
    assert len(invoked)==28
    assert json.loads((out/'selection_plan.json').read_text())==plan
    monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:_summary(kwargs['method'],kwargs['route']) | {'status':'waiting_for_complete_pressure_cohort'})
    waiting=module.build_public_screen_plan(**kwargs)
    assert waiting['units']==[] and len(waiting['pending'])==10
    assert module.run_public_screen(**kwargs,output_root=tmp_path/'pending')['status']=='waiting_for_simulation_and_pressure'
    assert len(invoked)==28


def test_public_plan_rejects_confirmatory_or_excess_selection(tmp_path,monkeypatch):
    registry=tmp_path/'registry.json';registry.write_text('{}')
    for changed in ({'confirmation_feedback_used':True},{'advance_to_public_development':['reference','capacity64','quality_gate','multihorizon']}):
        monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:_summary(kwargs['method'],kwargs['route']) | changed)
        with pytest.raises(ValueError):module.build_public_screen_plan(diagnostic_root=tmp_path,pressure_root=tmp_path,registry_path=registry)
