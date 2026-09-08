import json
import pytest

from chronaris.evaluation.application_tasks import v4_native_confirmation_cohort as module
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_public_screen import METHODS, ROUTES


def test_native_matrix_preserves_routes_and_counts_blocked_sorties(monkeypatch):
    methods={route:{method:candidate_options(method,'reference') for method in METHODS} for route in ROUTES}
    for route in ROUTES:methods[route]['naive_time_sync']={'name':'reference','nonparametric':True}
    methods['task_guided']['chronaris']=candidate_options('chronaris','capacity64')
    frozen={'methods':methods,'domain_status':{'cogpilot':'enabled','clare':'enabled','dingxin':'blocked_vehicle_content_overlap'}}
    monkeypatch.setattr(module,'read_frozen_configuration',lambda *args:frozen)
    plan=module.build_native_confirmation_plan('unused','a'*64)
    assert len(plan['units'])==210 and plan['enabled_evaluation_units']==360
    assert plan['intended_native_evaluation_units']==432 and plan['blocked_domains'][0]['evaluation_units']==72
    assert sum(u['backend']=='nonparametric' for u in plan['units'])==30
    assert all(u['routes']==['task_guided'] for u in plan['units'] if u['candidate_name']=='capacity64')
    frozen['domain_status']['dingxin']='enabled'
    frozen['native_fold_counts']={'cogpilot':5,'clare':5,'dingxin':1}
    amended=module.build_native_confirmation_plan('unused','a'*64)
    assert amended['enabled_evaluation_units']==396 and not amended['blocked_domains']
    assert {u['fold_index'] for u in amended['units'] if u['domain']=='dingxin'}=={0}


def test_native_queue_uses_one_child_per_unit_and_resumes_gpu_wait(tmp_path,monkeypatch):
    units=[dict(domain='clare',method='physiology_only',candidate_name='reference',fold_index=0,seed=seed,
                routes=['self_supervised','task_guided'],backend='neural') for seed in (17,29)]
    plan={'units':units,'blocked_domains':[],'freeze_sha256':'a'*64}
    monkeypatch.setattr(module,'build_native_confirmation_plan',lambda *args:plan)
    calls=[]
    class Child:
        pid=123
        def __init__(self,command,**kwargs):
            calls.append(command);seed=int(command[command.index('--seed')+1])
            waiting=len(calls)==1
            kwargs['stdout'].write(json.dumps({'status':'waiting_gpu' if waiting else 'completed'})+'\n')
            kwargs['stdout'].flush()
            if not waiting:
                root=tmp_path/'clare/physiology_only/reference/fold01'/f'seed{seed}';root.mkdir(parents=True)
                (root/'confirmation_unit.json').write_text(json.dumps({'completed':True,'freeze_sha256':'a'*64}))
        def wait(self,timeout):return 0
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    kwargs=dict(freeze_path='frozen.json',freeze_sha256='a'*64,output_root=tmp_path,backend='neural')
    assert module.run_native_confirmation_cohort(**kwargs)['status']=='waiting_gpu'
    state=module.run_native_confirmation_cohort(**kwargs)
    assert state['status']=='completed' and len(state['completed_units'])==2 and len(calls)==3
    assert module.run_native_confirmation_cohort(**kwargs)['status']=='completed' and len(calls)==3

    receipt=tmp_path/'clare/physiology_only/reference/fold01/seed17/confirmation_unit.json'
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='completed unit receipt changed'):
        module.run_native_confirmation_cohort(**kwargs)
