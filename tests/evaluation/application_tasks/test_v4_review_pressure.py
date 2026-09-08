import fcntl
import json

import pytest

from chronaris.evaluation.application_tasks import v4_review_pressure as module
from chronaris.evaluation.application_tasks import v4_public_screen as shared
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def _setup(root,monkeypatch):
    diagnostic=root/'review';diagnostic.mkdir()
    conditions=root/'conditions';conditions.mkdir();(conditions/'development_condition_audit.json').write_text('{}')
    units=[]
    for seed in (17,29,43):
        units.append(dict(domain='simulation',method='chronaris',candidate_name='reference',seed=seed,
                          routes=['self_supervised','task_guided']))
        unit=diagnostic/'simulation/chronaris/reference/review'/f'seed{seed}';unit.mkdir(parents=True)
        checkpoint=unit/'best.pt';checkpoint.write_bytes(b'fixture checkpoint')
        (unit/'run_state.json').write_text(json.dumps({'completed_consumers':['self_supervised:1500','task_guided:500'],
            **{route+'_training':{'best_checkpoint_path':str(checkpoint)} for route in ('self_supervised','task_guided')}}))
    plan={'plan_sha256':'a'*64,'units':units}
    monkeypatch.setattr(module,'load_verified_review_plan',lambda *args,**kwargs:plan)
    monkeypatch.setattr(shared,'GPU_LOCK_PATH',str(root/'gpu.lock'))
    (diagnostic/'run_state.json').write_text(json.dumps(dict(plan_sha256='a'*64,completed_units=[],failed_units=[])))
    return dict(diagnostic_root=diagnostic,condition_root=conditions)


def test_pressure_queue_is_serial_resumable_and_keeps_failures(tmp_path,monkeypatch):
    kwargs=_setup(tmp_path,monkeypatch);output=tmp_path/'pressure'
    plan=module.build_review_pressure_plan(**kwargs)
    assert len(plan['units'])==6 and plan['status']=='ready_for_review_pressure'
    calls=[]
    def run(**options):
        key=(options['seed'],options['route']);calls.append(key)
        assert options['phase']=='review' and options['device']=='cuda'
        if key==(29,'self_supervised'):raise ValueError('fixture replay mismatch')
        checkpoint=kwargs['diagnostic_root']/'simulation/chronaris/reference/review'/f"seed{options['seed']}"/'best.pt'
        return {'completed':True,'source':{'checkpoint_sha256':sha256_file(checkpoint)}}
    monkeypatch.setattr(module,'run_development_pressure',run)
    with open(shared.GPU_LOCK_PATH,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.run_review_pressure_cohort(output_root=output,**kwargs)['status']=='waiting_gpu'
        assert not calls and not output.exists()
    state=module.run_review_pressure_cohort(output_root=output,**kwargs)
    assert state['status']=='completed_with_failures' and len(state['completed_units'])==5
    assert state['failed_units']==['chronaris/reference/seed29/self_supervised']
    assert 'fixture replay mismatch' in state['errors'][state['failed_units'][0]]
    assert len(calls)==6
    assert module.run_review_pressure_cohort(output_root=output,**kwargs)['completed_units']==state['completed_units']
    assert len(calls)==6 and (output/'progress.json').exists()
    (kwargs['condition_root']/'development_condition_audit.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='pressure plan changed'):module.run_review_pressure_cohort(output_root=output,**kwargs)


def test_pressure_queue_replays_interrupted_unit_and_waits_for_clean_consumers(tmp_path,monkeypatch):
    kwargs=_setup(tmp_path,monkeypatch);output=tmp_path/'pressure'
    calls=[]
    def run(**options):
        calls.append((options['seed'],options['route']))
        if len(calls)==2:raise KeyboardInterrupt()
        checkpoint=kwargs['diagnostic_root']/'simulation/chronaris/reference/review'/f"seed{options['seed']}"/'best.pt'
        return {'completed':True,'source':{'checkpoint_sha256':sha256_file(checkpoint)}}
    monkeypatch.setattr(module,'run_development_pressure',run)
    with pytest.raises(KeyboardInterrupt):module.run_review_pressure_cohort(output_root=output,**kwargs)
    state=module.run_review_pressure_cohort(output_root=output,**kwargs)
    assert state['status']=='completed' and len(state['completed_units'])==6
    assert len(calls)==7 and calls[1]==calls[2] and calls.count((17,'self_supervised'))==1
    path=kwargs['diagnostic_root']/'simulation/chronaris/reference/review/seed43/run_state.json'
    saved=json.loads(path.read_text());saved['completed_consumers']=[];path.write_text(json.dumps(saved))
    waiting=module.build_review_pressure_plan(**kwargs)
    assert waiting['status']=='waiting_for_simulation_review_units' and len(waiting['pending'])==2
