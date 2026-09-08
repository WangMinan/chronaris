import json
import fcntl

import pytest

from chronaris.evaluation.application_tasks import v4_public_screen as executor
from chronaris.evaluation.application_tasks import v4_review_plan as module


def _results():
    rankings={}
    for method in module.METHODS:
        for route in module.ROUTES:
            names=(['quality_gate','cosine_temperature','reference'] if route=='self_supervised' else
                   ['single_stream_fidelity','projected_attention','reference']) if method=='chronaris' else ['capacity64','reference']
            rankings[f'{method}/{route}']=[{'candidate':name} for name in names]
    return dict(status='public_scores_verified_not_final_adoption',rankings=rankings,pending=[],failed=[],
                confirmation_feedback_used=False,selection_plan_sha256='a'*64)


def test_review_merges_initialization_and_runs_each_fold_seed_once(tmp_path,monkeypatch):
    monkeypatch.setattr(executor,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    registry=tmp_path/'registry.json';registry.write_text('{}')
    results=_results()
    monkeypatch.setattr(module,'collect_public_screen_results',lambda **kwargs:results)
    kwargs=dict(screen_root=tmp_path/'screen',registry_path=registry)
    plan=module.build_review_plan(**kwargs)
    assert plan['status']=='ready_for_three_seed_review'
    assert len(plan['units'])==231  # 11 method/configurations x (1 + 3 + 3) folds x 3 seeds.
    for method in module.METHODS:
        for route in module.ROUTES:
            assert len({u['candidate_name'] for u in plan['units'] if u['method']==method and route in u['routes']})<=2
    keys={(u['domain'],u['method'],u['candidate_name'],u['fold_index'],u['seed']) for u in plan['units']}
    assert len(keys)==len(plan['units'])
    for unit in plan['units']:
        assert unit['seed'] in (17,29,43) and unit['phase']=='review'
        assert unit['pretraining_updates']==1500
        assert unit['joint_updates']==(500 if 'task_guided' in unit['routes'] else 0)
        assert unit['head_warmup_updates']==(50 if 'task_guided' in unit['routes'] else 0)
        assert unit['fold_index'] in (range(1) if unit['domain']=='simulation' else range(3))
        if unit['candidate_name']=='reference':assert unit['routes']==list(module.ROUTES)
        if unit['candidate_name']=='quality_gate':assert unit['routes']==['self_supervised']
    calls=[]
    monkeypatch.setattr(executor,'run_candidate_development',lambda **kwargs:calls.append(kwargs) or {'completed':True})
    root=tmp_path/'run'
    with open(executor.GPU_LOCK_PATH,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.run_review_cohort(output_root=root,**kwargs)['status']=='waiting_gpu'
        assert not calls and not root.exists()
    assert module.run_review_cohort(output_root=root,**kwargs)['status']=='completed'
    assert len(calls)==231
    assert module.run_review_cohort(output_root=root,**kwargs)['status']=='completed' and len(calls)==231
    assert json.loads((root/'selection_plan.json').read_text())==plan
    results['selection_plan_sha256']='b'*64
    with pytest.raises(ValueError,match='selections or source changed'):
        module.run_review_cohort(output_root=root,**kwargs)


def test_review_waits_and_rejects_incomplete_or_confirmatory_rankings(tmp_path,monkeypatch):
    registry=tmp_path/'registry.json';registry.write_text('{}')
    kwargs=dict(screen_root=tmp_path,registry_path=registry)
    results=dict(status='blocked_by_public_execution_failure',pending=[],failed=['bad unit'])
    monkeypatch.setattr(module,'collect_public_screen_results',lambda **kwargs:results)
    blocked=module.run_review_cohort(output_root=tmp_path/'never',**kwargs)
    assert blocked['status']=='blocked_by_public_execution_failure' and not (tmp_path/'never').exists()
    results=_results();results['rankings'].pop('chronaris/self_supervised')
    with pytest.raises(ValueError,match='complete verified'):module.build_review_plan(**kwargs)
    results=_results();results['confirmation_feedback_used']=True
    with pytest.raises(ValueError,match='complete verified'):module.build_review_plan(**kwargs)
    results=_results();results['rankings']['chronaris/self_supervised']=[{'candidate':'quality_gate'}]
    with pytest.raises(ValueError,match='repaired reference'):module.build_review_plan(**kwargs)
