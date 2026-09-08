import json

import pytest

from chronaris.evaluation.application_tasks import v4_simulation_review_results as module
from chronaris.evaluation.application_tasks.v4_candidate_results import _pressure_p95
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_development_conditions import DEVELOPMENT_CONDITIONS
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def test_simulation_review_separates_clean_pressure_and_failed_seeds(tmp_path,monkeypatch):
    plan={'plan_sha256':'a'*64,'source_code_sha256':'b'*64,'units':[]}
    for seed in (17,29,43):
        plan['units'].append(dict(domain='simulation',method='physiology_only',candidate_name='reference',seed=seed,routes=['self_supervised']))
        directory=tmp_path/'simulation/physiology_only/reference/review'/f'seed{seed}';directory.mkdir(parents=True)
        (directory/'run_state.json').write_text(json.dumps(dict(source_code_sha256='b'*64,confirmation_opened=False,
            candidate_options=candidate_options('physiology_only','reference'),fold={'fold_id':'fixture__training512'},
            data_manifest_sha256='c'*64,completed_consumers=['self_supervised:1500'])))
    (tmp_path/'selection_plan.json').write_text(json.dumps(plan))
    (tmp_path/'run_state.json').write_text(json.dumps(dict(plan_sha256='a'*64,completed_units=[],failed_units=[])))
    monkeypatch.setattr(module,'load_verified_review_plan',lambda *args,**kwargs:plan)
    seen=[]
    def scores(*args,**kwargs):
        seen.append(kwargs)
        assert args[0].name==f"seed{kwargs['seed']}" and kwargs['phase']=='review'
        return {'candidate':'reference','scores':[.5,.2,.6]}
    monkeypatch.setattr(module,'_completed_scores',scores)
    monkeypatch.setattr(module,'_pressure_p95',lambda *args,**kwargs:None if kwargs['seed']==43 else .3)
    pressure=tmp_path/'pressure';pressure.mkdir()
    kwargs=dict(output_root=tmp_path,pressure_root=pressure)
    result=module.collect_simulation_review_results(**kwargs)
    assert len(result['completed'])==3 and len(result['task_rows'])==9
    assert result['status']=='waiting_for_simulation_review_pressure'
    assert result['pressure_pending']==['physiology_only/reference/seed43/self_supervised']
    (pressure/'queue_state.json').write_text(json.dumps({'failed_units':result['pressure_pending']}))
    failed=module.collect_simulation_review_results(**kwargs)
    assert failed['status']=='blocked_by_simulation_review_execution_failure'
    assert failed['pressure_failed']==result['pressure_pending'] and not failed['pressure_pending']
    monkeypatch.setattr(module,'_pressure_p95',lambda *args,**kwargs:.3)
    assert module.collect_simulation_review_results(**kwargs)['status']=='simulation_review_verified_not_final_selection'


def test_review_pressure_tail_rejects_another_seed_or_changed_artifact(tmp_path):
    root=tmp_path/'physiology_only/reference/review/seed29/self_supervised/1500';root.mkdir(parents=True)
    checkpoint=tmp_path/'best.pt';checkpoint.write_bytes(b'checkpoint')
    options=candidate_options('physiology_only','reference')
    state={'data_manifest_sha256':'a'*64,'candidate_options':options,
           'self_supervised_training':{'best_checkpoint_path':str(checkpoint)}}
    record={'consumer_prediction_sha256':'b'*64}
    source=dict(method='physiology_only',route='self_supervised',update=1500,phase='review',seed=29,
        evaluation_role='validation',confirmation_opened=False,consumer_refit=False,inference_device='cuda',
        data_manifest_sha256='a'*64,clean_prediction_sha256='b'*64,checkpoint_sha256=sha256_file(checkpoint),candidate_options=options)
    conditions={}
    for name in DEVELOPMENT_CONDITIONS:
        folder=root/name;folder.mkdir()
        result=folder/'result.json';result.write_text(json.dumps({'grouped':{'all_windows_retained':True,
            'regression_tails':[{'consumer':'linear','profile_id':'all_windows','p95_absolute_error':.4}]}}))
        prediction=folder/'predictions.npz';prediction.write_bytes(b'predictions')
        representation=folder/'representation/fusion_stream.npz';representation.parent.mkdir();representation.write_bytes(b'representation')
        conditions[name]=dict(result_path=str(result),result_sha256=sha256_file(result),prediction_path=str(prediction),
                              prediction_sha256=sha256_file(prediction),representation_sha256=sha256_file(representation))
    path=root/'run_state.json';saved=dict(completed=True,source=source,conditions=conditions)
    path.write_text(json.dumps(saved))
    args=(tmp_path,state,record,'physiology_only','reference','self_supervised',1500)
    assert _pressure_p95(*args,phase='review',seed=29)==.4
    source['seed']=43;path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='pressure roles'):_pressure_p95(*args,phase='review',seed=29)
    source['seed']=29;path.write_text(json.dumps(saved));result.write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):_pressure_p95(*args,phase='review',seed=29)
