import numpy as np
import pytest
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from chronaris.evaluation.application_tasks.v4_mechanism_data import timing_scenarios
from chronaris.evaluation.application_tasks.v4_mechanism_evaluation import mechanism_values, timing_metrics
from chronaris.evaluation.application_tasks.v4_final_reporting import safety_checks, write_final_report, METHODS
from chronaris.simulation.aviation_dual_stream.config import ObservationScenarioConfig


def test_clock_sign_and_response_delay_remain_distinct_and_all_errors_are_kept(tmp_path):
    np.savez(tmp_path/'ground_truth.npz',realized_physiology_lag_s=np.asarray([5.,9.]))
    rows=[dict(sample_id='sample',trajectory_id='trajectory',observed_path=str(tmp_path/'raw_dual_stream.npz'))]
    values,manifest=mechanism_values(rows,ObservationScenarioConfig('negative_clock',physiology_clock_offset_s=-1.))
    np.testing.assert_array_equal(values,[[1.,-1.,5.]])
    assert manifest['oracle_opened_after_representation']
    assert len(timing_scenarios())==12
    assert {s.physiology_clock_offset_s for s in timing_scenarios()}=={-3.,-1.,-.25,0.,.25,1.,3.}
    metric=timing_metrics(np.repeat(values,8,axis=0),np.zeros((8,3)))
    assert [row['rmse'] for row in metric]==[1.,1.,5.]
    assert all(row['sample_count']==8 and row['largest_five_squared_error_fraction']==5/8 for row in metric)
    with pytest.raises(ValueError):timing_metrics(values,np.full((1,3),np.nan))


def _pressure():
    result=[]
    for method,f1,error in [('chronaris',.60,1.20),('physiology_only',.7,1.),('vehicle_only',.65,1.1)]:
        result.append(dict(method=method,route='self_supervised',seed=17,condition='clean_asynchronous',
            grouped={'profile_metrics':[{'consumer':'linear','metric':'macro_f1','task':'workload_classification','value':f1}],
                     'regression_tails':[{'consumer':'linear','profile_id':'all_windows','rmse':error}]}))
    return result


def test_report_retains_failed_protection_and_renders_chinese_task_labels(tmp_path):
    checks=safety_checks(_pressure())
    assert len(checks)==1 and not checks[0]['f1_passed'] and not checks[0]['regression_passed']
    public=[]
    for domain,task in [('cogpilot','difficulty'),('clare','workload_classification')]:
        for route in ('self_supervised','task_guided'):
            for method in METHODS:
                public.append(dict(domain=domain,task=task,route=route,method=method,consumer='linear',metric='macro_f1',value=.6))
    evidence=dict(main_evaluation_units=432,public_scalar_rows=public,native_subject_rows=[],simulation_rows=[],
        simulation_paired_statistics=[],public_paired_statistics=[],pressure=_pressure(),mechanisms=[])
    write_final_report(evidence,tmp_path)
    report=(tmp_path/'report.md').read_text()
    assert '其中 1 项' in report and '不对单记录作跨架次显著性推断' in report
    assert (tmp_path/'公开分类主表.png').stat().st_size>1000


def test_timing_pipeline_fits_only_training_features_and_resumes_all_routes(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import v4_mechanism_evaluation as module
    from chronaris.evaluation.application_tasks import v4_public_screen
    from chronaris.representation import FoldLineage
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    formal=tmp_path/'formal';formal.mkdir()
    for filename in ('frozen_configuration.json','simulation_frozen_models.json'):(formal/filename).write_text('{}')
    timing=tmp_path/'timing';timing.mkdir();(timing/'timing_audit.json').write_text('{}')
    heldroot=tmp_path/'held';heldroot.mkdir();(heldroot/'confirmation_generation_audit.json').write_text('{}')
    checkpoint=tmp_path/'encoder.pt';checkpoint.write_bytes(b'engineering fixture')
    digest=sha256_file(checkpoint)
    units=[dict(method='chronaris',candidate_name=f'fixture_{i}',seed=seed,routes=['self_supervised','task_guided'],
                checkpoints={route:str(checkpoint) for route in ('self_supervised','task_guided')}) for i in range(6) for seed in (17,29,43)]
    monkeypatch.setattr(module,'read_simulation_model_freeze',lambda *a,**k:{'records':units,'simulation_root':'unused'})
    fold=FoldLineage('timing_fixture',('t0','t1','t2','t3'),('v0','v1'),('h0','h1'))
    roles={role:getattr(fold,role+'_sample_ids') for role in ('train','validation','held_out')}
    def rows(condition,role):
        directory=timing/condition;directory.mkdir(exist_ok=True)
        if not (directory/'ground_truth.npz').exists():np.savez(directory/'ground_truth.npz',realized_physiology_lag_s=[5.])
        return [dict(sample_id=sample,trajectory_id=sample,profile_id='p'+str(i%2),role=role,
                     observed_path=str(directory/'raw_dual_stream.npz')) for i,sample in enumerate(roles[role])]
    monkeypatch.setattr(module,'load_v4_simulation_development',lambda **kw:(SimpleNamespace(sample_manifest_rows=[]),fold))
    monkeypatch.setattr(module,'timing_batch',lambda root,condition,manifest:(None,rows(condition,'train')+rows(condition,'validation')))
    monkeypatch.setattr(module,'load_simulation_confirmation',lambda root,**kw:SimpleNamespace(sample_manifest_rows=rows(kw['condition'],'held_out')))
    monkeypatch.setattr(module,'load_frozen_application_encoder',lambda *a,**kw:(None,None,None))
    monkeypatch.setattr(v4_public_screen,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    def output(role):
        offset={'train':0.,'validation':10.,'held_out':100.}[role]
        return SimpleNamespace(sample_ids=roles[role],checkpoint_sha256=digest,
                               pooled_embedding=torch.arange(len(roles[role])*3).reshape(-1,3).float()+offset)
    def exports(**kw):
        for role in ('train','validation'):
            directory=Path(kw['root'])/role;directory.mkdir(parents=True,exist_ok=True)
            for name in ('representation_manifest.json','fusion_stream.npz'):(directory/name).write_text('{}')
        return {role:output(role) for role in ('train','validation')}
    monkeypatch.setattr(module,'export_loaded_application_encoder',exports)
    for unit in units:
        for route in unit['routes']:
            for scenario in timing_scenarios():
                directory=module.simulation_unit_root(formal,unit)/'evaluation'/route/'pressure'/scenario.scenario_id/'representation/held_out'
                directory.mkdir(parents=True,exist_ok=True)
                for name in ('representation_manifest.json','fusion_stream.npz'):(directory/name).write_text('{}')
    monkeypatch.setattr(module,'load_fusion_stream_batch',lambda path:output('held_out'))
    fit=module.fit_regressor;calls=[]
    def checked_fit(train,y,validation,vy,**kw):
        assert train.max()<12 and validation.max()<16 and kw['alpha_values']==(1.,)
        calls.append(1)
        return fit(train,y,validation,vy,**kw)
    monkeypatch.setattr(module,'fit_regressor',checked_fit)
    kwargs=dict(formal_root=formal,timing_root=timing,confirmation_root=heldroot,output_root=tmp_path/'results')
    result=module.run_final_mechanisms(**kwargs)
    assert result['status']=='completed' and len(result['records'])==36 and len(calls)==108
    assert all(len(row['condition_metrics'])==12 and row['metrics'][0]['sample_count']==24 for row in result['records'])
    assert module.run_final_mechanisms(**kwargs)['status']=='completed' and len(calls)==108
    record=result['records'][0];Path(record['prediction_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='timing evidence changed'):module.run_final_mechanisms(**kwargs)
