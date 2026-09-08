import pytest
from chronaris.evaluation.application_tasks.v4_public_results import TASKS,rank_development_rows,collect_public_screen_results


def _rows(seeds=(17,)):
    rows=[]
    for seed in seeds:
        for candidate in ('reference','quality_gate'):
            for domain,task,metric,direction in TASKS:
                # Reference wins all three simulation tasks; candidate wins both public domains.
                better=(domain=='simulation') == (candidate=='reference')
                value=(.9 if better else .5) if direction=='higher' else (.1 if better else .5)
                rows.append(dict(candidate=candidate,seed=seed,domain=domain,task=task,metric=metric,role='validation',value=value))
    return rows


def test_domain_balancing_and_seed_ranking_use_complete_fixed_tasks():
    metadata={c:dict(encoder_parameters=10,training_elapsed_s=20.) for c in ('reference','quality_gate')}
    ranked=rank_development_rows(_rows(),metadata)
    assert ranked[0]['candidate']=='quality_gate'
    assert ranked[0]['domain_task_ranks']['17']=={'simulation':2.,'cogpilot':1.,'clare':1.}
    repeated=rank_development_rows(_rows((17,29,43)),metadata,seeds=(17,29,43))
    assert repeated[0]['median_seed_rank']==1 and repeated[0]['seed_ranks']==[1.,1.,1.]
    tied=[row | {'value':.5} for row in _rows()]
    metadata['reference']['encoder_parameters']=9
    assert rank_development_rows(tied,metadata)[0]['candidate']=='reference'
    for bad in (_rows()[:-1],_rows()+_rows()[:1],[row | {'role':'held_out'} for row in _rows()]):
        with pytest.raises(ValueError):rank_development_rows(bad,metadata)


def test_public_result_reader_does_not_rank_before_a_selected_plan_exists(tmp_path):
    result=collect_public_screen_results(output_root=tmp_path)
    assert result['rankings']=={} and result['status']=='waiting_for_public_screen_plan'


def test_domain_weight_is_not_the_number_of_tasks():
    rows=[row | {'value':.5} if row['domain']=='clare' else row for row in _rows()]
    metadata={'reference':dict(encoder_parameters=10,training_elapsed_s=20.),'quality_gate':dict(encoder_parameters=9,training_elapsed_s=20.)}
    result=rank_development_rows(rows,metadata)
    assert result[0]['candidate']=='quality_gate'
    assert result[0]['median_seed_rank']==result[1]['median_seed_rank']==1.5


def test_public_reader_replays_a_completed_unit_and_waits_for_the_other_domain(tmp_path,monkeypatch):
    import hashlib,json
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from chronaris.evaluation.application_tasks import v4_public_results as module
    from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS
    from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
    from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskTargets
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import run_native_method_consumers
    from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
    from chronaris.representation import FoldLineage,write_fusion_stream_batch
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    from tests.evaluation.application_tasks.test_v4_grouped_consumers import _public_inputs
    outputs,old_targets,_,context=_public_inputs()
    fold=FoldLineage('fixture',outputs['train'].sample_ids,outputs['validation'].sample_ids,(),development_only=True)
    definitions=PUBLIC_TASKS['clare']
    values={'workload_classification':torch.arange(18)%2,'workload_regression':torch.arange(18.)}
    targets=ApplicationTaskTargets(old_targets.sample_ids,values,{k:torch.ones_like(v,dtype=torch.bool) for k,v in values.items()},{})
    directory=tmp_path/'clare/chronaris/reference/fold01';directory.mkdir(parents=True)
    checkpoint=directory/'fixture.pt';checkpoint.write_bytes(b'unit fixture checkpoint')
    outputs={role:replace(output,fold_id=fold.fold_id,checkpoint_sha256=sha256_file(checkpoint)) for role,output in outputs.items()}
    for role,output in outputs.items():
        write_fusion_stream_batch(output,root=directory/'representations/self_supervised/300'/role,export_role=role)
    result=run_native_method_consumers(outputs=outputs,targets=targets,definitions=definitions,context=context,
        output_root=directory/'consumers',label_used_for_encoder_training=False,minirocket_kernels=84)
    (directory/'self_supervised_300_consumers.json').write_text(json.dumps(result))
    source=v4_workflow_source_sha256()
    (directory/'run_state.json').write_text(json.dumps(dict(method='chronaris',domain='clare',phase='screen',seed=17,
        candidate_options=candidate_options('chronaris','reference'),source_code_sha256=source,data_manifest_sha256='a'*64,
        fold=fold.to_dict(),confirmation_opened=False,completed_consumers=['self_supervised:300'],
        self_supervised_training=dict(optimizer_updates=300,best_checkpoint_path=str(checkpoint),parameter_count=100,training_elapsed_s=1.))))
    registry=tmp_path/'registry.json';registry.write_text('{}')
    simulation={'completed':[]}
    plan=dict(source_code_sha256=source,public_registry_sha256=sha256_file(registry),confirmation_feedback_used=False,
        simulation_diagnostic_root=str(tmp_path),simulation_pressure_root=str(tmp_path),simulation_summaries={'chronaris/self_supervised':simulation},
        units=[dict(domain=domain,method='chronaris',candidate_name='reference',routes=['self_supervised']) for domain in ('clare','cogpilot')])
    digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    (tmp_path/'selection_plan.json').write_text(json.dumps(plan | {'plan_sha256':digest}))
    (tmp_path/'run_state.json').write_text(json.dumps(dict(plan_sha256=digest,completed_units=['clare/chronaris/reference'],failed_units=[])))
    data=SimpleNamespace(sample_manifest=[{'sample_id':s,'subject_id':g} for s,g in context['groups'].items()])
    monkeypatch.setattr(module,'collect_simulation_screen',lambda **kwargs:simulation)
    monkeypatch.setattr(module,'load_development_inputs',lambda *args,**kwargs:(None,None,fold,None,'a'*64,targets,definitions,data))
    audited=module.collect_public_screen_results(output_root=tmp_path,registry_path=registry)
    assert len(audited['public_task_rows'])==2 and audited['rankings']=={}
    assert audited['pending']==['cogpilot/chronaris/reference/self_supervised']
    (directory/'fixture.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='selected checkpoint'):
        module.collect_public_screen_results(output_root=tmp_path,registry_path=registry)
