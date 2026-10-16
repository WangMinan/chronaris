from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chronaris.evaluation.application_tasks import chapter5_delivery as delivery
from chronaris.evaluation.application_tasks import chapter5_delivery_diagnostics as diagnostics
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import (
    ApplicationConsumerSmokeData,build_guarded_application_consumer_targets)
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample


def references(tmp_path):
    rows=[]
    for method in ('chronaris','physiology_only','vehicle_only','mult','contiformer'):
        checkpoint=tmp_path/(method+'.pt')
        torch.save(dict(format='chronaris.common_pretraining_checkpoint.v2',training_status='completed',
                        method_name=method,label_used_for_encoder_training=False),checkpoint)
        rows.append(dict(method=method,candidate_name='reference',seed=17,
                         checkpoints={r:str(checkpoint) for r in ('self_supervised','task_guided')}))
    return rows


def test_target_references_ignore_grid_and_ablation_counts_but_reject_missing_or_duplicate_methods(tmp_path):
    rows=references(tmp_path)
    extras=[rows[0]|dict(candidate_name=name) for name in ('no_physics_residual','no_explicit_shift','weight_0025')]
    expected=[r['checkpoints']['self_supervised'] for r in rows]
    assert diagnostics.reference_initializations(rows+extras,17)==expected
    for bad in (rows[:-1]+extras,rows+[rows[0]],rows[:-1]+[rows[0]],rows[1:]+[rows[0]|dict(seed=29)]):
        with pytest.raises(ValueError,match='five distinct'):
            diagnostics.reference_initializations(bad,17)


def test_sensitivity_four_weights_use_real_five_checkpoint_guard_and_training_only_thresholds(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import chapter5_subset_evaluation as subset
    from chronaris.evaluation.application_tasks import application_consumer_runtime as runtime,v4_public_screen
    rows=references(tmp_path);zero=rows[0]|dict(candidate_name='no_physics_residual')
    rows.append(zero)
    for i in (30,33):
        write(tmp_path/'extra_units'/f'{i}.json',dict(files={},training={
            r+'_training':dict(best_checkpoint_path=rows[0]['checkpoints'][r]) for r in ('self_supervised','task_guided')}))
    samples=[_sample('train'),_sample('validation')];batch=collate_observation_samples(samples)
    manifest=[]
    for role,value in [('train',.2),('validation',.9)]:
        folder=tmp_path/role;folder.mkdir()
        np.savez(folder/'ground_truth.npz',true_time_s=np.arange(800)*.05,workload=np.full(800,value),maneuver_state=np.zeros(800))
        manifest.append(dict(sample_id=role,role=role,observed_path=str(folder/'observed.npz'),context_start_s=0.,context_end_s=30.))
    data=ApplicationConsumerSmokeData(batch,samples[0].schema,dict(train=('train',),validation=('validation',),held_out=()),tuple(manifest))
    monkeypatch.setattr(subset,'simulation_records',lambda root:dict(records=rows))
    monkeypatch.setattr(diagnostics,'load_v4_simulation_development',lambda **kw:(data,SimpleNamespace(fold_id='f')))
    monkeypatch.setattr(diagnostics,'load_frozen_application_encoder',lambda *a,**kw:(object(),None,None))
    monkeypatch.setattr(diagnostics,'export_loaded_application_encoder',lambda **kw:{})
    monkeypatch.setattr(v4_public_screen,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    seen=[];result_type=make_dataclass('Result',[('status',str)])
    def consume(**kw):
        targets=kw['targets'];seen.append(targets)
        assert targets.sample_ids==('train','validation') and 'held_out' not in targets.roles
        assert targets.manifest['oracle_opened_after_checkpoint_count']==5
        assert targets.manifest['workload_thresholds_train_only']==pytest.approx([.2,.2])
        return result_type('completed')
    monkeypatch.setattr(runtime,'run_application_method_consumers',consume)
    result=diagnostics.sensitivity(tmp_path,dict(subset_root='unused',simulation_root='unused',sensitivity_weights=[0.,.025,.05,.1]))
    assert len(seen)==8 and len(result['rows'])==8
    assert {r['weight'] for r in result['rows']}=={0.,.025,.05,.1} and not result['confirmation_opened']
    source=rows[0]['checkpoints']['self_supervised'];payload=torch.load(source,weights_only=True)
    for change in ({'training_status':'running'},{'label_used_for_encoder_training':True}):
        torch.save(payload|change,source)
        with pytest.raises(ValueError,match='checkpoint guard failed'):
            build_guarded_application_consumer_targets(data,completed_pretraining_checkpoints=diagnostics.reference_initializations(rows,17))


def test_execution_revision_preserves_receipts_binds_source_and_rejects_completed_outer_gate(tmp_path,monkeypatch):
    source=tmp_path/'new_code.py';source.write_text('repaired')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'frozen weights')
    plan=dict(plan_sha256='plan',input_files={})
    monkeypatch.setattr(delivery,'validated_plan',lambda root:plan)
    monkeypatch.setattr(delivery,'execution_files',lambda:{str(source):sha(source)})
    monkeypatch.setattr(delivery,'steps',lambda p:['train:24','sensitivity','freeze'])
    write(tmp_path/'state.json',dict(status='failed',phase='sensitivity',completed=['train:24'],plan_sha256='plan'))
    (tmp_path/'controller.log').write_text('original failure')
    (tmp_path/'logs').mkdir();(tmp_path/'logs/sensitivity.log').write_text('five checkpoints')
    receipt=tmp_path/'steps/train_24.json'
    write(receipt,dict(stage='train:24',plan_sha256='plan',exit_code=0,files={str(checkpoint):sha(checkpoint)}))
    original=receipt.read_bytes()
    with pytest.raises(ValueError,match='without a bound revision'):delivery.execution_binding(tmp_path,plan)
    revision=delivery.revise(tmp_path)
    assert revision['preserved_stages']==['train:24'] and receipt.read_bytes()==original
    assert str(tmp_path/'execution_revision.json') in delivery.execution_binding(tmp_path,plan)
    source.write_text('changed')
    with pytest.raises(ValueError,match='revision differs'):delivery.execution_binding(tmp_path,plan)
    source.write_text('repaired');checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):delivery.execution_binding(tmp_path,plan)
    checkpoint.write_bytes(b'frozen weights');write(tmp_path/'delivery_models.json',{})
    with pytest.raises(ValueError,match='stopped pre-sensitivity'):delivery.revise(tmp_path)


def test_resume_skips_verified_training_and_runs_only_pending_sensitivity(tmp_path,monkeypatch):
    import signal
    from chronaris.evaluation.application_tasks import v4_public_screen
    plan={'plan_sha256':'fixed'}
    monkeypatch.setattr(delivery,'plan_at',lambda root:plan)
    monkeypatch.setattr(delivery,'steps',lambda plan:['train:24','sensitivity'])
    monkeypatch.setattr(delivery,'execution_binding',lambda *a:{})
    monkeypatch.setattr(signal,'signal',lambda *a:None)
    monkeypatch.setattr(v4_public_screen,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    output=tmp_path/'training.json';write(output,{'status':'completed'})
    receipt=tmp_path/'steps/train_24.json'
    write(receipt,dict(plan_sha256='fixed',exit_code=0,files={str(output):sha(output)}))
    original=receipt.read_bytes();called=[]
    def spawn(args,**kw):
        stage=args[args.index('--stage')+1];called.append(stage)
        write(args[args.index('--result')+1],dict(status='completed',files={}))
        return SimpleNamespace(pid=987,wait=lambda:0)
    monkeypatch.setattr(delivery.subprocess,'Popen',spawn)
    result=delivery.run(tmp_path,'launcher.py',tmp_path)
    assert called==['sensitivity'] and result['completed']==['train:24','sensitivity']
    assert receipt.read_bytes()==original
