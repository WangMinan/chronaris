from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_confirmation_training as training
from chronaris.evaluation.application_tasks import v4_simulation_confirmation as simulation
from chronaris.evaluation.application_tasks import v4_simulation_confirmation_evaluation as evaluation
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeData, ApplicationConsumerSmokeTargets
from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from tests.representation.test_contracts import _sample
from tests.evaluation.application_tasks.test_v4_simulation_confirmation import _configuration


def test_shared_simulation_training_and_all_pressure_consumers_keep_confirmation_isolated(tmp_path,monkeypatch):
    frozen,kwargs=_configuration(tmp_path)
    batch=collate_observation_samples([_sample(f'sample_{i}',shift=i/10) for i in range(13)])
    fold=FoldLineage('engineering__training512',batch.sample_ids[:6],batch.sample_ids[6:9],batch.sample_ids[9:])
    roles={role:getattr(fold,role+'_sample_ids') for role in ('train','validation','held_out')}
    rows=tuple(dict(sample_id=sample,profile_id='held' if sample in roles['held_out'] else 'training',role=role)
               for role,ids in roles.items() for sample in ids)
    development=ApplicationConsumerSmokeData(select_observation_batch(batch,batch.sample_ids[:9]),_sample('schema').schema,
        roles,rows[:9])
    held=ApplicationConsumerSmokeData(select_observation_batch(batch,fold.held_out_sample_ids),development.schema,
        dict(train=(),validation=(),held_out=fold.held_out_sample_ids),rows[9:])
    states=torch.arange(96).repeat(13,1)%5
    boundaries=torch.cat((torch.zeros(13,1,dtype=torch.bool),states[:,1:]!=states[:,:-1]),dim=1)
    targets=ApplicationConsumerSmokeTargets(batch.sample_ids,('train',)*6+('validation',)*3+('held_out',)*4,
        torch.linspace(0,1,13),torch.arange(13)%3,states,boundaries,
        dict(smoke_only=True,oracle_files=[],workload_thresholds_train_only=[.2,.4]))
    def select_targets(data,**kw):
        selected=[batch.sample_ids.index(sample) for sample in data.batch.sample_ids]
        return replace(targets,sample_ids=data.batch.sample_ids,roles=tuple(targets.roles[i] for i in selected),
                       **{name:getattr(targets,name)[selected] for name in ('future_workload_mean','workload_class','maneuver_state','boundary_mask')})
    accessed=[]
    def provider(ids):
        assert not set(ids)&set(fold.held_out_sample_ids)
        accessed.extend(ids);return select_observation_batch(development.batch,ids)
    normalizer=TrainOnlyRobustNormalizer().fit(batch,train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids)
    inputs=(provider,development.schema,fold,{s:(s,) for s in fold.train_sample_ids},'d'*64,None,SIMULATION_TASKS,development)
    monkeypatch.setattr(training,'load_development_inputs',lambda *a,**k:inputs)
    monkeypatch.setattr(training,'development_normalization',lambda *a,**k:(normalizer,None))
    monkeypatch.setattr(simulation,'development_normalization',lambda *a,**k:(normalizer,None))
    from chronaris.evaluation.application_tasks import application_consumer_smoke_data as target_module
    monkeypatch.setattr(target_module,'build_guarded_application_consumer_targets',select_targets)
    monkeypatch.setattr(simulation,'_require_diagnostic_device',lambda seed:None)
    from chronaris.evaluation.application_tasks import v4_public_screen as locks
    monkeypatch.setattr(locks,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    pre,guided,loader=training.CandidateScreenConfig,training.EndToEndFineTuningConfig,training.load_common_pretraining_checkpoint
    monkeypatch.setattr(training,'CandidateScreenConfig',lambda **kw:pre(**(kw|dict(max_updates=2,device='cpu',effective_batch_size=4,early_stopping=False))))
    monkeypatch.setattr(training,'EndToEndFineTuningConfig',lambda **kw:guided(**(kw|dict(max_updates=1,head_warmup_updates=1,device='cpu',effective_batch_size=4,early_stopping=False))))
    monkeypatch.setattr(training,'load_common_pretraining_checkpoint',lambda path,**kw:loader(path,device='cpu'))
    result=simulation.train_simulation_confirmation(**kwargs,method='physiology_only',candidate_name='reference',seed=17)
    assert result['completed'] and result['training']['self_supervised_training']['optimizer_updates']==2
    assert result['training']['task_guided_training']['optimizer_updates']==2
    assert set(accessed)==set(fold.train_sample_ids+fold.validation_sample_ids)
    baseline=simulation.train_simulation_confirmation(**kwargs,method='naive_time_sync',candidate_name='reference',seed=17)
    assert baseline['neural_optimizer_updates']==0
    unit=dict(method='naive_time_sync',candidate_name='reference',seed=17,routes=['self_supervised','task_guided'],
        checkpoints={route:baseline['checkpoint'] for route in ('self_supervised','task_guided')},fold=fold.to_dict(),data_manifest_sha256='d'*64)
    models={'simulation_root':'engineering','records':[unit]+[dict(seed=17,method=m,routes=['self_supervised'],checkpoints={'self_supervised':'fixture'})
        for m in ('physiology_only','vehicle_only','mult','contiformer','chronaris')]}
    monkeypatch.setattr(evaluation,'load_v4_simulation_development',lambda **kw:(development,fold))
    def observed(root,**kw):
        if kw.get('condition','clean_asynchronous')=='clean_asynchronous':return held
        from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations
        return replace(held,batch=remove_future_observations(held.batch,cutoff_s=-1.))
    monkeypatch.setattr(evaluation,'load_simulation_confirmation',observed)
    monkeypatch.setattr(evaluation,'build_guarded_application_consumer_targets',select_targets)
    rocket,tcn=evaluation.MiniRocketConsumerConfig,evaluation.TCNConsumerConfig
    monkeypatch.setattr(evaluation,'MiniRocketConsumerConfig',lambda **kw:rocket(**kw,n_kernels=84))
    monkeypatch.setattr(evaluation,'TCNConsumerConfig',lambda **kw:tcn(**(kw|dict(epochs=1,hidden_channels=8,device='cpu'))))
    from chronaris.evaluation.application_tasks.application_consumers import LinearFrozenConsumer,MiniRocketFrozenConsumer
    original_fits=(LinearFrozenConsumer.fit,MiniRocketFrozenConsumer.fit)
    frozen_evaluate=evaluation.evaluate_frozen_application_consumers
    def no_fit(*args,**kw):pytest.fail('pressure must reuse the clean consumers')
    def evaluate(**kw):
        monkeypatch.setattr(LinearFrozenConsumer,'fit',no_fit)
        monkeypatch.setattr(MiniRocketFrozenConsumer,'fit',no_fit)
        return frozen_evaluate(**kw)
    monkeypatch.setattr(evaluation,'evaluate_frozen_application_consumers',evaluate)
    arguments=dict(models=models,unit=unit,freeze_sha256=kwargs['freeze_sha256'],model_freeze_sha256='m'*64,
                   output_root=kwargs['output_root'],confirmation_root='engineering')
    actual=evaluation._evaluate_simulation_unit(**arguments)
    assert actual['completed'] and len(actual['routes']['self_supervised']['conditions'])==38
    assert actual['routes']['task_guided']['shared_with']=='self_supervised'
    missing=actual['routes']['self_supervised']['conditions']['vehicle_missing']
    record=json.loads(Path(missing['result_path']).read_text())
    assert record['grouped']['no_observation_count']==4 and record['grouped']['all_windows_retained']
    assert all(row['role']=='held_out' for row in record['evaluation']['metric_rows'])
    assert evaluation._evaluate_simulation_unit(**arguments)['completed']
    Path(missing['result_path']).write_text('changed')
    with pytest.raises(ValueError,match='saved formal pressure'):
        evaluation._evaluate_simulation_unit(**arguments)

    monkeypatch.setattr(LinearFrozenConsumer,'fit',original_fits[0])
    monkeypatch.setattr(MiniRocketFrozenConsumer,'fit',original_fits[1])
    core=evaluation._evaluate_simulation_unit(**(arguments | {'output_root':tmp_path/'core','include_pressure':False}))
    assert core['completed'] and core['source']['pressure_condition_count']==0
    assert set(core['routes']['self_supervised']['conditions'])=={'clean_asynchronous'}
