from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_confirmation_training as module
from chronaris.evaluation.application_tasks import v4_public_screen as shared
from chronaris.evaluation.application_tasks.application_task_heads import ApplicationTaskDefinition, ApplicationTaskTargets
from chronaris.representation import FoldLineage, TrainOnlyRobustNormalizer, collate_observation_samples, select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.evaluation.application_tasks.test_v4_diagnostic_run import _sample


def test_native_confirmation_trains_both_routes_without_opening_outer_samples(tmp_path,monkeypatch):
    batch=collate_observation_samples([_sample(f'formal_{i}',shift=i/10) for i in range(12)])
    fold=FoldLineage('formal_fixture',batch.sample_ids[:6],batch.sample_ids[6:9],batch.sample_ids[9:])
    allowed=fold.train_sample_ids+fold.validation_sample_ids;accessed=[]
    def provider(ids):
        assert set(ids)<=set(allowed)
        accessed.extend(ids);return select_observation_batch(batch,ids)
    normalizer=TrainOnlyRobustNormalizer().fit(batch,train_sample_ids=fold.train_sample_ids,
        held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids)
    values={'classify':torch.arange(9)%3,'response':torch.arange(9,dtype=torch.float32)/10}
    targets=ApplicationTaskTargets(allowed,values,{k:torch.ones_like(v,dtype=torch.bool) for k,v in values.items()},
                                   {'source_role':'fixed_confirmation_subjects','smoke_only':True})
    definitions=(ApplicationTaskDefinition('classify','classification',3),ApplicationTaskDefinition('response','regression',1))
    full_values={'classify':torch.arange(12)%3,'response':torch.arange(12,dtype=torch.float32)/10}
    full_targets=ApplicationTaskTargets(batch.sample_ids,full_values,{k:torch.ones_like(v,dtype=torch.bool) for k,v in full_values.items()},targets.manifest)
    outer_accessed=[]
    def outer_provider(ids):
        outer_accessed.extend(ids);return select_observation_batch(batch,ids)
    data=SimpleNamespace(targets=full_targets,task_definitions=definitions,dataset=SimpleNamespace(batch_provider=outer_provider),
        sample_manifest=[{'sample_id':sample,'subject_id':role} for role in ('train','validation','held_out')
                         for sample in getattr(fold,role+'_sample_ids')])
    def inputs(*args,**kwargs):
        assert kwargs['subject_role']=='confirmation'
        return provider,_sample('schema').schema,fold,{s:(s,) for s in fold.train_sample_ids},'a'*64,targets,definitions,data
    monkeypatch.setattr(module,'load_development_inputs',inputs)
    def normalization(*args,**kwargs):
        assert kwargs['cache_root'].endswith('v4-public-confirmation-normalizers')
        return normalizer,None
    monkeypatch.setattr(module,'development_normalization',normalization)
    monkeypatch.setattr(module,'_require_diagnostic_device',lambda seed:None)
    monkeypatch.setattr(shared,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    pre,guided,loader=module.CandidateScreenConfig,module.EndToEndFineTuningConfig,module.load_common_pretraining_checkpoint
    scheduled=[]
    def pre_config(**kwargs):
        scheduled.append(kwargs);return pre(**(kwargs | {'max_updates':2,'device':'cpu','effective_batch_size':4,'validation_interval':1,'early_stopping':False}))
    def guided_config(**kwargs):
        scheduled.append(kwargs);return guided(**(kwargs | {'max_updates':1,'head_warmup_updates':1,'device':'cpu','effective_batch_size':4,'validation_interval':1,'early_stopping':False}))
    monkeypatch.setattr(module,'CandidateScreenConfig',pre_config)
    monkeypatch.setattr(module,'EndToEndFineTuningConfig',guided_config)
    monkeypatch.setattr(module,'load_common_pretraining_checkpoint',lambda path,**kwargs:loader(path,device='cpu'))
    registry=tmp_path/'registry.json';registry.write_text('{}')
    evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    options=module.candidate_options('physiology_only','reference')
    frozen=dict(format='chronaris.v4_frozen_configuration.v1',status='frozen',source_code_sha256=module.v4_workflow_source_sha256(),
        budget=module.CONFIRMATION_BUDGET,confirmation_feedback_used=False,evidence_files={str(evidence):sha256_file(evidence)},
        public_registry_sha256=sha256_file(registry),domain_status={'clare':'enabled'},
        normalizer_root='artifacts/application_evaluation/2026-09-08_v4-public-confirmation-normalizers',
        methods={r:{'physiology_only':options,'naive_time_sync':{'name':'reference','nonparametric':True}} for r in ('self_supervised','task_guided')})
    path=tmp_path/'frozen.json';path.write_text(json.dumps(frozen))
    kwargs=dict(freeze_path=path,freeze_sha256=sha256_file(path),domain='clare',fold_index=0,method='physiology_only',
                candidate_name='reference',seed=29,output_root=tmp_path/'trained',registry_path=registry)
    result=module.run_native_confirmation_training(**kwargs)
    assert result['completed'] and result['self_supervised_training']['optimizer_updates']==2
    assert result['task_guided_training']['optimizer_updates']==2
    assert set(accessed)==set(allowed) and not set(accessed)&set(fold.held_out_sample_ids)
    assert scheduled[0]['max_updates']==1500 and scheduled[1]['max_updates']==500 and scheduled[1]['head_warmup_updates']==50
    assert all(row['seed']==29 and row['device']=='cuda' for row in scheduled)
    for route in ('self_supervised','task_guided'):
        checkpoint=torch.load(result[route+'_training']['last_checkpoint_path'],map_location='cpu',weights_only=True)
        assert checkpoint['seed']==29
    repeated=module.run_native_confirmation_training(**kwargs)
    assert repeated['self_supervised_training']['optimizer_updates']==2
    from chronaris.evaluation.application_tasks import v4_native_frozen_evaluation as evaluation
    evaluator=evaluation.run_native_frozen_evaluation
    monkeypatch.setattr(evaluation,'load_development_inputs',inputs)
    monkeypatch.setattr(evaluation,'run_native_frozen_evaluation',lambda **kw:evaluator(**(kw | {'device':'cpu','engineering_only':True,'minirocket_kernels':84})))
    completed=module.run_native_confirmation_training(**kwargs,evaluate=True)
    assert completed['completed'] and set(completed['evaluations'])=={'self_supervised','task_guided'}
    assert set(fold.held_out_sample_ids)<=set(outer_accessed)
    assert not set(accessed)&set(fold.held_out_sample_ids)
    baseline=module.run_naive_native_confirmation(**{k:v for k,v in kwargs.items() if k not in ('method','candidate_name')})
    assert baseline['completed'] and baseline['neural_optimizer_updates']==0
    assert baseline['routes']['self_supervised']==baseline['routes']['task_guided']
    assert baseline['consumer_artifacts_shared_between_routes']
    with pytest.raises(ValueError,match='not selected'):module.run_native_confirmation_training(**(kwargs | {'candidate_name':'capacity64'}))
    evidence.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='evidence changed'):module.run_native_confirmation_training(**kwargs)
