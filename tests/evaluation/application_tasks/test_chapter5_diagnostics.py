from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import chapter5_delivery as delivery
from chronaris.evaluation.application_tasks import chapter5_delivery_diagnostics as diagnostics
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
from chronaris.modeling.fusion_encoders import NaiveTimeSyncEncoder
from chronaris.representation import collate_observation_samples
from tests.representation.test_contracts import _sample


def test_diagnostics_times_real_non_neural_adapter_and_preserves_neural_path(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import v4_naive_baseline,v4_public_screen
    from chronaris.modeling import training
    batch=collate_observation_samples([_sample('train')]+[_sample(f'v{i}',shift=i/10) for i in range(32)])
    naive=NaiveTimeSyncEncoder().fit(batch,train_sample_ids=('train',),held_out_sample_ids=batch.sample_ids[1:])
    assert not hasattr(naive,'parameters')
    neural=object();calls=[]
    def neural_adapter(**kw):
        assert kw['encoder'] is neural
        return lambda batch:calls.append(batch.sample_ids)
    monkeypatch.setattr(training,'TrainedFusionAdapter',neural_adapter)
    monkeypatch.setattr(v4_naive_baseline,'load_v4_naive_encoder',lambda *a,**k:(naive,None,None))
    monkeypatch.setattr(diagnostics,'load_frozen_application_encoder',lambda *a,**k:(neural,None,None))
    monkeypatch.setattr(diagnostics,'load_v4_simulation_development',lambda **kw:(SimpleNamespace(batch=batch),
        SimpleNamespace(validation_sample_ids=batch.sample_ids[1:],fold_id='f')))
    monkeypatch.setattr(v4_public_screen,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    for name in ('synchronize','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'test')
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'frozen')
    records=[dict(method=m,candidate_name='reference',checkpoints={'self_supervised':str(checkpoint)},
        data_manifest_sha256='d',training_source_sha256='s') for m in ('naive_time_sync','physiology_only')]
    result=diagnostics.diagnostics(tmp_path,{'simulation_root':'unused'},{'records':records})
    assert len(result['timings'])==20 and len(calls)==12 and all(len(ids)==4 for ids in calls)
    assert {r['method'] for r in result['timings']}=={'naive_time_sync','physiology_only'}
    assert {r['device'] for r in result['timings'] if r['method']=='naive_time_sync'}=={'cpu'}
    assert not result['confirmation_opened']


def test_diagnostics_revision_keeps_prior_revision_and_freeze_bytes(tmp_path,monkeypatch):
    source=tmp_path/'new.py';source.write_text('new')
    old_source=tmp_path/'old.py';old_source.write_text('old')
    plan={'plan_sha256':'p','input_files':{}}
    monkeypatch.setattr(delivery,'validated_plan',lambda root:plan)
    monkeypatch.setattr(delivery,'execution_files',lambda:{str(source):sha(source)})
    monkeypatch.setattr(delivery,'steps',lambda plan:['freeze','diagnostics','generate'])
    prior=tmp_path/'execution_revision.json'
    write(prior,dict(plan_sha256='p',execution_files={str(old_source):sha(old_source)},preserved_files={}))
    frozen=tmp_path/'delivery_models.json';write(frozen,dict(status='frozen',files={str(prior):sha(prior)}))
    write(tmp_path/'steps/freeze.json',dict(stage='freeze',plan_sha256='p',exit_code=0,files={str(frozen):sha(frozen)}))
    write(tmp_path/'state.json',dict(status='failed',phase='diagnostics',completed=['freeze'],plan_sha256='p'))
    (tmp_path/'controller.log').write_text('failure');(tmp_path/'logs').mkdir()
    (tmp_path/'logs/diagnostics.log').write_text('wrong adapter')
    original=(prior.read_bytes(),frozen.read_bytes())
    delivery.revise(tmp_path,phase='diagnostics')
    assert (prior.read_bytes(),frozen.read_bytes())==original
    assert str(tmp_path/'diagnostics_revision.json') in delivery.execution_binding(tmp_path,plan)
    def freeze(root,p,*,execution_evidence):
        assert execution_evidence=={str(old_source):sha(old_source),str(prior):sha(prior)}
        return read(frozen)
    monkeypatch.setattr(delivery,'freeze',freeze)
    assert delivery.require_freeze(tmp_path,plan)==read(frozen)
    frozen.write_text('modified')
    with pytest.raises(ValueError,match='source changed'):delivery.execution_binding(tmp_path,plan)
