from copy import deepcopy
import json
import time

import pytest
import torch

from chronaris.evaluation.application_tasks.stage45_resume import execution_checkpoint, inherited_step, read_evidence
from chronaris.evaluation.application_tasks.v4_pipeline import stage45_budget_expired
from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
from tests.evaluation.application_tasks.test_execution_migration import _payload


def test_mid_training_execution_checkpoint_keeps_all_state_and_can_restore_eager(tmp_path):
    payload = _payload()
    payload.update(optimizer_updates=75,step_count=75,
        data_cursor={'samples_seen':1200,'micro_batches_seen':300,'sampling_order_sha256':'fixed'})
    path = tmp_path/'parent.pt';torch.save(payload,path)
    graph = execution_checkpoint(payload,parent_path=path,graph=True,evidence_sha256='proof')
    assert graph['canonical_training_state_sha256']==payload['canonical_training_state_sha256']
    assert graph['data_cursor']==payload['data_cursor'] and graph['optimizer_updates']==75
    assert graph['protocol_sha256']!=payload['protocol_sha256']
    torch.save(graph,tmp_path/'graph.pt')
    ordinary = execution_checkpoint(graph,parent_path=tmp_path/'graph.pt',graph=False,evidence_sha256='fallback')
    assert ordinary['protocol_sha256']==payload['protocol_sha256']
    assert len(ordinary['execution_history'])==2
    broken = deepcopy(payload);broken['optimizer_state_dict']['tensor'][0]=5
    with pytest.raises(ValueError,match='digest'):
        execution_checkpoint(broken,parent_path=path,graph=True,evidence_sha256='proof')


def test_explicit_unlimited_budget_survives_old_deadline():
    state={'started_at_unix_s':time.time()-100*3600}
    assert stage45_budget_expired({},state)
    assert not stage45_budget_expired({'stage45_budget_hours':None},state)
    assert not stage45_budget_expired({'stage45_budget_hours':120},state)
    for value in (-1,0,float('nan')):
        with pytest.raises(ValueError):stage45_budget_expired({'stage45_budget_hours':value},state)


def test_inherited_qualification_keeps_failure_and_rejects_tampering(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    result=parent/'result.json';result.write_text(json.dumps({'status':'completed','passed':False}))
    stage='stage45_perf__clare__thesis_reference__check'
    (parent/'pipeline_state.json').write_text(json.dumps({'completed':{stage:{'path':str(result),'sha256':sha256_file(result)}}}))
    evidence=tmp_path/'evidence.json'
    evidence.write_text(json.dumps(dict(status='completed',confirmation_opened=False,parent_root=str(parent),
        candidate_source_sha256=candidate_source_code_sha256(),bindings={},resume_pretraining_graph=False)))
    config=dict(stage45_resume_parent=str(parent),stage45_resume_evidence=str(evidence))
    inherited=inherited_step(stage,config)
    assert inherited['passed'] is False and inherited['inherited_evidence']['parent_sha256']==sha256_file(result)
    result.write_text(json.dumps({'status':'completed','passed':True}))
    with pytest.raises(ValueError,match='changed'):inherited_step(stage,config)


@pytest.mark.skipif(not __import__('os').environ.get('CHRONARIS_STAGE45_MIXED_CHECK') or not torch.cuda.is_available(),
                   reason='explicit native graph-pretraining and eager-finetuning boundary test')
def test_cuda_actual_mixed_execution_keeps_pretraining_weights_and_guided_lineage(tmp_path,monkeypatch):
    from contextlib import nullcontext
    from chronaris.evaluation.application_tasks import common_downstream_smoke as entry
    monkeypatch.setattr(entry,'development_gpu_lock',lambda:nullcontext(True))
    result=entry.run_common_contract_smoke(domain='clare',methods=('chronaris',),recipe='thesis_reference',
        output_root=tmp_path,cuda_graph_recurrence=True,finetuning_graph_recurrence=False)
    rows={r['route']:r for r in result['results'] if 'training' in r}
    source=torch.load(rows['self_supervised']['training']['best_checkpoint_path'],map_location='cpu',weights_only=True)
    guided=torch.load(rows['task_guided']['training']['last_checkpoint_path'],map_location='cpu',weights_only=True)
    initializer=torch.load(guided['source_checkpoint_path'],map_location='cpu',weights_only=True)
    assert source['config']['cuda_graph_recurrence'] is True
    assert initializer['config']['cuda_graph_recurrence'] is False
    assert initializer['canonical_training_state_sha256']==source['canonical_training_state_sha256']
    assert guided['stage_update_counts']['joint_adaptation']==2
