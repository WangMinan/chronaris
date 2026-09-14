from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from chronaris.evaluation.application_tasks.stage45 import compare_candidate, stage45_groups, screen_units
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.evaluation.application_tasks.stage45_diagnostics import branch_features, fidelity_trigger
from chronaris.evaluation.application_tasks.v4_pipeline import execute_pipeline


def test_recipes_change_only_declared_model_settings(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    def recipe(name, **kw):
        return training_recipe(name, method='chronaris', full=True, seed=17, digest='a'*64, train_count=366, **kw)
    ref = recipe('stage4_reference')
    assert ref[0].learning_rate == 1e-3 and ref[1].weight_decay == 1e-5
    assert ref[1].max_updates == 300 and ref[2].max_updates == 200
    assert not ref[1].semantic_event_enabled and not ref[3]['chronaris_mechanism_enabled']
    for name, component, key, value in [('pretraining_lr',0,'learning_rate',3e-4),
        ('finetuning_lr',2,'learning_rate',3e-5), ('cosine_temperature',1,'attention_kind','cosine_temperature'),
        ('single_stream_fidelity',1,'single_stream_fidelity_weight',.2)]:
        actual = recipe(name)
        for i in range(3):
            expected = asdict(ref[i]) | ({key:value} if i == component else {})
            assert asdict(actual[i]) == expected
    with pytest.raises(ValueError, match='calibration'):
        recipe('thesis_reference')
    thesis = recipe('thesis_reference', calibration={'relations':[]})
    assert thesis[1].semantic_event_enabled and thesis[3]['chronaris_explicit_shift_enabled']
    assert not thesis[-1]['confirmation_opened']
    with pytest.raises(ValueError,match='structural'):
        training_recipe('thesis_reference', method='mult', full=True, seed=17, digest='a'*64, train_count=10)


def test_selection_requires_complete_coverage_and_keeps_negative_tasks():
    reference = [dict(domain=d,route=r,task=t,metric=m,value=v) for d in ('clare','cogpilot')
        for r in ('self_supervised','task_guided') for t,m,v in [('classification','macro_f1',.4),('regression','rmse',2.)]]
    candidate = [r | {'value':r['value']*.9 if r['metric']=='rmse' else r['value']} for r in reference]
    assert compare_candidate(candidate,reference)['eligible']
    candidate[0] = candidate[0] | {'value':.3}
    result = compare_candidate(candidate,reference)
    assert not result['eligible'] and result['rows'][0]['improvement'] < 0
    with pytest.raises(ValueError,match='coverage'):
        compare_candidate(candidate[:-1],reference)
    with pytest.raises(ValueError,match='duplicated'):
        compare_candidate(candidate+[candidate[0]],reference)


def test_stage45_is_finite_and_never_freezes_or_confirms():
    stages = [s for g in stage45_groups() for s,_ in g]
    assert len(stages) == len(set(stages))
    assert stages[-1] == 'stage45_report'
    assert not any('confirmation' in s or 'freeze' in s for s in stages)
    assert len(screen_units()) == 18
    assert sum('stage45_review' in s for s in stages) == 20


def test_expired_budget_does_not_launch_a_worker(tmp_path):
    import time
    config = {'root':str(tmp_path)}
    (tmp_path/'pipeline_state.json').write_text(json.dumps(dict(completed={},failures=[],attempts={},
        started_at_unix_s=time.time()-73*3600)))
    result = execute_pipeline(config,until='stage45')
    assert result['status']=='budget_exhausted' and not result['children'] and not result['completed']


def test_branch_probe_uses_real_masked_parts_and_fidelity_trigger():
    from types import SimpleNamespace as NS
    valid = torch.tensor([[True,False]])
    state = NS(reference_hidden_states=torch.ones(1,2,32),reference_valid_mask=valid)
    fusion = NS(physiology_private=torch.ones(1,2,24),vehicle_private=torch.ones(1,2,24)*2)
    output = NS(sequence_embedding=torch.ones(1,2,64),modality_available_mask=valid,
                auxiliary={'alignment_output':NS(physiology=state,vehicle=state),'fusion_output':fusion})
    features = branch_features(output)
    assert features['branches_concatenated'][0].shape[-1]==48
    assert features['full'][1] is valid
    def score(v):
        return {'components':{'linear':{'task_summary':{'validation':[dict(task='regression',metric='rmse',value=v)]}}}}
    probe = dict(route='self_supervised',branches={s+p:score(v) for s in ('physiology','vehicle')
        for p,v in [('_before_projection',1.),('_after_projection',1.1)]})
    assert fidelity_trigger([probe])['enabled']


def test_frozen_contract_mismatch_stops_before_training(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from chronaris.evaluation.application_tasks import common_downstream_smoke as entry
    from tests.evaluation.application_tasks.test_common_downstream_contract import _case
    kwargs, _, _, _ = _case(tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(entry,'development_gpu_lock',lambda:nullcontext(True))
    monkeypatch.setattr(entry,'contract_development_inputs',lambda *a,**k: (None,None,
        kwargs['fold'],kwargs['data_manifest_sha256'],kwargs['targets'],kwargs['definitions'],
        kwargs['context'],kwargs['observations']))
    with pytest.raises(ValueError,match='frozen development plan'):
        entry.run_common_contract_smoke(domain='clare', output_root=tmp_path/'run',
            recipe='stage4_reference', methods=('chronaris',),expected_contract_sha256='0'*64)
    assert not (tmp_path/'run/clare/data_contract.json').exists()


@pytest.mark.skipif(not __import__('os').environ.get('CHRONARIS_STAGE45_ENTRY_CHECK') or not torch.cuda.is_available(),
                   reason='explicit short real CUDA entry check')
def test_cuda_stage45_real_mechanism_entry(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from chronaris.evaluation.application_tasks import common_downstream_smoke as entry
    monkeypatch.setattr(entry,'development_gpu_lock',lambda:nullcontext(True))
    result = entry.run_common_contract_smoke(domain='clare', output_root=tmp_path, methods=('chronaris',),
        recipe='thesis_reference',cuda_graph_recurrence=True)
    assert result['status']=='completed'
    guided = next(r for r in result['results'] if r.get('route')=='task_guided' and 'training' in r)
    payload = torch.load(guided['training']['last_checkpoint_path'],map_location='cpu',weights_only=True)
    rows = [r for r in payload['update_rows'] if r['stage']=='joint_adaptation']
    assert all(any(t.get('weight',0)>0 and t.get('count',0)>0 for t in r['mechanism_terms']) for r in rows)
    assert any(v>0 for r in rows for v in r['gradient_groups'].values())
    assert all(any(t['term_name']=='explicit_time_shift' and t['weight']>0 and t['count']>0
                   for t in r['mechanism_terms']) for r in rows)
    assert any(r['gradient_groups']['explicit_shift_head']>0 for r in rows)
    assert any(r['gradient_groups']['encoder.backbone.semantic_event_fusion']>0 for r in rows)
    assert payload['stage_update_counts']['joint_adaptation']==2 and not result['confirmation_opened']


@pytest.mark.parametrize('domain', ['clare', 'cogpilot', 'dingxin'])
@pytest.mark.parametrize('recipe', ['stage4_reference', 'thesis_reference'])
@pytest.mark.skipif(not __import__('os').environ.get('CHRONARIS_STAGE45_PERFORMANCE_CHECK') or not torch.cuda.is_available(),
                   reason='explicit real native graph and independent restart validation')
def test_cuda_stage45_execution_and_independent_restart(tmp_path, domain, recipe):
    import os
    import subprocess
    import sys
    from chronaris.evaluation.application_tasks.stage45_performance import compare_execution
    project = Path(__file__).resolve().parents[3]
    config = dict(root=str(tmp_path),
        data_root=str(project/'artifacts/application_evaluation/2026-09-06_v4-public-development'),
        registry_path=str(project/'docs/requirements/thesis-v4-public-subjects.json'))
    for mode in ('eager','graph','interrupt','resume'):
        command = ('import json; from chronaris.evaluation.application_tasks.stage45_performance import performance_entry; '
                   'performance_entry(**json.loads(__import__("sys").argv[1]))')
        with (tmp_path/(mode+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-c',command,json.dumps(dict(config=config,domain=domain,recipe=recipe,mode=mode))],
                cwd=project,env=os.environ | {'PYTHONPATH':str(project/'src')},stdout=log,stderr=subprocess.STDOUT,check=True)
    check = compare_execution(tmp_path/'performance'/domain/recipe)
    (tmp_path/'comparison.json').write_text(json.dumps(check,indent=2))
    # Qualification may legitimately reject graph execution; do not require a
    # performance experiment to succeed scientifically in order to record it.
    assert check['checks'] and check['independent_resume']['passed'], str(tmp_path/'comparison.json')


def test_rejected_graph_qualification_uses_eager_training(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks import stage45
    directory = tmp_path/'performance/clare/stage4_reference'
    directory.mkdir(parents=True)
    (directory/'check.json').write_text(json.dumps({'passed':False}))
    (tmp_path/'stage45_plan.json').write_text(json.dumps({'contracts':{'clare/0/17':'a'*64}}))
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {'status':'completed'}
    monkeypatch.setattr(stage45,'run_common_contract_smoke',run)
    result = stage45._run_unit(dict(root=str(tmp_path),data_root='data',registry_path='registry'),
        dict(domain='clare',method='chronaris',recipe='cosine_temperature'),tmp_path/'unit')
    assert result['graph_execution'] is False
    assert calls[0]['cuda_graph_recurrence'] is False and calls[0]['expected_contract_sha256']=='a'*64
