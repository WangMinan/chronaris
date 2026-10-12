from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from chronaris.evaluation.application_tasks import chapter5_subset as subset
from chronaris.evaluation.application_tasks.chapter5_assets import write, sha
from chronaris.evaluation.application_tasks.chapter5_core import core_units
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe


def recipe(name):
    return training_recipe(name, method='chronaris', full=True, seed=17, digest='a'*64,
        train_count=5601, graph=True, calibration={'relations': []}, device='cpu')


def test_complete_and_semantic_removal_are_training_recipes_with_one_component_difference():
    full, removed = recipe('thesis_reference'), recipe('thesis_no_event_semantics')
    a, b = asdict(full[1]), asdict(removed[1])
    assert {k for k in a if a[k]!=b[k]} == {'semantic_event_enabled', 'learnable_semantic_queries'}
    assert a['semantic_event_enabled'] and not b['semantic_event_enabled']
    assert full[:1] == removed[:1] and full[2:4] == removed[2:4]
    assert full[3]['chronaris_mechanism_enabled'] and full[3]['chronaris_explicit_shift_enabled']
    assert full[1].max_updates == full[2].max_updates == 351
    assert full[2].head_warmup_updates == 50 and full[1].effective_batch_size == 16
    with pytest.raises(ValueError, match='calibration'):
        training_recipe('thesis_no_event_semantics', method='chronaris', full=True, seed=17, digest='a'*64, train_count=5, device='cpu')


def test_exact_scope_pairing_order_and_legacy_protocol_unchanged():
    units = subset.subset_units()
    assert len(units)==18 and {u['seed'] for u in units}=={17}
    for fold in range(5):
        pair = units[fold*2:fold*2+2]
        assert {u['recipe_name'] for u in pair}=={'thesis_reference','thesis_no_event_semantics'}
        assert all(u['domain']=='clare' and u['fold_index']==fold for u in pair)
    assert [u['fold_index'] for u in units[10:15]]==list(range(5))
    assert [u['options']['name'] for u in units[15:]]==['no_continuous_evolution','no_physics_residual','no_event_semantics']
    assert len(core_units())==36 and sum(len(u['routes']) for u in core_units())==69
    assert not any('semantic_event_enabled' in u['options'] for u in core_units())


def test_resume_count_excludes_head_warmup():
    assert subset.resume_counts({'step_count':125,'config':{'head_warmup_updates':50}})['joint_remaining']==425
    for count in (-1, 49, 551, 125.0):
        with pytest.raises(ValueError): subset.resume_counts({'step_count':count,'config':{'head_warmup_updates':50}})


def test_formal_thesis_requires_recipe_and_contract_before_data_loading(monkeypatch,tmp_path):
    from chronaris.evaluation.application_tasks import common_downstream_smoke as common
    monkeypatch.setattr(common.torch.cuda,'is_available',lambda:True)
    for name in ('thesis_reference','thesis_no_event_semantics'):
        with pytest.raises(ValueError,match='frozen recipe and data contract'):
            common.run_common_contract_smoke(domain='clare',output_root=tmp_path,methods=('chronaris',),
                recipe=name,full=True,subject_role='confirmation')
    assert not list(tmp_path.iterdir())


def test_plan_rejects_source_inventory_and_file_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(subset,'v4_workflow_source_sha256',lambda:'current')
    source=tmp_path/'input';source.write_text('original')
    def save(**changes):
        value=dict(root=str(tmp_path),source_code_sha256='current',units=subset.subset_units(),
            input_files={str(source):sha(source)})|changes
        write(tmp_path/'subset_plan.json',subset.seal_development_plan(value))
    save(); subset.verified_subset(tmp_path)
    save(source_code_sha256='other')
    with pytest.raises(ValueError,match='source changed'):subset.verified_subset(tmp_path)
    save(units=subset.subset_units()[:-1])
    with pytest.raises(ValueError,match='inventory changed'):subset.verified_subset(tmp_path)
    save(); source.write_text('modified')
    with pytest.raises(ValueError,match='source changed'):subset.verified_subset(tmp_path)


def test_subset_freeze_requires_all_units_and_preserves_original_gate(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks.chapter5_native import require_outer_freeze
    units=subset.subset_units();plan={'units':units,'input_files':{},'plan_sha256':'p','source_code_sha256':'s'}
    monkeypatch.setattr(subset,'verified_subset',lambda root:plan)
    checked=[]
    monkeypatch.setattr(subset,'validate_unit',lambda result,unit,plan:checked.append(unit['index']))
    for i in range(17):write(tmp_path/'units'/f'{i}.json',{'files':{}})
    with pytest.raises(ValueError,match='missing unit 17'):subset.freeze_subset(tmp_path)
    assert not (tmp_path/'subset_models.json').exists()
    write(tmp_path/'units/17.json',{'files':{}})
    gate=subset.freeze_subset(tmp_path)
    assert len(gate['unit_receipts'])==18 and gate['subset_models_frozen'] and not gate['legacy_core_models_frozen']
    assert subset.require_subset_freeze(tmp_path)==gate
    with pytest.raises(ValueError,match='core-model freeze'):require_outer_freeze(None,tmp_path)
    write(tmp_path/'units/17.json',{'files':{},'changed':True})
    with pytest.raises(ValueError,match='artifact changed'):subset.require_subset_freeze(tmp_path)


def test_final_evaluation_rejects_incomplete_subset_before_loading_data(tmp_path):
    from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import evaluate_native, generate_simulation, evaluate_simulation
    for call in (lambda:evaluate_native(tmp_path,0,role='held_out',sample_limit=0),
                 lambda:generate_simulation(tmp_path),lambda:evaluate_simulation(tmp_path,0)):
        with pytest.raises(ValueError,match='complete model freeze'):call()
    assert not list(tmp_path.iterdir())


def test_training_dispatch_binds_fold_recipe_contract_and_does_not_start_other_units(tmp_path,monkeypatch):
    from chronaris.evaluation.application_tasks import common_downstream_smoke as common
    unit=subset.subset_units()[3]|dict(recipe={'frozen':True},expected_contract_sha256='contract')
    monkeypatch.setattr(subset,'verified_subset',lambda root:{'units':[unit],'data_root':'data','registry_path':'registry'})
    calls=[]
    def run(**kwargs):calls.append(kwargs);return {'status':'waiting_gpu'}
    monkeypatch.setattr(common,'run_common_contract_smoke',run)
    assert subset.train_subset_unit(tmp_path,0)['status']=='waiting_gpu'
    assert len(calls)==1 and calls[0]['fold_index']==1 and calls[0]['recipe']=='thesis_no_event_semantics'
    assert calls[0]['expected_contract_sha256']=='contract' and calls[0]['expected_recipe']=={'frozen':True}
    assert calls[0]['subject_role']=='confirmation' and calls[0]['full']


def test_checkpoint_validation_rejects_inference_only_module_enablement(tmp_path,monkeypatch):
    import torch
    from chronaris.evaluation.application_tasks.chapter5_subset_checkpoints import validate_checkpoints
    unit=subset.subset_units()[17]
    checkpoints={r:str(tmp_path/f'{r}.pt') for r in unit['routes']}
    for p in checkpoints.values():Path(p).write_text('checkpoint')
    result={'training':{r+'_training':{'best_checkpoint_path':p} for r,p in checkpoints.items()},
        'files':{p:sha(p) for p in checkpoints.values()}}
    payload=dict(training_status='completed',seed=17,config={'device':'cuda','semantic_event_enabled':False,
        'learnable_semantic_queries':False,'physics_weight':.05},chronaris_mechanism_enabled=True,
        chronaris_explicit_shift_enabled=True,chronaris_variant='full',source_checkpoint_path=checkpoints['self_supervised'],
        source_checkpoint_sha256=sha(checkpoints['self_supervised']))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:payload)
    validate_checkpoints(result,unit)
    payload['config']['semantic_event_enabled']=True
    with pytest.raises(ValueError,match='declared thesis mechanisms'):validate_checkpoints(result,unit)
    payload['config']['semantic_event_enabled']=False
    payload['source_checkpoint_sha256']='different'
    with pytest.raises(ValueError,match='initialization'):validate_checkpoints(result,unit)


def test_budget_probe_stops_only_its_children_and_records_timeout(tmp_path,monkeypatch):
    import subprocess
    from chronaris.evaluation.application_tasks import chapter5_budget_probe as probe
    killed=[]
    class Child:
        pid=12345
        returncode=-15
        calls=0
        def wait(self,timeout=None):
            self.calls+=1
            if self.calls==1:raise subprocess.TimeoutExpired('owned probe',timeout)
            return -15
    monkeypatch.setattr(probe.subprocess,'Popen',lambda *a,**kw:Child())
    monkeypatch.setattr(probe.os,'killpg',lambda pid,sig:killed.append(pid))
    result=probe.probe_pair(tmp_path/'new')
    assert result['status']=='partial' and [r['status'] for r in result['domains']]==['timed_out','timed_out']
    assert killed==[12345,12345] and not result['confirmation_opened']
