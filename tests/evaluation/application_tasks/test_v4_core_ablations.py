from types import SimpleNamespace

import pytest
import torch

from chronaris.evaluation.application_tasks import v4_core_ablations as module
from chronaris.evaluation.application_tasks import v4_confirmation_training as training
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import ApplicationConsumerSmokeTargets
from chronaris.evaluation.application_tasks.application_task_heads import SIMULATION_TASKS
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.representation import FoldLineage,TrainOnlyRobustNormalizer,collate_observation_samples,select_observation_batch
from tests.representation.test_contracts import _sample
from tests.evaluation.application_tasks.test_v4_simulation_confirmation import _configuration


def test_core_ablation_plan_removes_one_component_and_skips_absent_objectives(tmp_path):
    frozen,_=_configuration(tmp_path)
    plan=module.build_core_ablation_plan(frozen)
    assert plan['evaluation_units']==24 and len(plan['units'])==12 and len(plan['skipped'])==6
    assert all(unit['routes']==['self_supervised','task_guided'] for unit in plan['units'])
    assert not plan['confirmation_feedback_used'] and plan['observation_anchor_retained']
    for route in frozen['methods']:frozen['methods'][route]['chronaris']=candidate_options('chronaris','independent_pairing')
    paired=module.build_core_ablation_plan(frozen)
    removed=next(u for u in paired['units'] if u['ablation']=='no_independent_pair_loss')
    assert removed['options']['training']['independent_pair_weight']==0
    assert removed['options']['training']['independent_pairing_enabled']
    assert paired['evaluation_units']==30
    for route in frozen['methods']:frozen['methods'][route]['chronaris']=candidate_options('chronaris','quality_gate')
    assert any(u['ablation']=='no_quality_gate' for u in module.build_core_ablation_plan(frozen)['units'])
    for route in frozen['methods']:frozen['methods'][route]['chronaris']=candidate_options('chronaris','missingness_mixture')
    assert any(u['ablation']=='no_expanded_missingness' and not u['options']['missingness_mixture']
               for u in module.build_core_ablation_plan(frozen)['units'])


@pytest.mark.parametrize('ablation,seed',[('no_physics_residual',17),('no_explicit_shift',29),('no_continuous_evolution',43)])
def test_core_options_survive_real_pretraining_and_guidance(tmp_path,monkeypatch,ablation,seed):
    frozen,kwargs=_configuration(tmp_path)
    samples=[_sample(f's_{i}',shift=i/10) for i in range(9)]
    batch=collate_observation_samples(samples)
    fold=FoldLineage('engineering__training512',batch.sample_ids[:6],batch.sample_ids[6:],('sealed',))
    roles={r:getattr(fold,r+'_sample_ids') for r in ('train','validation','held_out')}
    accessed=[]
    def provider(ids):
        assert 'sealed' not in ids;accessed.extend(ids);return select_observation_batch(batch,ids)
    states=torch.arange(96).repeat(9,1)%5
    targets=ApplicationConsumerSmokeTargets(batch.sample_ids,('train',)*6+('validation',)*3,torch.linspace(0,1,9),
        torch.arange(9)%3,states,torch.cat((torch.zeros(9,1,dtype=torch.bool),states[:,1:]!=states[:,:-1]),dim=1),{'smoke_only':True})
    inputs=(provider,samples[0].schema,fold,{s:(s,) for s in fold.train_sample_ids},'d'*64,None,SIMULATION_TASKS,SimpleNamespace(role_sample_ids=roles))
    normalizer=TrainOnlyRobustNormalizer().fit(batch,train_sample_ids=fold.train_sample_ids,held_out_sample_ids=fold.validation_sample_ids)
    monkeypatch.setattr(training,'load_development_inputs',lambda *a,**k:inputs)
    from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
    calibration=fit_physics_calibration(normalizer,provider,train_sample_ids=fold.train_sample_ids,
        vehicle_feature_names=samples[0].schema.vehicle_feature_names,relations=())
    monkeypatch.setattr(training,'development_normalization',lambda *a,**k:(normalizer,calibration))
    from chronaris.evaluation.application_tasks import application_consumer_smoke_data as target_module,v4_public_screen as locks
    monkeypatch.setattr(target_module,'build_guarded_application_consumer_targets',lambda *a,**k:targets)
    monkeypatch.setattr(locks,'GPU_LOCK_PATH',str(tmp_path/'gpu.lock'))
    monkeypatch.setattr(module,'_require_diagnostic_device',lambda seed:None)
    pre,guided,loader=training.CandidateScreenConfig,training.EndToEndFineTuningConfig,training.load_common_pretraining_checkpoint
    monkeypatch.setattr(training,'CandidateScreenConfig',lambda **kw:pre(**(kw|dict(max_updates=2,device='cpu',effective_batch_size=4,early_stopping=False))))
    monkeypatch.setattr(training,'EndToEndFineTuningConfig',lambda **kw:guided(**(kw|dict(max_updates=1,head_warmup_updates=1,device='cpu',effective_batch_size=4,early_stopping=False))))
    monkeypatch.setattr(training,'load_common_pretraining_checkpoint',lambda path,**kw:loader(path,device='cpu'))
    result=module.train_core_ablation(**kwargs,ablation=ablation,base_candidate='reference',seed=seed)
    assert result['completed'] and set(accessed)==set(batch.sample_ids)
    pre_payload=torch.load(result['training']['self_supervised_training']['best_checkpoint_path'],map_location='cpu',weights_only=True)
    guided_payload=torch.load(result['training']['task_guided_training']['best_checkpoint_path'],map_location='cpu',weights_only=True)
    assert pre_payload['seed']==guided_payload['seed']==seed
    assert guided_payload['source_checkpoint_path']==result['training']['self_supervised_training']['best_checkpoint_path']
    assert pre_payload['chronaris_variant']==('no_continuous_evolution' if ablation=='no_continuous_evolution' else 'full')
    if ablation=='no_physics_residual':
        assert pre_payload['config']['physics_weight']==0
        encoder,_,fitted,_=loader(result['training']['self_supervised_training']['best_checkpoint_path'],device='cpu')
        output=encoder(fitted.transform(batch),compute_chronaris_diagnostics=True)
        anchor=output.auxiliary['physical_consistency'].observation_anchor
        assert anchor is not None and float(anchor.detach())>0
        anchor.backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in encoder.parameters())
    if ablation=='no_explicit_shift':assert pre_payload['chronaris_explicit_shift_enabled'] is False
    with pytest.raises(ValueError,match='not applicable'):
        module.train_core_ablation(**kwargs,ablation='no_independent_pair_loss',base_candidate='reference',seed=seed)
