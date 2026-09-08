"""Native outer-fold training using the existing pretext and adaptation trainers."""
from dataclasses import asdict
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method)
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_development_data import (
    load_development_inputs, development_normalization, v4_workflow_source_sha256)
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.training import (
    CandidateScreenConfig, EncoderCandidateConfig, load_common_pretraining_checkpoint, train_pretext_candidate)
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import AugmentationPolicy
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

CONFIRMATION_BUDGET=dict(pretraining_max_updates=1500,head_warmup_updates=50,joint_max_updates=500,
    batch_size=4,effective_batch_size=32,dingxin_effective_batch_size=16,pretraining_minimum_updates=500,pretraining_patience=5,
    pretraining_validation_interval=100,joint_minimum_updates=200,joint_patience=4,joint_validation_interval=50)


def read_frozen_configuration(path, expected_sha256):
    if sha256_file(path)!=expected_sha256:raise ValueError('frozen configuration file changed')
    frozen=json.loads(Path(path).read_text())
    if (frozen['format']!='chronaris.v4_frozen_configuration.v1' or frozen['status']!='frozen'
        or frozen['source_code_sha256']!=v4_workflow_source_sha256() or frozen['budget']!=CONFIRMATION_BUDGET
        or frozen['confirmation_feedback_used'] or not frozen['evidence_files']):
        raise ValueError('configuration is not frozen under the current source and approved budget')
    for filename,digest in frozen['evidence_files'].items():
        if sha256_file(filename)!=digest:raise ValueError('configuration freeze evidence changed')
    return frozen


def run_native_confirmation_training(*, freeze_path, freeze_sha256, domain, fold_index, method, candidate_name,
                                     seed, output_root,
                                     data_root='artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared',
                                     registry_path='docs/requirements/thesis-v4-public-subjects.json', evaluate=False):
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    if domain not in ('cogpilot','clare','dingxin') or seed not in (17,29,43):
        raise ValueError('unsupported native confirmation domain or seed')
    if frozen['domain_status'].get(domain)!='enabled':raise ValueError('confirmation domain is blocked by its data audit')
    if frozen['public_registry_sha256']!=sha256_file(registry_path):raise ValueError('frozen subject registry changed')
    options=candidate_options(method,candidate_name)
    routes=[route for route in ('self_supervised','task_guided') if frozen['methods'][route][method]==json.loads(json.dumps(options))]
    if not routes:raise ValueError('candidate was not selected for this formal method')
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu',confirmation_started=False)
        _require_diagnostic_device(seed)
        training=_train_native_unit(domain=domain,fold_index=fold_index,method=method,options=options,routes=routes,
            seed=seed,output_root=output_root,data_root=data_root,registry_path=registry_path,freeze_sha256=freeze_sha256,
            normalizer_root=frozen['normalizer_root'])
        if not evaluate:return training
        from chronaris.evaluation.application_tasks.v4_native_frozen_evaluation import run_native_frozen_evaluation
        root=Path(output_root)/domain/method/candidate_name/f'fold{fold_index+1:02d}'/f'seed{seed}'
        evaluations={}
        for route in routes:
            checkpoint=training[route+'_training']['best_checkpoint_path']
            evaluations[route]=run_native_frozen_evaluation(domain=domain,fold_index=fold_index,checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),route=route,output_root=root/'evaluation'/route,
                data_root=data_root,registry_path=registry_path,device='cuda',method=method)
        result=dict(status='completed',completed=True,training=training,evaluations=evaluations,freeze_sha256=freeze_sha256)
        temporary=root/'confirmation_unit.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(root/'confirmation_unit.json')
        return result


def _native_training_inputs(domain,fold_index,data_root,registry_path,root):
    inputs=load_development_inputs(
        domain,data_root,registry_path,fold_index=fold_index,subject_role='confirmation')
    _,_,fold,_,_,_,_,data=inputs
    if domain=='dingxin':
        from chronaris.evaluation.application_tasks.v4_dingxin_data import audit_dingxin_vehicle_reuse
        audit=audit_dingxin_vehicle_reuse(data)
        root.mkdir(parents=True,exist_ok=True)
        (root/'vehicle_content_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        if not audit['outer_roles_disjoint']:raise ValueError('Dingxin outer roles share vehicle contents; training is blocked')
    elif data.targets.manifest.get('source_role')!='fixed_confirmation_subjects':
        raise ValueError('formal training cannot use the public development subjects')
    if not fold.held_out_sample_ids:raise ValueError('native confirmation requires an outer held-out role')
    return inputs


def _train_native_unit(*, domain, fold_index, method, options, routes, seed, output_root, data_root, registry_path, freeze_sha256, normalizer_root):
    root=Path(output_root)/domain/method/options['name']/f'fold{fold_index+1:02d}'/f'seed{seed}'
    provider,schema,fold,hierarchy,digest,targets,definitions,data=_native_training_inputs(domain,fold_index,data_root,registry_path,root)
    source=dict(format='chronaris.v4_native_confirmation_training.v1',source_code_sha256=v4_workflow_source_sha256(),
        freeze_sha256=freeze_sha256,domain=domain,method=method,candidate_options=options,seed=seed,routes=routes,
        fold=fold.to_dict(),data_manifest_sha256=digest,outer_outcomes_used_for_training=False)
    source=json.loads(json.dumps(source))
    root.mkdir(parents=True,exist_ok=True);path=root/'run_state.json'
    state=json.loads(path.read_text()) if path.exists() else dict(source=source,completed=False)
    if state['source']!=source:raise ValueError('frozen native training source, fold or configuration changed')
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
    normalizer,calibration=development_normalization(domain,provider,schema,fold,digest,
        **({'cache_root':normalizer_root} if domain!='dingxin' else {}))
    state['train_only_normalization']=dict(normalizer=normalizer.to_manifest(),physics_calibration=calibration)
    save()
    with _periodic_training_heartbeat(f'confirmation_{domain}_{method}',30,root=root) as progress:
        chronaris=method=='chronaris';effective=16 if domain=='dingxin' else 32
        progress['phase']='pretraining'
        pretraining=train_pretext_candidate(method,candidate=EncoderCandidateConfig(candidate_id='C',hidden_dim=options['hidden_dim'],learning_rate=3e-4),
            batch=None,batch_provider=provider,fold=fold,physiology_feature_names=schema.physiology_feature_names,
            vehicle_feature_names=schema.vehicle_feature_names,vehicle_field_labels=(),normalizer=normalizer,
            output_root=root/'self_supervised',config=CandidateScreenConfig(max_updates=1500,batch_size=4,effective_batch_size=effective,
                weight_decay=1e-4,device='cuda',early_stopping=True,seed=seed,minimum_updates=500,patience=5,validation_interval=100,
                retained_updates=(1500,),semantic_event_enabled=chronaris,learnable_semantic_queries=chronaris,
                physics_calibration=calibration if chronaris else None,physics_weight=.05,sampling_hierarchy=hierarchy,
                data_manifest_sha256=digest,cuda_graph_recurrence=chronaris,**options['training']),
            augmentation_policy=AugmentationPolicy(missingness_mixture=options['missingness_mixture']),
            chronaris_fusion_kind='safe_lag' if chronaris else 'multiscale',chronaris_mechanism_enabled=chronaris,
            chronaris_explicit_shift_enabled=chronaris,chronaris_explicit_shift_weight=.1 if chronaris else 0.,chronaris_event_pair_weight=0.)
        state['self_supervised_training']=asdict(pretraining);save()
        if 'task_guided' in routes:
            progress['phase']='task_guided'
            encoder,_,normalizer=load_common_pretraining_checkpoint(pretraining.best_checkpoint_path,device='cuda')[:3]
            with isolated_training_rng(seed):
                model=EndToEndApplicationModel(method_name=method,encoder=encoder,normalizer=normalizer,naive_encoder=None,task_definitions=definitions)
            guided=train_end_to_end_application_method(model=model,batch=None,batch_provider=provider,targets=targets,
                role_sample_ids={role:getattr(fold,role+'_sample_ids') for role in ('train','validation','held_out')},
                source_checkpoint_path=pretraining.best_checkpoint_path,output_root=root/'task_guided',
                config=EndToEndFineTuningConfig(max_updates=500,head_warmup_updates=50,batch_size=4,effective_batch_size=effective,
                    weight_decay=1e-4,device='cuda',early_stopping=True,seed=seed,minimum_updates=200,patience=4,
                    validation_interval=50,retained_updates=(500,),sampling_hierarchy=hierarchy,data_manifest_sha256=digest))
            state['task_guided_training']=asdict(guided);save()
        state['completed']=True;state['status']='completed';save()
    return state


def run_naive_native_confirmation(*, freeze_path, freeze_sha256, domain, fold_index, seed, output_root,
                                 data_root='artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared',
                                 registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    from chronaris.evaluation.application_tasks.v4_naive_baseline import fit_v4_naive_encoder
    from chronaris.evaluation.application_tasks.v4_native_frozen_evaluation import run_native_frozen_evaluation
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    if domain not in ('cogpilot','clare','dingxin') or seed not in (17,29,43) or frozen['domain_status'].get(domain)!='enabled':
        raise ValueError('unsupported frozen baseline domain or seed')
    if sha256_file(registry_path)!=frozen['public_registry_sha256']:raise ValueError('frozen subject registry changed')
    if any(frozen['methods'][route]['naive_time_sync']!={'name':'reference','nonparametric':True} for route in ('self_supervised','task_guided')):
        raise ValueError('nonparametric formal baseline configuration changed')
    root=Path(output_root)/domain/'naive_time_sync/reference'/f'fold{fold_index+1:02d}'/f'seed{seed}'
    provider,schema,fold,_,digest,_,_,_=_native_training_inputs(domain,fold_index,data_root,registry_path,root)
    normalizer,_=development_normalization(domain,provider,schema,fold,digest,
        **({'cache_root':frozen['normalizer_root']} if domain!='dingxin' else {}))
    checkpoint,fit=fit_v4_naive_encoder(provider=provider,fold=fold,normalizer=normalizer,data_manifest_sha256=digest,
                                      output_root=root/'checkpoint',seed=seed)
    evaluation_root=root/'evaluation/shared'
    evaluation=run_native_frozen_evaluation(domain=domain,fold_index=fold_index,checkpoint=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),route='self_supervised',output_root=evaluation_root,
        data_root=data_root,registry_path=registry_path,device='cpu',method='naive_time_sync')
    result=dict(status='completed',completed=True,freeze_sha256=freeze_sha256,consumers=evaluation,
                routes={route:str(evaluation_root) for route in ('self_supervised','task_guided')},consumer_artifacts_shared_between_routes=True,
                neural_optimizer_updates=0,projection_checkpoint_sha256=sha256_file(checkpoint),projection_shared_between_routes=True,
                projection_fit_elapsed_s=fit['training_elapsed_s'])
    temporary=root/'confirmation_unit.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(root/'confirmation_unit.json')
    return result
