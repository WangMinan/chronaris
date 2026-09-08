"""Clean and 35 plus two pressure evaluations reuse the same frozen consumers."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
from chronaris.evaluation.application_tasks.application_consumer_runtime import ApplicationConsumerProtocol, run_application_method_consumers
from chronaris.evaluation.application_tasks.application_consumers import LinearConsumerConfig, MiniRocketConsumerConfig, TCNConsumerConfig
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder, export_loaded_application_encoder
from chronaris.evaluation.application_tasks.application_frozen_evaluation import evaluate_frozen_application_consumers
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_encoding_diagnostics import collect_encoding_diagnostics
from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
from chronaris.evaluation.application_tasks.v4_pressure_run import summarize_pressure_predictions, _verify_clean_predictions
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY, simulation_unit_root
from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import read_simulation_model_freeze, load_simulation_confirmation
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import select_observation_batch
from chronaris.simulation.aviation_dual_stream.config import locked_stress_observation_scenarios
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def evaluate_simulation_confirmation(*, freeze_path, freeze_sha256, model_freeze_path, model_freeze_sha256,
                                     method, candidate_name, seed, output_root, confirmation_root):
    models = read_simulation_model_freeze(model_freeze_path,model_freeze_sha256,freeze_path=freeze_path,freeze_sha256=freeze_sha256)
    matches = [row for row in models['records']
               if (row['method'],row['candidate_name'],row['seed']) == (method,candidate_name,seed)]
    if len(matches)!=1:
        raise ValueError('evaluation unit is absent from the frozen simulation model inventory')
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu',completed=False)
        _require_diagnostic_device(seed)
        return _evaluate_simulation_unit(models=models,unit=matches[0],freeze_sha256=freeze_sha256,
            model_freeze_sha256=model_freeze_sha256,output_root=output_root,confirmation_root=confirmation_root)


def _evaluate_simulation_unit(*, models, unit, freeze_sha256, model_freeze_sha256, output_root, confirmation_root):
    method, seed = unit['method'], unit['seed']
    root = simulation_unit_root(output_root,unit)
    development, fold = load_v4_simulation_development(simulation_root=models['simulation_root'],registry_path=SIMULATION_REGISTRY)
    if fold.to_dict()!=unit['fold']:
        raise ValueError('simulation evaluation changed the encoder training or confirmation roles')
    held = load_simulation_confirmation(confirmation_root,model_freeze_sha256=model_freeze_sha256)
    if held.batch.sample_ids!=fold.held_out_sample_ids or held.schema!=development.schema:
        raise ValueError('confirmation samples or schema differ from the frozen encoder contract')
    initializations = [row['checkpoints']['self_supervised'] for row in models['records']
                       if row['seed']==seed and row['method']!='naive_time_sync' and 'self_supervised' in row['routes']]
    train_targets = build_guarded_application_consumer_targets(development,completed_pretraining_checkpoints=initializations,smoke_only=False)
    held_targets = build_guarded_application_consumer_targets(held,completed_pretraining_checkpoints=initializations,smoke_only=False,
        workload_thresholds=tuple(train_targets.manifest['workload_thresholds_train_only']))
    targets = replace(train_targets,sample_ids=train_targets.sample_ids+held_targets.sample_ids,
        roles=train_targets.roles+held_targets.roles,
        **{name:torch.cat((getattr(train_targets,name),getattr(held_targets,name))) for name in
           ('future_workload_mean','workload_class','maneuver_state','boundary_mask')},
        manifest=train_targets.manifest | dict(sample_count=len(train_targets.sample_ids)+len(held_targets.sample_ids),
            oracle_files=train_targets.manifest['oracle_files']+held_targets.manifest['oracle_files'],
            scope='frozen_confirmation',model_freeze_sha256=model_freeze_sha256))
    def provider(ids):
        for data in (development,held):
            if set(ids)<=set(data.batch.sample_ids):
                return select_observation_batch(data.batch,ids)
        raise ValueError('formal simulation export requested unknown or mixed data roles')
    conditions = ['clean_asynchronous']+[scenario.scenario_id for scenario in locked_stress_observation_scenarios()]+['physiology_missing','vehicle_missing']
    if len(conditions)!=38 or len(set(conditions))!=38:
        raise ValueError('formal pressure scope must have 35 fixed scenarios and two whole-modality checks')
    path=root/'confirmation_unit.json'
    source=dict(freeze_sha256=freeze_sha256,model_freeze_sha256=model_freeze_sha256,unit=unit,conditions=conditions,pressure_condition_count=37,clean_replay_count=1)
    state=json.loads(path.read_text()) if path.exists() else dict(source=source,freeze_sha256=freeze_sha256,completed=False,routes={})
    if state['source']!=source:
        raise ValueError('simulation evaluation sources changed')
    def save():
        root.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    with _periodic_training_heartbeat('simulation_confirmation',30,root=root) as progress:
        for route in (('self_supervised',) if method=='naive_time_sync' else unit['routes']):
            checkpoint=unit['checkpoints'][route];supervised=route=='task_guided'
            if method=='naive_time_sync':
                encoder,normalizer,_=load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256=unit['data_manifest_sha256'])
            else:
                encoder,normalizer,_=load_frozen_application_encoder(checkpoint,route=route,fold=fold,device='cuda')
            destination=root/'evaluation'/route
            progress.update(phase='clean_export',route=route)
            outputs=export_loaded_application_encoder(encoder=encoder,normalizer=normalizer,checkpoint=checkpoint,
                provider=provider,fold=fold,root=destination/'representations',export_roles=('train','validation','held_out'),
                export_prefix='simulation_confirmation',label_used_for_encoder_training=supervised)
            progress['phase']='clean_consumers'
            clean=run_application_method_consumers(method_name=method,outputs=outputs,targets=targets,
                output_root=destination/'consumers',fold_id=fold.fold_id,
                protocol=ApplicationConsumerProtocol(linear=LinearConsumerConfig(random_state=seed,tune_on_validation=True),
                    minirocket=MiniRocketConsumerConfig(random_state=seed,tune_on_validation=True),
                    tcn=TCNConsumerConfig(epochs=40,patience=6,seed=seed,device='cuda'),label_used_for_encoder_training=supervised))
            record=state['routes'].setdefault(route,dict(clean=asdict(clean),conditions={}))
            save()
            for condition in conditions:
                progress.update(phase='pressure_evaluation',condition=condition)
                if condition in record['conditions']:
                    for filename,digest in record['conditions'][condition]['files'].items():
                        if sha256_file(filename)!=digest:
                            raise ValueError('saved formal pressure evidence changed')
                    continue
                observed=held if condition=='clean_asynchronous' else load_simulation_confirmation(
                    confirmation_root,model_freeze_sha256=model_freeze_sha256,condition=condition)
                location=destination/'pressure'/condition
                pressure=export_loaded_application_encoder(encoder=encoder,normalizer=normalizer,checkpoint=checkpoint,
                    provider=lambda ids:select_observation_batch(observed.batch,ids),fold=fold,root=location/'representation',
                    export_roles=('held_out',),export_prefix='simulation_confirmation_pressure',label_used_for_encoder_training=supervised)['held_out']
                if condition=='clean_asynchronous' and (not torch.equal(outputs['held_out'].valid_mask,pressure.valid_mask)
                    or float((outputs['held_out'].sequence_embedding-pressure.sequence_embedding).abs().max())>1e-6):
                    raise ValueError('formal pressure did not reproduce the clean representation')
                evaluated=evaluate_frozen_application_consumers(method_name=method,output=pressure,targets=targets,
                    model_root=destination/'consumers',output_root=location/'predictions',fold_id=fold.fold_id,
                    evaluation_id=condition,seed=seed,evaluation_role='held_out')
                if condition=='clean_asynchronous':
                    _verify_clean_predictions(clean.model_manifest['prediction_path'],evaluated.prediction_path)
                result=dict(condition=condition,route=route,seed=seed,evidence_scope='frozen_confirmation',evaluation=asdict(evaluated),
                    grouped=summarize_pressure_predictions(evaluated,pressure,observed.sample_manifest_rows,evaluation_role='held_out'),
                    encoding_diagnostics=collect_encoding_diagnostics(encoder=encoder,normalizer=normalizer,batch=observed.batch))
                result_path=location/'result.json';result_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
                archive=location/'representation/held_out/fusion_stream.npz'
                record['conditions'][condition]=dict(result_path=str(result_path),files={str(p):sha256_file(p)
                    for p in (result_path,archive,Path(evaluated.prediction_path))});save()
            del encoder
        if method=='naive_time_sync':
            state['routes']['task_guided']=dict(shared_with='self_supervised',encoder_labels_used=False)
        state.update(status='completed',completed=True);save()
    return state
