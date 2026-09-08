"""Verify archived native consumer results against saved models and authoritative targets."""
from dataclasses import asdict
from pathlib import Path
import json

import joblib

from chronaris.evaluation.application_tasks.v4_grouped_consumers import (
    evaluate_native_consumers, native_consumer_protocol_sha256, _validate_inputs)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def audit_native_consumer_result(result, *, outputs, targets, definitions, context, label_used_for_encoder_training):
    _validate_inputs(outputs,targets,definitions,context)
    path=Path(result['manifest_path'])
    manifest=json.loads(path.read_text())
    declared_hash=manifest.pop('protocol_sha256')
    artifact_hashes=manifest.pop('artifacts',{})
    expected_roles={role:dict(sample_ids=list(output.sample_ids),source_sample_hashes=list(output.source_sample_hashes),
        method_name=output.method_name,fold_id=output.fold_id,checkpoint_sha256=output.checkpoint_sha256) for role,output in outputs.items()}
    if (manifest['roles']!=expected_roles or manifest['context']!=context or manifest['target_manifest']!=targets.manifest
        or manifest['task_definitions']!=[asdict(task) for task in definitions]
        or manifest['label_used_for_encoder_training'] is not label_used_for_encoder_training
        or native_consumer_protocol_sha256(manifest,outputs,targets)!=declared_hash or result['protocol_sha256']!=declared_hash):
        raise ValueError('native consumer data, labels or representation provenance changed')
    expected_families={'linear'} if context['domain']=='dingxin' else {'linear','minirocket'}
    if set(result['components'])!=expected_families:
        raise ValueError('native consumer family inventory changed')
    files={str(path):sha256_file(path)}
    metrics=[]
    for family,component in result['components'].items():
        for kind in ('model','result'):
            item=component[kind+'_path'];digest=sha256_file(item)
            expected=component.get(kind+'_sha256',artifact_hashes.get(family,{}).get(kind+'_sha256'))
            if expected is not None and digest!=expected:
                raise ValueError('native consumer saved file hash changed')
            files[item]=digest
        saved=json.loads(Path(component['result_path']).read_text())
        model=joblib.load(component['model_path'])
        if (saved['protocol_sha256']!=declared_hash or model['protocol_sha256']!=declared_hash
            or saved['fit_rows']!=model['consumer']['fit_rows'] or saved['fit_elapsed_s']!=model['fit_elapsed_s']):
            raise ValueError('native consumer model and score provenance differ')
        evaluated={role:evaluate_native_consumers(model['consumer'],output=output,targets=targets,context=context)
                   for role,output in outputs.items() if role!='train'}
        if evaluated!=saved['evaluations'] or component['task_summary']!={role:values['task_summary'] for role,values in evaluated.items()}:
            raise ValueError('native consumer archived results differ from frozen model replay')
        for role,values in evaluated.items():
            metrics.extend(dict(domain=context['domain'],role=role,consumer=family,method=outputs['train'].method_name,
                seed=manifest['seed'],fold=outputs['train'].fold_id,**row) for row in values['group_metrics'])
    return dict(protocol_sha256=declared_hash,source_code_sha256=manifest['source_code_sha256'],files=files,
                metrics=metrics,model_replay_exact=True,consumers_refitted=False)


def collect_fixed_native_development(*, run_root, output_root,
                                    data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                                    registry_path='docs/requirements/thesis-v4-public-subjects.json', domain_run_roots=None):
    from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
    from chronaris.modeling.fusion_encoders import load_naive_time_sync_checkpoint
    from chronaris.representation import load_fusion_stream_batch
    run_root,output_root=Path(run_root),Path(output_root)
    output_root.mkdir(parents=True,exist_ok=True)
    units,metrics,pending,failed=[],[],[],[]
    domain_run_roots=domain_run_roots or {}
    if set(domain_run_roots)-{'cogpilot','clare','dingxin'}:raise ValueError('unknown native domain source override')
    for domain in ('cogpilot','clare','dingxin'):
        domain_root=Path(domain_run_roots.get(domain,run_root))
        queue=json.loads((domain_root/f'{domain}_queue.json').read_text())
        for fold_index in range(2 if domain=='dingxin' else 3):
            inputs=None
            for seed in (17,29,43):
                key=f'fold{fold_index+1:02d}/seed{seed}'
                if key not in queue['completed_units']:
                    (failed if key in queue['failed_units'] else pending).append(f'{domain}/{key}')
                    continue
                if inputs is None:
                    inputs=load_development_inputs(domain,data_root,registry_path,fold_index=fold_index)
                _,_,fold,_,digest,targets,definitions,data=inputs
                root=domain_root/domain/key
                state=json.loads((root/'run_state.json').read_text())
                fit=json.loads((root/'checkpoint/fit_manifest.json').read_text())
                checkpoint=root/'checkpoint/best.pt'
                if (not state['completed'] or state['confirmation_opened'] or state['neural_optimizer_updates']!=0
                    or state['seed']!=seed or state['fold']!=fold.to_dict() or state['training']!=fit
                    or fit['source_code_sha256']!=queue['source_code_sha256'] or fit['config']['data_manifest_sha256']!=digest
                    or fit['checkpoint_sha256']!=sha256_file(checkpoint) or fit['label_used_for_encoder_training'] is not False):
                    raise ValueError('fixed baseline unit provenance differs from completed development queue')
                encoder=load_naive_time_sync_checkpoint(checkpoint)
                expected=tuple(sorted(fold.train_sample_ids))
                if (encoder.normalizer.fit_sample_ids!=expected or encoder.projector.fit_sample_ids!=expected
                    or encoder.config.random_state!=seed or encoder.config.validity_policy!='query_history'):
                    raise ValueError('fixed baseline checkpoint fit roles, seed or masks changed')
                outputs={role:load_fusion_stream_batch(root/'representations'/role) for role in ('train','validation')}
                audit=audit_native_consumer_result(state['consumers'],outputs=outputs,targets=targets,definitions=definitions,
                    context=native_consumer_context(domain,data,fold),label_used_for_encoder_training=False)
                if audit['source_code_sha256']!=queue['source_code_sha256']:
                    raise ValueError('fixed baseline consumer source differs from encoder fit source')
                files=audit['files'] | {str(path):sha256_file(path) for path in
                    (root/'run_state.json',root/'checkpoint/fit_manifest.json',checkpoint)}
                for role in outputs:
                    for name in ('fusion_stream.npz','representation_manifest.json'):
                        path=root/'representations'/role/name;files[str(path)]=sha256_file(path)
                units.append(dict(domain=domain,fold=fold.fold_id,seed=seed,files=files,projection_fit_s=fit['training_elapsed_s'],
                    consumer_fit_s={family:item['fit_elapsed_s'] for family,item in state['consumers']['components'].items()},
                    shared_between_routes=True,neural_optimizer_updates=0,model_replay_exact=True))
                metrics.extend(audit['metrics'])
    result=dict(scope='fixed_baseline_development_not_confirmation',completed_units=len(units),expected_units=24,
        pending=pending,failed=failed,units=units,group_metrics=metrics,neural_optimizer_updates=0,
        shared_routes=['self_supervised','task_guided'],complete=not pending and not failed,
        projection_fit_s=sum(row['projection_fit_s'] for row in units),
        consumer_fit_s=sum(sum(row['consumer_fit_s'].values()) for row in units))
    path=output_root/'fixed_native_development.json'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    return result
