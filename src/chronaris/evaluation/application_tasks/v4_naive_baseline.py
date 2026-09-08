"""Train-role-only fixed synchronization baseline with zero neural updates."""
from pathlib import Path
import json
import time

from chronaris.modeling.fusion_encoders import (NaiveTimeSyncEncoder, NaiveTimeSyncConfig,
    load_naive_time_sync_checkpoint, save_naive_time_sync_checkpoint)
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_v4_naive_encoder(checkpoint, *, fold, data_manifest_sha256):
    checkpoint = Path(checkpoint)
    payload = json.loads((checkpoint.parent / 'fit_manifest.json').read_text())
    if (payload['format'] != 'chronaris.v4_naive_fit.v1' or payload['training_status'] != 'completed'
        or payload['checkpoint_sha256'] != sha256_file(checkpoint) or payload['fold'] != fold.to_dict()
        or payload['config']['data_manifest_sha256'] != data_manifest_sha256
        or payload['source_code_sha256'] != v4_workflow_source_sha256()
        or payload['label_used_for_encoder_training'] is not False or payload['optimizer_updates'] != 0):
        raise ValueError('fixed synchronization checkpoint source/data/roles changed')
    encoder = load_naive_time_sync_checkpoint(checkpoint)
    expected = tuple(sorted(fold.train_sample_ids))
    if (encoder.normalizer.fit_sample_ids != expected or encoder.projector.fit_sample_ids != expected
        or encoder.config.validity_policy != 'query_history' or encoder.config.random_state != payload['seed']
        or encoder.projector.random_state != payload['seed']):
        raise ValueError('fixed synchronization fit roles or validity semantics changed')
    return encoder, encoder.normalizer, payload


def fit_v4_naive_encoder(*, provider, fold, normalizer, data_manifest_sha256, output_root, seed=17):
    if seed not in (17, 29, 43):
        raise ValueError('v4 fixed synchronization requires an approved seed')
    root = Path(output_root)
    checkpoint, manifest_path = root / 'best.pt', root / 'fit_manifest.json'
    if manifest_path.exists():
        encoder, _, payload = load_v4_naive_encoder(checkpoint, fold=fold, data_manifest_sha256=data_manifest_sha256)
        if payload['seed'] != seed or encoder.normalizer.to_manifest() != normalizer.to_manifest():
            raise ValueError('fixed synchronization seed or shared normalization changed')
        return checkpoint, payload
    if checkpoint.exists():
        raise ValueError('uncommitted fixed synchronization checkpoint; retain it and use a new output root')
    allowed = set(fold.train_sample_ids)
    def train_provider(ids):
        if not set(ids) <= allowed:
            raise ValueError('fixed synchronization fit attempted to read a non-training window')
        return provider(ids)
    started = time.perf_counter()
    encoder = NaiveTimeSyncEncoder(NaiveTimeSyncConfig(random_state=seed)).fit_from_batch_provider(train_provider,
        train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids,
        normalizer=normalizer, batch_size=4)
    elapsed = time.perf_counter()-started
    save_naive_time_sync_checkpoint(checkpoint, encoder=encoder)
    payload = dict(format='chronaris.v4_naive_fit.v1',training_status='completed',method_name='naive_time_sync',seed=seed,
        checkpoint_sha256=sha256_file(checkpoint),fold=fold.to_dict(),source_code_sha256=v4_workflow_source_sha256(),
        config=dict(data_manifest_sha256=data_manifest_sha256,device='cpu'),normalizer=normalizer.to_manifest(),
        label_used_for_encoder_training=False,encoder_checkpoint_selection_uses_validation_labels=False,
        optimizer_updates=0,stage_update_counts=dict(pretraining=0,head_warmup=0,joint_adaptation=0),
        training_elapsed_s=elapsed,fit_algorithm='train_only_randomized_pca',representation_identical_between_routes=True)
    temporary=manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(manifest_path)
    return checkpoint,payload


def run_native_naive_development(*, domain, fold_index=0, seed=17, output_root,
                                 data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                                 registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, development_normalization
    from chronaris.evaluation.application_tasks.application_finetuning_export import export_loaded_application_encoder
    from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context, run_native_method_consumers
    from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
    if domain not in {'cogpilot','clare','dingxin'}:
        raise ValueError('native fixed baseline development requires CogPilot, CLARE or Dingxin')
    root=Path(output_root)/domain/f'fold{fold_index+1:02d}'/f'seed{seed}'
    with _periodic_training_heartbeat('native_naive_development',30.,root=root) as progress:
        progress['phase']='load_native_development'
        provider,schema,fold,_,digest,targets,definitions,data=load_development_inputs(domain,data_root,registry_path,fold_index=fold_index)
        normalizer,_=development_normalization(domain,provider,schema,fold,digest)
        progress['phase']='train_only_pca_fit'
        checkpoint,fit=fit_v4_naive_encoder(provider=provider,fold=fold,normalizer=normalizer,
            data_manifest_sha256=digest,output_root=root/'checkpoint',seed=seed)
        encoder,_,_=load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256=digest)
        progress.update(phase='development_export',checkpoint=str(checkpoint),optimizer_updates=0)
        outputs=export_loaded_application_encoder(encoder=encoder,normalizer=normalizer,checkpoint=checkpoint,
            provider=provider,fold=fold,root=root/'representations',export_roles=('train','validation'),
            export_prefix='fixed_baseline_development')
        progress['phase']='cpu_consumers'
        consumers=run_native_method_consumers(outputs=outputs,targets=targets,definitions=definitions,
            context=native_consumer_context(domain,data,fold),output_root=root/'consumers',
            label_used_for_encoder_training=False,seed=seed)
        result=dict(completed=True,domain=domain,fold=fold.to_dict(),seed=seed,training=fit,consumers=consumers,
            scope='fixed_baseline_development_not_confirmation',confirmation_opened=False,
            routes=dict(self_supervised=str(root/'consumers'),task_guided=str(root/'consumers')),
            representation_identical_between_routes=True,neural_optimizer_updates=0)
        temporary=root/'run_state.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(root/'run_state.json')
        return result
