"""Official recent-model representations through the shared real-development contract."""
from dataclasses import asdict, replace
from inspect import getfile
import json
from pathlib import Path
import shutil
import time
import traceback

import torch

from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract, run_common_downstream
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.recent_model_training import short_fit, encoder_checkpoint_state, load_encoder_checkpoint
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders import recent_models
from chronaris.modeling.fusion_encoders.recent_models import (
    Chronos2WindowEncoder, TimeCMAWindowEncoder, historical_channels, timecma_prompt_cache, verify_assets)
from chronaris.modeling.training.candidate_checkpoint import atomic_save_candidate
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file, write_deterministic_npz


def _export(encoder, inputs, prompts):
    encoder.eval().requires_grad_(False)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    outputs = {}
    with torch.inference_mode():
        for role, (values, mask) in inputs.items():
            outputs[role] = torch.cat([encoder(values[i:i+1], mask[i:i+1],
                None if prompts[role] is None else prompts[role][i:i+1]).detach().float().cpu()
                for i in range(len(values))])
            if not torch.isfinite(outputs[role]).all():
                raise ValueError('nonfinite recent-model representation')
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    return outputs, dict(seconds=seconds, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        windows_per_second=sum(len(x) for x in outputs.values())/seconds)


def _run(*, domain, method, output_root, assets_path, data_root, registry_path, progress):
    root = Path(output_root)/domain/method
    started = time.perf_counter()
    assets = verify_assets(assets_path)
    provider, schema, fold, digest, targets, definitions, context, raw = contract_development_inputs(
        domain, data_root=data_root, registry_path=registry_path)
    kwargs = dict(fold=fold, observations=raw, targets=targets, definitions=definitions,
        context=context, data_manifest_sha256=digest)
    contract = build_common_contract(domain=domain, **kwargs)
    binding = dict(contract=contract, assets_sha256=sha256_file(assets_path), method=method,
        training_budget=2, resume_replay_updates=1, seed=17, cuda_allocator_fraction=.95)
    binding_path = root/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('recent model input/source changed; preserve this root and use a new one')
    write_result(binding_path, binding)
    normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
        train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids)
    inputs = {r: historical_channels(normalizer.transform(b)) for r, b in raw.items()}
    names = tuple('physiology.'+s for s in schema.physiology_feature_names) + tuple('vehicle.'+s for s in schema.vehicle_feature_names)
    prompts = {r: None for r in raw}
    prompt_costs = {}
    if method == 'timecma':
        for role, (values, mask) in inputs.items():
            progress.update(phase='gpt2_cache', role=role)
            cache_args = dict(values=values, mask=mask, times=raw[role].query_timestamps_s,
                channel_names=names, assets=assets, output_root=root/'language_cache'/role)
            prompts[role], first = timecma_prompt_cache(**cache_args)
            second, warm = timecma_prompt_cache(**cache_args)
            if not torch.equal(prompts[role], second):
                raise ValueError('language cache replay differs')
            prompt_costs[role] = dict(first=first, warm=warm)
    progress.update(phase='load_official_encoder')
    load_started = time.perf_counter()
    if method == 'sensorllm_deepseek':
        from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
        factory = SensorLLMWindowEncoder
    else:
        factory = TimeCMAWindowEncoder if method == 'timecma' else Chronos2WindowEncoder
    encoder = factory(assets, len(names))
    load_seconds = time.perf_counter()-load_started
    parameters = sum(p.numel() for p in encoder.parameters())
    alignment = None
    if method == 'sensorllm_deepseek':
        progress.update(phase='sensor_language_alignment')
        alignment = encoder.align_history(values=inputs['train'][0], mask=inputs['train'][1], names=names,
            train_ids=fold.train_sample_ids, root=root/'alignment', binding=binding, progress=progress)
    routes = ('frozen', 'history_adapted') if method == 'chronos2' else ('summary_adapted',)
    results = []
    for route in routes:
        progress.update(phase='fit_and_export', route=route)
        unit = root/route
        unit.mkdir(parents=True, exist_ok=True)
        adapted = route != 'frozen'
        signal = 'none' if method == 'chronos2' else 'summary_labels'
        training_definitions = tuple(t for t in definitions if method!='sensorllm_deepseek' or t.kind=='classification')
        training_targets = replace(targets, values={t.name: targets.values[t.name] for t in training_definitions},
            valid_masks={t.name: targets.valid_masks[t.name] for t in training_definitions})
        tasks = [asdict(t) for t in training_definitions] if signal == 'summary_labels' else []
        metadata = dict(method_name=method, target_supervision=signal, label_used_for_encoder_training=signal!='none',
            task_definitions=tasks, role_sample_ids={'train': list(fold.train_sample_ids) if adapted else []},
            normalizer=normalizer.to_manifest(), config={'data_manifest_sha256': digest},
            contract_sha256=contract['contract_sha256'], assets_sha256=binding['assets_sha256'],
            adaptation='summary_label_heads_removed' if signal=='summary_labels' else
                ('native_quantile_loss_on_last_16_points_inside_train_history' if adapted else 'none'),
            channel_names=list(names), source_code_sha256=contract['source_code_sha256'], alignment=alignment)
        training = None
        if adapted:
            fit_kwargs = dict(values=inputs['train'][0], mask=inputs['train'][1], prompts=prompts['train'],
                targets=training_targets, definitions=training_definitions, train_ids=fold.train_sample_ids,
                metadata=metadata, progress=progress)
            anchor = unit/'resume_anchor.pt'
            if not (unit/'fit'/'training.pt').exists():
                first = short_fit(encoder, **fit_kwargs, root=unit/'fit', stop_after=1)
                shutil.copyfile(unit/'fit'/'training.pt', anchor)
                write_result(unit/'first_update.json', first)
            primary = short_fit(encoder, **fit_kwargs, root=unit/'fit')
            if not anchor.exists():
                raise ValueError('training resume anchor missing')
            replay_dir = unit/'resume_replay'
            replay_dir.mkdir(exist_ok=True)
            if not (replay_dir/'training.pt').exists():
                shutil.copyfile(anchor, replay_dir/'training.pt')
            replay = short_fit(encoder, **fit_kwargs, root=replay_dir)
            if primary['state_sha256'] != replay['state_sha256'] or primary['history'] != replay['history']:
                raise ValueError('optimizer/RNG resume changed actual adapter training')
            training = dict(first_update=json.loads((unit/'first_update.json').read_text()),
                primary=primary, replay=replay, resume_identical=True)
            if signal == 'summary_labels':
                metadata['task_supervision_counts'] = {t['name']: sum(h['task_counts'][t['name']] for h in primary['history']) for t in tasks}
                if any(count <= 0 for count in metadata['task_supervision_counts'].values()):
                    raise ValueError('a declared encoder task received no actual supervision')
        checkpoint = unit/'encoder.pt'
        if not checkpoint.exists():
            atomic_save_candidate(checkpoint, metadata | {'encoder': encoder_checkpoint_state(encoder)})
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if any(saved[k] != v for k, v in metadata.items()):
            raise ValueError('frozen encoder provenance differs')
        load_encoder_checkpoint(encoder, saved['encoder'])
        features, export_cost = _export(encoder, inputs, prompts)
        # Reload the persisted encoder and recompute genuine representations, not cached predictions.
        load_encoder_checkpoint(encoder, saved['encoder'])
        repeated, replay_cost = _export(encoder, inputs, prompts)
        if any(not torch.equal(features[r], repeated[r]) for r in raw):
            raise ValueError('frozen model reloading changed representations')
        spread = features['train'].std(0, unbiased=False)
        if not (spread > 1e-8).any():
            raise ValueError('recent-model features collapsed to a constant')
        checkpoint_sha = sha256_file(checkpoint)
        outputs = {r: WindowFeatureBatch(b.sample_ids, b.context_durations_s, features[r],
            inputs[r][1].any((1, 2)), method, fold.fold_id, checkpoint_sha, b.source_sample_hashes)
            for r, b in raw.items()}
        evidence = {str(Path(recent_models.__file__).resolve()): sha256_file(recent_models.__file__),
            str(Path(getfile(type(encoder))).resolve()): sha256_file(getfile(type(encoder))),
            str(Path(__file__).resolve()): sha256_file(__file__), str(Path(assets_path).resolve()): sha256_file(assets_path),
            str(binding_path.resolve()): sha256_file(binding_path)}
        external = dict(assets[{'timecma':'gpt2', 'chronos2':'chronos2', 'sensorllm_deepseek':'deepseek_llama'}[method]])
        external['training_signal'] = 'external_time_series_forecasting' if method=='chronos2' else 'language_pretraining'
        external['official_code_revision'] = assets[{'timecma':'timecma', 'chronos2':'chronos_code', 'sensorllm_deepseek':'sensorllm'}[method]]['revision']
        if method=='sensorllm_deepseek':
            external['time_series_backbone'] = assets['chronos_t5_large']
            external['variant'] = 'DeepSeek-R1-Distill-Llama-8B; frozen mean boundary embeddings; classification adaptation'
        declaration = dict(method=method, checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
            route='task_guided' if signal!='none' else 'self_supervised', target_supervision=signal,
            training_tasks=[t['name'] for t in tasks], external_pretraining=external,
            kind='window_end', feature_origin='hidden_state', feature_dim=encoder.feature_dim,
            extraction_location={'timecma':'official Dual.decoder output before c_to_length; channel flatten',
                'chronos2':'Chronos2Pipeline.embed -> encoder REG token per channel; channel flatten; no output patches',
                'sensorllm_deepseek':'official SensorLLMStage2LlamaModel.last_hidden_state final nonpadding token; classifier removed'}[method],
            encoder_frozen=True, task_heads_removed=True, encoder_fit_sample_ids=metadata['role_sample_ids']['train'],
            preprocessing_fit_sample_ids=list(normalizer.fit_sample_ids), evidence_files=evidence)
        write_result(unit/'declaration.json', declaration)
        for role, batch in outputs.items():
            path = unit/f'{role}_features.npz'
            arrays = dict(sample_ids=batch.sample_ids, pooled_embedding=batch.pooled_embedding.numpy(),
                valid_mask=batch.valid_mask.numpy(), timestamps_s=batch.timestamps_s.numpy(), source_sample_hashes=batch.source_sample_hashes)
            if path.exists():
                import numpy as np
                with np.load(path) as old:
                    if any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                        raise ValueError('existing frozen export differs')
            else:
                write_deterministic_npz(path, arrays)
        progress.update(phase='common_downstream', route=route)
        first = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration, output_root=unit/'evaluation')
        resumed = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration, output_root=unit/'evaluation')
        if first != resumed:
            raise ValueError('common downstream resume changed predictions')
        result = dict(route=route, training=training, evaluation=first, export_cost=export_cost, replay_export_cost=replay_cost,
            parameters=parameters, feature_dim=encoder.feature_dim, nonconstant_dimensions=int((spread>1e-8).sum()),
            representation_reload_identical=True, downstream_resume_identical=True,
            feature_files={r: {'path': str(unit/f'{r}_features.npz'), 'sha256': sha256_file(unit/f'{r}_features.npz')} for r in raw})
        result_path = unit/'summary.json'
        if not result_path.exists():
            write_result(result_path, result)
        results.append(json.loads(result_path.read_text()))
    if v4_workflow_source_sha256() != contract['source_code_sha256']:
        raise ValueError('source changed during recent model execution')
    result = dict(status='completed', domain=domain, method=method, scope='engineering_development',
        source_code_sha256=contract['source_code_sha256'], contract_sha256=contract['contract_sha256'],
        sample_counts={r: len(b.sample_ids) for r,b in raw.items()}, channel_names=list(names),
        load_seconds=load_seconds, prompt_costs=prompt_costs, results=results,
        total_seconds=time.perf_counter()-started, confirmation_opened=False)
    if not (root/'summary.json').exists():
        write_result(root/'summary.json', result)
    return json.loads((root/'summary.json').read_text())


def run_recent_model_smoke(*, domain, method, output_root, assets_path,
                           data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                           registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    if domain not in {'dingxin','cogpilot','clare'} or method not in {'timecma','chronos2','sensorllm_deepseek'} or not torch.cuda.is_available():
        raise ValueError('recent model smoke requires an approved real development domain, method and CUDA')
    root = Path(output_root)/domain/method
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        previous_memory_fraction = torch.cuda.get_per_process_memory_fraction()
        with _periodic_training_heartbeat('recent_model_'+method, 30., root=root) as progress, isolated_training_rng(17):
            try:
                torch.cuda.set_per_process_memory_fraction(.95)
                return _run(domain=domain, method=method, output_root=output_root, assets_path=assets_path,
                    data_root=data_root, registry_path=registry_path, progress=progress)
            except BaseException as error:
                write_result(root/f'failure_{time.time_ns()}.json', dict(status='failed', error=repr(error),
                    traceback=traceback.format_exc(), progress=dict(progress)))
                raise
            finally:
                torch.cuda.set_per_process_memory_fraction(previous_memory_fraction)


def revalidate_recent_exports(*, source_root, evidence_path, source_inventory, output_root):
    """Refit common consumers from immutable, source-attributed neural exports."""
    import numpy as np
    evidence_record = json.loads(Path(evidence_path).read_text())
    evidence = evidence_record['files']
    if any(sha256_file(p) != digest for p, digest in evidence.items()):
        raise ValueError('prior recent-model evidence changed')
    snapshots = json.loads(Path(source_inventory).read_text())['files']
    for item in snapshots.values():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('prior extraction source snapshot changed')
    results = []
    for domain in ('dingxin', 'cogpilot', 'clare'):
        _, _, fold, digest, targets, definitions, context, observations = contract_development_inputs(domain,
            data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
            registry_path='docs/requirements/thesis-v4-public-subjects.json')
        kwargs = dict(fold=fold, observations=observations, targets=targets, definitions=definitions,
            context=context, data_manifest_sha256=digest)
        contract = build_common_contract(domain=domain, **kwargs)
        summary_paths = [Path(p) for p in evidence_record.get('summary_paths',
            sorted((Path(source_root)/domain).glob('*/summary.json')))]
        for summary_path in summary_paths:
            if str(summary_path) not in evidence:
                raise ValueError('unregistered source summary')
            summary = json.loads(summary_path.read_text())
            if summary['domain'] != domain:
                continue
            for unit_summary in summary['results']:
                unit = summary_path.parent/unit_summary['route']
                original = unit/'evaluation'/'common_contract.json'
                if sha256_file(original) != unit_summary['evaluation']['contract_file_sha256']:
                    raise ValueError('prior shared contract changed')
                record = json.loads(original.read_text())
                ignored = {'source_code_sha256','contract_sha256'}
                if {k:v for k,v in record['contract'].items() if k not in ignored} != {k:v for k,v in contract.items() if k not in ignored}:
                    raise ValueError('revalidation cannot change data, tasks, history, roles or policy')
                declaration = dict(record['declaration'])
                if declaration['target_supervision'] == 'summary_labels':
                    visited = [s for h in unit_summary['training']['primary']['history'] for s in h['sample_ids']]
                    if not set(visited) <= set(fold.train_sample_ids):
                        raise ValueError('actual encoder training escaped the train role')
                    positions = [targets.sample_ids.index(s) for s in visited]
                    if any(not targets.valid_masks[t][positions].any() for t in declaration['training_tasks']):
                        raise ValueError('original task declaration exceeds actual training-label coverage')
                files = {}
                for path, digest in declaration['evidence_files'].items():
                    replacement = snapshots.get(path+'@'+digest, snapshots.get(path))
                    if replacement is None or replacement['sha256'] != digest:
                        replacement = {'path': path, 'sha256': digest}
                    if replacement['sha256'] != digest or sha256_file(replacement['path']) != digest:
                        raise ValueError('original extraction source cannot be substantiated')
                    files[replacement['path']] = digest
                files.update({str(Path(__file__).resolve()): sha256_file(__file__),
                    str(Path(evidence_path).resolve()): sha256_file(evidence_path),
                    str(original): sha256_file(original)})
                declaration['evidence_files'] = files
                outputs = {}
                for role, feature_file in unit_summary['feature_files'].items():
                    if sha256_file(feature_file['path']) != feature_file['sha256']:
                        raise ValueError('prior feature array changed')
                    with np.load(feature_file['path']) as data:
                        outputs[role] = WindowFeatureBatch(tuple(data['sample_ids'].tolist()), torch.from_numpy(data['timestamps_s']),
                            torch.from_numpy(data['pooled_embedding']), torch.from_numpy(data['valid_mask']),
                            declaration['method'], fold.fold_id, declaration['checkpoint_sha256'], tuple(data['source_sample_hashes'].tolist()))
                destination = Path(output_root)/domain/declaration['method']/unit_summary['route']
                first = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration, output_root=destination)
                second = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration, output_root=destination)
                old_path = unit_summary['evaluation']['consumers']['components']['linear']['result_path']
                new_path = first['consumers']['components']['linear']['result_path']
                old, new = (json.loads(Path(p).read_text()) for p in (old_path, new_path))
                if first != second or old['evaluations'] != new['evaluations'] or old['fit_rows'] != new['fit_rows']:
                    raise ValueError('revalidated predictions, metrics, selected fits or resume differ')
                results.append(dict(domain=domain, route=unit_summary['route'], method=declaration['method'],
                    original_summary_path=str(unit/'summary.json'), evaluation=first, predictions_identical=True,
                    extra_encoder_updates=0, resume_identical=True))
    if any(sha256_file(p) != digest for p, digest in evidence.items()):
        raise ValueError('prior evidence changed during revalidation')
    return write_result(Path(output_root)/'revalidation.json', dict(status='completed', results=results,
        source_code_sha256=v4_workflow_source_sha256(), prior_files_unchanged=len(evidence), confirmation_opened=False))
