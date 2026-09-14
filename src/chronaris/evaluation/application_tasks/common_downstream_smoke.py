"""Bounded real-development closure using existing encoders, exports and consumers."""
from dataclasses import asdict, replace
import json
import math
import time
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method)
from chronaris.evaluation.application_tasks.application_finetuning_export import (
    export_loaded_application_encoder, load_frozen_application_encoder)
from chronaris.evaluation.application_tasks.application_task_heads import select_application_targets
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract, run_common_downstream
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, _hash_prefix, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development
from chronaris.evaluation.application_tasks.v4_dingxin_deduplicated import deduplicated_dingxin_data
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
from chronaris.evaluation.application_tasks.v4_naive_baseline import fit_v4_naive_encoder, load_v4_naive_encoder
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.training import CandidateScreenConfig, EncoderCandidateConfig, train_pretext_candidate
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import TrainOnlyRobustNormalizer, select_observation_batch
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file, write_deterministic_npz


def contract_development_inputs(domain, *, data_root, registry_path, full=False, fold_index=0):
    if domain == 'dingxin':
        data, _ = deduplicated_dingxin_data(load_v4_dingxin_development())
        if fold_index != 0:
            raise ValueError('Dingxin keeps its single deduplicated development fold')
        fold = data.folds[0]
        targets, definitions = data.targets_by_fold[fold.fold_id], data.definitions_by_fold[fold.fold_id]
        provider, schema, digest = data.development_provider(fold), data.index.plan.schema, data.data_manifest_sha256
    else:
        provider, schema, fold, _, digest, targets, definitions, data = load_development_inputs(domain, data_root, registry_path, fold_index=fold_index)
        if not full:
            # Fix task availability, not label values or model outcomes, before selecting methods.
            index = {sample: i for i, sample in enumerate(targets.sample_ids)}
            roles = {}
            for role, count in (('train', 32), ('validation', 16)):
                allowed = getattr(fold, role+'_sample_ids')
                selected = set()
                for task in definitions:
                    ids = [s for s in allowed if bool(targets.valid_masks[task.name][index[s]].any())]
                    selected.update(_hash_prefix(ids, count))
                roles[role+'_sample_ids'] = tuple(s for s in allowed if s in selected)
            fold = replace(fold, fold_id=fold.fold_id+'__common_contract_smoke', **roles)
            ids = fold.train_sample_ids + fold.validation_sample_ids
            targets = replace(targets, sample_ids=ids, **select_application_targets(targets, ids, 'cpu'))
    context = native_consumer_context(domain, data, fold)
    observations = {role: provider(getattr(fold, role+'_sample_ids')) for role in ('train', 'validation')}
    def selected_provider(ids):
        selected = tuple(ids)
        for raw in observations.values():
            if set(selected) <= set(raw.sample_ids):
                return select_observation_batch(raw, selected)
        raise ValueError('contract provider rejects mixed, held-out or embargo roles')
    return selected_provider, schema, fold, digest, targets, definitions, context, observations


def run_common_contract_smoke(*, domain, output_root,
                            data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                            registry_path='docs/requirements/thesis-v4-public-subjects.json',
                            full=False, methods=('naive_time_sync', 'chronaris'), cuda_graph_recurrence=False,
                            recipe=None, seed=17, fold_index=0, micro_batch=4, diagnostic_snapshots=False, pretraining_source=None, expected_contract_sha256=None,
                            finetuning_graph_recurrence=None):
    if domain not in {'dingxin', 'cogpilot', 'clare'} or not torch.cuda.is_available():
        raise ValueError('contract smoke requires a real native development domain and CUDA')
    if not methods or not set(methods) <= {'naive_time_sync', 'chronaris', 'physiology_only', 'vehicle_only', 'mult', 'contiformer'}:
        raise ValueError('unsupported common comparison method')
    if cuda_graph_recurrence and (methods != ('chronaris',) or (recipe is None and domain != 'cogpilot')):
        raise ValueError('graph execution is validated only for CogPilot Chronaris')
    if recipe is None and (seed != 17 or fold_index != 0 or micro_batch != 4 or diagnostic_snapshots):
        raise ValueError('stage 4 changes require an explicit stage 4.5 recipe')
    if pretraining_source is not None and (recipe != 'finetuning_lr' or not full or seed != 17 or fold_index != 0):
        raise ValueError('pretraining reuse is limited to the fixed first-setting fine-tuning candidate')
    scope = 'development_comparison' if full else 'engineering_development'
    root = Path(output_root)/domain
    started = time.perf_counter()
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        with _periodic_training_heartbeat('common_contract_'+domain, 30., root=root) as progress:
            progress['phase'] = 'bind_real_development_inputs'
            provider, schema, fold, digest, targets, definitions, context, observations = contract_development_inputs(
                domain, data_root=data_root, registry_path=registry_path, full=full, fold_index=fold_index)
            contract = build_common_contract(domain=domain, fold=fold, observations=observations,
                targets=targets, definitions=definitions, context=context, data_manifest_sha256=digest, scope=scope, seed=seed)
            if expected_contract_sha256 is not None and contract['contract_sha256'] != expected_contract_sha256:
                raise ValueError('unit data or targets differ from the frozen development plan')
            contract_path = root/'data_contract.json'
            if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
                raise ValueError('contract smoke inputs or source changed; preserve this root')
            write_result(contract_path, contract)
            normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
                train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids,
                batch_size=4)
            recipe_configs = None
            if recipe is not None:
                if len(methods) != 1 or methods[0] == 'naive_time_sync':
                    raise ValueError('recipe requires exactly one trainable method')
                from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
                from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
                calibration = (fit_physics_calibration(normalizer, provider, train_sample_ids=fold.train_sample_ids,
                    vehicle_feature_names=schema.vehicle_feature_names, relations=()) if recipe == 'thesis_reference' else None)
                recipe_configs = training_recipe(recipe, method=methods[0], full=full, seed=seed, digest=digest,
                    train_count=len(fold.train_sample_ids), graph=cuda_graph_recurrence,
                    calibration=calibration, micro_batch=micro_batch)
                record = recipe_configs[-1] | {'fold_index': fold_index, 'diagnostic_snapshots': diagnostic_snapshots,
                    'source_code_sha256': contract['source_code_sha256'], 'contract_sha256': contract['contract_sha256'],
                    'pretraining_source': str(pretraining_source) if pretraining_source else None,
                    'finetuning_graph_recurrence': finetuning_graph_recurrence,
                    'pretraining_source_sha256': sha256_file(pretraining_source) if pretraining_source else None}
                path = root/'recipe.json'
                if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(record)):
                    raise ValueError('stage 4.5 recipe changed; use a new root')
                write_result(path, record)
            results = []
            for method in methods:
                progress.update(phase='encoder_fit', method=method)
                effective_batch = 16 if full else 4
                pretraining_updates = max(300, math.ceil(len(fold.train_sample_ids)/effective_batch)) if full else 2
                joint_updates = max(200, math.ceil(len(fold.train_sample_ids)/effective_batch)) if full else 2
                torch.cuda.reset_peak_memory_stats()
                if method == 'naive_time_sync':
                    checkpoint, training = fit_v4_naive_encoder(provider=provider, fold=fold, normalizer=normalizer,
                        data_manifest_sha256=digest, output_root=root/method/'checkpoint')
                    encoder, _, _ = load_v4_naive_encoder(checkpoint, fold=fold, data_manifest_sha256=digest)
                    routes = [('self_supervised', checkpoint, encoder, training)]
                elif pretraining_source is not None:
                    encoder, source_normalizer, payload = load_frozen_application_encoder(pretraining_source,
                        route='self_supervised', fold=fold, device='cuda')
                    from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
                    if (payload['source_code_sha256'] != candidate_source_code_sha256()
                        or payload['candidate_config'] != asdict(recipe_configs[0])
                        or source_normalizer.to_manifest() != normalizer.to_manifest()
                        or payload['config']['cuda_graph_recurrence'] != cuda_graph_recurrence
                        or payload['chronaris_mechanism_enabled'] or payload['chronaris_explicit_shift_enabled']):
                        raise ValueError('preserved pretraining is not the unchanged stage 4 reference')
                    checkpoint = Path(pretraining_source)
                    training = dict(best_checkpoint_path=str(checkpoint), last_checkpoint_path=str(checkpoint),
                        optimizer_updates=payload['optimizer_updates'], reused=True, new_optimizer_updates=0,
                        training_elapsed_s=0., historical_training_elapsed_s=payload['training_elapsed_s'])
                    routes = [('self_supervised', checkpoint, encoder, training)]
                else:
                    training = train_pretext_candidate(method, candidate=recipe_configs[0] if recipe_configs else EncoderCandidateConfig(candidate_id='C', hidden_dim=32),
                        batch=None, batch_provider=provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
                        vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer,
                        output_root=root/method/'self_supervised', config=recipe_configs[1] if recipe_configs else CandidateScreenConfig(max_updates=pretraining_updates,
                            batch_size=4, effective_batch_size=effective_batch, device='cuda', validation_interval=50 if full else 2, early_stopping=False,
                            data_manifest_sha256=digest, cuda_graph_recurrence=cuda_graph_recurrence), **(recipe_configs[3] if recipe_configs else
                            {'chronaris_fusion_kind': 'safe_lag' if method == 'chronaris' else 'multiscale'}))
                    encoder, _, _ = load_frozen_application_encoder(training.best_checkpoint_path,
                        route='self_supervised', fold=fold, device='cuda')
                    routes = [('self_supervised', Path(training.best_checkpoint_path), encoder, asdict(training))]
                for route, checkpoint, encoder, training in routes:
                    progress.update(phase='export_and_common_consumers', route=route)
                    task_counts = {}
                    if full and route == 'task_guided':
                        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                        task_counts = {t.name: sum(r['task_valid_counts'][t.name] for r in payload['update_rows']
                            if r['stage']=='joint_adaptation') for t in definitions}
                        if any(v <= 0 for v in task_counts.values()):
                            raise ValueError('a declared baseline task received no encoder supervision')
                    export_started = time.perf_counter()
                    outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
                        provider=provider, fold=fold, root=root/method/route/'representations',
                        export_roles=('train', 'validation'), export_prefix='common_contract_smoke',
                        label_used_for_encoder_training=route == 'task_guided')
                    torch.cuda.synchronize()
                    export_seconds = time.perf_counter()-export_started
                    spread = outputs['train'].pooled_embedding.std(0, unbiased=False)
                    if full and not (spread > 1e-8).any():
                        raise ValueError('full development representation collapsed')
                    declaration = dict(method=method, checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
                        route=route, target_supervision='summary_labels' if route == 'task_guided' else 'none',
                        training_tasks=[t.name for t in definitions] if route == 'task_guided' else [], external_pretraining={'source': 'none'},
                        kind='causal_sequence', feature_origin='unsupervised_projection' if method == 'naive_time_sync' else 'hidden_state',
                        extraction_location='NaiveTimeSyncFusionAdapter.__call__' if method == 'naive_time_sync' else 'TrainedFusionAdapter.__call__',
                        feature_dim=64, encoder_frozen=True, task_heads_removed=True,
                        encoder_fit_sample_ids=list(fold.train_sample_ids), preprocessing_fit_sample_ids=list(normalizer.fit_sample_ids),
                        evidence_files={str(Path(__file__).resolve()): sha256_file(__file__),
                            str(checkpoint.resolve()): sha256_file(checkpoint)})
                    kwargs = dict(contract=contract, targets=targets, definitions=definitions, context=context,
                        observations=observations, fold=fold, data_manifest_sha256=digest)
                    run_root = root/method/route/'evaluation'
                    families = ('linear',) if domain == 'dingxin' else ('linear', 'minirocket')
                    first = run_common_downstream(**kwargs, outputs=outputs, declaration=declaration, output_root=run_root, families=families)
                    resumed = run_common_downstream(**kwargs, outputs=outputs, declaration=declaration, output_root=run_root, families=families)
                    if first != resumed:
                        raise ValueError('common consumer resume changed predictions or provenance')
                    results.append(first | {'training': training, 'resume_identical': True,
                        'route': route, 'export_seconds': export_seconds, 'task_supervision_counts': task_counts,
                        'nonconstant_dimensions': int((spread > 1e-8).sum()),
                        'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
                    if method != 'naive_time_sync' and route == 'self_supervised':
                        if not full:
                            window = {r: WindowFeatureBatch(o.sample_ids, observations[r].context_durations_s,
                                o.pooled_embedding, o.valid_mask.any(1), o.method_name, o.fold_id, o.checkpoint_sha256,
                                o.source_sample_hashes) for r, o in outputs.items()}
                            for role, batch in window.items():
                                write_deterministic_npz(root/method/route/'window_features'/f'{role}.npz',
                                    dict(sample_ids=batch.sample_ids, pooled_embedding=batch.pooled_embedding.numpy(),
                                        valid_mask=batch.valid_mask.numpy(), timestamps_s=batch.timestamps_s.numpy(),
                                        source_sample_hashes=batch.source_sample_hashes))
                            window_result = run_common_downstream(**kwargs, outputs=window, declaration=declaration | {
                                'kind': 'window_end', 'extraction_location': 'pool_exported_sequence'},
                                output_root=root/method/route/'window_evaluation')
                            results.append(window_result | {'derived_from_sequence_export': True})
                        # Extend the existing route list only after freezing/exporting the unsupervised encoder.
                        progress.update(phase='task_guided_fit', route='task_guided')
                        if finetuning_graph_recurrence is not None:
                            if method != 'chronaris' or finetuning_graph_recurrence is not False:
                                raise ValueError('only explicit ordinary fine-tuning fallback is supported')
                            from chronaris.evaluation.application_tasks.stage45_resume import execution_checkpoint
                            from chronaris.modeling.training.candidate_checkpoint import atomic_save_candidate
                            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                            derived = execution_checkpoint(payload, parent_path=checkpoint, graph=False,
                                evidence_sha256=sha256_file(root/'recipe.json'))
                            destination = root/method/'task_guided_initialization/best.pt'
                            if destination.exists():
                                saved = torch.load(destination, map_location='cpu', weights_only=True)
                                if (saved['protocol_sha256'] != derived['protocol_sha256'] or
                                    saved['canonical_training_state_sha256'] != derived['canonical_training_state_sha256']):
                                    raise ValueError('fine-tuning execution lineage changed')
                            else:
                                atomic_save_candidate(destination, derived)
                            checkpoint = destination
                            encoder, _, _ = load_frozen_application_encoder(checkpoint,
                                route='self_supervised', fold=fold, device='cuda')
                        with isolated_training_rng(seed):
                            model = EndToEndApplicationModel(method_name=method, encoder=encoder, normalizer=normalizer,
                                naive_encoder=None, task_definitions=definitions)
                        guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider,
                            targets=targets, role_sample_ids={r: getattr(fold, r+'_sample_ids') for r in ('train', 'validation', 'held_out')},
                            source_checkpoint_path=checkpoint, output_root=root/method/'task_guided',
                            config=recipe_configs[2] if recipe_configs else EndToEndFineTuningConfig(max_updates=joint_updates, head_warmup_updates=50 if full else 2, batch_size=4,
                                effective_batch_size=effective_batch, device='cuda', validation_interval=50 if full else 2, early_stopping=False,
                                data_manifest_sha256=digest))
                        guided_encoder, _, _ = load_frozen_application_encoder(guided.best_checkpoint_path,
                            route='task_guided', fold=fold, device='cuda')
                        routes.append(('task_guided', Path(guided.best_checkpoint_path), guided_encoder, asdict(guided)))
            if diagnostic_snapshots:
                from chronaris.evaluation.application_tasks.stage45_diagnostics import checkpoint_diagnostics
                checkpoint_diagnostics(root=root, domain=domain, provider=provider, fold=fold,
                    contract=contract, targets=targets, definitions=definitions, context=context,
                    observations=observations, digest=digest, results=results)
            if v4_workflow_source_sha256() != contract['source_code_sha256']:
                raise ValueError('source changed during contract smoke')
            result = dict(status='completed', domain=domain, scope=scope, total_seconds=time.perf_counter()-started,
                contract_sha256=contract['contract_sha256'], source_code_sha256=contract['source_code_sha256'],
                results=results, sample_counts={r: len(b.sample_ids) for r, b in observations.items()}, confirmation_opened=False)
            return write_result(root/'summary.json', result)
