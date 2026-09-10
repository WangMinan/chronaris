"""Bounded real-development closure using existing encoders, exports and consumers."""
from dataclasses import asdict, replace
import json
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


def contract_development_inputs(domain, *, data_root, registry_path):
    if domain == 'dingxin':
        data, _ = deduplicated_dingxin_data(load_v4_dingxin_development())
        fold = data.folds[0]
        targets, definitions = data.targets_by_fold[fold.fold_id], data.definitions_by_fold[fold.fold_id]
        provider, schema, digest = data.development_provider(fold), data.index.plan.schema, data.data_manifest_sha256
    else:
        provider, schema, fold, _, digest, targets, definitions, data = load_development_inputs(domain, data_root, registry_path)
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
                            registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    if domain not in {'dingxin', 'cogpilot', 'clare'} or not torch.cuda.is_available():
        raise ValueError('contract smoke requires a real native development domain and CUDA')
    root = Path(output_root)/domain
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        with _periodic_training_heartbeat('common_contract_'+domain, 30., root=root) as progress:
            progress['phase'] = 'bind_real_development_inputs'
            provider, schema, fold, digest, targets, definitions, context, observations = contract_development_inputs(
                domain, data_root=data_root, registry_path=registry_path)
            contract = build_common_contract(domain=domain, fold=fold, observations=observations,
                targets=targets, definitions=definitions, context=context, data_manifest_sha256=digest)
            contract_path = root/'data_contract.json'
            if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
                raise ValueError('contract smoke inputs or source changed; preserve this root')
            write_result(contract_path, contract)
            normalizer = TrainOnlyRobustNormalizer().fit_from_batch_provider(provider,
                train_sample_ids=fold.train_sample_ids, held_out_sample_ids=fold.validation_sample_ids+fold.held_out_sample_ids,
                batch_size=4)
            results = []
            for method in ('naive_time_sync', 'chronaris'):
                progress.update(phase='encoder_fit', method=method)
                if method == 'naive_time_sync':
                    checkpoint, training = fit_v4_naive_encoder(provider=provider, fold=fold, normalizer=normalizer,
                        data_manifest_sha256=digest, output_root=root/method/'checkpoint')
                    encoder, _, _ = load_v4_naive_encoder(checkpoint, fold=fold, data_manifest_sha256=digest)
                    routes = [('self_supervised', checkpoint, encoder, training)]
                else:
                    training = train_pretext_candidate(method, candidate=EncoderCandidateConfig(candidate_id='C', hidden_dim=32),
                        batch=None, batch_provider=provider, fold=fold, physiology_feature_names=schema.physiology_feature_names,
                        vehicle_feature_names=schema.vehicle_feature_names, vehicle_field_labels=(), normalizer=normalizer,
                        output_root=root/method/'self_supervised', config=CandidateScreenConfig(max_updates=2,
                            batch_size=4, effective_batch_size=4, device='cuda', validation_interval=2, early_stopping=False,
                            data_manifest_sha256=digest), chronaris_fusion_kind='safe_lag')
                    encoder, _, _ = load_frozen_application_encoder(training.best_checkpoint_path,
                        route='self_supervised', fold=fold, device='cuda')
                    routes = [('self_supervised', Path(training.best_checkpoint_path), encoder, asdict(training))]
                for route, checkpoint, encoder, training in routes:
                    progress.update(phase='export_and_common_consumers', route=route)
                    outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
                        provider=provider, fold=fold, root=root/method/route/'representations',
                        export_roles=('train', 'validation'), export_prefix='common_contract_smoke',
                        label_used_for_encoder_training=route == 'task_guided')
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
                    results.append(first | {'training': training, 'resume_identical': True})
                    if method == 'chronaris' and route == 'self_supervised':
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
                        with isolated_training_rng(17):
                            model = EndToEndApplicationModel(method_name=method, encoder=encoder, normalizer=normalizer,
                                naive_encoder=None, task_definitions=definitions)
                        guided = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider,
                            targets=targets, role_sample_ids={r: getattr(fold, r+'_sample_ids') for r in ('train', 'validation', 'held_out')},
                            source_checkpoint_path=checkpoint, output_root=root/method/'task_guided',
                            config=EndToEndFineTuningConfig(max_updates=2, head_warmup_updates=2, batch_size=4,
                                effective_batch_size=4, device='cuda', validation_interval=2, early_stopping=False,
                                data_manifest_sha256=digest))
                        guided_encoder, _, _ = load_frozen_application_encoder(guided.best_checkpoint_path,
                            route='task_guided', fold=fold, device='cuda')
                        routes.append(('task_guided', Path(guided.best_checkpoint_path), guided_encoder, asdict(guided)))
            if v4_workflow_source_sha256() != contract['source_code_sha256']:
                raise ValueError('source changed during contract smoke')
            result = dict(status='completed', domain=domain, scope='engineering_development',
                contract_sha256=contract['contract_sha256'], source_code_sha256=contract['source_code_sha256'],
                results=results, sample_counts={r: len(b.sample_ids) for r, b in observations.items()}, confirmation_opened=False)
            return write_result(root/'summary.json', result)
