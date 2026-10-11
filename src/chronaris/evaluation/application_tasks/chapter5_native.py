"""Predict with existing main encoders and consumers; never refit on outer roles."""
from dataclasses import replace
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_grouped_consumers import evaluate_native_consumers, native_consumer_context
from chronaris.representation import TrainOnlyRobustNormalizer, select_observation_batch


def require_outer_freeze(gate_path, main_root):
    if gate_path is None:
        raise ValueError('outer evaluation requires the committed core-model freeze')
    gate_path = Path(gate_path)
    gate = read(gate_path)
    state = read(gate_path.parent/'pipeline_state.json')
    receipt = state['completed'].get('chapter5_core_models')
    if not receipt:
        raise ValueError('core freeze has no committed pipeline receipt')
    verify({receipt['path']: receipt['sha256']})
    if read(receipt['path']) != gate:
        raise ValueError('core freeze differs from its receipt')
    if (gate['status'] != 'completed' or not gate['core_models_frozen']
        or gate['source_code_sha256'] != v4_workflow_source_sha256()
        or gate['main_models_sha256'] != sha(Path(main_root)/'main_models.json')):
        raise ValueError('outer evaluation freeze provenance differs')
    verify(gate['files'])
    return gate


def unit_record(main_root, index):
    root = Path(main_root)
    state, models = read(root/'pipeline_state.json'), read(root/'main_models.json')
    receipt = state['completed'][f'stage5_formal_unit__{index}']
    if models['unit_receipts'].get(receipt['path']) != receipt['sha256']:
        raise ValueError('main unit was not frozen')
    verify({receipt['path']: receipt['sha256']})
    unit = read(receipt['path'])
    if unit['unit'] != read(root/'formal_plan.json')['units'][index]:
        raise ValueError('main unit specification changed')
    verify(unit['model_files']); verify(unit['consumer_files'])
    return unit


def _inputs(spec, contract, role, sample_limit):
    project = Path(__file__).parents[4]
    data_root = project/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'
    provider, schema, fold, _, digest, targets, _, data = load_development_inputs(
        spec['domain'], data_root, project/'docs/requirements/thesis-v4-public-subjects.json',
        fold_index=spec['fold_index'], subject_role='confirmation')
    if digest != contract['data_manifest_sha256'] or fold.to_dict() != contract['fold']:
        raise ValueError('evaluation data or roles differ from the frozen training contract')
    ids = getattr(fold, role+'_sample_ids')
    if sample_limit:
        ids = ids[:sample_limit]
    if not ids or set(ids) & set(fold.train_sample_ids):
        raise ValueError('empty evaluation role or training overlap')
    if role == 'held_out':
        if spec['domain'] == 'dingxin':
            from chronaris.evaluation.application_tasks.v4_dingxin_data import _task_targets
            # Retain the original 12-block fit and its field order; do not refit on train+validation.
            targets, _, _ = _task_targets(data.fitted_targets, fold,
                fold.train_sample_ids+fold.validation_sample_ids+fold.held_out_sample_ids, 'inner_training')
            provider = data.index.load_batch
        else:
            targets = data.targets
            provider = data.dataset.batch_provider
    context = native_consumer_context(spec['domain'], data, fold, include_held_out=role == 'held_out')
    raw = provider(ids)
    if raw.sample_ids != tuple(ids):
        raise ValueError('provider returned other samples')
    if role == 'validation':
        expected = contract['observations']['validation']
        if list(raw.source_sample_hashes) != expected['source_sample_hashes'][:len(ids)]:
            raise ValueError('validation observations changed since training')
    return schema, fold, targets, context, raw


def _encode(spec, row, declaration, contract, raw, fold, output_root, role):
    checkpoint = declaration['checkpoint_path']
    if declaration['kind'] == 'causal_sequence':
        from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder, export_loaded_application_encoder
        if spec['method'] == 'naive_time_sync':
            from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
            if read(Path(checkpoint).parent/'fit_manifest.json')!=row['training']:
                raise ValueError('fixed synchronization fit manifest differs from the frozen receipt')
            encoder, normalizer, _ = load_v4_naive_encoder(checkpoint, fold=fold,
                data_manifest_sha256=contract['data_manifest_sha256'],
                expected_training_source_sha256=contract['source_code_sha256'])
        else:
            encoder, normalizer, _ = load_frozen_application_encoder(checkpoint, route=row['route'], fold=fold, device='cuda')
        selected_fold = replace(fold, **{role+'_sample_ids': raw.sample_ids})
        return export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
            provider=lambda ids: select_observation_batch(raw, ids), fold=selected_fold,
            root=output_root, export_roles=(role,), export_prefix='chapter5_frozen',
            label_used_for_encoder_training=declaration['route']=='task_guided')[role]
    from chronaris.modeling.fusion_encoders.recent_models import (
        verify_assets, historical_channels, timecma_prompt_cache, TimeCMAWindowEncoder, Chronos2WindowEncoder)
    from chronaris.evaluation.application_tasks.development_comparison import ASSETS, SENSOR_ASSETS
    from chronaris.evaluation.application_tasks.recent_model_training import load_encoder_checkpoint
    from chronaris.evaluation.application_tasks.recent_model_smoke import _export
    from chronaris.representation.window_features import WindowFeatureBatch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assets_path = SENSOR_ASSETS if spec['method']=='sensorllm_deepseek' else ASSETS
    if (saved['assets_sha256'] != sha(assets_path) or saved['seed'] != spec['seed']
        or saved['config']['data_manifest_sha256'] != contract['data_manifest_sha256']):
        raise ValueError('recent encoder assets, seed or data changed')
    assets = verify_assets(assets_path)
    normalizer = TrainOnlyRobustNormalizer.from_manifest(saved['normalizer'])
    if not set(normalizer.fit_sample_ids) <= set(fold.train_sample_ids):
        raise ValueError('recent encoder normalization escaped training')
    values, mask = historical_channels(normalizer.transform(raw))
    prompts = None
    if spec['method'] == 'timecma':
        prompts, _ = timecma_prompt_cache(values=values, mask=mask, times=raw.query_timestamps_s,
            channel_names=tuple(saved['channel_names']), assets=assets, output_root=Path(output_root)/'language_cache')
        factory = TimeCMAWindowEncoder
    elif spec['method'] == 'chronos2':
        factory = Chronos2WindowEncoder
    else:
        from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
        factory = SensorLLMWindowEncoder
    from chronaris.modeling.training.rng import isolated_training_rng
    with isolated_training_rng(spec['seed']):
        encoder = factory(assets, len(saved['channel_names']))
    if spec['method']=='sensorllm_deepseek':
        # Alignment sets this transient attribute; standalone inference must restore it without training.
        encoder.channel_names=tuple(saved['channel_names'])
    load_encoder_checkpoint(encoder, saved['encoder'])
    features, _ = _export(encoder, {role:(values, mask)}, {role:prompts})
    return WindowFeatureBatch(raw.sample_ids, raw.context_durations_s, features[role], mask.any((1,2)),
        spec['method'], fold.fold_id, sha(checkpoint), raw.source_sample_hashes)


def evaluate_main_unit(*, main_root, index, output_root, role='validation', sample_limit=4, gate_path=None):
    if role not in ('validation', 'held_out') or sample_limit < 0:
        raise ValueError('unsupported evaluation role or sample limit')
    if role == 'held_out':
        require_outer_freeze(gate_path, main_root)
        if sample_limit:
            raise ValueError('formal outer evaluation must retain every window')
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
    from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        root = Path(output_root)
        unit = unit_record(main_root, index)
        binding = dict(source_code_sha256=v4_workflow_source_sha256(), main_models_sha256=sha(Path(main_root)/'main_models.json'),
            index=index, role=role, sample_limit=sample_limit)
        result_path = root/'result.json'
        if result_path.exists():
            old = read(result_path)
            if old['binding'] != binding:
                raise ValueError('frozen evaluation binding changed')
            verify(old['files'])
            return old
        files, results = {}, []
        with _periodic_training_heartbeat('chapter5_frozen_prediction', 30., root=root) as progress:
            for row in unit['results']:
                evaluation = row.get('evaluation', row)
                contract_path = Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
                verify({str(contract_path):evaluation['contract_file_sha256']})
                record = read(contract_path)
                contract, declaration = record['contract'], record['declaration']
                verify(declaration['evidence_files'])
                schema, fold, targets, context, raw = _inputs(unit['unit'], contract, role, sample_limit)
                progress.update(route=row['route'], phase='encoder_reload_and_export')
                started = time.perf_counter()
                output = _encode(unit['unit'], row, declaration, contract, raw, fold, root/row['route']/'representation', role)
                components = {}
                for family, component in evaluation['consumers']['components'].items():
                    saved = joblib.load(component['model_path'])
                    if saved['protocol_sha256'] != evaluation['consumers']['protocol_sha256']:
                        raise ValueError('frozen downstream protocol changed')
                    bundle = saved['consumer']
                    if tuple(bundle['train_sample_ids']) != fold.train_sample_ids:
                        raise ValueError('downstream training role changed')
                    prediction = evaluate_native_consumers(bundle, output=output, targets=targets, context=context)
                    delta = None
                    if role == 'validation':
                        original = read(component['result_path'])['evaluations']['validation']['prediction_rows']
                        selected = [r for r in original if r['sample_id'] in set(raw.sample_ids)]
                        delta = compare_predictions(selected, prediction['prediction_rows'])
                    target = root/row['route']/f'{family}_{role}.json'
                    write(target, prediction)
                    files[str(target.resolve())] = sha(target)
                    components[family] = dict(path=str(target), replay=delta)
                results.append(dict(route=row['route'], samples=len(raw.sample_ids), seconds=time.perf_counter()-started,
                    components=components))
        return write(result_path, dict(status='completed', binding=binding, results=results, files=files,
            confirmation_opened=role=='held_out', downstream_refitted=False))


def compare_predictions(expected, actual, *, atol=1e-5, rtol=1e-5):
    key = lambda row:(row['sample_id'], row['task'], row['field'])
    left, right = {key(r):r for r in expected}, {key(r):r for r in actual}
    if len(left)!=len(expected) or len(right)!=len(actual) or left.keys()!=right.keys():
        raise ValueError('frozen replay sample/task/field coverage changed')
    maximum = 0.
    for k, a in left.items():
        b = right[k]
        for field in ('target_valid','target','group_id','no_observation','sample_weight'):
            if a[field] != b[field]:
                raise ValueError(f'frozen replay target/context differs: {k}/{field}')
        for field in ('prediction', 'probability'):
            if a[field] is None or b[field] is None:
                if a[field] != b[field]:
                    raise ValueError('frozen replay probability availability changed')
                continue
            x, y = np.asarray(a[field]), np.asarray(b[field])
            if x.shape!=y.shape or not np.allclose(x,y,atol=atol,rtol=rtol):
                raise ValueError(f'frozen replay prediction differs: {k}/{field}')
            maximum=max(maximum,float(np.max(np.abs(x-y))))
    return dict(passed=True, rows=len(left), max_absolute_difference=maximum, atol=atol, rtol=rtol)
