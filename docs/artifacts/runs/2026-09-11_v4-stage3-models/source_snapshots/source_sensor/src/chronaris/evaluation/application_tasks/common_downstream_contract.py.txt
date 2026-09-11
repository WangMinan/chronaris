"""Executable, method-independent development contracts for common downstream fits."""
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.application_consumers import LinearConsumerConfig
from chronaris.evaluation.application_tasks.application_finetuning import _task_target_sha256
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_grouped_consumers import run_native_method_consumers
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_data import PUBLIC_TASKS
from chronaris.representation.window_features import WindowFeatureBatch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


TASK_MEANINGS = {
    'dingxin': {
        'maneuver_classification': '鼎新未来机动强度分类：训练角色拟合低、中、高阈值；真实数据组件验证',
        'maneuver_regression': '鼎新未来机动强度回归：现有运动学分数；真实数据组件验证',
        'physiology_regression': '鼎新未来生理字段回归：历史窗口之后5秒的有效字段中位数；真实数据组件验证'},
    'cogpilot': {
        'difficulty': 'CogPilot虚拟飞行难度分类：原有四类标签',
        'event_response': 'CogPilot离线事件条件响应：事件后8秒与事件前2秒皮肤电导中位数之差'},
    'clare': {
        'workload_classification': 'CLARE认知负荷分类：原10秒区间评分，分类阈值7',
        'workload_regression': 'CLARE认知负荷回归：原10秒标签区间评分'},
    'simulation': {'scope': '仿真仅承担时间机制、已知扰动及鲁棒性验证，沿用既有冻结仿真入口'},
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def downstream_policy():
    config = LinearConsumerConfig()
    return dict(format='chronaris.common_downstream_policy.v1', tasks=TASK_MEANINGS,
        primary_consumers=dict(classification='logistic_regression', regression='fieldwise_ridge'),
        public_parameters=dict(classification_c=list(config.classification_c_grid),
            regression_alpha=list(config.regression_alpha_grid), selection='validation_subject_mean'),
        dingxin_parameters=dict(classification_c=[1.], regression_alpha=[1.], selection='fixed',
            shared_vehicle_view_weight_sum=1, persistence_reference=True),
        supplementary_consumers=dict(public='minirocket_10000_kernels', dingxin=None),
        primary_metrics=dict(classification='macro_f1_over_declared_classes', regression='rmse_field_mean_then_group_mean'),
        fitting='independent_weights_per_representation; all_transforms_fit_on_training_role_only',
        grouping=dict(public='subject', dingxin='single_record_time_block_descriptive_only', simulation='parameter_profile'),
        representation_kinds=['causal_sequence', 'window_end'],
        feature_origin='hidden_state_not_final_task_predictions',
        supervision_strata=['self_supervised:none', 'task_guided:summary_labels', 'task_guided:future_trajectory'],
        preserve_all_windows=True, confirmation_opened=False)


def _observation_binding(raw):
    digest = hashlib.sha256()
    for field in fields(raw):
        value = getattr(raw, field.name)
        digest.update(field.name.encode())
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(json.dumps(value).encode())
    return dict(sample_ids=list(raw.sample_ids), source_sample_hashes=list(raw.source_sample_hashes),
        observation_sha256=digest.hexdigest(), query_timestamps_s=raw.query_timestamps_s.double().tolist(),
        exclusive_window_end_s=raw.context_durations_s.double().tolist())


def build_common_contract(*, domain, fold, observations, targets, definitions, context, data_manifest_sha256,
                          scope='engineering_development', seed=17):
    """Freeze real inputs and targets before any method is evaluated; never opens confirmation."""
    if domain not in {'dingxin', 'cogpilot', 'clare'} or context['domain'] != domain or seed not in (17, 29, 43):
        raise ValueError('common scalar contract requires an approved native development domain and seed')
    if scope not in {'engineering_development', 'development_comparison'}:
        raise ValueError('confirmation requires its later configuration freeze')
    if set(observations) != {'train', 'validation'}:
        raise ValueError('common development contract requires exactly train and validation')
    if {t.name for t in definitions} != set(TASK_MEANINGS[domain]):
        raise ValueError('task names differ from the approved common evaluation')
    if domain in PUBLIC_TASKS and tuple(definitions) != PUBLIC_TASKS[domain]:
        raise ValueError('public task types/classes changed')
    if domain == 'dingxin':
        dims = {t.name: (t.kind, t.output_dim) for t in definitions}
        if (context.get('evaluation_protocol') != 'single_record_chronological_blocks_v1'
            or dims['maneuver_classification'] != ('classification', 3)
            or dims['maneuver_regression'] != ('regression', 1)
            or dims['physiology_regression'] != ('regression', len(targets.manifest['physiology_field_order']))):
            raise ValueError('Dingxin requires the retained record and existing task dimensions')
    roles = {role: list(getattr(fold, role+'_sample_ids')) for role in observations}
    if targets.sample_ids != tuple(roles['train'] + roles['validation']):
        raise ValueError('target order or development roles changed')
    for role, raw in observations.items():
        if tuple(roles[role]) != raw.sample_ids:
            raise ValueError('observation roles changed')
        for stream in ('physiology', 'vehicle'):
            times = getattr(raw, stream+'_timestamps_s')
            mask = getattr(raw, stream+'_point_mask')
            if ((times < 0) & mask).any() or ((times >= raw.context_durations_s[:, None]) & mask).any():
                raise ValueError('observations exceed the exclusive historical cutoff')
    contract = dict(format='chronaris.common_downstream_contract.v1', policy=downstream_policy(), domain=domain,
        scope=scope, seed=seed, fold=fold.to_dict(), roles=roles, data_manifest_sha256=data_manifest_sha256,
        source_code_sha256=v4_workflow_source_sha256(), context=context,
        task_definitions=[asdict(t) for t in definitions], target_manifest=targets.manifest,
        target_sha256=_task_target_sha256(targets), observations={r: _observation_binding(b) for r, b in observations.items()},
        time_reference='seconds_since_context_start', confirmation_opened=False)
    # Normalize tuples to JSON lists, so a serialized contract compares identically on resume.
    contract = json.loads(json.dumps(contract, allow_nan=False))
    return contract | {'contract_sha256': _digest(contract)}


def validate_representation_declaration(contract, outputs, declaration, *, families):
    """Require explicit temporal capability, extraction evidence and target-domain supervision."""
    if set(outputs) != set(contract['roles']):
        raise ValueError('representation roles differ from the common contract')
    required = {'method', 'checkpoint_path', 'checkpoint_sha256', 'route', 'target_supervision', 'training_tasks',
        'external_pretraining', 'kind', 'feature_origin', 'extraction_location', 'feature_dim',
        'encoder_frozen', 'task_heads_removed', 'encoder_fit_sample_ids', 'preprocessing_fit_sample_ids', 'evidence_files'}
    if set(declaration) != required:
        raise ValueError('representation declaration fields are incomplete or unknown')
    if declaration['encoder_frozen'] is not True or declaration['task_heads_removed'] is not True:
        raise ValueError('downstream evaluation requires a frozen encoder with training heads removed')
    if declaration['feature_origin'] not in {'hidden_state', 'unsupervised_projection'} or not declaration['extraction_location']:
        raise ValueError('final predictions and broadcast window vectors are not intermediate features')
    if declaration['kind'] not in {'causal_sequence', 'window_end'}:
        raise ValueError('unknown representation time capability')
    if declaration['kind'] == 'window_end' and tuple(families) != ('linear',):
        raise ValueError('window-end features support scalar linear consumers only')
    route, signal = declaration['route'], declaration['target_supervision']
    stratum = f'{route}:{signal}'
    if stratum not in contract['policy']['supervision_strata']:
        raise ValueError('encoder route and actual target supervision disagree')
    tasks = declaration['training_tasks']
    if len(tasks) != len(set(tasks)):
        raise ValueError('declared training tasks must be unique')
    if signal == 'summary_labels':
        if not tasks or not set(tasks) <= {t['name'] for t in contract['task_definitions']}:
            raise ValueError('summary-label supervision must identify its actual tasks')
    elif tasks:
        raise ValueError('non-summary supervision cannot claim summary task labels')
    if not isinstance(declaration['external_pretraining'], dict) or not declaration['external_pretraining'].get('source'):
        raise ValueError('external pretraining source must be explicit, including none')
    external = declaration['external_pretraining']
    if external['source'] != 'none' and not all(external.get(k) for k in ('revision', 'weights_sha256', 'training_signal')):
        raise ValueError('external pretraining requires version, weights and training-signal provenance')
    train = set(contract['roles']['train'])
    for key in ('encoder_fit_sample_ids', 'preprocessing_fit_sample_ids'):
        ids = declaration[key]
        if len(ids) != len(set(ids)) or not set(ids) <= train:
            raise ValueError('encoder or preprocessing fitting escaped the training role')
    checkpoint = declaration['checkpoint_path']
    if sha256_file(checkpoint) != declaration['checkpoint_sha256']:
        raise ValueError('declared encoder checkpoint changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    checkpoint_data = payload.get('config', {}).get('data_manifest_sha256')
    if checkpoint_data is not None and checkpoint_data != contract['data_manifest_sha256']:
        raise ValueError('encoder checkpoint was fitted on a different data manifest')
    if payload.get('method_name', declaration['method']) != declaration['method']:
        raise ValueError('declared method disagrees with the encoder checkpoint')
    if payload.get('label_used_for_encoder_training') is not (signal != 'none'):
        raise ValueError('declared supervision disagrees with the actual encoder checkpoint')
    actual_signal = payload.get('target_supervision', 'summary_labels' if payload['label_used_for_encoder_training'] else 'none')
    if signal != actual_signal:
        raise ValueError('future-trajectory and summary-label supervision cannot be mixed')
    if signal == 'summary_labels' and set(tasks) != {t['name'] for t in payload['task_definitions']}:
        raise ValueError('declared training tasks disagree with the checkpoint')
    encoder_roles = payload.get('role_sample_ids', payload.get('fold', {}))
    actual_fit = encoder_roles.get('train', encoder_roles.get('train_sample_ids'))
    if actual_fit is not None and list(actual_fit) != declaration['encoder_fit_sample_ids']:
        raise ValueError('declared encoder fitting samples disagree with the checkpoint')
    actual_preprocessing = payload.get('normalizer', {}).get('fit_sample_ids')
    if actual_preprocessing is not None and set(actual_preprocessing) != set(declaration['preprocessing_fit_sample_ids']):
        raise ValueError('declared preprocessing samples disagree with the checkpoint')
    for transform in ('normalizer', 'pca_state'):
        if not set(payload.get(transform, {}).get('fit_sample_ids', ())) <= train:
            raise ValueError('checkpoint preprocessing fitting escaped the training role')
    if not declaration['evidence_files'] or any(sha256_file(p) != h for p, h in declaration['evidence_files'].items()):
        raise ValueError('representation extraction evidence changed or is missing')
    for role, output in outputs.items():
        raw = contract['observations'][role]
        if (list(output.sample_ids) != contract['roles'][role]
            or list(output.source_sample_hashes) != raw['source_sample_hashes']
            or output.fold_id != contract['fold']['fold_id'] or output.method_name != declaration['method']
            or output.checkpoint_sha256 != declaration['checkpoint_sha256']
            or output.pooled_embedding.shape[1] != declaration['feature_dim']):
            raise ValueError('representation samples, source, fold, encoder or dimension changed')
        window = declaration['kind'] == 'window_end'
        if window != isinstance(output, WindowFeatureBatch):
            raise ValueError('declared temporal capability differs from the actual representation')
        expected = raw['exclusive_window_end_s'] if window else raw['query_timestamps_s']
        if not torch.equal(output.timestamps_s.double().cpu(), torch.tensor(expected, dtype=torch.float64)):
            raise ValueError('representation historical cutoff or query grid changed')
    return stratum


def run_common_downstream(*, contract, outputs, targets, definitions, context, observations, fold,
                          data_manifest_sha256, declaration, output_root, families=('linear',)):
    expected = build_common_contract(domain=context['domain'], fold=fold, observations=observations,
        targets=targets, definitions=definitions, context=context, data_manifest_sha256=data_manifest_sha256,
        scope=contract['scope'], seed=contract['seed'])
    if expected != contract:
        raise ValueError('common data/target/history/policy contract changed')
    stratum = validate_representation_declaration(contract, outputs, declaration, families=families)
    root = Path(output_root)
    record = dict(contract=contract, declaration=declaration, families=list(families), supervision_stratum=stratum)
    record = json.loads(json.dumps(record, allow_nan=False))
    path = root/'common_contract.json'
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('common evaluation source or declaration changed; use a new root')
    write_result(path, record)
    results = run_native_method_consumers(outputs=outputs, targets=targets, definitions=definitions,
        context=context | {'common_contract_sha256': contract['contract_sha256'],
            'representation_declaration': declaration, 'supervision_stratum': stratum},
        output_root=root/'consumers', label_used_for_encoder_training=declaration['target_supervision'] != 'none',
        seed=contract['seed'], families=families)
    result = dict(status='completed', contract_sha256=contract['contract_sha256'],
        supervision_stratum=stratum, method=declaration['method'], representation_kind=declaration['kind'],
        scope=contract['scope'], comparison_key=f"{contract['contract_sha256']}:{stratum}:{_digest(sorted(declaration['training_tasks']))}", consumers=results,
        contract_file_sha256=sha256_file(path), confirmation_opened=False)
    return write_result(root/'result.json', result)
