"""One guided-loss ablation, paired with the completed grouped-selection run."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from chronaris.evaluation.application_tasks.application_finetuning import (
    EndToEndApplicationModel, EndToEndFineTuningConfig, train_end_to_end_application_method)
from chronaris.evaluation.application_tasks.application_finetuning_export import (
    load_frozen_application_encoder, export_loaded_application_encoder)
from chronaris.evaluation.application_tasks.checkpoint_selection import (
    selection_split, subset_targets, GroupedCheckpointSelector)
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract, run_common_downstream
from chronaris.evaluation.application_tasks.stage45 import compare_candidate, metric_rows
from chronaris.evaluation.application_tasks.stage45b import averaged_rows
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

PARENT_UNITS = (0, 1, 8, 6, 7, 11)


def read(path):
    return json.loads(Path(path).read_text())


def semantic_contract(contract):
    return {k:v for k,v in contract.items() if k not in ('source_code_sha256', 'contract_sha256')}


def groups():
    return [[(name, 'completed')] for name in ['stage45c_plan']+
            [f'stage45c_screen__{i}' for i in range(len(PARENT_UNITS))]+['stage45c_report']]


def parent_unit(parent, index):
    result = read(Path(parent)/'screen'/f'{index}.json')
    if result.get('status') != 'completed' or result.get('skipped'):
        raise ValueError('guided-only trial requires a completed parent unit')
    routes = {r['route']:r for r in result['results'] if 'training' in r}
    checkpoint = Path(routes['self_supervised']['training']['best_checkpoint_path'])
    unit_root = checkpoint.parents[4]
    return result, routes, checkpoint, unit_root


def trial_config(parent_config, selector, *, short=False):
    if parent_config['self_supervised_weight'] != .2:
        raise ValueError('paired reference must use the fixed public loss weight .2')
    old_selection = dict(parent_config['checkpoint_selection'])
    new_selection = dict(selector.manifest)
    old_selection.pop('source_code_sha256'); new_selection.pop('source_code_sha256')
    if old_selection != new_selection:
        raise ValueError('selection rule or data roles changed')
    config = replace(EndToEndFineTuningConfig(**parent_config), self_supervised_weight=0.,
                     checkpoint_selection=selector.manifest)
    if short:
        config = replace(config, max_updates=2, head_warmup_updates=2, effective_batch_size=4,
                         validation_interval=1, checkpoint_interval=1, retained_updates=())
    return config


def run_unit(config, index, *, short=False):
    parent_index = PARENT_UNITS[index]
    old, routes, source, parent_root = parent_unit(config['stage45c_parent'], parent_index)
    if not short:
        frozen = read(Path(config['root'])/'stage45c_plan.json')
        if any(sha256_file(p) != h for p,h in frozen['sources'].items()):
            raise ValueError('frozen parent or validation evidence changed before training')
    unit = old['unit']; domain, method = unit['domain'], unit['method']
    root = Path(config['root'])/'units'/str(index)/domain
    started = time.perf_counter()
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        provider, _, fold, digest, targets, definitions, context, _ = contract_development_inputs(domain,
            data_root=config['data_root'], registry_path=config['registry_path'], full=True)
        training, evaluation, _ = selection_split(fold, targets, definitions, context, inner_index=unit['inner_index'])
        fitting_targets = subset_targets(targets, training.train_sample_ids+training.validation_sample_ids)
        evaluation_targets = subset_targets(targets, evaluation.train_sample_ids+evaluation.validation_sample_ids)
        observations = {r:provider(getattr(evaluation, r+'_sample_ids')) for r in ('train','validation')}
        contract = build_common_contract(domain=domain, fold=evaluation, observations=observations,
            targets=evaluation_targets, definitions=definitions, context=context, data_manifest_sha256=digest,
            scope='development_comparison', seed=unit['seed'])
        if semantic_contract(contract) != semantic_contract(read(parent_root/'data_contract.json')):
            raise ValueError('parent evaluation data/targets/policy changed')
        encoder, normalizer, source_payload = load_frozen_application_encoder(source, route='self_supervised',
            fold=training, device='cuda')
        if source_payload['source_code_sha256'] != candidate_source_code_sha256():
            raise ValueError('pretraining implementation changed; no implicit weight migration')
        selector = GroupedCheckpointSelector(provider=provider, fold=training, targets=fitting_targets,
            definitions=definitions, context=context, output_root=root/'checkpoint_selection/task_guided', seed=unit['seed'])
        parent_config = read(parent_root/'recipe.json')['finetuning']
        resolved = trial_config(parent_config, selector, short=short)
        record = dict(status='completed', unit=unit, parent_unit=parent_index, pretraining_source=str(source),
            pretraining_source_sha256=sha256_file(source), config=asdict(resolved), new_pretraining_updates=0,
            single_factor={'self_supervised_weight':{'reference':.2,'candidate':0.}}, engineering_only=short,
            confirmation_opened=False)
        if (root/'recipe.json').exists() and read(root/'recipe.json') != json.loads(json.dumps(record)):
            raise ValueError('trial recipe changed; preserve this directory')
        write_result(root/'recipe.json',record); write_result(root/'data_contract.json',contract)
        with isolated_training_rng(unit['seed']):
            model = EndToEndApplicationModel(method_name=method, encoder=encoder, normalizer=normalizer,
                naive_encoder=None, task_definitions=definitions)
        trained = train_end_to_end_application_method(model=model, batch=None, batch_provider=provider,
            targets=fitting_targets, role_sample_ids={r:getattr(training,r+'_sample_ids') for r in ('train','validation','held_out')},
            source_checkpoint_path=source, output_root=root/method/'task_guided', config=resolved, checkpoint_selector=selector)
        declaration_path = parent_root/method/'task_guided/evaluation/common_contract.json'
        declaration = read(declaration_path)['declaration']
        results = {}
        for key, checkpoint in [('primary',Path(trained.best_checkpoint_path)),
                                ('original_selector',Path(trained.best_checkpoint_path).with_name('original_selector.pt'))]:
            encoder, _, payload = load_frozen_application_encoder(checkpoint, route='task_guided', fold=training, device='cuda')
            outputs = export_loaded_application_encoder(encoder=encoder, normalizer=normalizer, checkpoint=checkpoint,
                provider=provider, fold=evaluation, root=root/method/'task_guided'/key/'representations',
                export_roles=('train','validation'), export_prefix='stage45c', label_used_for_encoder_training=True)
            declared = declaration | dict(checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
                checkpoint_selection_contract=selector.manifest,
                evidence_files={str(checkpoint.resolve()):sha256_file(checkpoint), str(Path(__file__).resolve()):sha256_file(__file__)})
            results[key] = run_common_downstream(contract=contract, outputs=outputs, declaration=declared,
                targets=evaluation_targets, definitions=definitions, context=context, observations=observations,
                fold=evaluation, data_manifest_sha256=digest, output_root=root/method/'task_guided'/key/'evaluation',
                families=('linear',) if key=='original_selector' else ('linear','minirocket'))
        guided = results['primary'] | dict(route='task_guided', training=asdict(trained), original_selector=results['original_selector'])
        inherited = routes['self_supervised'] | {'inherited_unchanged': True, 'new_optimizer_updates':0}
        result = dict(status='completed', domain=domain, unit=unit, parent_unit=parent_index,
            results=[inherited,guided], total_seconds=time.perf_counter()-started, new_pretraining_updates=0,
            confirmation_opened=False, engineering_only=short)
        return write_result(Path(config['root'])/'screen'/f'{index}.json',result)


def plan(config):
    from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
    files = _validation_evidence(config['stage45c_validation'])
    parent = Path(config['stage45c_parent']); state = read(parent/'pipeline_state.json')
    if state['status'] != 'stage45b_completed' or state['failures'] or state['children']:
        raise ValueError('stage 4.5-B must be complete and stopped')
    for receipt in state['completed'].values():
        if sha256_file(receipt['path']) != receipt['sha256']:
            raise ValueError('completed parent receipt changed')
        files[receipt['path']] = receipt['sha256']
    for index in PARENT_UNITS:
        parent_result = parent/'screen'/f'{index}.json'
        if sha256_file(parent_result) != state['completed'][f'stage45b_screen__{index}']['sha256']:
            raise ValueError('parent screen result differs from its completed receipt')
        files[str(parent_result)] = sha256_file(parent_result)
        _, routes, source, parent_root = parent_unit(parent,index)
        for path in (source,parent_root/'data_contract.json',parent_root/'recipe.json',
                     parent_root/routes['task_guided']['training']['method_name']/'task_guided/evaluation/common_contract.json'):
            files[str(path)] = sha256_file(path)
        for row in routes.values():
            for name in ('best_checkpoint_path','last_checkpoint_path'):
                path=row['training'][name]; files[path]=sha256_file(path)
    return write_result(Path(config['root'])/'stage45c_plan.json',dict(status='completed',sources=files,
        parent_units=list(PARENT_UNITS),new_training_units=6,new_pretraining_updates=0,
        single_factor='guided public self-supervision weight .2 -> 0',budget_hours=config['stage45_budget_hours'],
        confirmation_opened=False))


def report(config):
    root=Path(config['root']); parent=Path(config['stage45c_parent'])
    frozen=read(root/'stage45c_plan.json')
    if any(sha256_file(p)!=h for p,h in frozen['sources'].items()):
        raise ValueError('parent or validation evidence changed')
    from collections import defaultdict
    by_method=defaultdict(lambda:defaultdict(list))
    for index in range(len(PARENT_UNITS)):
        result=read(root/'screen'/f'{index}.json')
        for row in metric_rows(result):
            by_method[result['unit']['variant']][(row['domain'],row['route'],row['task'])].append(row)
    mean=lambda groups:[rows[0]|{'value':sum(r['value'] for r in rows)/len(rows),'inner_training_replicas':len(rows)} for rows in groups.values()]
    candidates=mean(by_method['reference']); controls=mean(by_method['baseline'])
    return write_result(root/'stage45c_report.json',dict(status='completed',candidate=compare_candidate(candidates,averaged_rows(parent,'reference')),
        candidate_rows=candidates,strong_reference_rows=controls,
        strong_reference_change=compare_candidate(controls,averaged_rows(parent,'baseline')),
        parent_unchanged=True,confirmation_opened=False,model_adoption_automatic=False))


def run_stage45c(stage,config):
    if stage=='stage45c_plan':return plan(config)
    if stage=='stage45c_report':return report(config)
    if stage.startswith('stage45c_screen__'):return run_unit(config,int(stage.split('__')[1]))
    raise ValueError('unknown stage 4.5-C stage')
