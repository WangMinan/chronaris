"""Predeclared seed-17 thesis delivery, independent of the legacy 36-unit gate."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write, bind
from chronaris.evaluation.application_tasks.chapter5_core import core_units, _train
from chronaris.evaluation.application_tasks.chapter5_native import unit_record
from chronaris.evaluation.application_tasks.common_downstream_contract import _digest, downstream_policy
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.evaluation.application_tasks.v4_confirmation_training import CONFIRMATION_BUDGET
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan


def subset_units():
    units = [dict(domain=domain, method='chronaris', recipe_name=recipe, fold_index=fold, seed=17,
                  routes=['self_supervised', 'task_guided'])
             for domain, recipes in [('clare', ('thesis_reference', 'thesis_no_event_semantics')),
                                     ('cogpilot', ('thesis_reference',))]
             for fold in range(5) for recipe in recipes]
    for index in (18, 21):
        units.append(deepcopy(core_units()[index]) | dict(domain='simulation', legacy_index=index))
    semantic = deepcopy(core_units()[0])
    semantic.update(domain='simulation', purpose='ablation')
    semantic['options'].update(name='no_event_semantics', semantic_event_enabled=False)
    units.append(semantic)
    return [u | dict(index=i) for i, u in enumerate(units)]


def _immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError('subset artifact changed; preserve it and use a new root')
    return write(path, value)


def _bind_native(files, receipt, record):
    bind(files, receipt['path'], receipt['sha256'])
    for key in ('model_files', 'consumer_files'):
        for path, digest in record[key].items(): bind(files, path, digest)
    for row in record['results']:
        evaluation = row.get('evaluation', row)
        path = Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
        bind(files, path, evaluation['contract_file_sha256'])
        for name, digest in read(path)['declaration']['evidence_files'].items(): bind(files, name, digest)


def prepare_subset(*, main_root, core_root, legacy_workspace, pause_receipt, output_root, legacy_compatibility=None):
    """CPU-only metadata/checkpoint audit; never reads held-out observations or scores."""
    import torch
    from chronaris.representation import TrainOnlyRobustNormalizer
    from chronaris.models.alignment.calibrated_physics import fit_physics_calibration
    main, core, root = (Path(p).resolve() for p in (main_root, core_root, output_root))
    current_source = v4_workflow_source_sha256()
    legacy = Path(legacy_workspace).resolve()
    if root in (main, core) or main in root.parents or core in root.parents:
        raise ValueError('subset requires an independent output root')
    files = {}
    if legacy_compatibility:
        from chronaris.evaluation.application_tasks.device_resume import qualified_execution
        _, compatibility_files = qualified_execution(legacy_compatibility, legacy_workspace=legacy, core_root=core)
        files.update(compatibility_files)
    formal, models, state = (read(main/n) for n in ('formal_plan.json', 'main_models.json', 'pipeline_state.json'))
    if not models['main_models_frozen'] or models['confirmation_opened'] or models['plan_sha256'] != formal['plan_sha256']:
        raise ValueError('parent main models are not frozen and unopened')
    for name in ('formal_plan.json', 'main_models.json'): bind(files, main/name, sha(main/name))
    for name, digest in formal['input_files'].items(): bind(files, name, digest)
    core_plan, core_state = read(core/'chapter5_plan.json'), read(core/'pipeline_state.json')
    config = read(core/'pipeline_config.json')
    for name in ('chapter5_plan.json', 'pipeline_config.json'): bind(files, core/name, sha(core/name))
    for name, digest in config['input_files'].items(): bind(files, name, digest)
    pause = read(pause_receipt)
    if pause['parent_root'] != str(core) or pause['completed_units'] != 18 or not pause['stopped']:
        raise ValueError('unrecognized preserved simulation pause')
    bind(files, pause_receipt, sha(pause_receipt))
    bind(files, pause['preserved_checkpoint'], pause['preserved_sha256'])
    saved = torch.load(pause['preserved_checkpoint'], map_location='cpu', weights_only=True)
    bind(files, saved['source_checkpoint_path'], saved['source_checkpoint_sha256'])
    pretraining = torch.load(saved['source_checkpoint_path'], map_location='cpu', weights_only=True)
    resume = resume_counts(saved, pretraining_updates=pretraining['total_optimizer_updates'])
    if resume != dict(pretraining=1500, warmup=50, joint=75, joint_remaining=425):
        raise ValueError('paused update accounting changed')
    command = [sys.executable, '-c', 'from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256; print(v4_workflow_source_sha256())']
    import os
    env = os.environ | {'PYTHONPATH': str(legacy/'src'), 'CUDA_VISIBLE_DEVICES': ''}
    source = subprocess.check_output(command, cwd=legacy, env=env, text=True).strip()
    if source != config['source_code_sha256']:
        raise ValueError('legacy frozen workspace source changed')
    native_refs, lookup = [], {}
    for i, spec in enumerate(formal['units']):
        if spec['domain'] not in ('clare', 'cogpilot') or spec['seed'] != 17: continue
        record = unit_record(main, i)
        _bind_native(files, state['completed'][f'stage5_formal_unit__{i}'], record)
        native_refs.append(dict(index=i, unit=spec, receipt=state['completed'][f'stage5_formal_unit__{i}']))
        if spec['method'] == 'chronaris': lookup[spec['domain'], spec['fold_index']] = record
    simulation_refs = []
    for i in (0, 3, 6, 9, 12, 15):
        receipt = core_state['completed'][f'chapter5_core__{i}']
        verify({receipt['path']: receipt['sha256']})
        result = read(receipt['path'])
        if result['unit'] != core_plan['units'][i] or result['engineering_smoke'] or result['plan_sha256'] != core_plan['plan_sha256']:
            raise ValueError('simulation reference differs from its frozen plan')
        bind(files, receipt['path'], receipt['sha256'])
        for name, digest in result['files'].items(): bind(files, name, digest)
        simulation_refs.append(dict(index=i, receipt=receipt))
    units = subset_units()
    for unit in units:
        if unit['domain'] == 'simulation':
            unit['budget'] = CONFIRMATION_BUDGET
            if 'legacy_index' in unit:
                expected = core_plan['units'][unit['legacy_index']]
                if any(unit[k] != expected[k] for k in ('method', 'options', 'seed', 'routes')):
                    raise ValueError('legacy continuation configuration differs')
            continue
        old = lookup[unit['domain'], unit['fold_index']]
        row = next(r for r in old['results'] if r['route'] == 'self_supervised')
        path = Path(row['consumers']['manifest_path']).parents[1]/'common_contract.json'
        contract = deepcopy(read(path)['contract'])
        contract.pop('contract_sha256')
        contract['source_code_sha256'] = current_source
        checkpoint = read(path)['declaration']['checkpoint_path']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        normalizer = TrainOnlyRobustNormalizer.from_manifest(payload['normalizer'])
        calibration = fit_physics_calibration(normalizer, None, train_sample_ids=contract['fold']['train_sample_ids'],
            vehicle_feature_names=payload['vehicle_feature_names'], relations=())
        candidate, pre, guided, args, _ = training_recipe(unit['recipe_name'], method='chronaris', full=True,
            seed=17, digest=old['unit']['data_manifest_sha256'], train_count=old['unit']['counts']['train'],
            graph=True, calibration=calibration, device='cpu')
        unit.update(counts=old['unit']['counts'], data_manifest_sha256=old['unit']['data_manifest_sha256'],
            expected_contract_sha256=_digest(contract), reference_index=next(r['index'] for r in native_refs if r['unit']==old['unit']),
            recipe=dict(candidate=asdict(candidate), pretraining=asdict(pre)|{'device':'cuda'},
                finetuning=asdict(guided)|{'device':'cuda'}, mechanisms=args),
            physics=dict(applicability='not_applicable_no_calibrated_native_relations', valid_count=0),
            consumers=['linear', 'minirocket'])
    verify(files)
    if v4_workflow_source_sha256() != current_source:
        raise ValueError('source changed while preparing subset')
    result = seal_development_plan(dict(format='chronaris.chapter5_subset.v1', status='prepared',
        source_code_sha256=current_source, root=str(root), main_root=str(main), core_root=str(core),
        legacy_workspace=str(legacy), legacy_source_sha256=source, input_files=files,
        **({'legacy_compatibility': str(Path(legacy_compatibility).resolve())} if legacy_compatibility else {}),
        registry_path=str(Path('docs/requirements/thesis-v4-public-subjects.json').resolve()),
        data_root=str(Path('artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared').resolve()),
        simulation_root=core_plan['simulation_root'], units=units, model_units=18, public_units=15,
        native_references=native_refs, simulation_references=simulation_refs, resume_counts=resume,
        seed_policy=[17], downstream_policy=downstream_policy(), confirmation_opened=False,
        later_order=['same_scope_seed29', 'same_scope_seed43', 'explicit_shift_ablation',
                     'single_stream_bypass_ablation', 'physics_weight_scan', 'cogpilot_semantic_ablation'],
        legacy_scope_preserved=dict(public_units=45, core_units=36), stop_after='independent_subset_freeze'))
    return _immutable(root/'subset_plan.json', result)


def resume_counts(payload, *, pretraining_updates=1500):
    total, warmup = payload['step_count'], payload['config']['head_warmup_updates']
    if type(total) is not int or warmup != 50 or total < warmup or total > warmup+500:
        raise ValueError('invalid paused update counts')
    return dict(pretraining=pretraining_updates, warmup=warmup, joint=total-warmup, joint_remaining=500-(total-warmup))


def verified_subset(root):
    plan = read(Path(root)/'subset_plan.json')
    if (plan['source_code_sha256'] != v4_workflow_source_sha256() or plan['root'] != str(Path(root).resolve())
        or seal_development_plan({k:v for k,v in plan.items() if k!='plan_sha256'}) != plan):
        raise ValueError('subset plan or source changed')
    skeleton = subset_units()
    if len(plan['units']) != 18 or any(any(u.get(k)!=v for k,v in s.items()) for u,s in zip(plan['units'], skeleton)):
        raise ValueError('subset inventory changed')
    verify(plan['input_files'])
    return plan


def train_subset_unit(root, index):
    from chronaris.evaluation.application_tasks.common_downstream_smoke import run_common_contract_smoke
    root = Path(root); plan = verified_subset(root)
    if not 0 <= index < len(plan['units']): raise ValueError('unplanned subset unit')
    unit = plan['units'][index]; destination = root/'units'/f'{index}.json'
    if destination.exists():
        result = read(destination); validate_unit(result, unit, plan); return result
    if unit['domain'] != 'simulation':
        result = run_common_contract_smoke(domain=unit['domain'], output_root=root/'training'/str(index),
            data_root=plan['data_root'], registry_path=plan['registry_path'], full=True, methods=('chronaris',),
            recipe=unit['recipe_name'], seed=17, fold_index=unit['fold_index'], subject_role='confirmation',
            cuda_graph_recurrence=True, expected_recipe=unit['recipe'], expected_contract_sha256=unit['expected_contract_sha256'])
        if result['status'] == 'waiting_gpu': return result
        files = {}; _bind_native(files, dict(path=str(root/'training'/str(index)/unit['domain']/'summary.json'),
            sha256=sha(root/'training'/str(index)/unit['domain']/'summary.json')), result | dict(model_files={}, consumer_files={}))
        for row in result['results']:
            for component in row['consumers']['components'].values():
                for kind in ('model', 'result'): bind(files, component[kind+'_path'], component[kind+'_sha256'])
        result['files'] = files
    elif 'legacy_index' in unit:
        if plan.get('legacy_compatibility'):
            from chronaris.evaluation.application_tasks.device_resume import qualified_execution
            execution, bindings = qualified_execution(plan['legacy_compatibility'],
                legacy_workspace=plan['legacy_workspace'], core_root=plan['core_root'])
            runner = Path(__file__).parents[4]/'scripts/evaluation/application_tasks/run_device_resume.py'
            subprocess.run([sys.executable, str(runner), '--workspace', execution['workspace'],
                '--receipt', plan['legacy_compatibility'], '--mode', 'resumed', '--index', str(unit['legacy_index'])],
                cwd=execution['workspace'], check=True)
            result = read(Path(execution['root'])/'resumed'/f"unit_{unit['legacy_index']}_completed.json")
            if result['status'] != 'waiting_gpu':
                result['files'].update(bindings)
        else:
            legacy = Path(plan['legacy_workspace'])
            raw = root/'legacy_receipts'/f'{index}.json'; raw.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(legacy/'scripts/evaluation/application_tasks/run_v4_pipeline.py'),
                '--worker', f"chapter5_core__{unit['legacy_index']}", '--config', str(Path(plan['core_root'])/'pipeline_config.json'),
                '--result', str(raw)], cwd=legacy, check=True)
            result = read(raw)
        if result['status'] == 'waiting_gpu': return result
        if result['unit'] != read(Path(plan['core_root'])/'chapter5_plan.json')['units'][unit['legacy_index']]:
            raise ValueError('legacy worker returned another unit')
        result['legacy_plan_sha256'] = result['plan_sha256']
    else:
        result = _train(unit, plan, root/'simulation_training')
        if result['status'] == 'waiting_gpu': return result
    result = result | dict(unit=unit, plan_sha256=plan['plan_sha256'])
    validate_unit(result, unit, plan)
    return _immutable(destination, result)


def validate_unit(result, unit, plan):
    if (result['status']!='completed' or result['unit']!=unit or result['plan_sha256']!=plan['plan_sha256']
        or result.get('confirmation_opened', True) or result.get('engineering_smoke', False) or not result['files']):
        raise ValueError('subset unit incomplete or binding changed')
    if unit['domain'] != 'simulation':
        if (result['scope']!='confirmation_training' or result['contract_sha256']!=unit['expected_contract_sha256']
            or [r['route'] for r in result['results']]!=unit['routes']
            or result['sample_counts']!={k:unit['counts'][k] for k in ('train','validation')}):
            raise ValueError('subset native routes or contract changed')
    else:
        from chronaris.evaluation.application_tasks.v4_candidates import validate_candidate_training_budget
        for route in unit['routes']: validate_candidate_training_budget(result['training'], phase='review', route=route)
    verify(result['files'])
    from chronaris.evaluation.application_tasks.chapter5_subset_checkpoints import validate_checkpoints
    validate_checkpoints(result, unit)


def freeze_subset(root):
    root = Path(root); plan = verified_subset(root); files = dict(plan['input_files'])
    receipts = []
    for unit in plan['units']:
        path = root/'units'/f"{unit['index']}.json"
        if not path.is_file(): raise ValueError(f"subset incomplete: missing unit {unit['index']}")
        result = read(path); validate_unit(result, unit, plan)
        files.update(result['files']); bind(files, path, sha(path)); receipts.append(str(path))
    verify(files)
    return _immutable(root/'subset_models.json', dict(status='completed', subset_models_frozen=True,
        plan_sha256=plan['plan_sha256'], source_code_sha256=plan['source_code_sha256'],
        unit_receipts=receipts, files=files, confirmation_opened=False, legacy_core_models_frozen=False))


def require_subset_freeze(root):
    path = Path(root)/'subset_models.json'
    if not path.is_file(): raise ValueError('subset evaluation requires its complete model freeze')
    gate = read(path)
    # Reconstruct every inventory entry; a handwritten boolean cannot open evaluation.
    if freeze_subset(root) != gate: raise ValueError('subset freeze changed')
    return gate
