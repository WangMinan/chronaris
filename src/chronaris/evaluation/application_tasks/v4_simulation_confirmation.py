"""Frozen simulation training and model inventory, before generating G2 observations."""
import json
import hashlib
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.v4_candidates import (
    EXPANDED_SIMULATION_ROOT, candidate_options, validate_candidate_training_budget)
from chronaris.evaluation.application_tasks.v4_confirmation_training import (
    read_frozen_configuration, _train_confirmation_unit, _confirmation_training_inputs)
from chronaris.evaluation.application_tasks.v4_development_data import development_normalization, v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
from chronaris.evaluation.application_tasks.v4_naive_baseline import fit_v4_naive_encoder, load_v4_naive_encoder
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

SIMULATION_REGISTRY = 'docs/requirements/thesis-v4-simulation-manifest.json'


def simulation_confirmation_units(frozen):
    if frozen['domain_status'].get('simulation') != 'enabled':
        raise ValueError('simulation confirmation is not enabled')
    units = []
    for method in sorted(frozen['methods']['self_supervised']):
        for candidate in sorted({frozen['methods'][route][method]['name'] for route in ('self_supervised', 'task_guided')}):
            routes = [route for route in ('self_supervised', 'task_guided')
                      if frozen['methods'][route][method]['name'] == candidate]
            for seed in (17, 29, 43):
                units.append(dict(method=method, candidate_name=candidate, seed=seed, routes=routes))
    return units


def simulation_unit_root(root, unit):
    return Path(root)/'simulation'/unit['method']/unit['candidate_name']/'fold01'/f"seed{unit['seed']}"


def train_simulation_confirmation(*, freeze_path, freeze_sha256, method, candidate_name, seed, output_root,
                                  simulation_root=EXPANDED_SIMULATION_ROOT):
    frozen = read_frozen_configuration(freeze_path, freeze_sha256)
    matches = [unit for unit in simulation_confirmation_units(frozen)
               if (unit['method'], unit['candidate_name'], unit['seed']) == (method, candidate_name, seed)]
    if len(matches) != 1:
        raise ValueError('simulation unit was not selected before confirmation')
    unit = matches[0]
    root = simulation_unit_root(output_root, unit)
    if method == 'naive_time_sync':
        provider, schema, fold, _, digest, _, _, _ = _confirmation_training_inputs(
            'simulation', 0, simulation_root, SIMULATION_REGISTRY, root)
        normalizer, _ = development_normalization('simulation', provider, schema, fold, digest)
        checkpoint, fit = fit_v4_naive_encoder(provider=provider, fold=fold, normalizer=normalizer,
            data_manifest_sha256=digest, output_root=root/'checkpoint', seed=seed)
        result = dict(completed=True, status='completed', freeze_sha256=freeze_sha256,
            source_code_sha256=v4_workflow_source_sha256(), unit=unit, fold=fold.to_dict(),
            data_manifest_sha256=digest, checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
            training=fit, neural_optimizer_updates=0)
        (root/'training_complete.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    options = candidate_options(method, candidate_name)
    if any(frozen['methods'][route][method] != json.loads(json.dumps(options)) for route in unit['routes']):
        raise ValueError('simulation options differ from the frozen configuration')
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu', completed=False)
        _require_diagnostic_device(seed)
        result = _train_confirmation_unit(domain='simulation', fold_index=0, method=method,
            options=options, routes=unit['routes'], seed=seed, output_root=output_root, data_root=simulation_root,
            registry_path=SIMULATION_REGISTRY, freeze_sha256=freeze_sha256, normalizer_root=frozen['normalizer_root'])
        receipt=dict(status='completed',completed=True,freeze_sha256=freeze_sha256,training=result)
        (root/'training_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt


def seal_simulation_models(*, freeze_path, freeze_sha256, output_root, simulation_root=EXPANDED_SIMULATION_ROOT,
                           ablation_units=None):
    """Require all selected methods/routes/seeds, not merely a frozen architecture."""
    frozen = read_frozen_configuration(freeze_path, freeze_sha256)
    units = simulation_confirmation_units(frozen) if ablation_units is None else ablation_units
    pending = []; records = []; files = {}
    for unit in units:
        root = simulation_unit_root(output_root, unit)
        path = root/('training_complete.json' if unit['method'] == 'naive_time_sync' else 'run_state.json')
        if not path.exists():
            pending.append(unit); continue
        state = json.loads(path.read_text())
        if not state.get('completed'):
            pending.append(unit); continue
        source = state if unit['method'] == 'naive_time_sync' else state['source']
        if source['freeze_sha256'] != freeze_sha256 or source['source_code_sha256'] != v4_workflow_source_sha256():
            raise ValueError('simulation training source changed before model freeze')
        if unit['method'] == 'naive_time_sync':
            from chronaris.representation import FoldLineage
            fold = FoldLineage(state['fold']['fold_id'], *(tuple(state['fold'][role+'_sample_ids'])
                for role in ('train', 'validation', 'held_out')), development_only=state['fold'].get('development_only', False))
            _, _, payload = load_v4_naive_encoder(state['checkpoint'], fold=fold, data_manifest_sha256=state['data_manifest_sha256'])
            if state['unit'] != unit or payload['seed'] != unit['seed'] or sha256_file(state['checkpoint']) != state['checkpoint_sha256']:
                raise ValueError('simulation fixed projection changed before model freeze')
            checkpoints = {route: state['checkpoint'] for route in unit['routes']}
        else:
            if (source['method'] != unit['method'] or source['seed'] != unit['seed'] or source['routes'] != unit['routes']
                or source['candidate_options'] != json.loads(json.dumps(unit['options'] if ablation_units is not None
                    else candidate_options(unit['method'], unit['candidate_name'])))):
                raise ValueError('simulation model does not match the selected method, seed or options')
            checkpoints = {}
            for route in unit['routes']:
                validate_candidate_training_budget(state, phase='review', route=route)
                checkpoint = state[route+'_training']['best_checkpoint_path']
                payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if (payload['training_status'] != 'completed' or payload['seed'] != unit['seed']
                    or payload['method_name'] != unit['method'] or payload['config']['device'] != 'cuda'
                    or payload['config']['data_manifest_sha256'] != source['data_manifest_sha256']
                    or payload['label_used_for_encoder_training'] is not (route == 'task_guided')):
                    raise ValueError('simulation checkpoint is not a completed selected CUDA model')
                if route == 'self_supervised' and payload['fold'] != source['fold']:
                    raise ValueError('simulation checkpoint data roles changed')
                if route == 'task_guided' and (payload['fold_id'] != source['fold']['fold_id']
                    or payload['role_sample_ids'] != {role: source['fold'][role+'_sample_ids'] for role in ('train','validation','held_out')}):
                    raise ValueError('simulation guided checkpoint data roles changed')
                checkpoints[route] = checkpoint
        files[str(path)] = sha256_file(path)
        for checkpoint in checkpoints.values():
            files[str(checkpoint)] = sha256_file(checkpoint)
        records.append(unit | dict(checkpoints=checkpoints, data_manifest_sha256=source['data_manifest_sha256'],
                                   fold=source['fold']))
    if pending:
        return dict(status='waiting_for_simulation_models', pending=pending, confirmation_generated=False)
    if len({record['data_manifest_sha256'] for record in records}) != 1 or len({json.dumps(record['fold'], sort_keys=True) for record in records}) != 1:
        raise ValueError('simulation methods did not share the same data and fold')
    audit_path=Path(simulation_root)/'v4_generation_audit.json'
    digest=hashlib.sha256((sha256_file(SIMULATION_REGISTRY)+sha256_file(audit_path)).encode()).hexdigest()
    if records[0]['data_manifest_sha256']!=digest:
        raise ValueError('simulation model training inventory differs from the selected data root')
    for path in (audit_path,Path(simulation_root)/'simulation_manifest.json'):
        files[str(path)]=sha256_file(path)
    if ablation_units is None and frozen.get('core_ablation_scope_required',False):
        from chronaris.evaluation.application_tasks.v4_core_ablations import seal_core_ablation_models
        core=seal_core_ablation_models(freeze_path=freeze_path,freeze_sha256=freeze_sha256,output_root=output_root,
                                       simulation_root=simulation_root)
        if core['status']!='frozen':return dict(status='waiting_for_core_ablation_models',confirmation_generated=False)
        core_path=Path(output_root)/'core_ablations/ablation_frozen_models.json'
        files.update(core['files']);files[str(core_path)]=sha256_file(core_path)
    result = dict(format='chronaris.v4_simulation_model_freeze.v1' if ablation_units is None else 'chronaris.v4_simulation_ablation_model_freeze.v1', status='frozen', freeze_sha256=freeze_sha256,
        source_code_sha256=v4_workflow_source_sha256(), records=records, files=files,
        simulation_registry_sha256=sha256_file(SIMULATION_REGISTRY), simulation_root=str(simulation_root),
        evaluation_units=sum(len(record['routes']) for record in records), confirmation_generated=False)
    path = Path(output_root)/('simulation_frozen_models.json' if ablation_units is None else 'ablation_frozen_models.json')
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('previously frozen simulation models cannot be overwritten')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2)+'\n')
    return result


def run_simulation_confirmation_cohort(*, freeze_path, freeze_sha256, output_root, backend, stage,
                                      confirmation_root='artifacts/application_evaluation/2026-09-08_v4-simulation-confirmation'):
    from chronaris.evaluation.application_tasks.v4_native_confirmation_cohort import run_confirmation_units
    if stage not in ('train','evaluate') or backend not in ('neural','nonparametric'):
        raise ValueError('unknown simulation confirmation stage or backend')
    frozen=read_frozen_configuration(freeze_path,freeze_sha256)
    units=[unit | dict(domain='simulation',fold_index=0,
        backend='nonparametric' if stage=='train' and unit['method']=='naive_time_sync' else 'neural')
        for unit in simulation_confirmation_units(frozen)]
    models=Path(output_root)/'simulation_frozen_models.json'
    model_hash=sha256_file(models) if stage=='evaluate' else None
    plan=dict(format='chronaris.v4_simulation_confirmation_plan.v1',freeze_sha256=freeze_sha256,
              stage=stage,units=units,model_freeze_sha256=model_hash,confirmation_root=str(confirmation_root))
    def unit_args(unit):
        result=[f'simulation-confirmation-{stage}','--domain','simulation','--method',unit['method'],
                '--candidate-name',unit['candidate_name'],'--seed',str(unit['seed']),
                '--simulation-root',EXPANDED_SIMULATION_ROOT]
        if stage=='evaluate':
            result+=['--model-freeze-path',str(models),'--model-freeze-sha256',model_hash,'--data-root',str(confirmation_root)]
        return result
    return run_confirmation_units(plan=plan,freeze_path=freeze_path,freeze_sha256=freeze_sha256,output_root=output_root,
        backend=backend,queue_prefix='simulation_'+stage,unit_args=unit_args,
        result_name='training_complete.json' if stage=='train' else 'confirmation_unit.json')
