"""Frozen-subset evaluation adapters; never relax the original full-matrix gate."""
from pathlib import Path

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha
from chronaris.evaluation.application_tasks.chapter5_subset import require_subset_freeze, verified_subset


def evaluate_native(root, index, *, reference=False, role='validation', sample_limit=4):
    from chronaris.evaluation.application_tasks.chapter5_native import unit_record, _evaluate_native_record
    if role not in ('validation', 'held_out') or sample_limit < 0:
        raise ValueError('invalid subset evaluation role')
    gate = require_subset_freeze(root); plan = verified_subset(root)
    if role == 'held_out' and sample_limit:
        raise ValueError('subset outer evaluation requires every held-out window')
    if reference:
        if index not in {r['index'] for r in plan['native_references']}:
            raise ValueError('reference outside frozen subset')
        unit = unit_record(plan['main_root'], index)
    else:
        if not 0 <= index < 15: raise ValueError('native unit outside frozen subset')
        unit = read(Path(root)/'units'/f'{index}.json')
    binding = dict(subset_freeze_sha256=sha(Path(root)/'subset_models.json'),
        source_code_sha256=gate['source_code_sha256'], index=index, reference=reference, role=role, sample_limit=sample_limit)
    return _evaluate_native_record(unit, binding, Path(root)/'evaluation'/('reference' if reference else 'new')/str(index)/role,
        role=role, sample_limit=sample_limit, data_root=plan['data_root'], registry_path=plan['registry_path'])


def simulation_records(root):
    import torch
    plan = verified_subset(root)
    results = [read(r['receipt']['path']) for r in plan['simulation_references']]
    results += [read(Path(root)/'units'/f'{i}.json') for i in range(15, 18)]
    records = []
    for result in results:
        unit = result['unit']; naive = unit['method']=='naive_time_sync'
        paths = ({'self_supervised': next(p for p in result['files'] if p.endswith('.pt'))} if naive else
                 {r: result['training'][r+'_training']['best_checkpoint_path'] for r in unit['routes']})
        saved = read(Path(paths['self_supervised']).parent/'fit_manifest.json') if naive else torch.load(
            paths['self_supervised'], map_location='cpu', weights_only=True)
        records.append(dict(method=unit['method'], candidate_name=unit['options']['name'], seed=unit['seed'],
            routes=unit['routes'], checkpoints=paths, fold=saved['fold'], data_manifest_sha256=saved['config']['data_manifest_sha256'],
            training_source_sha256=saved['source_code_sha256']))
    return dict(simulation_root=plan['simulation_root'], records=records)


def generate_simulation(root):
    from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import _generate_frozen_simulation
    gate = require_subset_freeze(root)
    return _generate_frozen_simulation(freeze_sha256=gate['plan_sha256'],
        model_freeze_sha256=sha(Path(root)/'subset_models.json'), output_root=Path(root)/'simulation_confirmation')


def evaluate_simulation(root, index):
    from chronaris.evaluation.application_tasks.v4_simulation_confirmation_evaluation import _evaluate_simulation_unit
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
    from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
    gate = require_subset_freeze(root); models = simulation_records(root)
    if not 0 <= index < len(models['records']): raise ValueError('simulation unit outside frozen subset')
    with development_gpu_lock() as acquired:
        if not acquired: return dict(status='waiting_gpu')
        _require_diagnostic_device(17)
        return _evaluate_simulation_unit(models=models, unit=models['records'][index], freeze_sha256=gate['plan_sha256'],
            model_freeze_sha256=sha(Path(root)/'subset_models.json'), output_root=Path(root)/'simulation_evaluation',
            confirmation_root=Path(root)/'simulation_confirmation', include_pressure=False)
