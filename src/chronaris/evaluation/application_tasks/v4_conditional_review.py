"""One predeclared decay-cell comparison, activated only by development evidence."""
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.v4_public_screen import ROUTES, seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def build_conditional_review_plan(*, parent_root, parent_pressure_root, data_root, registry_path):
    from chronaris.evaluation.application_tasks.v4_review_plan import load_verified_review_plan
    from chronaris.evaluation.application_tasks.v4_adoption import collect_adoption_decisions
    from chronaris.evaluation.application_tasks.v4_simulation_review_results import collect_simulation_review_results
    from chronaris.evaluation.application_tasks.v4_configuration_freeze import _simulation_freeze_evidence
    parent_root, parent_pressure_root = Path(parent_root), Path(parent_pressure_root)
    parent = load_verified_review_plan(parent_root, data_root=data_root, registry_path=registry_path)
    if parent is None or parent['format'] != 'chronaris.v4_three_seed_review_plan.v1':
        raise ValueError('conditional comparison requires the original completed review; no recursive search')
    adoption = collect_adoption_decisions(output_root=parent_root, pressure_root=parent_pressure_root,
                                        data_root=data_root, registry_path=registry_path)
    if adoption['status'] != 'single_factor_decisions_ready_not_frozen':
        return dict(status='waiting_for_complete_three_seed_review', units=[])
    simulation = collect_simulation_review_results(output_root=parent_root, pressure_root=parent_pressure_root,
                                                   data_root=data_root, registry_path=registry_path)
    files, triggers = _simulation_freeze_evidence(simulation, parent_root, parent_pressure_root,
        {route: adoption['decisions'][f'chronaris/{route}']['recommended_candidate'] for route in ROUTES})
    if not triggers:
        return dict(status='not_applicable', units=[], instability=[])
    routes = [route for route in ROUTES if any(row['route'] == route for row in triggers)]
    units = list(parent['units'])
    for domain, folds in (('simulation', range(1)), ('cogpilot', range(3)), ('clare', range(3))):
        for fold in folds:
            for seed in (17, 29, 43):
                units.append(dict(domain=domain, method='chronaris', candidate_name='analytic_decay',
                    routes=routes, purposes={route: 'conditional_continuous_cell' for route in routes},
                    fold_index=fold, seed=seed, phase='review', pretraining_updates=1500,
                    head_warmup_updates=50 if 'task_guided' in routes else 0,
                    joint_updates=500 if 'task_guided' in routes else 0,
                    prefetch_cpu_consumers=domain == 'simulation' and len(routes) == 2))
    files.update(simulation['files'])
    files[str(parent_root/'selection_plan.json')] = sha256_file(parent_root/'selection_plan.json')
    base = {key: value for key, value in parent.items() if key != 'plan_sha256'}
    return seal_development_plan(base | dict(format='chronaris.v4_conditional_review_plan.v1',
        status='ready_for_three_seed_review', parent_root=str(parent_root), parent_pressure_root=str(parent_pressure_root),
        instability=triggers, trigger_files=files, units=units,
        conditional_candidate='analytic_decay', conditional_search_limit=1))


def prepare_conditional_review(plan, *, output_root, pressure_root):
    """Reuse completed immutable unit directories; never rewrite the parent plan/state."""
    if plan['format'] != 'chronaris.v4_conditional_review_plan.v1':
        raise ValueError('expected a triggered conditional review')
    for filename, digest in plan['trigger_files'].items():
        if sha256_file(filename) != digest:
            raise ValueError('conditional trigger evidence changed')
    root, pressure = Path(output_root), Path(pressure_root)
    parent, parent_pressure = Path(plan['parent_root']), Path(plan['parent_pressure_root'])
    if root.resolve() == parent.resolve() or pressure.resolve() == parent_pressure.resolve():
        raise ValueError('conditional review must preserve its parent run')
    root.mkdir(parents=True, exist_ok=True)
    saved_plan = root/'selection_plan.json'
    if saved_plan.exists() and json.loads(saved_plan.read_text()) != plan:
        raise ValueError('conditional selection changed')
    if not saved_plan.exists():
        saved_plan.write_text(json.dumps(plan, indent=2)+'\n')
    for unit in plan['units']:
        if unit['candidate_name'] == 'analytic_decay':
            continue
        relative = Path(unit['domain'])/unit['method']/unit['candidate_name']
        pairs = [(root/relative, parent/relative)]
        if unit['domain'] == 'simulation':
            relative = Path(unit['method'])/unit['candidate_name']
            pairs.append((pressure/relative, parent_pressure/relative))
        for target, source in pairs:
            if not source.is_dir():
                raise ValueError(f'missing completed parent unit: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError('conditional result alias changed')
            else:
                target.symlink_to(source.resolve(), target_is_directory=True)
    path = root/'run_state.json'
    if not path.exists():
        state = json.loads((parent/'run_state.json').read_text())
        if state['status'] != 'completed' or state['failed_units']:
            raise ValueError('cannot inherit failed parent units')
        state.update(plan_sha256=plan['plan_sha256'], status='pending', current_unit=None)
        path.write_text(json.dumps(state, indent=2)+'\n')


def inherit_completed_pressure(plan, *, output_root):
    """Seed only verified parent completions after the new pressure plan is frozen."""
    if not plan.get('inherited_pressure_root'):
        return
    path = Path(output_root)/'queue_state.json'
    if path.exists():
        return
    state = json.loads((Path(plan['inherited_pressure_root'])/'queue_state.json').read_text())
    if state['status'] != 'completed' or state['failed_units']:
        raise ValueError('conditional pressure cannot reuse failed parent units')
    state.update(plan_sha256=plan['plan_sha256'], status='pending', current_unit=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2)+'\n')
