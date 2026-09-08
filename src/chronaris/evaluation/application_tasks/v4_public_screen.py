"""Bounded public first-fold screening from completed simulation-only selections."""
from pathlib import Path
import hashlib
import json

from chronaris.evaluation.application_tasks.v4_candidate_results import collect_simulation_screen
from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES, run_candidate_development
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

ROUTES=('self_supervised','task_guided')
METHODS=('chronaris','physiology_only','vehicle_only','mult','contiformer')


def build_public_screen_plan(*, diagnostic_root, pressure_root, registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    summaries={(method,route):collect_simulation_screen(diagnostic_root=diagnostic_root,pressure_root=pressure_root,
               method=method,route=route) for method in METHODS for route in ROUTES}
    pending={f'{method}/{route}':summary['status'] for (method,route),summary in summaries.items()
             if summary['status']!='development_screen_complete_not_final_adoption'}
    base=dict(format='chronaris.v4_public_first_fold_plan.v1',source_code_sha256=v4_workflow_source_sha256(),
              public_registry_sha256=sha256_file(registry_path),seed=17,fold_index=0,phase='screen',
              simulation_diagnostic_root=str(diagnostic_root),simulation_pressure_root=str(pressure_root),
              confirmation_feedback_used=False,pending=pending,units=[])
    if pending:
        return base | dict(status='blocked_by_simulation_or_pressure_failure'
                           if any(status.startswith('blocked_') for status in pending.values())
                           else 'waiting_for_simulation_and_pressure')
    if any(summary['confirmation_feedback_used'] or summary['seed']!=17 for summary in summaries.values()):
        raise ValueError('public screening selection used unapproved evidence')
    if len({s['data_manifest_sha256'] for s in summaries.values()})!=1 or len({s['cohort_source_code_sha256'] for s in summaries.values()})!=1:
        raise ValueError('public selection requires one shared simulation cohort')
    selected={route:list(summaries[('chronaris',route)]['advance_to_public_development']) for route in ROUTES}
    for names in selected.values():
        if not names or len(names)>3 or len(set(names))!=len(names) or not set(names)<=set(CANDIDATE_CHANGES):
            raise ValueError('public screening exceeded the fixed three-candidate shortlist')
    units=[]
    for domain in ('cogpilot','clare'):
        for method in METHODS:
            for candidate in (CANDIDATE_CHANGES if method=='chronaris' else ('reference','capacity64')):
                routes=[route for route in ROUTES if method!='chronaris' or candidate=='reference' or candidate in selected[route]]
                if not routes:continue
                if method!='chronaris' and any(candidate not in {row['candidate'] for row in summaries[(method,route)]['completed']} for route in routes):
                    raise ValueError('baseline capacity candidate did not complete its simulation screen')
                purposes={route:('repaired_reference_comparator' if candidate=='reference' and candidate not in selected[route]
                                else 'shortlisted_candidate' if method=='chronaris' else 'baseline_capacity_comparison') for route in routes}
                units.append(dict(domain=domain,method=method,candidate_name=candidate,routes=routes,purposes=purposes,
                    fold_index=0,seed=17,phase='screen',pretraining_updates=300,
                    head_warmup_updates=50 if 'task_guided' in routes else 0,joint_updates=200 if 'task_guided' in routes else 0))
    plan=base | dict(status='ready_for_public_development',selected_chronaris_candidates=selected,units=units,
                     simulation_summaries={f'{m}/{r}':s for (m,r),s in summaries.items()},
                     shared_initialization='one_pretraining_per_domain_method_candidate_fold_seed')
    plan=json.loads(json.dumps(plan))
    plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    return plan


def run_public_screen(*, output_root, diagnostic_root, pressure_root,
                      data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                      registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    plan=build_public_screen_plan(diagnostic_root=diagnostic_root,pressure_root=pressure_root,registry_path=registry_path)
    if plan['status']!='ready_for_public_development':return plan
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    path=root/'selection_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('frozen public screening selections or source changed')
    if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
    state_path=root/'run_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(plan_sha256=plan['plan_sha256'],completed_units=[],failed_units=[])
    if state['plan_sha256']!=plan['plan_sha256']:raise ValueError('public screen state differs from frozen plan')
    def save():
        temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    for unit in plan['units']:
        key='/'.join(unit[k] for k in ('domain','method','candidate_name'))
        if key in state['completed_units'] or key in state['failed_units']:continue
        state.update(status='running',current_unit=key);save()
        try:
            result=run_candidate_development(**{k:unit[k] for k in ('domain','method','candidate_name','routes','fold_index','seed','phase')},
                output_root=root,data_root=data_root,registry_path=registry_path)
            if not result['completed']:raise ValueError('public candidate returned incomplete training/evaluation')
            state['completed_units'].append(key)
        except Exception as error:
            state['failed_units'].append(key)
            state.setdefault('errors',{})[key]=f'{type(error).__name__}: {error}'
        save()
    state.update(status='completed_with_failures' if state['failed_units'] else 'completed',current_unit=None);save()
    return state
