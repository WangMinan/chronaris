"""Three-seed simulation review from the same clean and pressure evidence readers."""
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.v4_candidate_results import _completed_scores, _pressure_p95, PRIMARY_METRICS
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_review_plan import load_verified_review_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def collect_simulation_review_results(*, output_root, pressure_root,
                                     data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                                     registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    root=Path(output_root)
    base=dict(scope='simulation_three_seed_development_only',completed=[],task_rows=[],pending=[],failed=[],
              pressure_pending=[],pressure_failed=[],confirmation_feedback_used=False,final_selection=False)
    plan=load_verified_review_plan(root,data_root=data_root,registry_path=registry_path)
    if plan is None:return base | dict(status='waiting_for_three_seed_review_plan')
    path=root/'run_state.json'
    if not path.exists():return base | dict(status='waiting_for_three_seed_review_state')
    queue=json.loads(path.read_text())
    if queue['plan_sha256']!=plan['plan_sha256']:raise ValueError('simulation review state differs from selected plan')
    pressure_path=Path(pressure_root)/'queue_state.json'
    pressure_queue=json.loads(pressure_path.read_text()) if pressure_path.exists() else {}
    files={str(p):sha256_file(p) for p in (root/'selection_plan.json',path)}
    if pressure_path.exists():files[str(pressure_path)]=sha256_file(pressure_path)
    data_hash=None
    for unit in plan['units']:
        if unit['domain']!='simulation':continue
        method,candidate,seed=(unit[k] for k in ('method','candidate_name','seed'))
        key=f'simulation/{method}/{candidate}/fold01/seed{seed}'
        unit_root=root/'simulation'/method/candidate/'review'/f'seed{seed}'
        state_path=unit_root/'run_state.json'
        saved=json.loads(state_path.read_text()) if state_path.exists() else {}
        for route in unit['routes']:
            update=1500 if route=='self_supervised' else 500
            identity=f'{method}/{candidate}/seed{seed}/{route}'
            if f'{route}:{update}' not in saved.get('completed_consumers',[]):
                if key in queue['completed_units']:raise ValueError('completed simulation review lacks a selected route result')
                base['failed' if key in queue['failed_units'] else 'pending'].append(identity)
                continue
            if (saved['source_code_sha256']!=plan['source_code_sha256'] or saved['confirmation_opened']
                or saved['candidate_options']!=json.loads(json.dumps(candidate_options(method,candidate)))
                or '__training512' not in saved['fold']['fold_id']):
                raise ValueError('simulation review source, candidate or data role changed')
            if data_hash is not None and saved['data_manifest_sha256']!=data_hash:
                raise ValueError('simulation review candidates do not share the expanded data')
            data_hash=saved['data_manifest_sha256']
            record=_completed_scores(unit_root,saved,route,update,method,candidate,phase='review',seed=seed)
            record.update(method=method,route=route,seed=seed)
            record['missingness_p95']=_pressure_p95(pressure_root,saved,record,method,candidate,route,update,phase='review',seed=seed)
            if record['missingness_p95'] is None:
                base['pressure_failed' if identity in pressure_queue.get('failed_units',()) else 'pressure_pending'].append(identity)
            base['completed'].append(record);files[str(state_path)]=sha256_file(state_path)
            for (task,_,metric,_),value in zip(PRIMARY_METRICS,record['scores'],strict=True):
                base['task_rows'].append(dict(method=method,route=route,candidate=candidate,seed=seed,domain='simulation',
                                             task=task,metric=metric,role='validation',value=value))
    blocked=base['failed'] or base['pressure_failed'] or queue.get('status')=='failed' or pressure_queue.get('status')=='failed'
    status=('blocked_by_simulation_review_execution_failure' if blocked else 'waiting_for_simulation_review_units'
            if base['pending'] else 'waiting_for_simulation_review_pressure' if base['pressure_pending']
            else 'simulation_review_verified_not_final_selection')
    return base | dict(status=status,files=files,data_manifest_sha256=data_hash,selection_plan_sha256=plan['plan_sha256'])
