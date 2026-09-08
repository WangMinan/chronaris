"""Serial, resumable eight-condition pressure for the frozen review cohort."""
import json
import os
import time
from pathlib import Path

from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pressure_run import run_development_pressure
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock, seal_development_plan
from chronaris.evaluation.application_tasks.v4_review_plan import load_verified_review_plan
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def build_review_pressure_plan(*, diagnostic_root, condition_root,
                              data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                              registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    plan=load_verified_review_plan(diagnostic_root,data_root=data_root,registry_path=registry_path)
    base=dict(format='chronaris.v4_review_pressure_queue.v1',units=[],pending=[],failed=[],
              scope='eight_condition_development_only',confirmation_feedback_used=False,consumer_refit=False)
    if plan is None:return base | dict(status='waiting_for_three_seed_review_plan')
    root=Path(diagnostic_root);path=root/'run_state.json'
    if not path.exists():return base | dict(status='waiting_for_three_seed_review_state')
    state=json.loads(path.read_text())
    if state['plan_sha256']!=plan['plan_sha256']:raise ValueError('pressure cohort differs from its review selection')
    for unit in plan['units']:
        if unit['domain']!='simulation':continue
        method,candidate,seed=(unit[k] for k in ('method','candidate_name','seed'))
        key=f'simulation/{method}/{candidate}/fold01/seed{seed}'
        saved_path=root/'simulation'/method/candidate/'review'/f'seed{seed}'/'run_state.json'
        saved=json.loads(saved_path.read_text()) if saved_path.exists() else {}
        for route in unit['routes']:
            update=1500 if route=='self_supervised' else 500
            identity=f'{method}/{candidate}/seed{seed}/{route}'
            if f'{route}:{update}' not in saved.get('completed_consumers',()):
                if key in state['completed_units']:raise ValueError('completed review unit lacks its pressure consumer')
                base['failed' if key in state['failed_units'] else 'pending'].append(identity)
            else:
                base['units'].append(dict(method=method,candidate_name=candidate,seed=seed,route=route,update=update,
                    checkpoint_sha256=sha256_file(saved[route+'_training']['best_checkpoint_path'])))
    if base['failed'] or state.get('status')=='failed':return base | dict(status='blocked_by_review_execution_failure')
    if base['pending']:return base | dict(status='waiting_for_simulation_review_units')
    if not base['units']:raise ValueError('review pressure requires the fixed simulation cohort')
    return seal_development_plan(base | dict(status='ready_for_review_pressure',source_code_sha256=v4_workflow_source_sha256(),
        review_selection_plan_sha256=plan['plan_sha256'],diagnostic_root=str(diagnostic_root),condition_root=str(condition_root),
        condition_audit_sha256=sha256_file(Path(condition_root)/'development_condition_audit.json')))


def run_review_pressure_cohort(*, output_root, diagnostic_root, condition_root,
                              data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                              registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    plan=build_review_pressure_plan(diagnostic_root=diagnostic_root,condition_root=condition_root,
                                  data_root=data_root,registry_path=registry_path)
    if plan['status']!='ready_for_review_pressure':return plan
    with development_gpu_lock() as acquired:
        if not acquired:return plan | dict(status='waiting_gpu')
        root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
        plan_path=root/'pressure_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('frozen review pressure plan changed')
        if not plan_path.exists():plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        path=root/'queue_state.json'
        state=json.loads(path.read_text()) if path.exists() else dict(plan_sha256=plan['plan_sha256'],completed_units=[],failed_units=[],errors={})
        if state['plan_sha256']!=plan['plan_sha256']:raise ValueError('review pressure queue source changed')
        state['pid']=os.getpid()
        def save():
            state['updated_at_unix_s']=time.time()
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
        with _periodic_training_heartbeat('review_pressure_cohort',30,root=root) as progress:
            for unit in plan['units']:
                key=f"{unit['method']}/{unit['candidate_name']}/seed{unit['seed']}/{unit['route']}"
                if key in state['completed_units'] or key in state['failed_units']:continue
                state.update(status='running',current_unit=key);progress.update(current_unit=key,pid=os.getpid());save()
                try:
                    result=run_development_pressure(**{k:unit[k] for k in ('method','candidate_name','seed','route','update')},
                        output_root=root,diagnostic_root=diagnostic_root,condition_root=condition_root,phase='review',device='cuda')
                    if not result['completed'] or result['source']['checkpoint_sha256']!=unit['checkpoint_sha256']:
                        raise ValueError('pressure unit incomplete or selected checkpoint changed')
                    state['completed_units'].append(key)
                except Exception as error:
                    state['failed_units'].append(key);state['errors'][key]=f'{type(error).__name__}: {error}'
                progress.update(completed_units=len(state['completed_units']),failed_units=len(state['failed_units']));save()
            state.update(status='completed_with_failures' if state['failed_units'] else 'completed',current_unit=None)
            progress.update(status=state['status'],current_unit=None);save()
        return state
