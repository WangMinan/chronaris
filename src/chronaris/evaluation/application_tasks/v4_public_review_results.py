"""Verified subject-level public results for the bounded three-seed review."""
import json
from pathlib import Path
from statistics import mean

import numpy as np

from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_public_results import TASKS, read_public_candidate_unit
from chronaris.evaluation.application_tasks.v4_review_plan import load_verified_review_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def aggregate_review_subject_rows(rows, expected_subjects):
    tasks={(domain,task,metric) for domain,task,metric,_ in TASKS if domain!='simulation'}
    values={};configurations=set()
    for row in rows:
        configuration=tuple(row[k] for k in ('method','route','candidate'))
        key=configuration+tuple(row[k] for k in ('seed','domain','task','metric'))
        if (row['role']!='validation' or row['seed'] not in (17,29,43) or row['fold_index'] not in (0,1,2)
            or tuple(row[k] for k in ('domain','task','metric')) not in tasks or not np.isfinite(row['value'])):
            raise ValueError('review contains an unapproved subject metric, seed or role')
        subjects=values.setdefault(key,{})
        if row['subject'] in subjects:raise ValueError('review subject appears in more than one development fold')
        subjects[row['subject']]=row['value'];configurations.add(configuration)
    expected={configuration+(seed,domain,task,metric) for configuration in configurations
              for seed in (17,29,43) for domain,task,metric in tasks}
    if not expected or set(values)!=expected:raise ValueError('review requires all public tasks and three seeds')
    result=[]
    for key,subjects in sorted(values.items()):
        if set(subjects)!=set(expected_subjects[key[4]]):raise ValueError('review does not cover every fixed development subject')
        result.append(dict(zip(('method','route','candidate','seed','domain','task','metric'),key)) |
                      dict(role='validation',value=mean(subjects.values()),subject_count=len(subjects)))
    return result


def collect_public_review_results(*, output_root,
                                 data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                                 registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    root=Path(output_root);path=root/'selection_plan.json'
    base=dict(scope='public_three_seed_development_only',subject_rows=[],task_rows=[],units=[],pending=[],failed=[],
              confirmation_feedback_used=False,requires_simulation_and_pressure_review=True)
    plan=load_verified_review_plan(root,data_root=data_root,registry_path=registry_path)
    if plan is None:return base | dict(status='waiting_for_three_seed_review_plan')
    state_path=root/'run_state.json'
    if not state_path.exists():return base | dict(status='waiting_for_three_seed_review_state')
    state=json.loads(state_path.read_text())
    if state['plan_sha256']!=plan['plan_sha256']:raise ValueError('review state differs from selected plan')
    inputs={};files={str(p):sha256_file(p) for p in (path,state_path)}
    pretraining_updates=supervised_updates=0
    for unit in plan['units']:
        domain,method,candidate=(unit[k] for k in ('domain','method','candidate_name'))
        if domain=='simulation':continue
        fold_index,seed=unit['fold_index'],unit['seed']
        key=f'{domain}/{method}/{candidate}/fold{fold_index+1:02d}/seed{seed}'
        unit_root=root/domain/method/candidate/'review'/f'seed{seed}'/f'fold{fold_index+1:02d}'
        unit_state_path=unit_root/'run_state.json'
        saved=json.loads(unit_state_path.read_text()) if unit_state_path.exists() else {}
        counted=False
        for route in unit['routes']:
            update=1500 if route=='self_supervised' else 500
            identity=f'{key}/{route}'
            if f'{route}:{update}' not in saved.get('completed_consumers',[]):
                if key in state['completed_units']:raise ValueError('completed review unit lacks a selected route result')
                base['failed' if key in state['failed_units'] else 'pending'].append(identity)
                continue
            data_key=(domain,fold_index)
            if data_key not in inputs:inputs[data_key]=load_development_inputs(domain,data_root,registry_path,fold_index=fold_index)
            audit=read_public_candidate_unit(unit_root=unit_root,saved=saved,unit=unit,route=route,
                source_code_sha256=plan['source_code_sha256'],inputs=inputs[data_key])
            base['subject_rows'].extend(audit['subject_rows']);files.update(audit['files'])
            base['units'].append(dict(unit=key,route=route,training_updates=audit['training_updates'],
                                     metadata=audit['metadata'],pretraining_shared_between_routes=True))
            if not counted:pretraining_updates+=audit['training_updates']['pretraining'];counted=True
            supervised_updates+=audit['training_updates']['supervised']
    status='blocked_by_public_review_execution_failure' if base['failed'] else 'waiting_for_public_review_units'
    if not base['failed'] and not base['pending']:
        registry=json.loads(Path(registry_path).read_text())
        base['task_rows']=aggregate_review_subject_rows(base['subject_rows'],
            {domain:registry['domains'][domain]['development_subjects'] for domain in ('cogpilot','clare')})
        status='public_review_verified_not_final_selection'
    return base | dict(status=status,files=files,selection_plan_sha256=plan['plan_sha256'],
                       audited_pretraining_updates=pretraining_updates,audited_supervised_updates=supervised_updates)
