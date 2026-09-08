"""Verified public first-fold scores and domain-balanced candidate rankings."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.stats import rankdata

from chronaris.evaluation.application_tasks.v4_candidate_results import PRIMARY_METRICS, collect_simulation_screen
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
from chronaris.evaluation.application_tasks.v4_native_result_audit import audit_native_consumer_result
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

TASKS = [('simulation',task,metric,direction) for task,_,metric,direction in PRIMARY_METRICS] + [
    ('cogpilot','difficulty','macro_f1','higher'),('cogpilot','event_response','rmse','lower'),
    ('clare','workload_classification','macro_f1','higher'),('clare','workload_regression','rmse','lower')]


def rank_development_rows(rows, metadata, *, seeds=(17,)):
    """Rank tasks within each domain, domains equally, then take median seed ranks."""
    if seeds not in ((17,),(17,29,43)):
        raise ValueError('ranking requires the fixed initial or three-seed set')
    candidates=sorted(metadata)
    if not candidates or 'reference' not in candidates:
        raise ValueError('candidate ranking requires its repaired reference')
    if any(not np.isfinite(row[key]) or row[key]<0 for row in metadata.values() for key in ('encoder_parameters','training_elapsed_s')):
        raise ValueError('candidate complexity and training time must be finite and nonnegative')
    expected={(candidate,seed,domain,task,metric) for candidate in candidates for seed in seeds for domain,task,metric,_ in TASKS}
    values={}
    for row in rows:
        key=(row['candidate'],row['seed'],row['domain'],row['task'],row['metric'])
        if key in values or key not in expected or row['role']!='validation' or not np.isfinite(row['value']):
            raise ValueError('ranking contains duplicated, nonfinite or unapproved task/role rows')
        values[key]=row['value']
    if set(values)!=expected:
        raise ValueError('ranking requires every fixed task, domain and seed for every candidate')
    seed_ranks={candidate:[] for candidate in candidates}
    details={candidate:{} for candidate in candidates}
    for seed in seeds:
        domain_scores={domain:[] for domain in ('simulation','cogpilot','clare')}
        for domain,task,metric,direction in TASKS:
            scores=[values[(candidate,seed,domain,task,metric)] for candidate in candidates]
            ranks=rankdata(np.asarray(scores)*(-1 if direction=='higher' else 1),method='average')
            domain_scores[domain].append(ranks)
        domain_scores={domain:np.mean(scores,axis=0) for domain,scores in domain_scores.items()}
        overall=rankdata(np.mean(list(domain_scores.values()),axis=0),method='average')
        for index,candidate in enumerate(candidates):
            seed_ranks[candidate].append(float(overall[index]))
            details[candidate][str(seed)]={domain:float(scores[index]) for domain,scores in domain_scores.items()}
    result=[dict(candidate=candidate,median_seed_rank=float(np.median(seed_ranks[candidate])),
        seed_ranks=seed_ranks[candidate],domain_task_ranks=details[candidate],**metadata[candidate]) for candidate in candidates]
    result.sort(key=lambda row:(row['median_seed_rank'],row['encoder_parameters'],row['training_elapsed_s'],row['candidate']))
    return result


def read_public_candidate_unit(*, unit_root, saved, unit, route, source_code_sha256, inputs):
    """Replay a public unit, preserving subjects and recorded optimizer updates."""
    domain,method,candidate=(unit[k] for k in ('domain','method','candidate_name'))
    phase,seed=unit.get('phase','screen'),unit.get('seed',17)
    if phase not in ('screen','review') or seed not in ((17,) if phase=='screen' else (17,29,43)):
        raise ValueError('unapproved public development phase or seed')
    reviewing=phase=='review'
    update=(1500 if reviewing else 300) if route=='self_supervised' else (500 if reviewing else 200)
    _,_,fold,_,data_hash,targets,definitions,data=inputs
    pretraining=saved['self_supervised_training']
    count=pretraining['optimizer_updates']
    if (saved['method']!=method or saved['domain']!=domain or saved['phase']!=phase
        or saved['candidate_options']!=json.loads(json.dumps(candidate_options(method,candidate)))
        or saved['source_code_sha256']!=source_code_sha256 or saved['data_manifest_sha256']!=data_hash
        or saved['fold']!=fold.to_dict() or saved['seed']!=seed or saved['confirmation_opened']
        or type(count) is not int or (not 500<=count<=1500 if reviewing else count!=300)):
        raise ValueError('public candidate training source, roles or update counts changed')
    if route=='task_guided':
        guided=saved['task_guided_training'];joint=guided['joint_updates']
        if (type(joint) is not int or guided['head_warmup_updates']!=50 or guided['optimizer_updates']!=50+joint
            or (not 200<=joint<=500 if reviewing else joint!=200)):
            raise ValueError('public candidate supervised update counts changed')
    checkpoint=Path(saved[route+'_training']['best_checkpoint_path']);checkpoint_hash=sha256_file(checkpoint)
    representation_root=unit_root/'representations'/route/str(update)
    if route=='task_guided':representation_root/=method
    outputs={role:load_fusion_stream_batch(representation_root/role) for role in ('train','validation')}
    if any(output.checkpoint_sha256!=checkpoint_hash or output.method_name!=method or output.fold_id!=fold.fold_id
           or output.sample_ids!=getattr(fold,role+'_sample_ids') for role,output in outputs.items()):
        raise ValueError('public representations do not use the selected checkpoint')
    result_path=unit_root/f'{route}_{update}_consumers.json'
    result=json.loads(result_path.read_text())
    context=native_consumer_context(domain,data,fold)
    audit=audit_native_consumer_result(result,outputs=outputs,targets=targets,definitions=definitions,context=context,
                                      label_used_for_encoder_training=route=='task_guided')
    if audit['source_code_sha256']!=source_code_sha256:raise ValueError('public consumer source differs from training')
    if any(row['seed']!=seed or row['role']!='validation' for row in audit['metrics']):
        raise ValueError('public consumer seed or evaluation role differs from training')
    files=dict(audit['files']);files.update({str(p):sha256_file(p) for p in (unit_root/'run_state.json',checkpoint,result_path)})
    groups={context['groups'][sample] for sample in fold.validation_sample_ids}
    records=[];subjects=[]
    for expected_domain,task,metric,_ in TASKS:
        if expected_domain!=domain:continue
        scores=[row for row in audit['metrics'] if row['consumer']=='linear' and row['task']==task]
        if (len(scores)!=len(groups) or {r['group_id'] for r in scores}!=groups
            or any(r['status']!='completed' or r['field']!=0 or not np.isfinite(r[metric]) for r in scores)):
            raise ValueError('public task does not cover the fixed validation subjects')
        base=dict(method=method,route=route,candidate=candidate,seed=seed,domain=domain,task=task,
                  metric=metric,role='validation')
        records.append(base | dict(value=float(np.mean([row[metric] for row in scores]))))
        subjects.extend(base | dict(subject=row['group_id'],fold_index=unit.get('fold_index',0),value=float(row[metric])) for row in scores)
    return dict(task_rows=records,subject_rows=subjects,files=files,
        metadata=dict(encoder_parameters=pretraining['parameter_count'],training_elapsed_s=pretraining['training_elapsed_s']
            +(saved['task_guided_training']['training_elapsed_s'] if route=='task_guided' else 0.)),
        training_updates=dict(pretraining=count,supervised=saved['task_guided_training']['optimizer_updates'] if route=='task_guided' else 0))


def collect_public_screen_results(*, output_root, data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                                  registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    root=Path(output_root);plan_path=root/'selection_plan.json'
    if not plan_path.exists():return dict(status='waiting_for_public_screen_plan',rankings={},pending=[],failed=[])
    plan=json.loads(plan_path.read_text());digest=plan.pop('plan_sha256')
    if (hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()!=digest
        or plan['confirmation_feedback_used'] or plan['public_registry_sha256']!=sha256_file(registry_path)):
        raise ValueError('public selection plan or registry changed')
    if not (root/'run_state.json').exists():
        return dict(status='waiting_for_public_run_state',rankings={},pending=[],failed=[])
    for key,expected in plan['simulation_summaries'].items():
        method,route=key.split('/')
        actual=collect_simulation_screen(diagnostic_root=plan['simulation_diagnostic_root'],pressure_root=plan['simulation_pressure_root'],method=method,route=route)
        if json.dumps(actual,sort_keys=True)!=json.dumps(expected,sort_keys=True):
            raise ValueError('frozen simulation screening evidence changed')
    state=json.loads((root/'run_state.json').read_text())
    if state['plan_sha256']!=digest:raise ValueError('public run differs from its selected plan')
    inputs={};records=[];metadata={};pending=[];failed=[];files={str(plan_path):sha256_file(plan_path)}
    route_candidates={}
    for unit in plan['units']:
        domain,method,candidate=(unit[k] for k in ('domain','method','candidate_name'))
        key=f'{domain}/{method}/{candidate}'
        unit_root=root/domain/method/candidate/'fold01'
        unit_state_path=unit_root/'run_state.json'
        saved=json.loads(unit_state_path.read_text()) if unit_state_path.exists() else {}
        for route in unit['routes']:
            route_candidates.setdefault((method,route),set()).add(candidate)
            update=300 if route=='self_supervised' else 200
            identity=f'{key}/{route}'
            if f'{route}:{update}' not in saved.get('completed_consumers',[]):
                if key in state['completed_units']:raise ValueError('completed public unit lacks a selected route result')
                (failed if key in state['failed_units'] else pending).append(identity)
                continue
            if domain not in inputs:inputs[domain]=load_development_inputs(domain,data_root,registry_path,fold_index=0)
            audited=read_public_candidate_unit(unit_root=unit_root,saved=saved,unit=unit,route=route,
                source_code_sha256=plan['source_code_sha256'],inputs=inputs[domain])
            records.extend(audited['task_rows']);files.update(audited['files'])
            entry=metadata.setdefault((method,route,candidate),dict(encoder_parameters=0,training_elapsed_s=0.))
            for name,value in audited['metadata'].items():entry[name]+=value
    rankings={}
    if not pending and not failed:
        for (method,route),candidates in route_candidates.items():
            selected=[row for row in records if row['method']==method and row['route']==route and row['candidate'] in candidates]
            meta={candidate:dict(metadata[(method,route,candidate)]) for candidate in candidates}
            simulation=plan['simulation_summaries'][f'{method}/{route}']
            for candidate in candidates:
                source=next(row for row in simulation['completed'] if row['candidate']==candidate)
                for (task,_,metric,_),score in zip(PRIMARY_METRICS,source['scores'],strict=True):
                    selected.append(dict(candidate=candidate,seed=17,domain='simulation',task=task,metric=metric,role='validation',value=score))
                meta[candidate]['encoder_parameters']+=source['encoder_parameters']
                meta[candidate]['training_elapsed_s']+=source['training_elapsed_s']
            rankings[f'{method}/{route}']=rank_development_rows(selected,meta)
    return dict(status='blocked_by_public_execution_failure' if failed else
                'waiting_for_public_units' if pending else 'public_scores_verified_not_final_adoption',
        selection_plan_sha256=digest,pending=pending,failed=failed,excluded_candidates={},rankings=rankings,
        public_task_rows=records,files=files,ranking_scope='task_mean_then_equal_simulation_cogpilot_clare',
        confirmation_feedback_used=False,requires_three_seed_review=True)
