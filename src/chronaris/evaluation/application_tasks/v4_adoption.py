"""Predeclared three-seed single-factor decisions; no confirmation feedback."""
import hashlib
import json
from collections import defaultdict
from statistics import mean

from chronaris.evaluation.application_tasks.v4_public_results import TASKS, rank_development_rows
from chronaris.evaluation.application_tasks.v4_public_review_results import collect_public_review_results
from chronaris.evaluation.application_tasks.v4_simulation_review_results import collect_simulation_review_results
from chronaris.evaluation.application_tasks.v4_public_screen import METHODS, ROUTES
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options

SEEDS=(17,29,43)


def assess_candidate_adoption(rows, metadata, tails):
    """Use seed-mean task effects; enforce the long-tail ceiling in every seed."""
    rankings=rank_development_rows(rows,metadata,seeds=SEEDS)
    values={(r['candidate'],r['seed'],r['domain'],r['task'],r['metric']):r['value'] for r in rows}
    for (_,_,_,_,metric),value in values.items():
        if value<0 or ('f1' in metric and value>1):raise ValueError('invalid F1 or regression error')
    tail_values={}
    for row in tails:
        key=(row['candidate'],row['seed']);value=row['missingness_p95']
        if key in tail_values or value is None or not 0<=value<float('inf'):
            raise ValueError('missing, duplicated or invalid long-tail error')
        tail_values[key]=value
    if set(tail_values)!={(candidate,seed) for candidate in metadata for seed in SEEDS}:
        raise ValueError('adoption requires all three pressure seeds for every candidate')
    assessments={}
    for candidate in metadata:
        effects=[]
        for domain,task,metric,direction in TASKS:
            reference=[values[('reference',seed,domain,task,metric)] for seed in SEEDS]
            actual=[values[(candidate,seed,domain,task,metric)] for seed in SEEDS]
            ref,score=mean(reference),mean(actual)
            gains=[a-b if direction=='higher' else b-a for a,b in zip(actual,reference)]
            same_direction=sum(value>0 for value in gains)
            if direction=='higher':
                benefit=score-ref>=.01-1e-12 and same_direction>=2
                guard=score>=ref-.02-1e-12
            else:
                benefit=ref>0 and score<=(.98+1e-12)*ref and same_direction>=2
                guard=score<=(1.05+1e-12)*ref
            effects.append(dict(domain=domain,task=task,metric=metric,reference_seed_values=reference,
                candidate_seed_values=actual,reference_seed_mean=ref,candidate_seed_mean=score,
                directional_seed_count=same_direction,benefit=benefit,clean_guard_passed=guard,
                absolute_gain=mean(gains),relative_error_reduction=(ref-score)/ref if direction=='lower' and ref>0 else None))
        tail_checks=[dict(seed=seed,reference_p95=tail_values[('reference',seed)],candidate_p95=tail_values[(candidate,seed)],
                          passed=tail_values[(candidate,seed)]<=(1.2+1e-12)*tail_values[('reference',seed)]) for seed in SEEDS]
        eligible=(candidate=='reference' or any(r['benefit'] for r in effects)) and all(r['clean_guard_passed'] for r in effects) and all(r['passed'] for r in tail_checks)
        assessments[candidate]=dict(eligible=eligible,effects=effects,tail_checks=tail_checks)
    recommendation=next(row['candidate'] for row in rankings if assessments[row['candidate']]['eligible'])
    return dict(rankings=rankings,assessments=assessments,recommended_candidate=recommendation,
        effect_aggregation='equal_seed_mean_after_subject_or_simulation_task_aggregation',
        tail_guard_aggregation='each_seed',configuration_frozen=False)


def collect_adoption_decisions(*, output_root, pressure_root,
                              data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                              registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    kwargs=dict(output_root=output_root,data_root=data_root,registry_path=registry_path)
    public=collect_public_review_results(**kwargs)
    simulation=collect_simulation_review_results(**kwargs,pressure_root=pressure_root)
    statuses=dict(public=public['status'],simulation=simulation['status'])
    base=dict(scope='three_seed_single_factor_development_decisions',upstream_statuses=statuses,decisions={},
              confirmation_feedback_used=False,configuration_frozen=False)
    if (public['status']!='public_review_verified_not_final_selection' or
        simulation['status']!='simulation_review_verified_not_final_selection'):
        return base | dict(status='blocked_by_review_execution_failure' if any(v.startswith('blocked_') for v in statuses.values())
                           else 'waiting_for_complete_three_seed_review')
    if (public['selection_plan_sha256']!=simulation['selection_plan_sha256'] or
        public['confirmation_feedback_used'] or simulation['confirmation_feedback_used']):
        raise ValueError('adoption sources have different selections or confirmation feedback')
    rows=public['task_rows']+simulation['task_rows']
    if {(r['method'],r['route']) for r in rows}!={(m,r) for m in METHODS for r in ROUTES}:
        raise ValueError('adoption requires all methods and both representation routes')
    metadata=defaultdict(lambda:defaultdict(list))
    for item in public['units']:
        domain,method,candidate,_,_=item['unit'].split('/')
        metadata[(method,item['route'],candidate)][domain].append(item['metadata'])
    for item in simulation['completed']:
        metadata[(item['method'],item['route'],item['candidate'])]['simulation'].append(
            {k:item[k] for k in ('encoder_parameters','training_elapsed_s')})
    for method in METHODS:
        for route in ROUTES:
            selected=[row for row in rows if row['method']==method and row['route']==route]
            candidates={row['candidate'] for row in selected};complexity={}
            for candidate in candidates:
                candidate_options(method,candidate)
                domains=metadata[(method,route,candidate)]
                if set(domains)!={'simulation','cogpilot','clare'}:
                    raise ValueError('candidate complexity or training time evidence is incomplete')
                complexity[candidate]=dict(encoder_parameters=sum(mean(r['encoder_parameters'] for r in records) for records in domains.values()),
                    training_elapsed_s=sum(r['training_elapsed_s'] for records in domains.values() for r in records))
            tails=[row for row in simulation['completed'] if row['method']==method and row['route']==route]
            base['decisions'][f'{method}/{route}']=assess_candidate_adoption(selected,complexity,tails)
    return base | dict(status='single_factor_decisions_ready_not_frozen',selection_plan_sha256=public['selection_plan_sha256'],
        source_summaries_sha256={name:hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
                                for name,value in (('public',public),('simulation',simulation))})
