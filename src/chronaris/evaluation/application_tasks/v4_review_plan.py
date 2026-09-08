"""Fixed three-seed development review, selected from verified first-fold results."""
import hashlib
import json

from chronaris.evaluation.application_tasks.v4_candidates import candidate_options
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_public_results import collect_public_screen_results
from chronaris.evaluation.application_tasks.v4_public_screen import (
    METHODS, ROUTES, run_development_plan, seal_development_plan,
)
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def build_review_plan(*, screen_root, data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                      registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    results=collect_public_screen_results(output_root=screen_root,data_root=data_root,registry_path=registry_path)
    base=dict(format='chronaris.v4_three_seed_review_plan.v1',phase='review',
        source_code_sha256=v4_workflow_source_sha256(),public_registry_sha256=sha256_file(registry_path),
        screen_root=str(screen_root),data_root=str(data_root),confirmation_feedback_used=False,units=[])
    if results['status']!='public_scores_verified_not_final_adoption':
        return base | dict(status=results['status'],pending=results['pending'],failed=results['failed'])
    expected={f'{method}/{route}' for method in METHODS for route in ROUTES}
    if (set(results['rankings'])!=expected or results['confirmation_feedback_used']
        or results['pending'] or results['failed']):
        raise ValueError('review requires complete verified development rankings for both routes and all methods')
    selected={}
    for key,ranking in results['rankings'].items():
        method,_=key.split('/')
        names=[row['candidate'] for row in ranking]
        if not names or len(names)!=len(set(names)) or 'reference' not in names:
            raise ValueError('review shortlist requires unique candidates and its repaired reference')
        for name in names:candidate_options(method,name)
        # The fixed reference counts toward the two-configuration review budget.
        selected[key]=['reference']+[name for name in names if name!='reference'][:1]
    units=[]
    for domain,folds in (('simulation',range(1)),('cogpilot',range(3)),('clare',range(3))):
        for method in METHODS:
            candidates=sorted({'reference'} | {name for route in ROUTES for name in selected[f'{method}/{route}']})
            for candidate in candidates:
                routes=[route for route in ROUTES if candidate=='reference' or candidate in selected[f'{method}/{route}']]
                purposes={route:('shortlisted_candidate' if candidate in selected[f'{method}/{route}']
                                else 'repaired_reference_comparator') for route in routes}
                for fold in folds:
                    for seed in (17,29,43):
                        units.append(dict(domain=domain,method=method,candidate_name=candidate,routes=routes,purposes=purposes,
                            fold_index=fold,seed=seed,phase='review',pretraining_updates=1500,
                            prefetch_cpu_consumers=domain=='simulation' and len(routes)==2,
                            head_warmup_updates=50 if 'task_guided' in routes else 0,
                            joint_updates=500 if 'task_guided' in routes else 0))
    return seal_development_plan(base | dict(status='ready_for_three_seed_review',selected_candidates=selected,units=units,
        public_selection_plan_sha256=results['selection_plan_sha256'],
        verified_screen_results_sha256=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest(),
        shared_initialization='one_pretraining_per_domain_method_candidate_fold_seed',
        budget_semantics='maximum_updates_with_frozen_internal_validation_early_stopping',
        reference_policy='reference_and_best_ranked_alternative_at_most_two_configurations_per_route',
        dingxin_status='outer_evaluation_blocked_by_cross_sortie_vehicle_content_overlap'))


def run_review_cohort(*, output_root, screen_root,
                      data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',
                      registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    plan=build_review_plan(screen_root=screen_root,data_root=data_root,registry_path=registry_path)
    if plan['status']!='ready_for_three_seed_review':return plan
    return run_development_plan(plan,output_root=output_root,data_root=data_root,registry_path=registry_path)
