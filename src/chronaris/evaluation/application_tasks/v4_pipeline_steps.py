"""Fixed v4 stages, reusing the existing data, trainers and evidence readers."""
import json
import time
import traceback
from pathlib import Path

from chronaris.evaluation.application_tasks.v4_candidates import CANDIDATE_CHANGES, BASELINE_CANDIDATES
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_public_screen import METHODS, ROUTES, run_development_plan, seal_development_plan, development_gpu_lock
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def write_result(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return value


def initial_plan(config):
    units = [dict(domain='simulation', method=method, candidate_name=candidate,
        routes=list(ROUTES), fold_index=0, seed=17, phase='screen', pretraining_updates=300,
        head_warmup_updates=50, joint_updates=200, prefetch_cpu_consumers=True)
        for method in METHODS for candidate in (CANDIDATE_CHANGES if method == 'chronaris' else BASELINE_CANDIDATES)]
    return seal_development_plan(dict(format='chronaris.v4_initial_pipeline_plan.v1',
        status='ready', source_code_sha256=v4_workflow_source_sha256(), phase='screen', units=units,
        public_registry_sha256=sha256_file(config['registry_path']), confirmation_feedback_used=False,
        common_objective_opportunity='both_fixed_objectives_for_every_trainable_baseline'))


def active_review(root):
    decision = json.loads((root/'conditional_plan.json').read_text())
    return (root/'conditional_review', root/'conditional_pressure') if decision['status'] != 'not_applicable' else (root/'review', root/'review_pressure')


def run_pipeline_step(stage, config, *, attempt=1):
    root = Path(config['root'])
    initial, pressure, public, review = (root/name for name in ('initial', 'initial_pressure', 'public_screen', 'review'))
    common = dict(data_root=config['data_root'], registry_path=config['registry_path'])
    if stage.startswith('comparison_'):
        from chronaris.evaluation.application_tasks.development_comparison import run_comparison_step
        return run_comparison_step(stage, config)
    if stage == 'cuda_validation':
        from chronaris.evaluation.application_tasks.v4_configuration_freeze import validate_current_cuda
        return validate_current_cuda(root/'cuda_validation'/f'attempt_{attempt}')
    if stage == 'initial':
        plan = initial_plan(config)
        state = run_development_plan(plan, output_root=initial, **common)
        if state['status'] == 'waiting_gpu':
            return state
        cohort = dict(source_code_sha256=plan['source_code_sha256'], status=state['status'],
            units=[[u['method'], u['candidate_name']] for u in plan['units']], current_child_pid=None,
            completed_units=[key.removeprefix('simulation/') for key in state['completed_units']],
            failed_units=[key.removeprefix('simulation/') for key in state['failed_units']])
        return write_result(initial/'cohort_state.json', cohort)
    if stage == 'initial_pressure':
        from chronaris.evaluation.application_tasks.v4_pressure_run import run_development_pressure
        cohort = json.loads((initial/'cohort_state.json').read_text())
        if cohort['status'] != 'completed' or cohort['failed_units']:
            raise ValueError('initial pressure requires the complete clean cohort')
        path = pressure/'queue_state.json'
        state = json.loads(path.read_text()) if path.exists() else dict(
            source_code_sha256=v4_workflow_source_sha256(), completed_units=[], failed_units=[])
        if state['source_code_sha256'] != v4_workflow_source_sha256():
            raise ValueError('initial pressure source changed')
        if state.get('failed_units'):
            return state
        with development_gpu_lock() as acquired:
            if not acquired:
                return dict(status='waiting_gpu')
            for method, candidate in cohort['units']:
                for route in ROUTES:
                    key = f'{method}/{candidate}/{route}'
                    if key in state['completed_units']:
                        continue
                    state.update(status='running', current_unit=key)
                    attempts = state.setdefault('attempts', {})
                    attempts[key] = attempts.get(key, 0) + 1
                    write_result(path, state)
                    try:
                        result = run_development_pressure(method=method, candidate_name=candidate, route=route,
                            update=300 if route == 'self_supervised' else 200, output_root=pressure,
                            diagnostic_root=initial, condition_root=config['condition_root'], device='cuda')
                        if not result['completed']:
                            raise ValueError(f'incomplete initial pressure: {key}')
                    except BaseException:
                        error = traceback.format_exc()
                        state['failed_units'].append(key)
                        state.setdefault('errors', {})[key] = error
                        state.setdefault('failures', []).append(dict(unit=key, attempt=attempts[key],
                            time_unix_s=time.time(), error=error))
                        state.update(status='failed', current_unit=None, current_child_pid=None)
                        write_result(path, state)
                        raise
                    state['completed_units'].append(key)
                    write_result(path, state)
        state.update(status='completed', current_unit=None, current_child_pid=None)
        return write_result(path, state)
    if stage == 'public_screen':
        from chronaris.evaluation.application_tasks.v4_public_screen import run_public_screen
        return run_public_screen(output_root=public, diagnostic_root=initial, pressure_root=pressure, **common)
    if stage == 'public_results':
        from chronaris.evaluation.application_tasks.v4_public_results import collect_public_screen_results
        return write_result(public/'results_summary.json', collect_public_screen_results(output_root=public, **common))
    if stage == 'review':
        from chronaris.evaluation.application_tasks.v4_review_plan import run_review_cohort
        return run_review_cohort(output_root=review, screen_root=public, **common)
    conditional = stage.startswith('conditional_') and stage != 'conditional_plan'
    if conditional:
        plan = json.loads((root/'conditional_plan.json').read_text())
        if plan['status'] == 'not_applicable':
            return dict(status='not_applicable')
        review = root/'conditional_review'
        if stage == 'conditional_review':
            from chronaris.evaluation.application_tasks.v4_conditional_review import prepare_conditional_review
            prepare_conditional_review(plan, output_root=review, pressure_root=root/'conditional_pressure')
            return run_development_plan(plan, output_root=review, **common)
    review_pressure = root/('conditional_pressure' if conditional else 'review_pressure')
    name = stage.removeprefix('conditional_')
    if name == 'public_review_results':
        from chronaris.evaluation.application_tasks.v4_public_review_results import collect_public_review_results
        return write_result(review/'public_results_summary.json', collect_public_review_results(output_root=review, **common))
    if name == 'review_pressure':
        from chronaris.evaluation.application_tasks.v4_review_pressure import run_review_pressure_cohort
        return run_review_pressure_cohort(output_root=review_pressure, diagnostic_root=review,
            condition_root=config['condition_root'], **common)
    if name == 'simulation_review_results':
        from chronaris.evaluation.application_tasks.v4_simulation_review_results import collect_simulation_review_results
        return write_result(review/'simulation_results_summary.json', collect_simulation_review_results(
            output_root=review, pressure_root=review_pressure, **common))
    if name == 'adoption':
        from chronaris.evaluation.application_tasks.v4_adoption import collect_adoption_decisions
        return write_result(review/'adoption_decisions.json', collect_adoption_decisions(
            output_root=review, pressure_root=review_pressure, **common))
    if stage == 'conditional_plan':
        from chronaris.evaluation.application_tasks.v4_conditional_review import build_conditional_review_plan
        return write_result(root/'conditional_plan.json', build_conditional_review_plan(
            parent_root=review, parent_pressure_root=review_pressure, **common))
    formal = root/'confirmation'
    freeze = formal/'frozen_configuration.json'
    if stage == 'freeze':
        from chronaris.evaluation.application_tasks.v4_configuration_freeze import freeze_reviewed_configuration
        review, review_pressure = active_review(root)
        validation=json.loads((root/'pipeline_state.json').read_text())['completed']['cuda_validation']
        if sha256_file(validation['path'])!=validation['sha256']:
            raise ValueError('selected CUDA validation receipt changed')
        return write_result(formal/'freeze_readiness.json', freeze_reviewed_configuration(
            review_root=review, pressure_root=review_pressure, validation_receipt=validation['path'],
            output_path=freeze, registry_path=config['registry_path']))
    frozen = dict(freeze_path=freeze, freeze_sha256=sha256_file(freeze), output_root=formal)
    if stage == 'mechanism_data':
        from chronaris.evaluation.application_tasks.v4_mechanism_data import generate_mechanism_data
        return generate_mechanism_data(formal_root=formal, output_root=root/'timing_data')
    if stage == 'mechanisms':
        from chronaris.evaluation.application_tasks.v4_mechanism_evaluation import run_final_mechanisms
        return run_final_mechanisms(formal_root=formal, timing_root=root/'timing_data',
            confirmation_root=root/'simulation_confirmation', output_root=root/'mechanisms')
    if stage == 'report':
        from chronaris.evaluation.application_tasks.v4_final_evidence import collect_final_evidence
        return collect_final_evidence(formal_root=formal, mechanism_root=root/'mechanisms', output_root=root/'report',
            data_root=config['confirmation_data_root'], registry_path=config['registry_path'])
    if stage.startswith('native_'):
        from chronaris.evaluation.application_tasks.v4_native_confirmation_cohort import run_native_confirmation_cohort
        return run_native_confirmation_cohort(**frozen, backend=stage.removeprefix('native_'),
            data_root=config['confirmation_data_root'], registry_path=config['registry_path'])
    if stage.startswith('simulation_train_') or stage == 'simulation_evaluate':
        from chronaris.evaluation.application_tasks.v4_simulation_confirmation import run_simulation_confirmation_cohort
        return run_simulation_confirmation_cohort(**frozen,
            backend='neural' if stage == 'simulation_evaluate' else stage.removeprefix('simulation_train_'),
            stage='evaluate' if stage == 'simulation_evaluate' else 'train', confirmation_root=root/'simulation_confirmation')
    if stage in ('core_train', 'core_evaluate'):
        from chronaris.evaluation.application_tasks.v4_core_ablations import run_core_ablation_cohort
        return run_core_ablation_cohort(**frozen, stage=stage.removeprefix('core_'), confirmation_root=root/'simulation_confirmation')
    if stage == 'core_model_freeze':
        from chronaris.evaluation.application_tasks.v4_core_ablations import seal_core_ablation_models
        return seal_core_ablation_models(**frozen)
    if stage == 'simulation_model_freeze':
        from chronaris.evaluation.application_tasks.v4_simulation_confirmation import seal_simulation_models
        return seal_simulation_models(**frozen)
    if stage == 'simulation_generation':
        from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import generate_simulation_confirmation
        models = formal/'simulation_frozen_models.json'
        return generate_simulation_confirmation(freeze_path=freeze, freeze_sha256=frozen['freeze_sha256'],
            model_freeze_path=models, model_freeze_sha256=sha256_file(models), output_root=root/'simulation_confirmation')
    raise ValueError(f'unknown pipeline stage: {stage}')
