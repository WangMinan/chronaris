import json
from pathlib import Path

import pytest

from chronaris.evaluation.application_tasks import v4_pipeline as module
from chronaris.evaluation.application_tasks.v4_pipeline_steps import initial_plan


def test_single_entry_retries_only_gpu_wait_resumes_and_preserves_failure(tmp_path, monkeypatch):
    config = dict(root=str(tmp_path/'run'), source_code_sha256='a'*64)
    expected = dict(step for group in module.pipeline_groups('confirmation') for step in group)
    calls = []
    fail = {'public_screen': True}
    class Child:
        pid = 99999999
        def __init__(self, command, **kwargs):
            stage = command[command.index('--worker')+1]
            calls.append(stage)
            if stage in module.CPU_STAGES:
                assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            status = expected[stage]
            if stage == 'cuda_validation' and calls.count(stage) == 1:
                status = 'waiting_gpu'
            if stage == 'public_screen' and fail['public_screen']:
                status = 'completed_with_failures'
            if stage.startswith('conditional_'):
                status = 'not_applicable'
            module.write_result(command[command.index('--result')+1], {'status': status})
        def poll(self):
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    monkeypatch.setattr(module.time, 'sleep', lambda value: None)
    with pytest.raises(ValueError, match='successful completion'):
        module.execute_pipeline(config, until='freeze')
    before = list(calls)
    stopped = module.execute_pipeline(config, until='freeze')
    assert stopped['status'] == 'failed' and calls == before
    assert len(stopped['failures']) == 1
    fail['public_screen'] = False
    result = module.execute_pipeline(config, until='freeze', retry_failed=True)
    assert result['status'] == 'freeze_completed' and len(result['failures']) == 1
    assert calls.count('initial') == 1 and calls.count('public_screen') == 2
    assert all(name not in calls for group in module.CONFIRMATION for name, _ in group)
    result = module.execute_pipeline(config, until='confirmation')
    assert result['status'] == 'confirmation_completed'
    assert calls.count('initial') == 1 and calls.count('freeze') == 1
    assert calls.index('core_model_freeze') < calls.index('simulation_generation')
    assert calls.index('simulation_model_freeze') < calls.index('simulation_generation')
    before = list(calls)
    module.execute_pipeline(config, until='confirmation')
    assert calls == before
    receipt = Path(result['completed']['review']['path'])
    receipt.write_text('{"status":"changed"}')
    with pytest.raises(ValueError, match='receipt changed'):
        module.execute_pipeline(config, until='confirmation')
    with pytest.raises(ValueError, match='source, inputs or paths changed'):
        module.execute_pipeline(config | {'source_code_sha256': 'b'*64}, until='freeze')


def test_initial_matrix_covers_baseline_objectives_without_untriggered_decay(tmp_path):
    registry = tmp_path/'registry.json'; registry.write_text('{}')
    plan = initial_plan({'registry_path': str(registry)})
    assert len(plan['units']) == 26
    keys = {(u['method'], u['candidate_name']) for u in plan['units']}
    assert len(keys) == 26 and not any(name == 'analytic_decay' for _, name in keys)
    assert all(u['seed'] == 17 and u['pretraining_updates'] == 300 for u in plan['units'])
    for method in ('physiology_only', 'vehicle_only', 'mult', 'contiformer'):
        assert {(method, name) for name in ('reference', 'capacity64', 'multihorizon', 'missingness_mixture')} <= keys


@pytest.mark.parametrize('error', [ValueError('replay mismatch'), KeyboardInterrupt('interrupted')])
def test_initial_pressure_failure_and_explicit_worker_retry(tmp_path, monkeypatch, error):
    from contextlib import nullcontext
    from chronaris.evaluation.application_tasks import v4_pipeline_steps as steps, v4_pressure_run as pressure
    root = tmp_path/'run'
    config = dict(root=str(root), data_root='unused', registry_path='unused', condition_root='unused',
        source_code_sha256='a'*64, entry_sha256='b'*64, input_files={})
    steps.write_result(root/'initial/cohort_state.json', dict(status='completed', failed_units=[],
        units=[['chronaris', 'reference'], ['chronaris', 'capacity64']]))
    monkeypatch.setattr(steps, 'v4_workflow_source_sha256', lambda: 'a'*64)
    monkeypatch.setattr(steps, 'development_gpu_lock', lambda: nullcontext(True))
    calls = []
    def run(**kwargs):
        calls.append((kwargs['candidate_name'], kwargs['route']))
        if len(calls) == 2:
            raise error
        return dict(completed=True)
    monkeypatch.setattr(pressure, 'run_development_pressure', run)
    with pytest.raises(type(error)):
        steps.run_pipeline_step('initial_pressure', config)
    path = root/'initial_pressure/queue_state.json'
    failed = json.loads(path.read_text())
    key = 'chronaris/reference/task_guided'
    assert failed['status'] == 'failed' and failed['current_unit'] is None
    assert failed['failed_units'] == [key] and len(failed['failures']) == 1
    assert steps.run_pipeline_step('initial_pressure', config) == failed and len(calls) == 2
    config_path = root/'config.json'
    steps.write_result(config_path, config)
    monkeypatch.setattr(module, 'v4_workflow_source_sha256', lambda: 'a'*64)
    monkeypatch.setattr(module, 'sha256_file', lambda path: 'b'*64)
    handlers = {}
    monkeypatch.setattr(module.signal, 'signal', lambda signum, handler: handlers.update({signum: handler}))
    monkeypatch.setattr(module.sys, 'argv', ['pipeline', '--worker', 'initial_pressure', '--config', str(config_path),
        '--result', str(root/'receipt.json'), '--retry-failed'])
    module.main()
    with pytest.raises(KeyboardInterrupt, match='termination requested'):
        handlers[module.signal.SIGTERM](module.signal.SIGTERM, None)
    resumed = json.loads(path.read_text())
    assert resumed['status'] == 'completed' and len(resumed['completed_units']) == 4
    assert resumed['failed_units'] == [] and resumed['attempts'][key] == 2
    assert resumed['failures'] == failed['failures']
    assert resumed['failed_attempts'][0]['units'] == [key]
    assert calls[:3] == [('reference', 'self_supervised'), ('reference', 'task_guided'), ('reference', 'task_guided')]


def test_core_evaluation_forwards_the_new_confirmation_directory(tmp_path, monkeypatch):
    from chronaris.evaluation.application_tasks import v4_core_ablations as core
    monkeypatch.setattr(core, 'read_frozen_configuration', lambda *args: {})
    unit = dict(ablation='no_physics_residual', options={'name': 'analytic_decay__no_physics_residual'}, seed=29)
    monkeypatch.setattr(core, 'build_core_ablation_plan', lambda frozen: {'units': [unit]})
    def queue(**kwargs):
        args = kwargs['unit_args'](unit)
        assert args[args.index('--data-root')+1] == str(tmp_path/'new_confirmation')
        assert args[args.index('--candidate-name')+1] == 'analytic_decay'
        return {'status': 'completed'}
    monkeypatch.setattr(core, 'run_confirmation_units', queue)
    assert core.run_core_ablation_cohort(freeze_path='unused', freeze_sha256='a'*64, output_root=tmp_path,
        stage='evaluate', confirmation_root=tmp_path/'new_confirmation')['status'] == 'completed'
