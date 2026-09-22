"""One resumable process entry for the fixed development and confirmation sequence."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import run_pipeline_step, write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

DEVELOPMENT = [
    ('cuda_validation', 'completed'), ('initial', 'completed'), ('initial_pressure', 'completed'),
    ('public_screen', 'completed'), ('public_results', 'public_scores_verified_not_final_adoption'),
    ('review', 'completed'), ('public_review_results', 'public_review_verified_not_final_selection'),
    ('review_pressure', 'completed'), ('simulation_review_results', 'simulation_review_verified_not_final_selection'),
    ('adoption', 'single_factor_decisions_ready_not_frozen'), ('conditional_plan', 'ready_for_three_seed_review'),
    ('conditional_review', 'completed'), ('conditional_public_review_results', 'public_review_verified_not_final_selection'),
    ('conditional_review_pressure', 'completed'),
    ('conditional_simulation_review_results', 'simulation_review_verified_not_final_selection'),
    ('conditional_adoption', 'single_factor_decisions_ready_not_frozen'),
]
CONFIRMATION = [
    [('native_nonparametric', 'completed'), ('simulation_train_nonparametric', 'completed'), ('simulation_train_neural', 'completed')],
    [('core_train', 'completed')], [('core_model_freeze', 'frozen')], [('simulation_model_freeze', 'frozen')],
    [('mechanism_data', 'completed'), ('simulation_generation', 'completed')], [('simulation_evaluate', 'completed')],
    [('core_evaluate', 'completed')], [('native_neural', 'completed')], [('mechanisms', 'completed')], [('report', 'completed')],
]
CPU_STAGES = {'public_results', 'public_review_results', 'simulation_review_results', 'adoption', 'conditional_plan',
    'conditional_public_review_results', 'conditional_simulation_review_results', 'conditional_adoption',
    'freeze', 'native_nonparametric', 'simulation_train_nonparametric', 'simulation_generation', 'mechanism_data', 'report'}


def pipeline_groups(until):
    if until == 'stage45c':
        from chronaris.evaluation.application_tasks.stage45c import groups
        return groups()
    if until == 'stage45b':
        from chronaris.evaluation.application_tasks.stage45b import groups
        return groups()
    if until == 'stage45':
        from chronaris.evaluation.application_tasks.stage45 import stage45_groups
        return stage45_groups()
    if until == 'comparison':
        from chronaris.evaluation.application_tasks.development_comparison import comparison_groups
        return comparison_groups()
    if until not in ('development', 'freeze', 'confirmation'):
        raise ValueError('unknown pipeline endpoint')
    groups = [[step] for step in DEVELOPMENT]
    if until != 'development':
        groups.append([('freeze', 'frozen')])
    if until == 'confirmation':
        groups.extend(CONFIRMATION)
    return groups


def _verify_receipt(path, expected_digest, expected_status, optional):
    if sha256_file(path) != expected_digest:
        raise ValueError('completed pipeline receipt changed')
    result = json.loads(Path(path).read_text())
    allowed = {expected_status, 'not_applicable'} if optional else {expected_status}
    if result.get('status') not in allowed:
        raise ValueError('pipeline receipt is not a successful completion')


def stage45_budget_expired(config, state):
    hours = config.get('stage45_budget_hours', 72)
    if hours is None:
        return False
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
        raise ValueError('stage 4.5 budget must be positive hours or explicit unlimited')
    return time.time() >= state['started_at_unix_s'] + hours*3600


def execute_pipeline(config, *, until, retry_failed=False):
    """Subprocesses release CUDA resources between stages; existing trainers own GPU locks."""
    root = Path(config['root']); root.mkdir(parents=True, exist_ok=True)
    with (root/'pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='already_running', root=str(root))
        manifest_path = root/'pipeline_config.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != config:
            raise ValueError('pipeline source, inputs or paths changed; preserve this run and use another root')
        if not manifest_path.exists():
            write_result(manifest_path, config)
        path = root/'pipeline_state.json'
        state = json.loads(path.read_text()) if path.exists() else dict(completed={}, failures=[], attempts={})
        if state.get('config_sha256', sha256_file(manifest_path)) != sha256_file(manifest_path):
            raise ValueError('pipeline configuration changed')
        # A killed controller can leave workers alive and holding the GPU lock.
        for child in state.get('children', {}).values():
            try:
                command = Path(f'/proc/{child}/cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if str(manifest_path).encode() in command:
                raise RuntimeError('recorded worker is still live; do not duplicate its work')
        if state.get('status') == 'failed' and not retry_failed:
            return state | dict(resume_hint='inspect failure, then explicitly use --retry-failed for an unchanged-source transient failure')
        state.setdefault('started_at_unix_s', time.time())
        state.update(pid=os.getpid(), config_sha256=sha256_file(manifest_path), status='running', until=until, children={})
        children = {}
        project = Path(__file__).parents[4]
        entry = project/'scripts/evaluation/application_tasks/run_v4_pipeline.py'

        def save():
            state['updated_at_unix_s'] = time.time()
            write_result(path, state)

        def launch(stage, expected):
            attempt = state['attempts'].get(stage, 0)+1
            state['attempts'][stage] = attempt
            result_path = root/'stage_results'/f'{stage}.{attempt}.json'
            log_path = root/'logs'/f'{stage}.{attempt}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(project/'src'))
            if stage in CPU_STAGES:
                env['CUDA_VISIBLE_DEVICES'] = ''
            command = [sys.executable, str(entry), '--worker', stage, '--config', str(manifest_path),
                       '--result', str(result_path), '--attempt', str(attempt)]
            if retry_failed:
                command.append('--retry-failed')
            log = log_path.open('w')
            child = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[stage] = (child, log, result_path, expected, time.perf_counter())
            state['children'][stage] = child.pid
            state['current_stage'] = stage
            state['current_log'] = str(log_path)
            save()

        try:
            save()
            for group in pipeline_groups(until):
                if until in ('stage45', 'stage45b', 'stage45c') and stage45_budget_expired(config, state):
                    state.update(status='budget_exhausted', confirmation_opened=False); save()
                    return state
                for stage, expected in group:
                    if stage in state['completed']:
                        receipt = state['completed'][stage]
                        _verify_receipt(receipt['path'], receipt['sha256'], expected, stage.startswith('conditional_'))
                    else:
                        launch(stage, expected)
                while children:
                    if until in ('stage45', 'stage45b', 'stage45c') and stage45_budget_expired(config, state):
                        state.update(status='budget_exhausted', confirmation_opened=False); save()
                        return state
                    for stage, (child, log, result_path, expected, started) in list(children.items()):
                        code = child.poll()
                        if code is None:
                            continue
                        write_result(root/'attempt_costs'/f"{stage}.{state['attempts'][stage]}.json",
                            dict(stage=stage, pid=child.pid, exit_code=code, status='exited', seconds=time.perf_counter()-started))
                        log.close(); children.pop(stage); state['children'].pop(stage)
                        result = json.loads(result_path.read_text()) if result_path.exists() else {}
                        if code == 0 and result.get('status') == 'waiting_gpu':
                            state['status'] = 'waiting_gpu'; save()
                            time.sleep(30)
                            launch(stage, expected)
                            continue
                        state['current_stage'] = stage
                        state['current_log'] = log.name
                        digest = sha256_file(result_path) if result_path.exists() else None
                        if code or digest is None:
                            raise RuntimeError(f'{stage} failed (exit {code}); see its attempt log')
                        _verify_receipt(result_path, digest, expected, stage.startswith('conditional_'))
                        state['completed'][stage] = dict(path=str(result_path), sha256=digest)
                        state['status'] = 'running'; save()
                    if children:
                        time.sleep(1)
                        if time.time()-state['updated_at_unix_s'] >= 30:
                            save()
            state.update(status=f'{until}_completed', current_stage=None)
            save()
            return state
        except BaseException:
            state.update(status='failed', error=traceback.format_exc())
            state['failures'].append(dict(stage=state.get('current_stage'), time_unix_s=time.time(), error=state['error']))
            raise
        finally:
            for stage, (child, log, _, _, started) in children.items():
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=10)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                write_result(root/'attempt_costs'/f"{stage}.{state['attempts'][stage]}.json",
                    dict(stage=stage, pid=child.pid, exit_code=child.returncode, status='interrupted', seconds=time.perf_counter()-started))
                log.close()
            state['children'] = {}; save()


def main(*, default_until='freeze'):
    parser = argparse.ArgumentParser(description='Fixed v4 development, selection and confirmation; resumes one immutable run root.')
    parser.add_argument('--root')
    parser.add_argument('--until', choices=('development', 'freeze', 'confirmation', 'comparison', 'stage45', 'stage45b', 'stage45c'), default=default_until)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--stage45-budget-hours', type=float, default=72, help='0 removes the stage 4.5 wall-clock limit')
    parser.add_argument('--stage45-resume-parent', help='Stopped stage 4.5 root to preserve and resume in a new root')
    parser.add_argument('--stage45-resume-evidence', help='Verified pause/recovery evidence manifest')
    parser.add_argument('--stage45-parent', help='Completed stage 4 parent; only for the stage45 endpoint')
    parser.add_argument('--stage45-acceptance', default='docs/artifacts/runs/2026-09-14_v4-stage4-closeout/acceptance.json')
    parser.add_argument('--comparison-parent', help='Preserved stage-4 run to verify and reuse in a new comparison root')
    parser.add_argument('--execution-parent', help='Stopped stage-4 run for validated CUDA graph migration')
    parser.add_argument('--execution-evidence', help='Completed checkpoint performance trial summary')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--worker', choices=[stage for group in pipeline_groups('confirmation')+pipeline_groups('comparison')+pipeline_groups('stage45')+pipeline_groups('stage45b')+pipeline_groups('stage45c') for stage, _ in group], help=argparse.SUPPRESS)
    parser.add_argument('--config', help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if bool(args.execution_parent) != bool(args.execution_evidence) or (args.execution_parent and
            (args.comparison_parent or args.until != 'comparison')):
        parser.error('execution migration requires parent and evidence, comparison endpoint, and no comparison-parent')
    def terminate(signum, frame):
        raise KeyboardInterrupt('pipeline termination requested')
    signal.signal(signal.SIGTERM, terminate)
    if args.worker:
        config = json.loads(Path(args.config).read_text())
        if config['source_code_sha256'] != v4_workflow_source_sha256():
            raise ValueError('worker source differs from pipeline configuration')
        entry = Path(__file__).parents[4]/'scripts/evaluation/application_tasks/run_v4_pipeline.py'
        if sha256_file(entry) != config['entry_sha256'] or any(sha256_file(path) != digest for path, digest in config['input_files'].items()):
            raise ValueError('pipeline entry or input inventory changed')
        if args.retry_failed:
            # Preserve the failure history; only the explicit restart may retry failed units.
            queue_paths = {
                'initial': 'initial/run_state.json', 'public_screen': 'public_screen/run_state.json',
                'initial_pressure': 'initial_pressure/queue_state.json',
                'review': 'review/run_state.json', 'conditional_review': 'conditional_review/run_state.json',
                'review_pressure': 'review_pressure/queue_state.json',
                'conditional_review_pressure': 'conditional_pressure/queue_state.json',
                'native_neural': 'confirmation/native_neural_queue.json',
                'native_nonparametric': 'confirmation/native_nonparametric_queue.json',
                'simulation_train_neural': 'confirmation/simulation_train_neural_queue.json',
                'simulation_train_nonparametric': 'confirmation/simulation_train_nonparametric_queue.json',
                'simulation_evaluate': 'confirmation/simulation_evaluate_neural_queue.json',
                'core_train': 'confirmation/core_ablations/core_train_neural_queue.json',
                'core_evaluate': 'confirmation/core_ablations/core_evaluate_neural_queue.json',
            }
            if args.worker in queue_paths:
                path = Path(config['root'])/queue_paths[args.worker]
                if path.exists():
                    state = json.loads(path.read_text())
                    if state.get('failed_units'):
                        state.setdefault('failed_attempts', []).append(dict(units=state['failed_units'], errors=state.get('errors', {})))
                        state.update(failed_units=[], errors={}, status='pending')
                        write_result(path, state)
        result = run_pipeline_step(args.worker, config, attempt=args.attempt)
        write_result(args.result, result)
        return
    if not args.root:
        parser.error('--root is required; use a new directory for changed source')
    project = Path(__file__).parents[4]
    os.chdir(project)
    config = dict(format='chronaris.v4_pipeline.v1', root=str(Path(args.root).resolve()),
        source_code_sha256=v4_workflow_source_sha256(),
        entry_sha256=sha256_file(project/'scripts/evaluation/application_tasks/run_v4_pipeline.py'),
        registry_path=str(project/'docs/requirements/thesis-v4-public-subjects.json'),
        data_root=str(project/'artifacts/application_evaluation/2026-09-06_v4-public-development'),
        confirmation_data_root=str(project/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'),
        condition_root=str(project/'artifacts/application_evaluation/2026-09-06_v4-development-conditions-repair'))
    input_paths = [Path(config['registry_path']), project/'docs/requirements/thesis-v4-simulation-manifest.json',
        project/'artifacts/application_evaluation/2026-09-07_thesis-v4-simulation-expanded/v4_generation_audit.json',
        Path(config['condition_root'])/'development_condition_audit.json']
    input_paths += [Path(config[name])/domain/'summary.json' for name in ('data_root', 'confirmation_data_root')
                    for domain in ('cogpilot', 'clare')]
    if args.until == 'comparison':
        from chronaris.evaluation.application_tasks.development_comparison import ASSETS, SENSOR_ASSETS
        input_paths = [Path(config['registry_path']), Path(ASSETS), Path(SENSOR_ASSETS)]
        input_paths += [Path(config['data_root'])/domain/'summary.json' for domain in ('cogpilot', 'clare')]
    if bool(args.stage45_resume_parent) != bool(args.stage45_resume_evidence) or (args.stage45_resume_parent and args.until != 'stage45'):
        parser.error('stage 4.5 recovery requires parent and evidence together')
    if args.until == 'stage45':
        if not math.isfinite(args.stage45_budget_hours) or args.stage45_budget_hours < 0:
            parser.error('stage45 budget must be nonnegative; zero means unlimited')
        config['stage45_budget_hours'] = args.stage45_budget_hours or None
        if not args.stage45_parent or args.comparison_parent or args.execution_parent:
            parser.error('stage45 requires its completed parent and cannot migrate the old pipeline')
        config.update(stage45_parent=str(Path(args.stage45_parent).resolve()),
            stage45_acceptance=str(Path(args.stage45_acceptance).resolve()))
        input_paths = [Path(config['registry_path']), Path(config['stage45_acceptance']),
            Path(config['stage45_parent'])/'pipeline_state.json']
        input_paths += [Path(config['data_root'])/domain/'summary.json' for domain in ('cogpilot','clare')]
        if args.stage45_resume_parent:
            config.update(stage45_resume_parent=str(Path(args.stage45_resume_parent).resolve()),
                stage45_resume_evidence=str(Path(args.stage45_resume_evidence).resolve()))
            input_paths += [Path(config['stage45_resume_evidence']), Path(config['stage45_resume_parent'])/'pipeline_state.json']
    elif args.until == 'stage45b':
        if args.stage45_budget_hours <= 0 or args.stage45_budget_hours > 48 or not math.isfinite(args.stage45_budget_hours):
            parser.error('stage45b requires a positive budget no greater than 48 hours')
        if args.stage45_parent or args.comparison_parent or args.execution_parent:
            parser.error('stage45b never reuses historical training weights')
        config['stage45_budget_hours'] = args.stage45_budget_hours
        config['stage45b_validation'] = str(Path(args.stage45_acceptance).resolve())
        input_paths = [Path(config['stage45b_validation']), Path(config['registry_path']), project/'docs/requirements/thesis-stage45b-development-20260921.md',
            project/'docs/artifacts/runs/2026-09-21_v4-stage45-closeout/evidence.json']
        input_paths += [Path(config['data_root'])/d/'summary.json' for d in ('cogpilot','clare')]
    elif args.until == 'stage45c':
        if not args.stage45_parent or args.comparison_parent or args.execution_parent:
            parser.error('stage45c requires its completed stage45b parent')
        if not math.isfinite(args.stage45_budget_hours) or not 0 < args.stage45_budget_hours <= 24:
            parser.error('stage45c requires a positive budget no greater than 24 hours')
        config.update(stage45c_parent=str(Path(args.stage45_parent).resolve()),
            stage45c_validation=str(Path(args.stage45_acceptance).resolve()), stage45_budget_hours=args.stage45_budget_hours)
        input_paths = [Path(config['registry_path']), Path(config['stage45c_validation']),
            Path(config['stage45c_parent'])/'pipeline_state.json',
            project/'docs/requirements/thesis-stage45c-development-20260922.md']
        input_paths += [Path(config['data_root'])/d/'summary.json' for d in ('cogpilot','clare')]
    elif args.stage45_parent:
        parser.error('--stage45-parent requires --until stage45 or stage45c')
    if args.comparison_parent:
        if args.until != 'comparison':
            parser.error('--comparison-parent requires --until comparison')
        config['comparison_parent'] = str(Path(args.comparison_parent).resolve())
        input_paths += [Path(config['comparison_parent'])/name for name in ('pipeline_state.json', 'pipeline_config.json')]
    if args.execution_parent:
        config.update(execution_parent=str(Path(args.execution_parent).resolve()),
                      execution_evidence=str(Path(args.execution_evidence).resolve()))
        input_paths += [Path(config['execution_parent'])/name for name in ('pipeline_state.json', 'pipeline_config.json')]
        input_paths.append(Path(config['execution_evidence']))
    config['input_files'] = {str(path): sha256_file(path) for path in input_paths}
    if args.plan_only and args.until in ('stage45', 'stage45b', 'stage45c'):
        print(json.dumps(dict(config=config, groups=pipeline_groups(args.until), budget_hours=config['stage45_budget_hours'],
            confirmation_opened=False, executes_training=False), indent=2))
        return
    if args.plan_only and args.until == 'comparison':
        print(json.dumps(dict(config=config, groups=pipeline_groups(args.until), seed=17, fold_index=0,
            model_units=25, representation_routes=43, confirmation_opened=False, executes_training=False), indent=2))
        return
    if args.plan_only:
        print(json.dumps(dict(config=config, groups=pipeline_groups(args.until),
            initial_configurations=26, seeds=[17,29,43], public_review_folds=3,
            conditional_candidate_limit=1, executes_training=False), indent=2))
        return
    result = execute_pipeline(config, until=args.until, retry_failed=args.retry_failed)
    print(json.dumps(result, indent=2))
    if result['status'] == 'failed':
        raise SystemExit(1)
