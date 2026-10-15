"""Apply a separately recorded audit policy while retaining frozen training source."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_policy(policy_path):
    from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify
    from chronaris.evaluation.application_tasks import v4_correctness, chapter5_core
    policy = read(policy_path)
    verify(policy['files'])
    if sha(v4_correctness.__file__) not in policy['original_auditor_sha256']:
        raise ValueError('audit override requires the reviewed original auditor')
    precision, sweep = read(policy['precision_path']), read(policy['sweep_path'])
    controls = [r for r in precision if r.get('negative_control') == 'causal_mask_disabled']
    high_precision = [r for r in precision if r.get('dtype') == 'torch.float64' and 'deltas' in r]
    if (len(controls) != 2 or any(not r['delta'] > .001 for r in controls)
        or not high_precision or any(not r['deltas']['sequence'] <= 1e-10 or not r['mask_equal'] for r in high_precision)
        or len(sweep) != 48 or any(not r['revised_passed'] or r['revised_atol'] != 1e-5
            or not all(0 <= v <= 1e-5 for v in r['historical_max_delta'].values()) for r in sweep)):
        raise ValueError('precision or negative-control qualification failed')
    revised = load_module(policy['auditor_path'], 'qualified_causality_auditor')
    if revised.HISTORICAL_ATOL != policy['historical_atol'] or policy['historical_atol'] != 1e-5:
        raise ValueError('audit tolerance differs from qualification')
    binding = dict(path=str(policy_path), sha256=sha(policy_path), auditor_sha256=sha(policy['auditor_path']))
    def audit(*args, **kwargs):
        result = revised.audit_checkpoint_causality(*args, **kwargs)
        return result | dict(validation_policy=binding)
    original_train = chapter5_core._train
    def train(*args, **kwargs):
        result = original_train(*args, **kwargs)
        if result['status'] == 'completed':
            result['files'].update(policy['files'] | {str(policy_path): binding['sha256']})
            result['validation_policy'] = binding
        return result
    v4_correctness.audit_checkpoint_causality = audit
    chapter5_core._train = train
    return policy


def worker(*, policy_path, index, legacy):
    policy = install_policy(policy_path)
    if legacy:
        if index not in (15, 16):
            raise ValueError('unplanned legacy recovery')
        helper = load_module(policy['legacy_driver_module'], 'frozen_device_resume')
        helper.run_legacy(receipt_path=policy['execution_receipt'], mode='resumed', index={15:18, 16:21}[index])
    else:
        if index != 17:
            raise ValueError('unplanned new simulation')
        from chronaris.evaluation.application_tasks.chapter5_subset import train_subset_unit
        train_subset_unit(policy['subset_root'], index)


def continue_subset(policy_path, launcher):
    from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write
    from chronaris.evaluation.application_tasks.chapter5_subset import verified_subset, validate_unit, freeze_subset, _immutable
    from chronaris.evaluation.application_tasks.device_resume import qualified_execution
    policy = read(policy_path); verify(policy['files'])
    root = Path(policy['subset_root']); run = root.parent
    plan = verified_subset(root)
    execution, bindings = qualified_execution(plan['legacy_compatibility'],
        legacy_workspace=plan['legacy_workspace'], core_root=plan['core_root'])
    for index in range(15):
        validate_unit(read(root/'units'/f'{index}.json'), plan['units'][index], plan)
    state = dict(status='running', pid=os.getpid(), phase='causality_policy_recovery',
                 validation_policy=str(policy_path), commands=[], confirmation_opened=False)
    destination = Path(policy['recovery_root']); destination.mkdir(parents=True, exist_ok=True)
    try:
        for index in (15, 16, 17):
            target = root/'units'/f'{index}.json'
            if target.exists():
                validate_unit(read(target), plan['units'][index], plan)
                continue
            workspace = execution['workspace'] if index < 17 else policy['training_workspace']
            args = [sys.executable, str(launcher), '--policy', str(policy_path), '--workspace', workspace,
                    '--index', str(index), '--action', 'legacy' if index < 17 else 'native']
            state.update(phase=f'subset_unit_{index}')
            log = destination/f'unit_{index}.log'; started = time.time()
            with log.open('a') as handle:
                child = subprocess.Popen(args, cwd=workspace, stdout=handle, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid; write(run/'validation_state.json', state)
                try:
                    code = child.wait()
                except BaseException:
                    child.terminate(); child.wait()
                    raise
            receipt = destination/f'unit_{index}.json'
            write(receipt, dict(exit_code=code, command=args, wall_seconds=time.time()-started, log=str(log)))
            state['commands'].append(str(receipt)); state.pop('child_pid', None)
            if code:
                raise RuntimeError(f'qualified unit {index} failed: see {log}')
            if index < 17:
                unit = plan['units'][index]
                result = read(Path(execution['root'])/'resumed'/f"unit_{unit['legacy_index']}_completed.json")
                expected = read(Path(plan['core_root'])/'chapter5_plan.json')['units'][unit['legacy_index']]
                if result['unit'] != expected:
                    raise ValueError('legacy worker returned another unit')
                result['files'].update(bindings)
                result = result | dict(legacy_plan_sha256=result['plan_sha256'], unit=unit, plan_sha256=plan['plan_sha256'])
                validate_unit(result, unit, plan); _immutable(target, result)
            else:
                validate_unit(read(target), plan['units'][index], plan)
            write(run/'validation_state.json', state)
        freeze_subset(root)
        state.update(status='completed', phase='subset_frozen')
        write(run/'validation_state.json', state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(run/'validation_state.json', state)
        raise
