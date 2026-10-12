"""Serial local qualification, full-fold timing, then the already declared subset."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import subprocess
import sys
import time

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write


def calibrated_budget(prior, timings):
    factors = {}
    for index, case in ((0, 'clare_batched'), (10, 'cogpilot_batched')):
        old = next(r['hours'] for r in prior['rows'] if r['case'] == case and r['fold'] == 0)
        factors[case] = timings[str(index)]['wall_seconds']/3600/old
    factors['clare_no_event'] = max(1., factors['clare_batched'])
    rows = [dict(r, hours=r['hours']*factors.get(r['case'], 1.)) for r in prior['rows']]
    hours = sum(r['hours'] for r in rows)
    measured = sum(t['wall_seconds'] for t in timings.values())/3600
    remaining = max(0., hours-measured)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    return dict(status='calibrated_two_full_folds', measured_units=[0, 10], calibration_factors=factors,
        first_batch_compute_hours=hours, remaining_compute_hours=remaining, rows=rows,
        remaining_with_delivery_days=[(remaining*.8+24)/24, (remaining*1.5+48)/24],
        completion_window_beijing=[(now+timedelta(hours=h)).isoformat() for h in (remaining*.8+24, remaining*1.5+48)],
        public_45_unit_compute_hours=3*sum(r['hours'] for r in rows if not r['case'].startswith('sim_')),
        assumptions=['Only CLARE complete fold 1 and CogPilot complete fold 1 are measured end to end.',
            'CLARE semantic ablation uses a conservative complete-fold calibration; simulation retains earlier short probes.',
            'Remaining simulation estimate conservatively includes the 50 qualification updates already reused.',
            '0.8/1.5 are planning scenarios; add 24/48 hours for delivery, not confidence bounds.',
            'The 45-public-unit figure excludes remaining old simulation, later semantic seeds, evaluation and export.'])


def run_validation(*, root, replay_root, replay_pid, execution_receipt, main_root, prior_budget):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).parents[4]
    driver = project/'scripts/evaluation/application_tasks'
    execution = read(execution_receipt)
    config = dict(root=str(root), replay_root=str(replay_root), replay_pid=replay_pid,
        execution_receipt=str(execution_receipt), main_root=str(main_root),
        prior_budget=str(prior_budget), driver_workspace=str(project))
    config_path = root/'validation_config.json'
    if config_path.exists() and read(config_path) != config:
        raise ValueError('validation configuration changed')
    write(config_path, config)
    state = dict(status='running', phase='waiting_full_replay', pid=os.getpid(), commands=[], confirmation_opened=False)
    write(root/'validation_state.json', state)

    def command(name, args, *, cwd=project):
        receipt_path = root/'commands'/f'{name}.json'
        if receipt_path.exists():
            old = read(receipt_path)
            if old['exit_code'] != 0 or old['command'] != args:
                raise ValueError(f'preserve failed or changed command: {name}')
            return old
        state.update(phase=name); write(root/'validation_state.json', state)
        log = root/'logs'/f'{name}.log'; log.parent.mkdir(exist_ok=True)
        started = time.time(); peak = 0
        with log.open('w') as handle:
            child = subprocess.Popen(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                env=os.environ | {'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1', 'PYTHONNOUSERSITE':'1'})
            state['child_pid'] = child.pid; write(root/'validation_state.json', state)
            while child.poll() is None:
                gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                peak = max(peak, int(gpu.strip().splitlines()[0]))
                time.sleep(5)
        result = write(receipt_path, dict(command=args, exit_code=child.returncode, started_unix_s=started,
            wall_seconds=time.time()-started, sampled_device_peak_mib=peak,
            memory_scope='whole device including display; sampled every 5 seconds', log=str(log)))
        state['commands'].append(str(receipt_path)); state.pop('child_pid', None)
        write(root/'validation_state.json', state)
        if child.returncode:
            raise RuntimeError(f'{name} failed with {child.returncode}; see {log}')
        return result

    try:
        while True:
            process = Path(f'/proc/{replay_pid}/cmdline')
            if not process.exists() or b'run_device_replay.py' not in process.read_bytes():
                break
            time.sleep(10)
        summary = Path(replay_root)/'summary.json'
        if not summary.exists():
            raise RuntimeError('full replay stopped without a complete summary; inspect its log')
        replay = read(summary)
        if replay['status'] != 'completed' or len(replay['expected_units']) != 180:
            raise ValueError('full replay inventory is incomplete')
        for index in replay['expected_units']:
            verify(read(Path(replay_root)/'units'/str(index)/'result.json')['files'])
        write(root/'replay_gate.json', dict(summary=str(summary), sha256=sha(summary),
            strict_passed=replay['strict_passed'], class_flips=replay['class_flips'],
            changed_rank_panels=sum(bool(r['changed_pairs']) for r in replay['rankings']),
            historical_predictions_preserved=True))
        base = [sys.executable, str(driver/'run_device_resume.py'), '--workspace', execution['workspace'],
                '--receipt', str(execution_receipt)]
        for name, mode, stop in (('resume_continuous_175', 'continuous', 175),
                                ('resume_interrupt_150', 'resumed', 150), ('resume_restart_175', 'resumed', 175)):
            command(name, base+['--mode', mode, '--stop-at', str(stop)], cwd=execution['workspace'])
        from chronaris.evaluation.application_tasks.device_resume import compare_resume, qualified_execution
        qualification = Path(execution['root'])/'qualification.json'
        if not qualification.exists():
            compare_resume(execution_receipt)
        qualified_execution(execution_receipt, legacy_workspace=execution['legacy_workspace'], core_root=execution['core_root'])
        subset = root/'subset'
        entry = [sys.executable, str(driver/'run_chapter5_subset.py')]
        command('prepare_subset', entry+['prepare', '--root', str(subset), '--main-root', str(main_root),
            '--core-root', execution['core_root'], '--legacy-workspace', execution['legacy_workspace'],
            '--legacy-compatibility', str(execution_receipt), '--pause-receipt',
            str(Path(execution['core_root']).parent/'2026-10-12-semantic-qualification/pause/pause_receipt.json')])
        timings = {}
        for index in (0, 10):
            timings[str(index)] = command(f'full_fold_{index}', entry+['train', '--root', str(subset), '--unit', str(index)])
            from chronaris.evaluation.application_tasks.chapter5_subset import verified_subset, validate_unit
            plan = verified_subset(subset)
            validate_unit(read(subset/'units'/f'{index}.json'), plan['units'][index], plan)
        budget = calibrated_budget(read(prior_budget), timings)
        write(root/'calibrated_budget.json', budget)
        changed = any(r['changed_pairs'] for r in replay['rankings'])
        # A changed research ranking needs an explicit interpretation before mixing device results.
        if changed or budget['remaining_with_delivery_days'][1] > 7:
            state.update(status='needs_review', phase='two_full_folds_completed',
                reason='replay_ranking_changed' if changed else 'calibrated_budget_exceeds_seven_days')
            return write(root/'validation_state.json', state)
        for index in [i for i in range(18) if i not in (0, 10)]:
            command(f'subset_unit_{index}', entry+['train', '--root', str(subset), '--unit', str(index)])
        command('subset_freeze', entry+['freeze', '--root', str(subset)])
        state.update(status='completed', phase='subset_frozen')
        return write(root/'validation_state.json', state)
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(root/'validation_state.json', state)
        raise
