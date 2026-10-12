"""Resume an interrupted full fold, retaining its pre-outage timing evidence."""
from pathlib import Path
import os
import subprocess
import sys
import time

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write


def recovered_timing(manifest, *, started, elapsed, peak, exit_code, log):
    if manifest['elapsed_before_outage_lower_bound_s'] <= 0 or elapsed < 0:
        raise ValueError('invalid interrupted timing')
    return dict(command=manifest['command'], exit_code=exit_code, started_unix_s=started,
        wall_seconds=manifest['elapsed_before_outage_lower_bound_s']+elapsed,
        resumed_wall_seconds=elapsed, pre_outage_seconds_lower_bound=manifest['elapsed_before_outage_lower_bound_s'],
        timing_scope='active execution lower bound; excludes powered-off time; includes recomputation',
        sampled_device_peak_mib=peak, memory_scope='whole device sampled every 5 seconds after restart only',
        log=str(log), recovery_manifest=manifest['manifest_path'])


def resume_validation(manifest_path):
    manifest = read(manifest_path)
    if manifest['manifest_path'] != str(Path(manifest_path).resolve()):
        raise ValueError('recovery manifest path changed')
    verify(manifest['files'])
    root = Path(manifest['root']); config = read(root/'validation_config.json')
    workspace = Path(config['driver_workspace'])
    checkpoint = Path(manifest['checkpoint'])
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('interrupted checkpoint changed before restart')
    for pid in manifest['old_pids']:
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and str(root).encode() in command.read_bytes():
            raise ValueError('original experiment is still running')
    receipt = root/'commands'/f"{manifest['phase']}.json"
    if receipt.exists():
        raise ValueError('command receipt already exists; inspect before retrying')
    state = read(root/'validation_state.json')
    state.update(status='running', phase=manifest['phase'], pid=os.getpid(), recovery_manifest=str(manifest_path))
    state.pop('child_pid', None)
    write(root/'validation_state.json', state)
    log = root/'logs'/f"{manifest['phase']}.log"
    started, peak = time.time(), 0
    with log.open('a') as handle:
        child = subprocess.Popen(manifest['command'], cwd=workspace, stdout=handle, stderr=subprocess.STDOUT)
        state['child_pid'] = child.pid; write(root/'validation_state.json', state)
        try:
            while child.poll() is None:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                peak = max(peak, int(value.strip().splitlines()[0]))
                time.sleep(5)
        except BaseException:
            child.terminate(); child.wait()
            raise
    result = recovered_timing(manifest, started=started, elapsed=time.time()-started,
                             peak=peak, exit_code=child.returncode, log=log)
    write(receipt, result)
    if child.returncode:
        state.update(status='failed', error=f'restored worker exited {child.returncode}')
        state.pop('child_pid', None); write(root/'validation_state.json', state)
        raise RuntimeError(state['error'])
    # The frozen controller revalidates the completed fold and uses the combined active time.
    args = [sys.executable, str(workspace/'scripts/evaluation/application_tasks/run_local_validation.py')]
    for key in ('root', 'replay_root', 'replay_pid', 'execution_receipt', 'main_root', 'prior_budget'):
        args += ['--'+key.replace('_', '-'), str(config[key])]
    os.chdir(workspace)
    os.execv(sys.executable, args)
