"""Bounded native development probe, with phase timing and no final-role access."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

from chronaris.evaluation.application_tasks.chapter5_assets import read, write


def probe_domain(domain, root):
    import torch
    from chronaris.evaluation.application_tasks.common_downstream_smoke import run_common_contract_smoke
    result = run_common_contract_smoke(domain=domain, output_root=root, methods=('chronaris',),
        recipe='thesis_reference', cuda_graph_recurrence=True)
    if result['status'] != 'completed': return result
    phases = {}
    for row in result['results']:
        if 'training' not in row: continue
        training = row['training']
        saved = torch.load(training['last_checkpoint_path'], map_location='cpu', weights_only=True)
        phases[row['route']] = dict(optimizer_updates=training['optimizer_updates'],
            validation_seconds=saved.get('validation_elapsed_s'), training_seconds=saved.get('training_elapsed_s'),
            peak_cuda_bytes=row['peak_cuda_bytes'], export_seconds=row['export_seconds'])
    return write(Path(root)/'probe.json', dict(status='completed', domain=domain, phase_seconds=result['phase_seconds'],
        routes=phases, device=torch.cuda.get_device_name(), confirmation_opened=False,
        scope='2_pretraining_2_warmup_2_joint_development_updates_not_full_unit_cost'))


def probe_pair(output_root):
    root = Path(output_root).resolve()
    if root.exists(): raise ValueError('budget probe needs a new output root')
    root.mkdir(parents=True)
    entry = Path(__file__).parents[4]/'scripts/evaluation/application_tasks/run_chapter5_budget_probe.py'
    started = time.monotonic(); results = []
    for domain in ('clare', 'cogpilot'):
        seconds = min(897., max(0., 1797.-(time.monotonic()-started)))
        if seconds <= 0: break
        with (root/f'{domain}.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(entry), '--root', str(root/domain), '--domain', domain],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=seconds)
                status = 'completed' if code==0 and (root/domain/'probe.json').exists() else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                status = 'timed_out'; code = child.returncode
        results.append(dict(domain=domain, status=status, returncode=code,
            report=str(root/domain/'probe.json'), log=str(root/f'{domain}.log')))
        write(root/'budget_probe.json', dict(status='completed' if len(results)==2 and all(r['status']=='completed' for r in results) else 'partial',
            domains=results, total_seconds=time.monotonic()-started, confirmation_opened=False))
    return read(root/'budget_probe.json')
