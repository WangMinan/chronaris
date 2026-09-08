"""Frozen CLI succession from the active initial screen through reviewed decisions."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path.cwd()/'src'))
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

base=Path('artifacts/application_evaluation')
root=base/'2026-09-08_v4-development-chain'
upstream=base/'2026-09-08_v4-candidate-pressure/queue_state.json'
public=base/'2026-09-08_v4-public-screen'
review=base/'2026-09-08_v4-candidate-review'
pressure=base/'2026-09-08_v4-review-pressure'
path=root/'chain_state.json'
source=v4_workflow_source_sha256()
script_hash=sha256_file(__file__)
state=json.loads(path.read_text()) if path.exists() else dict(source_code_sha256=source,script_sha256=script_hash,completed_steps=[])
if state['source_code_sha256']!=source or state['script_sha256']!=script_hash:raise ValueError('frozen chain source changed')
state.update(pid=os.getpid(),worktree=str(Path.cwd()),current_child_pid=None,confirmation_started=False)

def save():
    state['updated_at_unix_s']=time.time()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)

def run(name,command):
    with (root/(name+'.log')).open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',current_step=name,current_child_pid=child.pid);save()
        while True:
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:save()
    state['current_child_pid']=None;save()
    if code:raise RuntimeError(f'{name} failed with exit code {code}; inspect its log')
    return root/(name+'.log')

try:
    state.update(status='waiting_for_initial_pressure',current_step='initial_pressure');save()
    while True:
        initial=json.loads(upstream.read_text())
        if initial['source_code_sha256']!='9bcaeaec8a4f06502ed9e6309ff34832cf61f7e25e504a4d1c2d4534a2ff9b43':
            raise ValueError('initial pressure source changed')
        if initial['status'] in ('failed','completed_with_failures'):
            raise RuntimeError('initial pressure has execution failures; repair before public selection')
        if initial['status']=='completed' and not initial['current_child_pid']:break
        try:os.kill(initial['pid'],0)
        except ProcessLookupError:raise RuntimeError('upstream pressure process stopped before completion')
        if b'2026-09-08_v4-candidate-pressure/run_queue.py' not in Path(f"/proc/{initial['pid']}/cmdline").read_bytes():
            raise RuntimeError('upstream PID no longer belongs to the frozen pressure queue')
        save();time.sleep(30)
    if 'full_cuda_tests' not in state['completed_steps']:
        with open('/tmp/chronaris-v4-gpu.lock','a') as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB);break
                except BlockingIOError:
                    state.update(status='waiting_gpu',current_step='full_cuda_tests');save();time.sleep(30)
            run('full_cuda_tests',[sys.executable,'-m','pytest','-q'])
        state['completed_steps'].append('full_cuda_tests');save()
    steps=[('public-screen',public/'run_state.json','completed',[]),
           ('public-screen-results',public/'results_summary.json','public_scores_verified_not_final_adoption',[]),
           ('candidate-review-cohort',review/'run_state.json','completed',[]),
           ('public-review-results',review/'public_results_summary.json','public_review_verified_not_final_selection',[]),
           ('review-pressure-cohort',pressure/'queue_state.json','completed',[]),
           ('simulation-review-results',review/'simulation_results_summary.json','simulation_review_verified_not_final_selection',['--domain','simulation']),
           ('candidate-adoption',review/'adoption_decisions.json','single_factor_decisions_ready_not_frozen',[])]
    for stage,result_path,expected,extra in steps:
        if stage in state['completed_steps']:continue
        while True:
            log=run(stage,[sys.executable,'scripts/evaluation/application_tasks/run_thesis_v4.py',stage,*extra])
            output=log.read_text();start=output.rfind('\n{')
            reply=json.loads(output[start+1:]) if start>=0 else json.loads(output)
            if reply.get('status')!='waiting_gpu':break
            state.update(status='waiting_gpu');save();time.sleep(30)
        result=json.loads(result_path.read_text())
        if result['status']!=expected:raise RuntimeError(f'{stage} returned {result["status"]}; inspect evidence before continuing')
        state.setdefault('results',{})[stage]={'path':str(result_path),'sha256':sha256_file(result_path)}
        state['completed_steps'].append(stage);save()
    state.update(status='development_complete_ready_for_freeze',current_step=None);save()
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save()
    raise
