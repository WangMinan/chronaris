"""Wait for reviewed development, freeze, then run CPU/GPU native confirmation."""
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
root=base/'2026-09-08_v4-confirmation-chain';root.mkdir(parents=True,exist_ok=True)
upstream=base/'2026-09-08_v4-development-chain/chain_state.json'
review=base/'2026-09-08_v4-candidate-review'
formal=base/'2026-09-08_v4-confirmation'
freeze=formal/'frozen_configuration.json'
path=root/'chain_state.json'
source=v4_workflow_source_sha256();script_hash=sha256_file(__file__)
state=json.loads(path.read_text()) if path.exists() else dict(source_code_sha256=source,script_sha256=script_hash,completed_steps=[])
if state['source_code_sha256']!=source or state['script_sha256']!=script_hash:raise ValueError('formal chain source changed')
state.update(pid=os.getpid(),worktree=str(Path.cwd()),current_child_pid=None)

def save():
    state['updated_at_unix_s']=time.time()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)

def run(stage):
    log_path=root/(stage+'.log')
    with log_path.open('a') as log:
        child=subprocess.Popen([sys.executable,'scripts/evaluation/application_tasks/run_thesis_v4.py',stage],stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',current_step=stage,current_child_pid=child.pid);save()
        while True:
            try:code=child.wait(timeout=30);break
            except subprocess.TimeoutExpired:save()
    state['current_child_pid']=None;save()
    if code:raise RuntimeError(f'{stage} failed; inspect {log_path}')
    output=log_path.read_text();start=output.rfind('\n{')
    return json.loads(output[start+1:] if start>=0 else output)

try:
    state.update(status='waiting_for_development',current_step='development');save()
    while True:
        development=json.loads(upstream.read_text())
        if development['source_code_sha256']!='e82738fa97687d482f9b0b3d5df66cfd0254ed86886ebea073768a8166884d86':
            raise ValueError('development chain source changed')
        if development['status']=='failed':raise RuntimeError('development chain failed; inspect and repair its unit')
        if development['status']=='development_complete_ready_for_freeze':break
        try:os.kill(development['pid'],0)
        except ProcessLookupError:raise RuntimeError('development chain process stopped before completion')
        if b'2026-09-08_v4-development-chain/run_chain.py' not in Path(f"/proc/{development['pid']}/cmdline").read_bytes():
            raise RuntimeError('development PID no longer belongs to the frozen chain')
        save();time.sleep(30)
    if 'cuda_validation' not in state['completed_steps']:
        while True:
            result=run('configuration-cuda-validation')
            if result['status']=='completed':break
            if result['status']!='waiting_gpu':raise RuntimeError('current-source CUDA validation failed')
            state['status']='waiting_gpu';save();time.sleep(30)
        state['completed_steps'].append('cuda_validation');save()
    if 'freeze' not in state['completed_steps']:
        while True:
            result=run('freeze-configuration')
            if result['status']=='frozen':break
            if not result['status'].startswith('waiting_'):raise RuntimeError('configuration freeze failed')
            previous=sha256_file(review/'adoption_decisions.json')
            state.update(status='waiting_for_development_resolution',freeze_status=result['status']);save()
            while sha256_file(review/'adoption_decisions.json')==previous:time.sleep(30);save()
        state['completed_steps'].append('freeze');save()
    digest=sha256_file(freeze)
    if state.get('freeze_sha256',digest)!=digest:raise ValueError('configuration changed after confirmation started')
    state.update(status='running_native_confirmation',freeze_sha256=digest,children={});save()
    children={}
    def launch(backend):
        command=[sys.executable,'scripts/evaluation/application_tasks/run_thesis_v4.py','native-confirmation-cohort',
                 '--freeze-path',str(freeze),'--freeze-sha256',digest,'--backend',backend]
        env=os.environ.copy()
        if backend=='nonparametric':env['CUDA_VISIBLE_DEVICES']=''
        log=(root/(backend+'.log')).open('a')
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        children[backend]=(child,log);state['children'][backend]=child.pid;save()
    for backend in ('neural','nonparametric'):
        if backend not in state['completed_steps']:launch(backend)
    failures={}
    while children:
        for backend,(child,log) in list(children.items()):
            code=child.poll()
            if code is None:continue
            log.close();children.pop(backend);state['children'].pop(backend);save()
            queue_path=formal/f'native_{backend}_queue.json'
            result=json.loads(queue_path.read_text()) if queue_path.exists() else {}
            if code==0 and result.get('status')=='waiting_gpu':launch(backend)
            elif code or result.get('status')!='completed':failures[backend]={'exit_code':code,'status':result.get('status')}
            else:state['completed_steps'].append(backend);save()
        if children:time.sleep(30);save()
    plan=json.loads((formal/'native_confirmation_plan.json').read_text())
    state.update(status='native_confirmation_completed_with_failures' if failures else 'native_confirmation_completed',
                 failures=failures,blocked_domains=plan['blocked_domains'],current_step=None);save()
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save()
    raise
