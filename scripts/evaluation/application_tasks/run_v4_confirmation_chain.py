"""Wait for reviewed development, freeze, then run simulation and native confirmation."""
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
root=Path(os.environ.get('CHRONARIS_V4_CONFIRMATION_CHAIN_ROOT',str(base/'2026-09-08_v4-confirmation-chain')))
root.mkdir(parents=True,exist_ok=True)
upstream=base/'2026-09-08_v4-development-chain/chain_state.json'
review=base/'2026-09-08_v4-candidate-review'
formal=base/'2026-09-08_v4-confirmation'
freeze=formal/'frozen_configuration.json'
path=root/'chain_state.json'
source=v4_workflow_source_sha256();script_hash=sha256_file(__file__)
state=json.loads(path.read_text()) if path.exists() else dict(source_code_sha256=source,script_sha256=script_hash,completed_steps=[])
if state['source_code_sha256']!=source or state['script_sha256']!=script_hash:raise ValueError('formal chain source changed')
if state.get('pid') != os.getpid():
    for previous in [state.get('pid'),state.get('current_child_pid'),*state.get('children',{}).values()]:
        if previous is None:continue
        try:command=Path(f'/proc/{previous}/cmdline').read_bytes()
        except FileNotFoundError:continue
        if b'2026-09-08_v4-confirmation-chain' in command or b'run_thesis_v4.py' in command:
            raise RuntimeError('a recorded confirmation process is still live; resume or inspect it before restarting')
state.update(pid=os.getpid(),worktree=str(Path.cwd()),current_child_pid=None)

def save():
    state['updated_at_unix_s']=time.time()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)

def run(stage, *extra):
    log_path=root/(stage+'.log')
    with log_path.open('a') as log:
        child=subprocess.Popen([sys.executable,'scripts/evaluation/application_tasks/run_thesis_v4.py',stage,*extra],stdout=log,stderr=subprocess.STDOUT)
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
    state.update(status='running_confirmation',freeze_sha256=digest,children={});save()
    def run_queues(jobs):
        children={};failures={}
        def launch(name):
            stage,backend,extra=jobs[name]
            command=[sys.executable,'scripts/evaluation/application_tasks/run_thesis_v4.py',stage,
                     '--freeze-path',str(freeze),'--freeze-sha256',digest,'--backend',backend,*extra]
            env=os.environ.copy()
            if backend=='nonparametric':env['CUDA_VISIBLE_DEVICES']=''
            log=(root/(name+'.log')).open('a')
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            children[name]=(child,log);state['children'][name]=child.pid;save()
        for name in jobs:
            if name not in state['completed_steps']:launch(name)
        while children:
            for name,(child,log) in list(children.items()):
                code=child.poll()
                if code is None:continue
                log.close();children.pop(name);state['children'].pop(name);save()
                queue_path=(formal/'core_ablations' if name.startswith('core_') else formal)/f'{name}_queue.json'
                result=json.loads(queue_path.read_text()) if queue_path.exists() else {}
                if code==0 and result.get('status')=='waiting_gpu':launch(name)
                elif code or result.get('status')!='completed':failures[name]={'exit_code':code,'status':result.get('status')}
                else:state['completed_steps'].append(name);save()
            if children:time.sleep(30);save()
        return failures
    failures=run_queues({
        'simulation_train_neural':('simulation-confirmation-train-cohort','neural',['--domain','simulation']),
        'core_train_neural':('core-ablation-train-cohort','neural',['--domain','simulation']),
        'simulation_train_nonparametric':('simulation-confirmation-train-cohort','nonparametric',['--domain','simulation']),
        'native_nonparametric':('native-confirmation-cohort','nonparametric',[])})
    if failures:
        state['failures']=failures;save();raise RuntimeError('formal training or fixed baseline has failed units')
    if 'core_model_freeze' not in state['completed_steps']:
        result=run('core-ablation-model-freeze','--domain','simulation','--freeze-sha256',digest)
        if result['status']!='frozen':raise RuntimeError('not all core ablation models completed')
        state['completed_steps'].append('core_model_freeze');save()
    model_path=formal/'simulation_frozen_models.json'
    if 'simulation_model_freeze' not in state['completed_steps']:
        result=run('simulation-model-freeze','--domain','simulation','--freeze-sha256',digest)
        if result['status']!='frozen':raise RuntimeError('not all selected simulation models completed')
        state['completed_steps'].append('simulation_model_freeze');save()
    model_digest=sha256_file(model_path)
    if 'simulation_generation' not in state['completed_steps']:
        result=run('simulation-confirmation-data','--domain','simulation','--freeze-sha256',digest,
                   '--model-freeze-path',str(model_path),'--model-freeze-sha256',model_digest)
        if result['status']!='completed':raise RuntimeError('formal simulation generation failed')
        state['completed_steps'].append('simulation_generation');save()
    failures=run_queues({
        'simulation_evaluate_neural':('simulation-confirmation-evaluate-cohort','neural',['--domain','simulation']),
        'core_evaluate_neural':('core-ablation-evaluate-cohort','neural',['--domain','simulation']),
        'native_neural':('native-confirmation-cohort','neural',[])})
    plan=json.loads((formal/'native_confirmation_plan.json').read_text())
    state.update(status='main_confirmation_completed_with_failures' if failures else 'main_confirmation_completed',
                 failures=failures,blocked_domains=plan['blocked_domains'],current_step=None,
                 remaining_scope=['separate_clock_and_response_mechanisms','combined_statistics_and_paper_evidence']);save()
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save()
    raise
