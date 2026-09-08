"""Outer-fold formal units; neural and nonparametric queues can run concurrently."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from chronaris.evaluation.application_tasks.v4_confirmation_training import read_frozen_configuration
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def build_native_confirmation_plan(freeze_path,freeze_sha256):
    frozen=read_frozen_configuration(freeze_path,freeze_sha256);units=[];blocked=[]
    fold_counts=frozen.get('native_fold_counts',{'cogpilot':5,'clare':5,'dingxin':2})
    for domain,folds in fold_counts.items():
        if frozen['domain_status'][domain]!='enabled':
            blocked.append(dict(domain=domain,reason=frozen['domain_status'][domain],evaluation_units=folds*3*6*2));continue
        for method in sorted(frozen['methods']['self_supervised']):
            names={frozen['methods'][route][method]['name'] for route in ('self_supervised','task_guided')}
            for candidate in sorted(names):
                routes=[route for route in ('self_supervised','task_guided') if frozen['methods'][route][method]['name']==candidate]
                for fold in range(folds):
                    for seed in (17,29,43):
                        units.append(dict(domain=domain,method=method,candidate_name=candidate,fold_index=fold,seed=seed,routes=routes,
                                          backend='nonparametric' if method=='naive_time_sync' else 'neural'))
    return dict(format='chronaris.v4_native_confirmation_plan.v1',freeze_sha256=freeze_sha256,units=units,blocked_domains=blocked,
                enabled_evaluation_units=sum(len(unit['routes']) for unit in units),
                intended_native_evaluation_units=sum(fold_counts.values())*3*6*2,
                evaluation_protocol=frozen.get('domain_evaluation_protocol',{}))


def run_native_confirmation_cohort(*, freeze_path, freeze_sha256, output_root, backend,
                                   data_root='artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared',
                                   registry_path='docs/requirements/thesis-v4-public-subjects.json'):
    if backend not in ('neural','nonparametric'):raise ValueError('unknown confirmation backend')
    plan=build_native_confirmation_plan(freeze_path,freeze_sha256)
    def unit_args(unit):
        stage='naive-confirmation-unit' if backend=='nonparametric' else 'native-confirmation-unit'
        arguments=[stage,'--domain',unit['domain'],'--fold-index',str(unit['fold_index']),'--seed',str(unit['seed']),
                   '--data-root',str(data_root),'--registry',str(registry_path)]
        if backend=='neural':arguments+=['--method',unit['method'],'--candidate-name',unit['candidate_name']]
        return arguments
    return run_confirmation_units(plan=plan,freeze_path=freeze_path,freeze_sha256=freeze_sha256,
        output_root=output_root,backend=backend,queue_prefix='native',unit_args=unit_args,result_name='confirmation_unit.json')


def run_confirmation_units(*,plan,freeze_path,freeze_sha256,output_root,backend,queue_prefix,unit_args,result_name):
    """Shared subprocess, heartbeat and resume handling for both native and simulation units."""
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    plan_path=root/f'{queue_prefix}_confirmation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('confirmation plan changed')
    if not plan_path.exists():
        temporary=plan_path.with_suffix(f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(plan,indent=2)+'\n');temporary.replace(plan_path)
    path=root/f'{queue_prefix}_{backend}_queue.json'
    state=json.loads(path.read_text()) if path.exists() else dict(freeze_sha256=freeze_sha256,completed_units=[],failed_units=[],errors={})
    if state['freeze_sha256']!=freeze_sha256:raise ValueError('queue belongs to another configuration freeze')
    state.update(pid=os.getpid(),backend=backend,blocked_domains=plan.get('blocked_domains',[]))
    state.setdefault('completed_receipts',{})
    def save():
        state['updated_at_unix_s']=time.time()
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
    project=Path(__file__).parents[4]
    for unit in plan['units']:
        if unit['backend']!=backend:continue
        key=f"{unit['domain']}/{unit['method']}/{unit['candidate_name']}/fold{unit['fold_index']+1:02d}/seed{unit['seed']}"
        if key in state['completed_units']:
            receipt=root/key/result_name
            if not receipt.exists() or state['completed_receipts'].get(key)!=sha256_file(receipt):
                raise ValueError('completed unit receipt changed; refuse silent reuse')
            continue
        if key in state['failed_units']:continue
        command=[sys.executable,str(project/'scripts/evaluation/application_tasks/run_thesis_v4.py'),*unit_args(unit),
                 '--freeze-path',str(freeze_path),'--freeze-sha256',freeze_sha256,'--output-root',str(root)]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(project/'src'))
        if backend=='nonparametric':env['CUDA_VISIBLE_DEVICES']=''
        log_path=root/(queue_prefix+'__'+key.replace('/','__')+'.log')
        with log_path.open('a') as log:
            child=subprocess.Popen(command,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',current_unit=key,current_child_pid=child.pid);save()
            while True:
                try:code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired:save()
        state['current_child_pid']=None
        result_path=root/key/result_name
        output=log_path.read_text();start=output.rfind('\n{')
        reply=json.loads(output[start+1:] if start>=0 else output) if code==0 else None
        if reply and reply.get('status')=='waiting_gpu':
            state['status']='waiting_gpu';save();return state
        if code or not result_path.exists():
            state['failed_units'].append(key);state['errors'][key]=f'exit_code={code}; see {log_path}'
        else:
            result=json.loads(result_path.read_text())
            if not result['completed'] or result['freeze_sha256']!=freeze_sha256:raise ValueError('unit returned an invalid completion')
            state['completed_units'].append(key)
            state['completed_receipts'][key]=sha256_file(result_path)
        save()
    state.update(status='completed_with_failures' if state['failed_units'] else 'completed',current_unit=None);save()
    return state
