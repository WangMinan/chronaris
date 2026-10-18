"""Gather only raw-source audit results; no model execution or remote file writes."""
import json
import shlex
import subprocess
import torch
from audit_io import HERE,ASSETS,PREP,manifest,sha,save_json


def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    output=ASSETS/'raw_response.json'
    if output.exists():raise FileExistsError('raw response already retained')
    target=torch.load(PREP/'cogpilot/task_targets.pt',weights_only=True)
    ys={s:float(v) for s,v in zip(target['sample_ids'],target['values']['event_response'].reshape(-1))}
    events=[]
    for r in manifest('cogpilot'):
        if '::event' in r['sample_id']:
            events.append(dict(sample_id=r['sample_id'],subject=r['subject_id'],record_id=r['record_id'],
                source_paths=r['source_paths'],cached_target=ys[r['sample_id']],
                start_native=r['record_time_origin_native']+r['context_start_s']/86400))
    clare={}
    for row in manifest('clare'):clare.setdefault(row['record_id'],[]).append(row)
    implementation=HERE.parents[3]/'src/chronaris/evaluation/application_tasks/event_target_validation.py'
    request=dict(events=events,clare_records=clare,implementation=implementation.read_text())
    code=(HERE/'raw_remote.py').read_text()+'\nprint(json.dumps(run(request),allow_nan=False))\n'
    request['code']=code
    remote='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
        'import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o',
        'ConnectTimeout=10','wangminan@10.69.120.91',remote],input=json.dumps(request),
        capture_output=True,text=True,timeout=7200)
    (HERE/'raw_remote.log').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    response=json.loads(result.stdout)
    save_json(output,response)
    save_json(HERE/'raw_receipt.json',dict(status='completed',response_path=str(output),
        response_sha256=sha(output),implementation_sha256=sha(implementation),
        event_count=len(response['events']),record_count=len(response['records']),
        clare_windows=len(response['clare']),remote_sources=response['sources'],encoder_updates=0))
    print('raw audit completed',len(response['events']),len(response['clare']),flush=True)


if __name__=='__main__':main()
