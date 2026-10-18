"""Send only the frozen development identities and read-only inspection code."""
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
ASSETS=ROOT/'artifacts/application_evaluation/2026-10-18-task-anchor'


def main():
    output=ASSETS/'remote_sources.json'
    if output.exists():raise FileExistsError('retain previous source inspection')
    raw=json.loads((ROOT/'artifacts/application_evaluation/2026-10-18-frozen-feasibility/raw_response.json').read_text())
    records={}
    for e in raw['events']:records.setdefault(e['record_id'],[]).append(e)
    request=dict(clare_subjects=sorted({r['subject'] for r in raw['clare']}),cog_records=records)
    request['code']=(HERE/'remote_sources.py').read_text()+'\nprint(json.dumps(inspect(request),allow_nan=False))'
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(
        'import sys,json;request=json.load(sys.stdin);exec(request["code"])')
    result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'wangminan@10.69.120.91',command],input=json.dumps(request),capture_output=True,text=True,timeout=1800)
    (HERE/'remote_sources.log').write_text(result.stderr)
    if result.returncode:raise RuntimeError(result.stderr)
    data=json.loads(result.stdout);output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (HERE/'remote_receipt.json').write_text(json.dumps(dict(sources=data['sources'],
        timestamp_files=len(data['timestamps']),headers=len(data['headers']),performance_records=len(data['performance']),
        zero_encoder_updates=True),ensure_ascii=False,indent=2)+'\n')
    print('read-only sources complete',len(data['sources']))


if __name__=='__main__':main()
