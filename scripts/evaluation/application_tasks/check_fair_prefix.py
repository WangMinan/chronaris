"""Read-only real-signal prefix isolation checks for fixed registered windows."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from chronaris.dataset.cogpilot_fair_residual import audit_fair_prefix

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.snapshot) as raw:meta=json.loads(str(raw['metadata']))
    first={}
    for row in sorted(meta['windows'],key=lambda r:(r['record_id'],r['cutoff_s'])):
        if row['accepted']:first.setdefault(row['subject'],row)
    code=(root/'src/chronaris/dataset/cogpilot_performance.py').read_text()+'\n'
    code+=(root/'src/chronaris/dataset/cogpilot_history_features.py').read_text()+'\n'
    code+=inspect.getsource(audit_fair_prefix)+'\nprint(json.dumps(audit_fair_prefix(request)))\n'
    request=dict(code=code,windows=list(first.values()),source_hashes=meta['sources'])
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10','wangminan@10.69.120.91',command],input=json.dumps(request),capture_output=True,text=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    payload=json.loads(result.stdout);args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(payload['status'],len(payload['checks']),max(r['max_delta'] for r in payload['checks']))
