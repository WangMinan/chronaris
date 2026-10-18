"""Check that the candidate official performance targets have finite source support."""
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    root=HERE.parents[3]/'artifacts/application_evaluation/2026-10-18-task-anchor'
    source=json.loads((root/'remote_sources.json').read_text())
    files={r['file']:source['sources'][r['file']] for r in source['performance']}
    code='''import json,sys,hashlib,pandas as pd,numpy as np
files=json.load(sys.stdin);rows=[]
for p,h in files.items():
 with open(p,'rb') as f:before=hashlib.file_digest(f,'sha256').hexdigest()
 assert before==h
 d=pd.read_csv(p);a=d[['time_dn','glideslope_error_deg','localizer_error_deg','airspeed_error_kts','total_error']].to_numpy(float)
 rows.append(dict(file=p,rows=len(d),nonfinite_cells=int((~np.isfinite(a)).sum()),sha256=before))
 with open(p,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==before
print(json.dumps(rows))'''
    remote='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(code)
    result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'wangminan@10.69.120.91',remote],input=json.dumps(files),capture_output=True,text=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    rows=json.loads(result.stdout)
    assert len(rows)==84 and all(x['nonfinite_cells']==0 for x in rows)
    (HERE/'performance_finite_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('84 performance records, all time and target cells finite')


if __name__=='__main__':main()
