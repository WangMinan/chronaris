import argparse,gzip,inspect,json,shlex,subprocess
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.clare_frequency import audit_clare_gaps

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.snapshot,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    code='from pathlib import Path\nimport hashlib,numpy as np,pandas as pd\n'
    code+=inspect.getsource(file_digest)+'\n'+inspect.getsource(audit_clare_gaps)
    code+='\nimport json,gzip,sys\nsys.stdout.buffer.write(gzip.compress(json.dumps(audit_clare_gaps(request),allow_nan=False).encode()))\n'
    request=dict(root='/home/wangminan/dataset/chronaris/clare',windows=meta['windows'],sources=meta['sources'],code=code)
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    with a.output.with_suffix('.partial').open('xb') as out:
        proc=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                             'wangminan@10.69.120.91',command],input=json.dumps(request).encode(),stdout=out,stderr=subprocess.PIPE,timeout=1800)
    a.output.with_suffix('.log').write_bytes(proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr.decode())
    a.output.with_suffix('.partial').rename(a.output);print(a.output)
