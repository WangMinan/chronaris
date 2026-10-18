import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.clare_frequency import frequency_features,extract_clare_frequency

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.partial').exists():raise FileExistsError(a.output)
    subjects=['1337','1419','1544','1688','1717','1892','1929','1933','1953','1981','1390','1624','1629','1674']
    code='from pathlib import Path\nimport hashlib,json,numpy as np,pandas as pd\nfrom scipy.signal import welch,butter,sosfilt\n'
    code+='\n'.join(inspect.getsource(f) for f in (file_digest,frequency_features,extract_clare_frequency))
    code+='\nimport io,sys\nb=io.BytesIO()\nnp.savez_compressed(b,**extract_clare_frequency(request))\nsys.stdout.buffer.write(b.getvalue())\n'
    request=dict(subjects=subjects,root='/home/wangminan/dataset/chronaris/clare',code=code)
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.partial').open('xb') as out,a.output.with_suffix('.log').open('xb') as log:
        r=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                          'wangminan@10.69.120.91',command],input=json.dumps(request).encode(),stdout=out,stderr=log,timeout=3600)
    if r.returncode:raise RuntimeError('source qualification failed; see log')
    a.output.with_suffix('.partial').rename(a.output);print(a.output)
