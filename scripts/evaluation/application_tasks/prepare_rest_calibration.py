import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from chronaris.dataset.cogpilot_performance import STREAM_COLUMNS,file_digest
from chronaris.dataset.cogpilot_rest_calibration import extract_rest_calibration

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.partial').exists():raise FileExistsError(a.output)
    with np.load(a.original,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    sessions={}
    for r in meta['windows']:
        key=(r['subject'],Path(r['record_id']).parent.name);stamp=r['source_origin_native']+(r['cutoff_s']-30)/86400
        sessions[key]=min(sessions.get(key,float('inf')),stamp)
    assert len(sessions)==23 and len({s[0] for s in sessions})==23
    requested=[dict(subject=s,session=session,first_input_native=stamp) for (s,session),stamp in sorted(sessions.items())]
    code='from pathlib import Path\nimport hashlib,numpy as np,pandas as pd\nSTREAM_COLUMNS='+repr(STREAM_COLUMNS)+'\n'
    code+=inspect.getsource(file_digest)+'\n'+inspect.getsource(extract_rest_calibration)
    code+='\nimport json,sys\njson.dump(extract_rest_calibration(request),sys.stdout,allow_nan=False)\n'
    request=dict(root='/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-rest',sessions=requested)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.request.json').write_text(json.dumps(dict(request,input_path=str(Path(a.original).resolve()),input_sha256=file_digest(a.original)),indent=2))
    request['code']=code
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    with a.output.with_suffix('.partial').open('xb') as out,a.output.with_suffix('.log').open('xb') as log:
        r=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                          'wangminan@10.69.120.91',command],input=json.dumps(request).encode(),stdout=out,stderr=log,timeout=3600)
    if r.returncode:raise RuntimeError('rest source audit failed; see log')
    a.output.with_suffix('.partial').rename(a.output);print(a.output)
