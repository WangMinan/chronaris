"""Read-only source check for all ambiguous zeros used by prospective auxiliary targets."""
import argparse
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import numpy as np
from chronaris.dataset.cogpilot_history_supervision import audit_zero_anchors
from chronaris.dataset.cogpilot_performance import file_digest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    with np.load(a.snapshot,allow_pickle=False) as raw:meta=json.loads(str(raw['metadata']));anchors=raw['anchor']
    rows=[]
    for i,j in np.argwhere(anchors[:,15:]==0):
        rows.append(dict(meta['windows'][i],sample_index=int(i),query_index=int(j+15)))
    root=Path(__file__).resolve().parents[3]
    code='\n'.join((root/'src/chronaris/dataset'/name).read_text() for name in ('cogpilot_performance.py','cogpilot_history_features.py'))
    code+='\n'+inspect.getsource(audit_zero_anchors)+'\nprint(json.dumps(audit_zero_anchors(request)))\n'
    request=dict(rows=rows,sources=meta['sources'],code=code)
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    result=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                           'wangminan@10.69.120.91',command],input=json.dumps(request),capture_output=True,text=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    receipt=json.loads(result.stdout);assert len(receipt['checks'])==len(rows)
    receipt.update(snapshot_sha256=file_digest(a.snapshot),sample_ids=[r['sample_id'] for r in meta['windows']],
                   examined_zero_queries=len(rows),query_start_index=15)
    mask=np.ones_like(anchors,bool);mask[:,:15]=False
    for r in receipt['checks']:mask[r['sample_index'],r['query_index']]=r['valid']
    mask_path=a.output.with_suffix('.npz');np.savez_compressed(mask_path,mask=mask)
    receipt.update(mask_path=str(mask_path.resolve()),mask_sha256=file_digest(mask_path),
                   valid_auxiliary_pairs=int((mask[:,15:80]&mask[:,31:96]).sum()),
                   missing_queries=sum(not r['valid'] for r in receipt['checks']))
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print('checked zeros',len(rows),'missing',receipt['missing_queries'],'valid pairs',receipt['valid_auxiliary_pairs'])
