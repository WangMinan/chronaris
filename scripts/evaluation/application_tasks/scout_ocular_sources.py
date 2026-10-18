import argparse,inspect,json,shlex,subprocess
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_ocular_scout import scout_ocular_sources
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    output=a.report/'inventory.json'
    if output.exists():raise FileExistsError(output)
    q=json.loads(Path('docs/review/stage/fair-residual-20261019/qualification.json').read_text())
    assert file_digest(q['snapshot_path'])==q['snapshot_sha256']
    with np.load(q['snapshot_path'],allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    records={r['record_id']:dict(record_id=r['record_id'],subject=r['subject']) for r in meta['windows']}
    assert len(records)==84
    code='from pathlib import Path\nimport hashlib,numpy as np,pandas as pd\n'
    code+=inspect.getsource(file_digest)+'\n'+inspect.getsource(scout_ocular_sources)
    code+='\nimport json,sys\njson.dump(scout_ocular_sources(request["records"]),sys.stdout,allow_nan=False)\n'
    request=dict(records=list(records.values()),code=code)
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote('import json,sys;request=json.load(sys.stdin);exec(request["code"])')
    proc=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                         'wangminan@10.69.120.91',command],input=json.dumps(request).encode(),capture_output=True,timeout=1800)
    (a.report/'extraction.log').write_bytes(proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stderr.decode())
    result=json.loads(proc.stdout);save_json(output,result)
    save_rows(a.report/'record_inventory.csv',[{k:v for k,v in r.items() if not k.endswith('_counts')} for r in result['records']])
    print('records',len(result['records']),'time series',sum(r['has_time_series'] for r in result['records']))
