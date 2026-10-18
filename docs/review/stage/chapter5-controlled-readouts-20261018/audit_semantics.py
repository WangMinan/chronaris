"""Raw-source quality details and metadata labels, without altering target definitions."""
import json
import shlex
import subprocess

import numpy as np
import torch

from common import HERE, PREP, ASSETS, manifest, save_json, save_csv, sha

REMOTE = r'''
import json,sys,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
request=json.load(sys.stdin)
records={r['record_id']:r for r in request}
quality=[]
for row in records.values():
 if row['domain']!='clare':continue
 for path,column in zip(row['source_paths'][1:],['GSR Conductance CAL','ECG LL-RA CAL']):
  f=pd.read_csv(path,usecols=['Timestamp',column]);finite=f.dropna(subset=[column])
  repeated=finite[finite.Timestamp.duplicated(keep=False)]
  conflict=int((repeated.groupby('Timestamp')[column].nunique()>1).sum())
  quality.append(dict(path=path,column=column,rows=len(f),finite=len(finite),
   finite_duplicate_rows=int(finite.Timestamp.duplicated().sum()),conflicting_timestamps=conflict,
   finite_start=float(finite.Timestamp.min()),finite_end=float(finite.Timestamp.max()),
   value_min=float(finite[column].min()),value_max=float(finite[column].max())))
root=Path('/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0')
p=root/'dataPackage/task-ils/PerfMetrics.csv'
f=pd.read_csv(p)
with p.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
print(json.dumps(dict(clare_quality=quality,perf_metrics_columns=list(f.columns),
 perf_metrics=f.to_dict('records'),perf_metrics_sha256=sha,
 clare_readme=Path('/home/wangminan/dataset/chronaris/clare/README.txt').read_text()),allow_nan=False))
'''


def main():
    output=ASSETS/'source_semantics.json'
    if not output.exists():
        command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(REMOTE)
        p=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                          'wangminan@10.69.120.91',command],input=(HERE/'raw_selection.json').read_text(),
                         text=True,capture_output=True,timeout=1200)
        if p.returncode:raise RuntimeError(p.stderr)
        save_json(output,json.loads(p.stdout))
    raw=json.loads(output.read_text())
    save_csv(HERE/'clare_source_quality.csv',raw['clare_quality'])
    # The original paper defines >=5; this is a descriptive disagreement count only.
    profile=[]
    for domain in ('clare','cogpilot'):
        target=torch.load(PREP/domain/'task_targets.pt',weights_only=True)
        rows=manifest(domain);assert list(target['sample_ids'])==[r['sample_id'] for r in rows]
        g=np.array([r['subject_id'] for r in rows])
        for task,value in target['values'].items():
            y=value.numpy().reshape(-1);m=target['valid_masks'][task].numpy().reshape(-1)
            for subject in sorted(set(g)):
                v=y[m&(g==subject)]
                profile.append(dict(domain=domain,task=task,subject=subject,samples=len(v),mean=float(np.mean(v)),
                    std=float(np.std(v)),minimum=float(np.min(v)),maximum=float(np.max(v)),
                    counts=json.dumps(dict(zip(*[a.tolist() for a in np.unique(v,return_counts=True)]))) if 'classification' in task or task=='difficulty' else '',
                    scores_5_or_6=int(((v>=5)&(v<7)).sum()) if task=='workload_regression' else '',
                    high_at_7=int((v>=7).sum()) if task=='workload_regression' else '',
                    high_at_5=int((v>=5).sum()) if task=='workload_regression' else ''))
    save_csv(HERE/'target_profile.csv',profile)
    lookup={(int(r['subject']),int(r['date']),int(r['run'])):int(r['difficulty']) for r in raw['perf_metrics']}
    difficulty=[]
    import re
    for r in manifest('cogpilot'):
        if not r['task_valid_mask']['difficulty']:continue
        sid=r['sample_id'];subject=int(sid.split('::')[0].removeprefix('sub-cp'))
        date=int(re.search(r'ses-(\d+)',sid)[1]);run=int(re.search(r'run-(\d+)',sid)[1])
        expected=int(re.search(r'level-(\d+)',sid)[1])
        actual=lookup.get((subject,date,run))
        difficulty.append(dict(sample_id=sid,folder_difficulty=expected,metadata_difficulty=actual,matches=actual==expected))
    save_csv(HERE/'difficulty_metadata.csv',difficulty)
    assert all(r['matches'] for r in difficulty), 'Difficulty metadata differs from prepared folder target'
    print(raw['perf_metrics_columns'])
    save_json(HERE/'semantics_sources.json',dict(source_semantics_path=str(output),source_semantics_sha256=sha(output),
        dictionary_path=str(ASSETS/'DataDictionary.pdf'),dictionary_sha256=sha(ASSETS/'DataDictionary.pdf'),
        clare_paper='https://arxiv.org/html/2404.17098v2',clare_paper_section='VI-B',
        original_binary_threshold=5,frozen_binary_threshold=7,labels_changed=False))


if __name__=='__main__':main()
