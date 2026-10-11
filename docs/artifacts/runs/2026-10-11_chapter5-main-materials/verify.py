"""Verify portable input hashes and descriptive summaries using the standard library."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def verify(root):
    evidence=json.loads((root/'evidence.json').read_text(encoding='utf-8'))
    for name,digest in evidence['data_files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('input outside package')
        with path.open('rb') as handle:
            if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('input changed: '+name)
    def rows(name):
        with (root/'data'/name).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
    keys=('domain','method','route','consumer','task','metric','error_units')
    grouped=defaultdict(list)
    for row in rows('setting_metrics.csv'):grouped[tuple(row[k] for k in keys)].append(row['value'])
    summary=rows('summary.csv')
    if len(summary)!=len(grouped):raise ValueError('summary coverage changed')
    for row in summary:
        raw=grouped[tuple(row[k] for k in keys)];values=[float(v) for v in raw if v!='']
        if len(raw)!=int(row['settings']) or len(values)!=int(row['available_settings']):raise ValueError('summary counts changed')
        for key,value in dict(mean=mean(values),std=pstdev(values),minimum=min(values),maximum=max(values)).items() if values else []:
            if not math.isclose(float(row[key]),value,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('summary differs: '+key)
    if evidence['confirmation_opened'] or evidence['data_role']!='inner_validation':raise ValueError('data role changed')
    print('PASS:',len(evidence['data_files']),'data files;',len(summary),'descriptive summaries; inner validation only')


if __name__=='__main__':verify(Path(__file__).resolve().parent)
