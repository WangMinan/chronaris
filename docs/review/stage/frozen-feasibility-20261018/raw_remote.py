"""Transport-only entrypoint: run the shipped read-only audit against raw recordings."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def run(request):
    namespace = {}; exec(request['implementation'], namespace)
    groups = {}
    for row in request['events']:
        groups.setdefault(row['record_id'], []).append(row)
    events, records, sources = [], [], {}
    for record, rows in sorted(groups.items()):
        paths = rows[0]['source_paths']; vpath, epath = paths[-1], paths[0]
        before = {p:digest(p) for p in (vpath, epath)}
        result, counts = namespace['audit_record'](pd.read_csv(vpath), pd.read_csv(epath),
            sorted(rows, key=lambda x:x['sample_id']))
        events.extend(result);records.append(dict(record_id=record, subject=rows[0]['subject'], **counts))
        assert all(digest(p)==h for p,h in before.items());sources.update(before)
    clare, clare_records = [], []
    for record, rows in request['clare_records'].items():
        first=rows[0]; paths=first['source_paths'];subject=first['subject_id']
        label=str(Path(paths[0]).parents[2]/'Labels'/f'{subject}.csv')
        files=paths+[label];before={p:digest(p) for p in files}
        frames=[pd.read_csv(p) for p in paths]; labels=pd.read_csv(label)
        exp=int(re.search(r'experiment(\d+)',first['sample_id'])[1])
        for row in rows:
            win=int(re.search(r'window(\d+)',row['sample_id'])[1]);value=float(labels.iloc[win][f'level_{exp}'])
            clare.append(dict(sample_id=row['sample_id'],subject=subject,record_id=record,
                score=value,standard_class=int(value>=5),legacy_class=int(value>=7)))
        clare_records.append(dict(record_id=record,subject=subject,windows=len(rows),label_rows=len(labels),
            label_columns=list(labels.columns),signal_columns=[list(f.columns) for f in frames],
            signal_start_s=[float(f.Timestamp.min()) for f in frames],
            signal_end_s=[float(f.Timestamp.max()) for f in frames],
            eeg_rows=len(frames[0]),label_nonfinite=int((~np.isfinite(labels[f'level_{exp}'].to_numpy(float))).sum())))
        assert all(digest(p)==h for p,h in before.items());sources.update(before)
    root=Path('/home/wangminan/dataset/chronaris/clare')
    documents={str(p):p.read_text() for p in [root/'README.txt',root/'MANIFEST.TXT']}
    sources.update({p:digest(p) for p in documents})
    return dict(events=events,records=records,clare=clare,clare_records=clare_records,
                documents=documents,sources=sources,encoder_updates=0)
