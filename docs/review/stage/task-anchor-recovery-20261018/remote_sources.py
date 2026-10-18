"""Read-only source inspection: timestamps, rest segments, event reversals and official task files."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def inspect(request):
    root=Path('/home/wangminan/dataset/chronaris/clare')
    sources={};timestamps=[];headers=[];labels=[]
    def read(path,**kwargs):
        sources[str(path)]=digest(path)
        return pd.read_csv(path,**kwargs)
    for subject in request['clare_subjects']:
        label=root/'Labels'/f'{subject}.csv';frame=read(label)
        labels.append(dict(subject=subject,rows=len(frame),columns=list(frame.columns),
            md5=hashlib.md5(label.read_bytes()).hexdigest()))
        for stream in ('EEG','EDA','ECG','Gaze'):
            for path in sorted((root/stream/subject).glob('*.csv')):
                head=read(path,nrows=1)
                names=[str(c) for c in head.columns if any(x in str(c).lower() for x in ('time','event','label','marker','stimulus'))]
                headers.append(dict(subject=subject,stream=stream,file=str(path),columns=list(head.columns),candidate_time_event_columns=names))
                if stream=='Gaze' or 'Timestamp' not in head:continue
                t=pd.read_csv(path,usecols=['Timestamp']).Timestamp.to_numpy(float)
                valid=t[np.isfinite(t)];diff=np.diff(valid)
                timestamps.append(dict(subject=subject,stream=stream,file=str(path),rows=len(t),
                    first=float(valid[0]),last=float(valid[-1]),minimum=float(valid.min()),maximum=float(valid.max()),
                    duration=float(valid.max()-valid.min()),nonfinite=int((~np.isfinite(t)).sum()),
                    reversals=int((diff<0).sum()),largest_gap=float(diff.max()) if len(diff) else None))
    extras=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.suffix.lower() not in ('.csv','.ds_store')]
    base=Path('/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0')
    summary=read(base/'dataPackage/task-ils/PerfMetrics.csv');performance=[];reversals=[]
    for record,events in request['cog_records'].items():
        run=Path(record);files=list(run.glob('*perfmetric*.csv'))
        subject=int(re.search(r'sub-cp(\d+)',record)[1]);run_id=int(re.search(r'run-(\d+)',run.name)[1]);date=int(re.search(r'ses-(\d+)',record)[1])
        selected=summary[(summary.subject==subject)&(summary.run==run_id)&(summary.date==date)]
        item=dict(record_id=record,subject=events[0]['subject'],files=len(files),summary_rows=len(selected))
        if len(files)==1:
            p=files[0];d=read(p);cols=['glideslope_error_deg','localizer_error_deg','airspeed_error_kts']
            reconstruction=d[cols].abs().sum(axis=1)
            item.update(file=str(p),rows=len(d),columns=list(d.columns),
                total_identity_max_error=float(abs(reconstruction-d.total_error).max()),
                total_min=float(d.total_error.min()),total_max=float(d.total_error.max()),
                first_time=float(d.time_dn.min()),last_time=float(d.time_dn.max()),
                time_reversals=int((np.diff(d.time_dn)<0).sum()),
                recorded_cumulative=float(selected.iloc[0].cumulative_total_error) if len(selected)==1 else None,
                sum_exported_total=float(d.total_error.sum()),
                rounding_bound=float(len(d)*3*0.00005+0.00005))
        performance.append(item)
        eda=read(Path(events[0]['source_paths'][0]),usecols=['time_dn']);t=eda.time_dn.to_numpy(float)
        for i in np.flatnonzero(np.diff(t)<0):
            lo,hi=float(t[i+1]),float(t[i]);affected=[];targets=[];controls=[]
            for e in events:
                at=e['event_native'];ct=e['control_native']
                if at+8/86400>lo and at-12/86400<hi:affected.append(e['sample_id'])
                if at+8/86400>lo and at-2/86400<hi:targets.append(e['sample_id'])
                if ct+8/86400>lo and ct-2/86400<hi:controls.append(e['sample_id'])
            reversals.append(dict(record_id=record,from_time=hi,to_time=lo,backward_s=(hi-lo)*86400,
                input_or_target_events=affected,target_events=targets,control_events=controls))
    documents={}
    for name in ('compute_flight_performance.m','assembleFeatureMatrix.py'):
        path=base/'starterCode'/name;sources[str(path)]=digest(path);documents[name]=path.read_text()
    for name in ('README.txt','MANIFEST.TXT'):
        path=root/name;sources[str(path)]=digest(path);documents['clare_'+name]=path.read_text()
    assert all(digest(p)==h for p,h in sources.items())
    return dict(timestamps=timestamps,headers=headers,labels=labels,extra_inventory=extras,
        performance=performance,reversals=reversals,documents=documents,sources=sources,
        clare_readme_md5=hashlib.md5((root/'README.txt').read_bytes()).hexdigest())
