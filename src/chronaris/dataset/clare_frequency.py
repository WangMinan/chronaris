"""Native-frequency, window-end diagnostics for the separately versioned CLARE task."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch,butter,sosfilt
from chronaris.dataset.cogpilot_performance import file_digest


def frequency_features(t,x,start,end,kind):
    keep=(t>=start)&(t<end)&np.isfinite(x);t,x=t[keep],x[keep]
    if len(t)<20 or not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('finite_time_order_or_points')
    if max(t[0]-start,end-t[-1],np.diff(t).max())>.25:raise ValueError('source_gap')
    fs=float(1/np.median(np.diff(t)));minimum=20 if kind=='eda' else 100
    if fs<minimum:raise ValueError('native_rate')
    if kind=='eeg':
        f,p=welch(x,fs=fs,nperseg=min(len(x),round(fs*2)),detrend='constant')
        total=float(p[(f>=.4)&(f<45)].sum());den=max(total,1e-30)
        bands=((.4,4),(4,8),(8,12),(12,31),(31,45))
        values=[float(p[(f>=a)&(f<b)].sum()/den) for a,b in bands]+[float(np.log1p(total))]
        high=float(p[(f>4.8)&(f<45)].sum()/den)
    else:
        band=(.05,3) if kind=='eda' else (5,15)
        filtered=sosfilt(butter(4,band,btype='bandpass',fs=fs,output='sos'),x-x[0])
        centered=t-t.mean();slope=float(np.dot(centered,x-x.mean())/np.dot(centered,centered))
        values=[float(x.mean()),float(x.std()),float(np.quantile(x,.75)-np.quantile(x,.25)),
                float(np.sqrt(np.mean(x*x))),slope,float(np.sqrt(np.mean(filtered*filtered)))]
        high=None
    assert np.isfinite(values).all()
    return values,dict(rate_hz=fs,points=len(t),higher_than_query_nyquist=high)


def extract_clare_frequency(request):
    import sys
    root=Path(request['root']);windows=[];features=[];quality=[];sources={};records=[]
    for subject in request['subjects']:
        label_path=root/'Labels'/(subject+'.csv');sources[str(label_path)]=file_digest(label_path)
        labels=pd.read_csv(label_path)
        for experiment in range(4):
            paths=[root/'EEG'/subject/f'eeg_data_exp_{experiment}.csv',
                   root/'EDA'/subject/f'eda_data_experiment_{experiment}.csv',
                   root/'ECG'/subject/f'ecg_data_experiment_{experiment}.csv']
            if not all(p.exists() for p in paths):
                count=0
                for index,label in enumerate(labels[f'level_{experiment}']):
                    if not np.isfinite(label):continue
                    if not 1<=label<=9:raise ValueError('label outside declared scale')
                    count+=1
                    windows.append(dict(sample_id=f'{subject}__experiment{experiment}_window{index}',subject=subject,
                                        experiment=experiment,index=index,start_s=None,end_s=None,score=float(label),
                                        target=int(label>=5),accepted=False,reason='missing_source'))
                records.append(dict(subject=subject,experiment=experiment,legal=count,accepted=0,
                                    missing_sources=[str(p) for p in paths if not p.exists()]))
                print(subject,experiment,0,count,'missing_source',file=sys.stderr,flush=True)
                continue
            before={str(p):file_digest(p) for p in paths};sources.update(before)
            tables=[pd.read_csv(p) for p in paths];origin=float(tables[0].Timestamp.iloc[0])
            if not np.isfinite(origin):raise ValueError('invalid recording origin')
            columns=[('TP9','AF7','AF8','TP10'),('GSR Conductance CAL',),('ECG LL-RA CAL',)]
            streams=[]
            for table,names,kind in zip(tables,columns,('eeg','eda','ecg')):
                for name in names:streams.append((table.Timestamp.to_numpy(float),table[name].to_numpy(float),kind))
            count=0;accepted=0
            for index,label in enumerate(labels[f'level_{experiment}']):
                if not np.isfinite(label):continue
                if not 1<=label<=9:raise ValueError('label outside declared scale')
                start=origin+index*10;end=start+10
                if end>float(tables[0].Timestamp.max()):continue
                count+=1;row=dict(sample_id=f'{subject}__experiment{experiment}_window{index}',subject=subject,
                                  experiment=experiment,index=index,start_s=start,end_s=end,score=float(label),
                                  target=int(label>=5),accepted=False)
                try:
                    output=[frequency_features(t,x,start,end,kind) for t,x,kind in streams]
                    features.append(np.concatenate([o[0] for o in output]));quality.append([o[1] for o in output])
                    row['accepted']=True;accepted+=1
                except ValueError as exc:row['reason']=str(exc)
                windows.append(row)
            if any(file_digest(p)!=h for p,h in before.items()):raise RuntimeError('source changed')
            records.append(dict(subject=subject,experiment=experiment,legal=count,accepted=accepted))
            print(subject,experiment,accepted,count,file=sys.stderr,flush=True)
    assert all(file_digest(p)==h for p,h in sources.items())
    coverage=[]
    for subject in request['subjects']:
        rows=[r for r in windows if r['subject']==subject];valid=[r for r in rows if r['accepted']]
        coverage.append(dict(subject=subject,legal=len(rows),accepted=len(valid),
                             class_counts=[sum(r['target']==k for r in valid) for k in (0,1)],
                             passed=len(valid)>=100 and len(valid)>=.8*len(rows)))
    return dict(features=np.asarray(features),metadata=np.array(json.dumps(dict(
        status='passed' if all(r['passed'] for r in coverage) else 'failed',windows=windows,quality=quality,
        sources=sources,records=records,coverage=coverage),allow_nan=False)))
