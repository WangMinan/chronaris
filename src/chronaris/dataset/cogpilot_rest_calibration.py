"""Source-bound, strictly preflight resting statistics for an explicit calibration condition."""
from pathlib import Path
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import STREAM_COLUMNS,file_digest


def extract_rest_calibration(request):
    import sys
    root=Path(request['root']);sources={};calibrations=[];inventory=[]
    for item in request['sessions']:
        candidates=[];folder=root/item['subject']/item['session']
        for run in sorted(folder.glob('level-*')):
            detail=dict(subject=item['subject'],session=item['session'],record_id=str(run),accepted=False)
            before={}
            try:
                paths=[]
                for token in list(STREAM_COLUMNS)[:3]:
                    found=list(run.glob(f'*stream-{token}*_dat.csv'))
                    if len(found)!=1:raise ValueError('missing or ambiguous source '+token)
                    paths+=found
                before={str(p):file_digest(p) for p in paths};sources.update(before)
                tables=[pd.read_csv(p,usecols=['time_dn',*columns]) for p,columns in zip(paths,list(STREAM_COLUMNS.values())[:3])]
                times=[t.time_dn.to_numpy(float) for t in tables]
                if any(not np.isfinite(t).all() or (np.diff(t)<=0).any() for t in times):raise ValueError('rest time order')
                if any(t.max()>=item['first_input_native'] for t in times):raise ValueError('rest not entirely before first input')
                end=min(t.max() for t in times);start=end-60/86400
                if max(t.min() for t in times)>start:raise ValueError('less than sixty seconds common support')
                centers=[];scales=[];qualities=[]
                for table,t,columns in zip(tables,times,list(STREAM_COLUMNS.values())[:3]):
                    use=(t>=start)&(t<end);tt=(t[use]-start)*86400
                    if len(tt)<50 or max(tt[0],60-tt[-1],np.diff(tt).max())>.25:raise ValueError('rest coverage')
                    for column in columns:
                        x=table[column].to_numpy(float)[use]
                        if column=='eda_hand_l_kOhms':
                            if (x<=0).any():raise ValueError('invalid rest resistance')
                            x=1000/x
                        if not np.isfinite(x).all():raise ValueError('nonfinite resting value')
                        centers.append(float(np.median(x)));scales.append(float(np.quantile(x,.75)-np.quantile(x,.25)))
                        qualities.append(dict(points=len(x),maximum_gap_s=float(np.diff(tt).max()),span_s=float(tt[-1]-tt[0])))
                detail.update(accepted=True,start_native=float(start),end_native=float(end),center=centers,scale=scales,
                              gap_to_first_input_s=float((item['first_input_native']-end)*86400),quality=qualities)
                candidates.append((max(t.min() for t in times),detail))
            except (ValueError,FileNotFoundError) as exc:detail['reason']=str(exc)
            finally:
                if any(file_digest(p)!=h for p,h in before.items()):raise RuntimeError('rest source changed')
            inventory.append(detail)
        if candidates:
            chosen=min(candidates,key=lambda pair:(pair[0],pair[1]['record_id']))[1];calibrations.append(chosen)
        print(item['subject'],len(candidates),'eligible rest records',file=sys.stderr,flush=True)
    assert all(file_digest(p)==h for p,h in sources.items())
    return dict(status='passed' if len(calibrations)==len(request['sessions']) else 'failed',
                calibrations=calibrations,records=inventory,sources=sources,sessions=request['sessions'],encoder_updates=0)
