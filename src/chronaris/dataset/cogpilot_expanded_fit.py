"""Read-only cohort extension; original confirmation roles never enter the request."""
import json
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import extract_performance_windows, file_digest, STREAM_COLUMNS
from chronaris.dataset.cogpilot_fair_residual import extract_fair_residual
from chronaris.dataset.cogpilot_shape_extract import extract_shape_native


def extract_expanded_fit(request):
    import sys
    root=Path(request['source_root']); inventory=[]; sources={}; windows=[]; pieces=[]; checks=[]
    assert file_digest(root/'PerfMetrics.csv')==request['metadata_sha256']
    for number,item in enumerate(request['records']):
        subject=f"sub-cp{int(item['subject']):03d}"
        assert subject in request['subjects']
        matches=list((root/subject/f"ses-{int(item['date'])}").glob(
            f"level-{int(item['difficulty']):02d}*_run-{int(item['run']):03d}"))
        row=dict(subject=subject,date=int(item['date']),run=int(item['run']),accepted=False)
        before={}
        try:
            if len(matches)!=1:raise ValueError('record directory missing or ambiguous')
            run=matches[0];row['record_id']=str(run)
            paths=[]
            for token in (*STREAM_COLUMNS,'lslxp11_feat-perfmetric'):
                found=list(run.glob(f'*stream-{token}*.csv'))
                if len(found)!=1:raise ValueError('source missing or ambiguous: '+token)
                paths+=found
            before={str(p):file_digest(p) for p in paths};sources.update(before)
            item_record=dict(subject=subject,record_id=str(run),file=str(paths[-1]))
            initial=extract_performance_windows([item_record]);meta=json.loads(str(initial['metadata']))
            candidates=len(meta['windows']);row['candidates']=candidates
            selected=[r for r in meta['windows'] if r['accepted']]
            if not selected:raise ValueError('no initial windows qualified')
            result=extract_fair_residual(dict(records=[item_record],windows={str(run):selected},
                                              source_hashes=before,include_shape_native=True))
            detail=json.loads(str(result['metadata']));accepted=[r for r in detail['windows'] if r['accepted']]
            row['accepted_windows']=len(accepted)
            row['window_rejections']=[dict(sample_id=r['sample_id'],reason=r['reason'])
                                      for r in meta['windows']+detail['windows'] if not r['accepted']]
            if len(accepted)<.8*candidates:raise ValueError('record qualification below 80 percent')
            row['accepted']=True;windows+=accepted;pieces.append(result);checks+=detail['reconstruction']
        except (ValueError,StopIteration,FileNotFoundError) as exc:
            row['reason']=str(exc)
        finally:
            if any(file_digest(p)!=h for p,h in before.items()):raise RuntimeError('source changed')
        inventory.append(row);print(number+1,len(request['records']),subject,row['accepted'],row.get('reason',''),file=sys.stderr,flush=True)
    coverage=[]
    for subject in request['subjects']:
        records=[r for r in inventory if r['subject']==subject]
        valid=[r for r in records if r['accepted']];count=sum(r['accepted_windows'] for r in valid)
        coverage.append(dict(subject=subject,expected_records=len(records),qualified_records=len(valid),windows=count,
                             passed=len(valid)>=8 and count>=100))
    passed=all(c['passed'] for c in coverage) and sum(r['accepted'] for r in inventory)>=.8*len(inventory)
    first={}
    for row in windows:first.setdefault(row['subject'],row)
    prefix=extract_shape_native(dict(windows=list(first.values()),sources={p:h for p,h in sources.items()
                                     if str(Path(p).parent) in {r['record_id'] for r in first.values()}})) if first else None
    prefix_checks=json.loads(str(prefix['metadata']))['prefix_checks'] if prefix else []
    assert all(file_digest(p)==h for p,h in sources.items())
    assert file_digest(root/'PerfMetrics.csv')==request['metadata_sha256']
    metadata=dict(status='passed' if passed else 'failed',windows=windows,records=inventory,sources=sources,
                  reconstruction=checks,coverage=coverage,prefix_checks=prefix_checks,source_metadata_sha256=request['metadata_sha256'])
    arrays={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0] if k!='metadata'} if pieces else {}
    return dict(arrays,metadata=np.array(json.dumps(metadata,allow_nan=False)))
