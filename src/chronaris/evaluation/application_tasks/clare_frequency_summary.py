"""Reproducible quality diagnosis; failed readiness never becomes a model result."""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json


def summarize_clare_frequency(report,assets):
    report,assets=Path(report),Path(assets)
    with np.load(assets/'observations.npz',allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    audit=json.loads(gzip.decompress((assets/'gaps.json.gz').read_bytes()))
    assert audit['sources']==meta['sources']
    d=pd.DataFrame(audit['rows']);assert not d[['sample_id','field']].duplicated().any()
    d['too_few']=d.finite_points<20;d['bad_order']=(d.repeated_time+d.reversed_time)>0
    d['gap']=(d[['initial_gap_s','final_gap_s','internal_gap_s']].max(axis=1)>.25)&~d.too_few
    d.to_csv(report/'gap_details.csv.gz',index=False)
    summary=d.groupby(['stream','field'])[['too_few','bad_order','gap']].sum().reset_index()
    summary.to_csv(report/'gap_summary.csv',index=False)
    frequencies=pd.read_csv(report/'native_frequencies.csv');brain=frequencies[frequencies.channel<4]
    assert len(brain)==4*len(meta['quality'])
    failures={reason:sum(r.get('reason')==reason for r in meta['windows'])
              for reason in sorted({r['reason'] for r in meta['windows'] if not r['accepted']})}
    result=dict(status='completed',readiness=meta['status'],legal_windows=len(meta['windows']),qualified_windows=len(meta['quality']),
                failed_subjects=[c for c in meta['coverage'] if not c['passed']],rejections=failures,
                field_windows=len(d),nonincreasing_field_windows=int(d.bad_order.sum()),
                peripheral_incomplete_windows=int(d[(d.stream!='EEG')&(d.too_few|d.bad_order|d.gap)].sample_id.nunique()),
                eeg_incomplete_windows=int(d[(d.stream=='EEG')&(d.too_few|d.bad_order|d.gap)].sample_id.nunique()),
                brain_above_4_8_hz_median=float(brain.higher_than_query_nyquist.median()),
                median_rates=frequencies.groupby('channel').rate_hz.median().to_dict(),
                encoder_updates=0,classifier_fits=0,outer_scored=False)
    assert not (report/'feasibility.json').exists()
    save_json(report/'quality_summary.json',result)
    save_json(report/'asset_manifest.json',dict(root=str(assets.resolve()),files={str(p.relative_to(assets)):file_digest(p) for p in sorted(assets.rglob('*')) if p.is_file()}))
    return result
