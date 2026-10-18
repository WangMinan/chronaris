"""Export raw-target validity, matched-control and timing-stability comparisons."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from audit_io import HERE,ASSETS,REVISED,sha,save_json
from chronaris.evaluation.application_tasks.public_readout_revision import load_readout_revision


def main():
    source=ASSETS/'raw_response.json';raw=json.loads(source.read_text())
    e=pd.DataFrame(raw['events']);c=pd.DataFrame(raw['clare']);records=pd.DataFrame(raw['records'])
    assert len(e)==1462 and e.sample_id.is_unique and e.target_matches_float32.all() and e.event_start_error_s.max()==0
    target=load_readout_revision(REVISED/'clare').targets
    indexed=c.set_index('sample_id').loc[list(target.sample_ids)]
    for name,col in [('workload_regression','score'),('workload_classification','legacy_class'),
                     ('workload_standard_classification','standard_class')]:
        assert np.array_equal(target.values[name].numpy().reshape(-1),indexed[col].to_numpy())
    e.drop(columns=['source_paths']).to_csv(HERE/'event_targets.csv',index=False)
    c.to_csv(HERE/'clare_targets.csv',index=False);records.to_csv(HERE/'event_records.csv',index=False)
    pd.DataFrame(raw['clare_records']).to_csv(HERE/'clare_records.csv',index=False)
    rows=[];stability=[]
    for subject,d in e.groupby('subject'):
        matched=d[d.matched];y=d.response.to_numpy();p=matched.response.to_numpy();q=matched.control_response.to_numpy()
        rows.append(dict(subject=subject,events=len(d),matched=len(matched),
            unique_controls=matched[['record_id','control_native']].drop_duplicates().shape[0],
            mean=float(y.mean()),median=float(np.median(y)),std=float(y.std()),
            positive_fraction=float(np.mean(y>0)),zero_fraction=float(np.mean(y==0)),
            event_zero_rmse=float(np.sqrt(np.mean(p*p))),control_zero_rmse=float(np.sqrt(np.mean(q*q))),
            event_mean_abs=float(np.mean(abs(p))),control_mean_abs=float(np.mean(abs(q))),
            paired_signed_difference=float(np.mean(p-q)),paired_absolute_difference=float(np.mean(abs(p)-abs(q))),
            match_distance_median=float(matched.match_distance.median()),match_distance_p95=float(matched.match_distance.quantile(.95))))
        for col in ('shift_-1.0','shift_-0.5','shift_0.5','shift_1.0','first_half','second_half'):
            mask=d[col].notna();a=d.loc[mask,'response'].to_numpy();b=d.loc[mask,col].to_numpy()
            stability.append(dict(subject=subject,comparison=col,samples=len(a),
                response_rmse=float(np.sqrt(np.mean(a*a))),change_rmse=float(np.sqrt(np.mean((a-b)**2))),
                spearman=float(spearmanr(a,b).statistic),sign_flip_fraction=float(np.mean(a*b<0)),
                median_abs_change=float(np.median(abs(a-b)))))
    subjects=pd.DataFrame(rows);subjects.to_csv(HERE/'event_subjects.csv',index=False)
    pd.DataFrame(stability).to_csv(HERE/'event_stability.csv',index=False)
    clare=c.groupby('subject').agg(windows=('score','size'),score_min=('score','min'),score_max=('score','max'),
        score_mean=('score','mean'),score_std=('score','std'),standard_high_fraction=('standard_class','mean'),
        legacy_high_fraction=('legacy_class','mean')).reset_index()
    clare.to_csv(HERE/'clare_subjects.csv',index=False)
    summaries=[]
    for col in ('paired_signed_difference','paired_absolute_difference'):
        v=subjects[col].to_numpy();rng=np.random.default_rng(17)
        samples=rng.choice(v,size=(10000,len(v)),replace=True).mean(axis=1)
        summaries.append(dict(contrast=col,subject_mean=float(v.mean()),
            bootstrap_low=float(np.quantile(samples,.025)),bootstrap_high=float(np.quantile(samples,.975)),
            positive_subjects=int(np.sum(v>0)),subjects=len(v)))
    pd.DataFrame(summaries).to_csv(HERE/'event_matched_contrasts.csv',index=False)
    save_json(HERE/'task_validity.json',dict(status='completed_with_semantic_gaps',source_sha256=sha(source),
        event_targets_exact=1462,clare_targets_exact=794,clare_task_checks=2382,
        matched_events=int(e.matched.sum()),unique_controls=int(e[['record_id','control_native']].drop_duplicates().shape[0]),
        eda_records_with_reversal=int((records.eda_time_reversals>0).sum()),
        max_reversal_s=float(records.largest_reversal_s.max()),
        clare_label_files_have_explicit_timestamps=False,clare_signal_units_fully_documented=False,
        intervals='descriptive subject bootstrap, 10000 resamples, seed17; 7 subjects, not randomized causal inference',
        target_changes=0,encoder_updates=0))
    print(subjects.to_string(index=False));print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':main()
