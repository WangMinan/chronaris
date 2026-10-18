"""No-fit, preregistered trend references on the unchanged event cohort."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    old=HERE.parent/'frozen-feasibility-20261018'
    events=pd.read_csv(old/'event_targets.csv').set_index('sample_id')
    history=ROOT/'artifacts/application_evaluation/2026-10-18-controlled-readouts/exclusive_endpoint/cogpilot__history.npz'
    with np.load(history) as z:
        series=pd.Series(z['trend'],index=z['sample_ids'])
    assert events.index.is_unique and series.index.is_unique
    events['past_trend']=series.reindex(events.index)
    assert np.isfinite(events.past_trend).all()
    folds=json.loads((ROOT/'docs/requirements/thesis-v4-public-subjects.json').read_text())['domains']['cogpilot']['folds']['development']
    rows=[];subject_rows=[]
    for i,fold in enumerate(folds):
        train=events[events.subject.isin(fold['train_subjects'])]
        test=events[events.subject.isin(fold['validation_subjects'])]
        assert not set(train.subject)&set(test.subject)
        predictions={'zero':np.zeros(len(test)),'train_mean':np.full(len(test),train.cached_target.mean()),
            'train_median':np.full(len(test),train.cached_target.median()),
            'past_difference':test.past_trend.to_numpy(),'linear_trend_2_5':2.5*test.past_trend.to_numpy()}
        for method,p in predictions.items():
            for subject in sorted(set(test.subject)):
                mask=test.subject.to_numpy()==subject;y=test.cached_target.to_numpy()[mask]
                error=float(np.sqrt(np.mean((y-p[mask])**2)))
                subject_rows.append(dict(fold=i,subject=subject,method=method,windows=len(y),rmse=error))
            rows.append(dict(fold=i,method=method,rmse=float(np.mean([r['rmse'] for r in subject_rows if r['fold']==i and r['method']==method]))))
    descriptive=[]
    for subject,d in events.groupby('subject'):
        descriptive.append(dict(subject=subject,events=len(d),past_trend_spearman=float(spearmanr(d.past_trend,d.response).statistic),
            baseline_spearman=float(spearmanr(d.baseline,d.response).statistic),
            both_decreasing_fraction=float(np.mean((d.past_trend<0)&(d.response<0))),
            target_negative_fraction=float(np.mean(d.response<0)),past_negative_fraction=float(np.mean(d.past_trend<0))))
    pd.DataFrame(rows).to_csv(HERE/'trend_folds.csv',index=False)
    pd.DataFrame(subject_rows).to_csv(HERE/'trend_subjects.csv',index=False)
    pd.DataFrame(descriptive).to_csv(HERE/'trend_correlations.csv',index=False)
    summary=pd.DataFrame(rows).groupby('method').rmse.mean().reset_index();summary.to_csv(HERE/'trend_summary.csv',index=False)
    events[['subject','cached_target','response','baseline','past_trend']].reset_index().to_csv(HERE/'trend_inputs.csv',index=False)
    print(summary.to_string(index=False));print(pd.DataFrame(descriptive).to_string(index=False))


if __name__=='__main__':main()
