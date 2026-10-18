"""Verify preserved sources and independently recompute the no-fit decision evidence."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    protected=json.loads((HERE/'protected_sources.json').read_text())
    changes=[p for p,h in protected.items() if digest(p)!=h]
    assert not changes,changes
    inputs=pd.read_csv(HERE/'trend_inputs.csv');subjects=pd.read_csv(HERE/'trend_subjects.csv')
    folds=json.loads((ROOT/'docs/requirements/thesis-v4-public-subjects.json').read_text())['domains']['cogpilot']['folds']['development']
    differences=[]
    for row in subjects.to_dict('records'):
        train=inputs[inputs.subject.isin(folds[row['fold']]['train_subjects'])]
        test=inputs[inputs.subject==row['subject']]
        candidates={'zero':np.zeros(len(test)),'train_mean':np.full(len(test),train.cached_target.mean()),
            'train_median':np.full(len(test),train.cached_target.median()),
            'past_difference':test.past_trend.to_numpy(),'linear_trend_2_5':2.5*test.past_trend.to_numpy()}
        value=float(np.sqrt(np.mean((test.cached_target.to_numpy()-candidates[row['method']])**2)))
        differences.append(abs(value-row['rmse']))
    assert max(differences)<1e-12
    table=pd.read_csv(HERE/'trend_summary.csv').set_index('method').rmse
    old=pd.read_csv(HERE.parent/'chapter5-readout-repair-20261018/comparison.csv')
    for method,arm in [('zero','zero'),('train_mean','mean'),('train_median','median'),('past_difference','history')]:
        previous=old[(old.domain=='cogpilot')&(old.task=='event_response')&(old.method=='constant')&(old.arm==arm)]
        assert len(previous)==2 and np.max(abs(previous['mean'].to_numpy()-table[method]))<1e-12
    finite=json.loads((HERE/'performance_finite_checks.json').read_text());assert len(finite)==84
    assert sum(x['rows'] for x in finite)==163768 and all(x['nonfinite_cells']==0 for x in finite)
    perf=pd.read_csv(HERE/'performance_sources.csv')
    assert (abs(perf.sum_exported_total-perf.recorded_cumulative)<=perf.rounding_bound).all()
    assert perf.total_identity_max_error.max()<1e-12
    cases=pd.read_csv(HERE/'direct_time_conditions.csv');assert len(cases)==25 and cases.condition_id.is_unique
    ambiguous=cases[(cases.group=='ambiguity_pair')&(~cases.sync_anchor)]
    assert len(ambiguous)==3 and (ambiguous.offset_s+ambiguous.response_delay_s).eq(4).all()
    checks=json.loads((HERE/'identifiability_checks.json').read_text());assert checks['status']=='passed'
    assert len(checks['affine_cases'])==9 and max(checks['visible_signal_max_differences'])==0
    missing=[]
    for p in HERE.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not link.startswith('http') and not (p.parent/link).exists():missing.append((p.name,link))
    # verification.json is the single output still being created by this invocation.
    assert all(link=='verification.json' for _,link in missing),missing
    result=dict(status='passed',initial_plan_commit='1b9f3db3',trend_rule_commit='5bef401b',
        protected_sources=len(protected),changed_sources=changes,no_fit_subject_metrics=len(differences),
        max_metric_recompute_error=max(differences),unchanged_constant_references=4,
        official_performance_records=84,official_performance_rows=163768,
        affine_formula_checks=9,indistinguishable_parameter_pairs=3,planned_base_conditions=25,
        new_encoder_updates=0,new_readout_fits=0,new_recovery_estimators_fit=0,
        unresolved=['CLARE per-record task/beep/report anchors','real clock calibration truth'],
        scope='source and algebra audit; no new model, target benchmark, or direct-recovery benchmark executed')
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
