"""Recheck recorded train means on the exact validation cohort; no fitting."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
REPO=PARENT.parents[3]
DATA=REPO/'docs/artifacts/runs/2026-10-18_chapter5-materials/data'


def main():
    sources={}
    def track(path):
        path=Path(path)
        with path.open('rb') as f:sources[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
        return path
    def csv_rows(path):
        with track(path).open() as f:return list(csv.DictReader(f))
    constants={(r['domain'],r['fold'],r['task']):float(r['train_constant'])
               for r in csv_rows(PARENT/'statistics/constant_baselines.csv') if r['baseline']=='train_mean'}
    probes=csv_rows(PARENT/'readouts/metrics.csv')
    groups=[];folds=[]
    for unit in csv_rows(DATA/'inventory.csv'):
        if unit['variant']!='thesis_reference':continue
        receipt=json.loads(track(unit['receipt']).read_text())
        route=next(r for r in receipt['results'] if r['route']=='task_guided')
        result=json.loads(track(route['consumers']['components']['linear']['result_path']).read_text())['evaluations']['validation']
        by_group=defaultdict(list)
        for r in result['prediction_rows']:
            if r['target_valid'] and (unit['domain'],unit['fold_index'],r['task']) in constants:
                by_group[(r['task'],r['group_id'])].append(r)
        for (task,subject),rows in by_group.items():
            value=constants[(unit['domain'],unit['fold_index'],task)]
            y=[r['target'] for r in rows];p=[r['prediction'] for r in rows]
            mse=mean((a-b)**2 for a,b in zip(y,p));reference=mean((a-value)**2 for a in y)
            groups.append(dict(domain=unit['domain'],fold=unit['fold_index'],task=task,subject=subject,
                samples=len(rows),train_mean=value,target_mean=mean(y),target_variance=pvariance(y),
                original_prediction_mean=mean(p),original_prediction_variance=pvariance(p),
                original_bias=mean(b-a for a,b in zip(y,p)),constant_bias=value-mean(y),
                original_rmse=math.sqrt(mse),constant_rmse=math.sqrt(reference),
                original_mse_skill=1-mse/reference if reference else None))
        for task in sorted({task for task,_ in by_group}):
            part=[r for r in groups if r['domain']==unit['domain'] and r['fold']==unit['fold_index'] and r['task']==task]
            original=mean(r['original_rmse'] for r in part)
            saved=next(r['value'] for r in result['task_summary'] if r['task']==task)
            assert abs(original-saved)<1e-12
            strong=next((float(r['value']) for r in probes if r['domain']==unit['domain'] and r['fold']==unit['fold_index']
                         and r['task']==task and r['route']=='task_guided' and r['role']=='validation' and r['block']=='full_alpha10000'),None)
            folds.append(dict(domain=unit['domain'],fold=unit['fold_index'],task=task,subjects=len(part),
                original_rmse=original,strong_ridge_rmse=strong,train_mean_rmse=mean(r['constant_rmse'] for r in part)))
    summary=[]
    for domain,task in sorted({(r['domain'],r['task']) for r in folds}):
        part=[r for r in folds if r['domain']==domain and r['task']==task]
        strong=[r for r in part if r['strong_ridge_rmse'] is not None]
        summary.append(dict(domain=domain,task=task,folds=len(part),original_rmse=mean(r['original_rmse'] for r in part),
            train_mean_rmse=mean(r['train_mean_rmse'] for r in part),
            strong_ridge_rmse=mean(r['strong_ridge_rmse'] for r in strong) if strong else None,
            train_mean_better_than_strong_folds=sum(r['train_mean_rmse']<r['strong_ridge_rmse'] for r in strong) if strong else None))
    for name,rows in [('validation_groups.csv',groups),('validation_folds.csv',folds),('validation_summary.csv',summary)]:
        with (HERE/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    for path,expected in sources.items():
        with Path(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
    c=next(r for r in summary if r['domain']=='cogpilot')
    assert c['train_mean_rmse']<c['strong_ridge_rmse']<c['original_rmse']
    (HERE/'verification.json').write_text(json.dumps(dict(sources=sources,source_files_unchanged=True,
        model_fits=0,encoder_updates=0,scope='same_validation_rows_subject_first_equal_fold_mean'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
