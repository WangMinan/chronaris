"""Audit frozen public predictions and paired statistics; never fit a model."""
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
DATA = REPO / 'docs/artifacts/runs/2026-10-18_chapter5-materials/data'
OUT = HERE / 'statistics'
SOURCES = {}


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def read_csv(path):
    SOURCES[str(path.relative_to(REPO))] = digest(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    assert rows
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(records, repeats=10000):
    """Positive difference always means the full method is better."""
    folds = defaultdict(list)
    for fold, delta in records:
        folds[fold].append(delta)
    rng = np.random.default_rng(20261018)
    draws = np.zeros(repeats)
    for values in folds.values():
        a = np.asarray(values)
        draws += a[rng.integers(0, len(a), size=(repeats, len(a)))].mean(1) / len(folds)
    return np.quantile(draws, [.025, .975])


def main():
    OUT.mkdir(exist_ok=True)
    groups = [r for r in read_csv(DATA/'native_groups.csv') if r['domain'] in ('clare', 'cogpilot')]
    summaries = read_csv(DATA/'native_summary.csv')
    inventory = read_csv(DATA/'inventory.csv')
    keyed = defaultdict(dict)
    classification = []
    for r in groups:
        metric = 'macro_f1' if r['macro_f1'] else 'rmse'
        assert r['status'] == 'completed'
        key = tuple(r[k] for k in ('domain', 'method', 'variant', 'route', 'consumer', 'task'))
        group_key = (r['fold_index'], r['group_id'])
        assert group_key not in keyed[key], 'Unexpected multiple fields per public task/group'
        keyed[key][group_key] = float(r[metric])
        if metric == 'macro_f1':
            cm = np.asarray(json.loads(r['confusion_matrix']))
            classification.append(dict(zip(('domain','method','variant','route','consumer','task'),key)) | {
                'fold': r['fold_index'], 'subject': r['group_id'], 'samples': int(cm.sum()),
                'macro_f1': float(r[metric]), 'accuracy': float(cm.trace()/cm.sum()),
                'true_counts': json.dumps(cm.sum(1).tolist()), 'predicted_counts': json.dumps(cm.sum(0).tolist()),
                'predicted_class_count': int((cm.sum(0)>0).sum()),
                'true_class_count': int((cm.sum(1)>0).sum()),
                'recalls': json.dumps(np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(len(cm)),where=cm.sum(1)>0).tolist())})
    for r in summaries:
        if r['domain'] not in ('clare','cogpilot'): continue
        key = tuple(r[k] for k in ('domain','method','variant','route','consumer','task'))
        assert len({subject for _,subject in keyed[key]})==len(keyed[key]), 'Subject appears in multiple outer folds'
        folds = defaultdict(list)
        for (fold,_), value in keyed[key].items(): folds[fold].append(value)
        means = [np.mean(x) for x in folds.values()]
        assert np.isclose(np.mean(means),float(r['mean']),atol=1e-12,rtol=0)
        assert np.isclose(np.std(means),float(r['std']),atol=1e-12,rtol=0)
    paired, detail = [], []
    for key, full in keyed.items():
        domain,method,variant,route,consumer,task = key
        if method!='chronaris' or variant!='thesis_reference': continue
        comparators = [('chronaris','historical_reference'),('chronaris','thesis_no_event_semantics'),
                       ('physiology_only','reference'),('vehicle_only','reference'),
                       ('mult','reference'),('contiformer','reference'),('naive_time_sync','reference')]
        for base_method,base_variant in comparators:
            base_route = 'self_supervised' if base_method=='naive_time_sync' else route
            other = keyed.get((domain,base_method,base_variant,base_route,consumer,task))
            if other is None: continue
            assert full.keys()==other.keys()
            direction = 1 if 'classification' in task or task=='difficulty' else -1
            meta = dict(domain=domain,route=route,consumer=consumer,task=task,
                        comparator=base_method,comparator_variant=base_variant,comparator_route=base_route)
            records = []
            for group,a in full.items():
                b=other[group];delta=(a-b)*direction;records.append((group[0],delta))
                detail.append(meta|dict(fold=group[0],subject=group[1],full=a,comparator_value=b,benefit=delta))
            by_fold=defaultdict(list)
            for fold,delta in records:by_fold[fold].append(delta)
            fold_deltas=[np.mean(v) for v in by_fold.values()]
            low,high=paired_bootstrap(records)
            paired.append(meta|dict(subjects=len(records),folds=len(by_fold),benefit=np.mean(fold_deltas),
                conditional_ci_low=low,conditional_ci_high=high,
                subjects_better=sum(d>1e-12 for _,d in records),subjects_worse=sum(d< -1e-12 for _,d in records),
                folds_better=sum(d>1e-12 for d in fold_deltas),folds_worse=sum(d< -1e-12 for d in fold_deltas)))
    # Independently recompute every public group metric from the frozen prediction tables.
    references={}; verified=0; max_error=0.; prediction_count=0
    for unit in inventory:
        if unit['domain'] not in ('clare','cogpilot'):continue
        for path in sorted((DATA/'native_predictions').glob(unit['unit_key']+'_*.csv.gz')):
            suffix=path.name.removeprefix(unit['unit_key']+'_').removesuffix('.csv.gz')
            route,consumer=suffix.rsplit('_',1)
            rows=read_csv(path);prediction_count+=len(rows)
            truth={}; grouped=defaultdict(list)
            for r in rows:
                identity=(r['sample_id'],r['task'],r['field'])
                assert identity not in truth
                truth[identity]=(r['group_id'],r['target'],r['target_valid'],r['sample_weight'])
                if r['target_valid']=='True':grouped[(r['task'],r['group_id'])].append(r)
            refkey=(unit['domain'],unit['fold_index'])
            if refkey in references:assert truth==references[refkey], 'Method sample/target mismatch'
            else:references[refkey]=truth
            for (task,subject),part in grouped.items():
                y=np.array([float(r['target']) for r in part]);pred=np.array([float(r['prediction']) for r in part])
                w=np.array([float(r['sample_weight']) for r in part]);assert np.all(w==1)
                classification_task='classification' in task or task=='difficulty'
                if classification_task:
                    labels=range(len(json.loads(part[0]['probability'])))
                    value=f1_score(y,pred,labels=list(labels),average='macro',zero_division=0)
                else:value=np.sqrt(np.mean((y-pred)**2))
                key=(unit['domain'],unit['method'],unit['variant'],route,consumer,task)
                error=abs(value-keyed[key][(unit['fold_index'],subject)])
                max_error=max(max_error,error);assert error<1e-10
                verified+=1
    write_csv('paired_comparisons.csv',paired);write_csv('paired_subjects.csv',detail)
    write_csv('classification_profiles.csv',classification)
    result=dict(public_prediction_rows=prediction_count,recomputed_group_metrics=verified,
        max_metric_recomputation_error=max_error,summary_reaggregation_passed=True,
        identical_public_samples_targets_weights=True,bootstrap_repeats=10000,bootstrap_seed=20261018,
        interval_scope='post_hoc_conditional_on_frozen_models_fold_stratified_subject_bootstrap_not_confirmatory',
        production_model_updates=0,sources=SOURCES)
    for name,expected in SOURCES.items():assert digest(REPO/name)==expected
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))


if __name__=='__main__':
    assert np.allclose(paired_bootstrap([('0',2.),('0',2.),('1',2.)]),[2.,2.])
    main()
