"""Nested subject-isolated linear readouts, fixed dimensional controls and constants."""
import argparse
import gzip
import json
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import f1_score

from common import HERE, PREP, REGISTRY, ASSETS, manifest, save_json, save_csv, sha
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier, fit_regressor
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development

C_GRID = (.1, 1., 10.)
A_GRID = (.1, 1., 10., 100., 1000., 10000.)
BLOCKS = {'full64': slice(None), 'first24': slice(0,24), 'second24': slice(24,48),
          'cross16': slice(48,64), 'branches48': slice(0,48)}


def metric(y, pred, classes):
    return float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)) if classes else float(np.sqrt(np.mean((y-pred)**2)))


def transform(train, score, variant):
    if variant in BLOCKS or variant=='full64_original_grid':
        block = BLOCKS.get(variant, slice(None))
        return train[:,block], score[:,block]
    scaler = StandardScaler().fit(train)
    a, b = scaler.transform(train), scaler.transform(score)
    if variant.startswith('pca'):
        d = int(variant[3:])
        assert min(a.shape) >= d
        model = PCA(n_components=d, svd_solver='full').fit(a)
    else:
        model = GaussianRandomProjection(n_components=24, random_state=int(variant.split('_')[-1])).fit(a)
    return model.transform(a), model.transform(b)


def fit(x, y, parameter, classes):
    if classes:
        assert set(y)==set(classes), 'Inner training missing target classes; no fallback'
        return fit_classifier(x, y, None, None, c_values=(parameter,), random_state=17,
                              scaler_with_mean=True, classification_labels=classes)[0]
    return fit_regressor(x, y, None, None, alpha_values=(parameter,), scaler_with_mean=True)[0]


def nested(x, y, g, train, test, variant, classes):
    grid = C_GRID if classes else A_GRID[:4] if variant=='full64_original_grid' else A_GRID
    inner = []
    subjects = sorted(set(g[train]))
    assert len(subjects)>=2 and not set(g[train]) & set(g[test])
    for subject in subjects:
        a = train[g[train]!=subject]
        b = train[g[train]==subject]
        assert len(a) and len(b) and not set(g[a]) & set(g[b])
        if classes and set(y[a]) != set(classes):
            return None, None, [], 'inner_training_missing_class'
        tx, vx = transform(x[a], x[b], variant)
        for parameter in grid:
            model = fit(tx, y[a], parameter, classes)
            inner.append(dict(subject=subject, parameter=parameter, value=metric(y[b], model.predict(vx), classes)))
    scores = [np.mean([r['value'] for r in inner if r['parameter']==p]) for p in grid]
    chosen = grid[int(np.argmax(scores) if classes else np.argmin(scores))]
    tx, vx = transform(x[train], x[test], variant)
    model = fit(tx, y[train], chosen, classes)
    return model.predict(vx), chosen, inner, 'completed'


def history_baseline(domain, rows, exclusive=False):
    asset_root=ASSETS/'exclusive_endpoint' if exclusive else ASSETS
    out = asset_root/f'{domain}__history.npz'
    if out.exists():
        with np.load(out) as z:
            assert list(z['sample_ids'])==[r['sample_id'] for r in rows]
            return z['trend']
    data = load_prepared_public_development(domain, output_root=PREP, registry_path=REGISTRY)
    trend = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if not row['task_valid_mask'].get('event_response'):
            continue
        sample = data.dataset.load_sample(row['sample_id'])
        t, value, mask = sample.physiology_timestamps_s, sample.physiology_values[:,1], sample.physiology_feature_mask[:,1]
        end=12.
        if exclusive:
            start=row['record_time_origin_native']+row['context_start_s']/86400
            end=((start+12/86400)-start)*86400
        a, b = mask & (t>=end-2) & (t<end), mask & (t>=end-4) & (t<end-2)
        if a.any() and b.any():
            trend[i] = float(np.median(value[a])-np.median(value[b]))
    np.savez_compressed(out, sample_ids=np.array([r['sample_id'] for r in rows]), trend=trend)
    return trend


def evaluate(domain, constants_only=False, exclusive=False):
    assert json.loads((HERE/'raw_verification.json').read_text())['status']=='passed'
    rows = manifest(domain)
    ids = np.array([r['sample_id'] for r in rows])
    groups = np.array([r['subject_id'] for r in rows])
    target = torch.load(PREP/domain/'task_targets.pt', weights_only=True)
    assert list(ids)==list(target['sample_ids'])
    registry = json.loads(REGISTRY.read_text())['domains'][domain]
    folds = registry['folds']['development']
    assert not exclusive or domain=='cogpilot'
    asset_root=ASSETS/'exclusive_endpoint' if exclusive else ASSETS
    trend = history_baseline(domain, rows,exclusive) if domain=='cogpilot' else np.full(len(ids), np.nan)
    variants = list(BLOCKS)+['pca16','pca24','pca48','rp24_17','rp24_29','rp24_43','full64_original_grid']
    methods = ['chronaris','physiology_only','vehicle_only','naive_time_sync','mult','contiformer']
    metrics, group_metrics, predictions, selections, status_rows = [], [], [], [], []
    began = time.monotonic()
    for task, value in target['values'].items():
        y = value.numpy().reshape(-1).astype(float)
        valid = target['valid_masks'][task].numpy().reshape(-1)
        classes = list(range(4 if domain=='cogpilot' else 2)) if task in ('difficulty','workload_classification') else None
        for fold_index, fold in enumerate(folds):
            tr = np.flatnonzero(valid & np.isin(groups, fold['train_subjects']))
            te = np.flatnonzero(valid & np.isin(groups, fold['validation_subjects']))
            assert len(tr) and len(te) and not set(groups[tr]) & set(groups[te])
            for route in ('self_supervised','task_guided'):
                candidates = [('constant','train_majority' if classes else 'train_mean',None),]
                if not classes:
                    candidates.append(('constant','train_median',None))
                    if domain=='cogpilot':
                        candidates += [('constant','zero_change',None),('constant','history_trend',None)]
                if not constants_only:
                    for method in methods:
                        path = asset_root/f"{domain}__{method}__{'self_supervised' if method=='naive_time_sync' else route}.npz"
                        with np.load(path) as z:
                            assert np.array_equal(z['sample_ids'],ids)
                            x = z['pooled'].astype(float)
                        candidates += [(method,v,x) for v in (variants if method=='chronaris' else ['full64'])]
                for method, variant, x in candidates:
                    if time.monotonic()-began > 14400:
                        raise TimeoutError('Readout CPU budget exceeded')
                    meta = dict(domain=domain, task=task, route=route, fold=fold_index, method=method, variant=variant)
                    parameter, inner, status = '', [], 'completed'
                    if method=='constant':
                        if variant=='train_majority':
                            pred = np.full(len(te), np.argmax(np.bincount(y[tr].astype(int), minlength=len(classes))))
                        elif variant=='history_trend':
                            pred = trend[te]
                            if not np.isfinite(pred).all():
                                status = 'history_missing_support'
                        else:
                            constant = {'train_mean':np.mean(y[tr]), 'train_median':np.median(y[tr]), 'zero_change':0.}[variant]
                            pred = np.full(len(te),constant)
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter('error',ConvergenceWarning)
                            pred, parameter, inner, status = nested(x,y,groups,tr,te,variant,classes)
                    status_rows.append(meta|dict(status=status))
                    if status!='completed':
                        continue
                    selections += [meta|r for r in inner]
                    vals = []
                    for subject in sorted(set(groups[te])):
                        m = groups[te]==subject
                        yt, pt = y[te][m], pred[m]
                        score = metric(yt,pt,classes)
                        vals.append(score)
                        corr = float(np.corrcoef(yt,pt)[0,1]) if np.std(yt)>1e-12 and np.std(pt)>1e-12 else ''
                        group_metrics.append(meta|dict(subject=subject, value=score, samples=int(m.sum()),
                            prediction_std=float(np.std(pt)), target_std=float(np.std(yt)),
                            bias=float(np.mean(pt-yt)), correlation=corr))
                    metrics.append(meta|dict(value=float(np.mean(vals)), parameter=parameter,
                        metric='macro_f1' if classes else 'rmse', samples=len(te), subjects=len(vals)))
                    predictions += [meta|dict(sample_id=str(s),subject=str(g),target=float(t),prediction=float(p))
                                    for s,g,t,p in zip(ids[te],groups[te],y[te],pred,strict=True)]
            print('scored', domain, task, fold_index, flush=True)
    prefix = domain+('_boundary' if exclusive else '')+('__constants' if constants_only else '')
    for suffix, data in [('folds',metrics),('subjects',group_metrics),('status',status_rows),('selection',selections)]:
        if data:
            save_csv(HERE/f'{prefix}_{suffix}.csv',data)
    import csv
    with gzip.open(HERE/f'{prefix}_predictions.csv.gz','wt',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(predictions[0]),lineterminator='\n');w.writeheader();w.writerows(predictions)
    save_json(HERE/f'{prefix}_verification.json',dict(status='completed',encoder_updates=0,
        scores=len(metrics),subject_scores=len(group_metrics),predictions=len(predictions),inner_scores=len(selections),
        unavailable=[r for r in status_rows if r['status']!='completed'],seconds=time.monotonic()-began,
        input_view='exclusive_event_endpoint_v1' if exclusive else 'original_frozen_input',
        registry_sha256=sha(REGISTRY),targets_sha256=sha(PREP/domain/'task_targets.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['clare','cogpilot']);p.add_argument('--constants-only',action='store_true')
    p.add_argument('--exclusive-boundary',action='store_true')
    a=p.parse_args();evaluate(a.domain,a.constants_only,a.exclusive_boundary)
