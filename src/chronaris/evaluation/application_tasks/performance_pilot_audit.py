"""Qualification and source-bound receipts for the registered performance task."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from chronaris.dataset.cogpilot_performance import file_digest


FIT_SUBJECTS = ('sub-cp004', 'sub-cp008', 'sub-cp019')
SELECT_SUBJECTS = ('sub-cp025',)
SCORE_SUBJECTS = ('sub-cp003', 'sub-cp011', 'sub-cp022')


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_rows(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def subject_rmse(y, prediction, subjects):
    return float(np.mean([np.sqrt(np.mean((y[subjects == s] - prediction[subjects == s])**2))
                          for s in sorted(set(subjects))]))


def qualify_performance_task(snapshot, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    receipt = output/'qualification.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    with np.load(snapshot, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        values, times, masks = (data[k] for k in ('values', 'observation_times', 'masks'))
    accepted = [r for r in metadata['windows'] if r['accepted']]
    assert len(accepted) == len(values) == len(metadata['target_points'])
    assert len({r['sample_id'] for r in metadata['windows']}) == len(metadata['windows'])
    subjects = np.array([r['subject'] for r in accepted])
    y = np.array([r['target'] for r in accepted])
    persistence = np.array([r['persistence'] for r in accepted])
    assert np.isfinite(y).all() and np.isfinite(values[masks]).all()
    assert np.isfinite(times[masks]).all() and (times[masks] >= 0).all() and (times[masks] < 30).all()
    queries = np.broadcast_to(np.linspace(0, 30, 96, endpoint=False)[None, :, None], times.shape)
    assert np.all((queries - times)[masks] >= -1e-9) and np.all((queries - times)[masks] <= 2 + 1e-9)
    target_deltas = []
    for row, source in zip(accepted, metadata['target_points'], strict=True):
        assert row['sample_id'] == source['sample_id']
        for prefix, lo, hi, expected in [('future', 0., 5., row['target']),
                                          ('past', -5., 0., row['persistence'])]:
            t, v = np.array(source[prefix+'_t']), np.array(source[prefix+'_y'])
            assert t.min() > lo and t.max() <= hi
            if prefix == 'past':
                assert t.max() < hi
            assert max(t[0]-lo, hi-t[-1], np.diff(t).max()) <= 1
            # Independent quadrature: trapezoids plus two boundary rectangles.
            mean = (np.trapezoid(v, t) + (t[0]-lo)*v[0] + (hi-t[-1])*v[-1])/(hi-lo)
            target_deltas.append(abs(mean - expected))
    assert max(target_deltas, default=0) < 1e-10
    summaries = []
    for s in sorted(set(subjects)):
        sel = subjects == s
        rows = [r for r in accepted if r['subject'] == s]
        summaries.append(dict(subject=s, samples=int(sel.sum()), records=len({r['record_id'] for r in rows}),
                              target_mean=float(y[sel].mean()), target_std=float(y[sel].std()),
                              target_min=float(y[sel].min()), target_max=float(y[sel].max()),
                              persistence_rmse=float(np.sqrt(np.mean((y[sel]-persistence[sel])**2)))))
    train_pool = np.isin(subjects, FIT_SUBJECTS + SELECT_SUBJECTS)
    baselines = []
    for s in FIT_SUBJECTS + SELECT_SUBJECTS:
        train = train_pool & (subjects != s); score = subjects == s
        for name, prediction in [('train_mean', np.full(score.sum(), y[train].mean())),
                                 ('train_median', np.full(score.sum(), np.median(y[train]))),
                                 ('persistence', persistence[score])]:
            baselines.append(dict(subject=s, method=name,
                                  rmse=float(np.sqrt(np.mean((prediction-y[score])**2)))))
    averages = {name: float(np.mean([r['rmse'] for r in baselines if r['method'] == name]))
                for name in ('train_mean', 'train_median', 'persistence')}
    ratio = averages['persistence']/averages['train_mean']
    gates = dict(
        records=len(metadata['records']) == 84,
        source_file_count=len(metadata['sources']) == 420,
        subjects=set(subjects) == set(FIT_SUBJECTS + SELECT_SUBJECTS + SCORE_SUBJECTS),
        per_subject_support=all(r['samples'] >= 100 and r['records'] >= 8 for r in summaries),
        retained_fraction=len(accepted)/max(1, len(metadata['windows'])) >= .8,
        nonconstant=all(r['target_std'] >= .01 for r in summaries),
        persistence_not_trivial=bool(ratio >= .2),
        target_recomputed=True, history_only=True)
    save_json(output/'sources.json', metadata['sources'])
    save_json(output/'record_inventory.json', metadata['records'])
    save_rows(output/'sample_manifest.csv', metadata['windows'])
    save_rows(output/'subject_qualification.csv', summaries)
    save_rows(output/'qualification_baselines.csv', baselines)
    payload = dict(status='passed' if all(gates.values()) else 'failed', gates=gates,
                   snapshot_sha256=file_digest(snapshot), snapshot_path=str(Path(snapshot).resolve()),
                   samples=len(accepted), candidates=len(metadata['windows']),
                   rejected_reasons=dict(Counter(r['reason'] for r in metadata['windows'] if not r['accepted'])),
                   target_recompute_max_delta=max(target_deltas, default=0),
                   training_pool_baselines=averages, persistence_to_mean_rmse_ratio=ratio,
                   fit_subjects=FIT_SUBJECTS, selection_subjects=SELECT_SUBJECTS, score_subjects=SCORE_SUBJECTS,
                   encoder_updates=0, evidence_scope='previously_exposed_development_subjects')
    save_json(receipt, payload)
    return payload
