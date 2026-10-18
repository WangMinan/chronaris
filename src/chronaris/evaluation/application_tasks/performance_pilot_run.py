"""Fit, lock, then score one qualified development comparison panel."""
import gzip
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.performance_pilot_data import PerformancePilotData
from chronaris.evaluation.application_tasks.performance_pilot_training import (
    METHODS, build_pilot_model, export_features, train_pilot_method,
)
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def fit_readout(data, features):
    train, select = data.indices('fit'), data.indices('select')
    return fit_regressor(features['fit'], data.y[train], features['select'], data.y[select],
                         alpha_values=(.1, 1., 10., 100., 1000.), scaler_with_mean=True,
                         train_sample_weight=data.weights[train].cpu().numpy(),
                         validation_groups=data.subjects[select])


def run_performance_pilot(snapshot, report, assets):
    report, assets = Path(report), Path(assets)
    if (report/'run_status.json').exists():
        raise FileExistsError('previous run is retained; no automatic replacement')
    if not torch.cuda.is_available():
        raise RuntimeError('the registered neural pilot requires CUDA')
    torch.set_num_threads(1)
    started = time.monotonic()
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('GPU is owned by another experiment')
        data = PerformancePilotData(Path(snapshot), report/'qualification.json', 'cuda')
        binding = dict(snapshot_sha256=data.snapshot_sha256,
                       qualification_sha256=file_digest(report/'qualification.json'),
                       plan_sha256=file_digest(report/'PLAN.md'),
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       device=torch.cuda.get_device_name(), role_subjects={
                           role: sorted(str(s) for s in set(data.subjects[ids])) for role, ids in data.roles.items()})
        save_json(report/'training_binding.json', binding)
        save_json(report/'normalization.json', data.normalizer)
        save_json(report/'run_status.json', dict(status='running', current_method=None, binding=binding))
        try:
            return _execute(data, report, assets, started, binding)
        except BaseException as exc:
            save_json(report/'run_status.json', dict(status='failed', error=repr(exc), binding=binding,
                                                    elapsed_s=time.monotonic()-started))
            raise


def _execute(data, report, assets, started, binding):
    training = assets/'training'; training.mkdir(exist_ok=False)
    feature_sets, receipts, readouts, choices = {}, {}, {}, {}
    for method in METHODS:
        save_json(report/'run_status.json', dict(status='running', current_method=method, binding=binding))
        feature_sets[method], receipts[method] = train_pilot_method(
            method, data, training, report, deadline=started+4*3600, binding=binding)
        if not receipts[method]['complete']:
            result = dict(status='budget_incomplete', completed_methods=list(receipts), score_role_opened=False)
            save_json(report/'run_status.json', result)
            return result
        arrays = {role: feature_sets[method][role][0] for role in ('fit', 'select')}
        readouts[method], choices[method] = fit_readout(data, arrays)
    combined = {role: np.concatenate([feature_sets[m][role][0] for m in ('physiology_only', 'vehicle_only')], axis=1)
                for role in ('fit', 'select')}
    readouts['independent_concat'], choices['independent_concat'] = fit_readout(data, combined)
    statistics = {role: data.historical_statistics(data.indices(role)) for role in ('fit', 'select')}
    for name, stream in [('physiology_statistics', 0), ('vehicle_statistics', 1), ('dual_statistics', None)]:
        arrays = {role: (np.concatenate(statistics[role], axis=1) if stream is None else statistics[role][stream])
                  for role in ('fit', 'select')}
        readouts[name], choices[name] = fit_readout(data, arrays)
    readout_path = assets/'readouts.joblib'; joblib.dump(readouts, readout_path)
    lock = dict(binding=binding, selected_ridge_alpha=choices, receipts=receipts,
                readout_sha256=file_digest(readout_path), score_role_opened=False)
    save_json(report/'locked_models.json', lock)
    data.score_open = True
    score = data.indices('score')
    predictions, score_features = {}, {}
    for method in METHODS:
        state = torch.load(receipts[method]['checkpoint_path'], map_location='cuda', weights_only=True)
        if file_digest(receipts[method]['checkpoint_path']) != receipts[method]['checkpoint_sha256']:
            raise RuntimeError('checkpoint changed after selection')
        model = build_pilot_model(method, 'cuda'); model.load_state_dict(state['encoder'])
        head = torch.nn.Linear(64, 1).cuda(); head.load_state_dict(state['head'])
        score_features[method], pred = export_features(model, head, method, data, score)
        predictions[method] = readouts[method].predict(score_features[method])
        predictions[method+'_task_head'] = pred*data.target_scale+data.target_center
        del model, head
    predictions['independent_concat'] = readouts['independent_concat'].predict(np.concatenate(
        [score_features[m] for m in ('physiology_only', 'vehicle_only')], axis=1))
    stats = data.historical_statistics(score)
    for name, stream in [('physiology_statistics', 0), ('vehicle_statistics', 1), ('dual_statistics', None)]:
        predictions[name] = readouts[name].predict(np.concatenate(stats, axis=1) if stream is None else stats[stream])
    predictions['train_mean'] = np.full(len(score), data.target_center)
    predictions['train_median'] = np.full(len(score), np.median(data.y[data.indices('fit')]))
    predictions['persistence'] = np.array([data.rows[i]['persistence'] for i in score])
    summary, per_subject, prediction_rows = [], [], []
    for method, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f'nonfinite scoring: {method}')
        group_rows = []
        for subject in sorted(set(data.subjects[score])):
            mask = data.subjects[score] == subject
            delta = pred[mask]-data.y[score][mask]
            row = dict(method=method, subject=subject, samples=int(mask.sum()),
                       rmse=float(np.sqrt(np.mean(delta**2))), mae=float(np.mean(np.abs(delta))))
            group_rows.append(row); per_subject.append(row)
        summary.append(dict(method=method, rmse=float(np.mean([r['rmse'] for r in group_rows])),
                            mae=float(np.mean([r['mae'] for r in group_rows])),
                            scope='task_head' if method.endswith('_task_head') else 'primary'))
        prediction_rows.extend(dict(method=method, sample_id=data.rows[i]['sample_id'], subject=data.subjects[i],
                                    truth=float(data.y[i]), prediction=float(value)) for i, value in zip(score, pred))
    save_rows(report/'summary.csv', summary); save_rows(report/'subject_metrics.csv', per_subject)
    temporary = report/'predictions.csv'
    save_rows(temporary, prediction_rows)
    with temporary.open('rb') as src, gzip.open(report/'predictions.csv.gz', 'wb') as dest:
        dest.write(src.read())
    temporary.unlink()
    candidate = next(r for r in summary if r['method'] == 'same_backbone')
    controls = [r for r in summary if r['scope'] == 'primary' and r['method'] != 'same_backbone']
    strongest = min(controls, key=lambda r: r['rmse'])
    improvements = {r['method']: 1-candidate['rmse']/r['rmse'] for r in controls}
    wins = sum(next(r for r in per_subject if r['method'] == 'same_backbone' and r['subject'] == s)['rmse'] <
               next(r for r in per_subject if r['method'] == strongest['method'] and r['subject'] == s)['rmse']
               for s in sorted(set(data.subjects[score])))
    result = dict(status='completed', success=min(improvements.values()) >= .02 and wins >= 2,
                  candidate_rmse=candidate['rmse'], strongest_control=strongest,
                  relative_improvements=improvements, subjects_improved_over_strongest=wins,
                  encoder_updates=sum(r['updates'] for r in receipts.values()),
                  model_count=len(receipts), score_samples=len(score), elapsed_s=time.monotonic()-started,
                  scope='one_previously_exposed_development_fold_seed17', binding=binding)
    save_json(report/'run_status.json', result)
    return result
