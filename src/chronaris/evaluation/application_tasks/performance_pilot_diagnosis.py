"""Read-only diagnosis of frozen task, subject and branch errors; no fitting."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.performance_pilot_data import PerformancePilotData
from chronaris.evaluation.application_tasks.performance_pilot_training import build_pilot_model, export_features
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def affine_readout(pipeline):
    scaler, ridge = pipeline.named_steps['standardscaler'], pipeline.named_steps['ridge']
    weight = ridge.coef_.reshape(-1)/scaler.scale_
    return weight, float(np.asarray(ridge.intercept_).item()-weight @ scaler.mean_)


def error_stats(y, prediction):
    delta = np.asarray(prediction)-np.asarray(y)
    return dict(rmse=float(np.sqrt(np.mean(delta**2))), mae=float(np.mean(np.abs(delta))),
                bias=float(delta.mean()), centered_rmse=float(delta.std()),
                prediction_std=float(np.std(prediction)), target_std=float(np.std(y)))


def inspect_source_sampling(request):
    """Read one identity-selected run per person on the source host."""
    result = []
    for record in request['records']:
        run = Path(record['record_id'])
        for token in ('lslshimmereda', 'lslshimmerresp', 'lslshimmerecg', 'lslxp11xpcac'):
            path = next(run.glob(f'*stream-{token}*_dat.csv'))
            before = file_digest(path)
            assert before == request['source_hashes'][str(path)]
            table = pd.read_csv(path, usecols=lambda name: name == 'time_dn')
            times = table.time_dn.to_numpy(float)
            dt = np.diff(times)*86400
            positive = dt[dt > 0]
            header = list(pd.read_csv(path, nrows=0).columns)
            result.append(dict(subject=record['subject'], stream=token, file=str(path), rows=len(times),
                               duration_s=float((times.max()-times.min())*86400),
                               median_positive_hz=float(1/np.median(positive)),
                               interval_p01_s=float(np.quantile(positive, .01)),
                               interval_p99_s=float(np.quantile(positive, .99)),
                               nonpositive_intervals=int(np.sum(dt <= 0)),
                               geometry_fields=[f for f in ('aircraft_longitude_deg', 'aircraft_latitude_deg',
                                                           'aircraft_elevation_m') if f in header],
                               sha256=before))
            assert file_digest(path) == before
    return result


def run_pilot_diagnosis(original, output):
    original, output = Path(original), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'verification.json').exists():
        raise FileExistsError('diagnosis already exists')
    frozen = json.loads((output/'protected_sources.json').read_text())
    assert all(file_digest(p) == h for p, h in frozen.items())
    qualification = json.loads((original/'qualification.json').read_text())
    snapshot = Path(qualification['snapshot_path'])
    with np.load(snapshot, allow_pickle=False) as raw:
        metadata = json.loads(str(raw['metadata']))
        raw_values, raw_mask = raw['values'], raw['masks']
    lock = json.loads((original/'locked_models.json').read_text())
    readouts = joblib.load(snapshot.parent/'readouts.joblib')
    existing = pd.read_csv(original/'predictions.csv.gz')
    score_wide = existing.pivot(index=['sample_id', 'subject', 'truth'], columns='method', values='prediction').reset_index()
    task_rows = []
    for subject, group in score_wide.groupby('subject'):
        y = group.truth.to_numpy()
        for method in ('same_backbone', 'vehicle_only', 'independent_concat', 'persistence'):
            task_rows.append(dict(subject=subject, method=method, **error_stats(y, group[method])))
    save_rows(output/'scoring_errors.csv', task_rows)
    gaps = []
    for subject, g in score_wide.groupby('subject'):
        y = g.truth.to_numpy()
        gaps.append(dict(subject=subject,
                         persistence_correlation=float(np.corrcoef(y, g.persistence)[0, 1]),
                         candidate_plane_mse_gap=float(np.mean((g.same_backbone-y)**2-(g.vehicle_only-y)**2)),
                         plane_persistence_mse_gap=float(np.mean((g.vehicle_only-y)**2-(g.persistence-y)**2))))
    save_rows(output/'error_decomposition.csv', gaps)
    subjects = np.array([r['subject'] for r in metadata['windows'] if r['accepted']])
    scale_rows = []
    norm = json.loads((original/'normalization.json').read_text())
    for subject in sorted(set(subjects)):
        for field in range(12):
            vals = raw_values[subjects == subject, :, field][raw_mask[subjects == subject, :, field]]
            standardized = (vals-norm['center'][field])/norm['scale'][field]
            scale_rows.append(dict(subject=subject, field=field, median=float(np.median(vals)),
                                   iqr=float(np.quantile(vals, .75)-np.quantile(vals, .25)),
                                   normalized_abs_p99=float(np.quantile(np.abs(standardized), .99)),
                                   normalized_abs_max=float(np.abs(standardized).max())))
    save_rows(output/'input_scale.csv', scale_rows)
    torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('respect existing GPU owner')
        data = PerformancePilotData(snapshot, original/'qualification.json', 'cuda')
        data.score_open = True
        features, branches, metrics = {}, {}, []
        for method in ('vehicle_only', 'physiology_only', 'same_backbone'):
            state = torch.load(lock['receipts'][method]['checkpoint_path'], map_location='cuda', weights_only=True)
            model = build_pilot_model(method, 'cuda'); model.load_state_dict(state['encoder']); model.eval()
            head = torch.nn.Linear(64, 1).cuda(); head.load_state_dict(state['head']); head.eval()
            features[method] = {}
            for role in ('fit', 'select', 'score'):
                ids = data.indices(role)
                f, h = export_features(model, head, method, data, ids)
                features[method][role] = f
                for label, pred in [('ridge', readouts[method].predict(f)),
                                    ('head', h*data.target_scale+data.target_center)]:
                    if role == 'score':
                        previous = existing[existing.method == method+('_task_head' if label == 'head' else '')]
                        assert list(previous.sample_id) == [data.rows[i]['sample_id'] for i in ids]
                        np.testing.assert_allclose(pred, previous.prediction.to_numpy(), atol=1e-5, rtol=1e-5)
                    for subject in sorted(set(data.subjects[ids])):
                        sel = data.subjects[ids] == subject
                        metrics.append(dict(method=method, role=role, readout=label, subject=subject,
                                            **error_stats(data.y[ids][sel], pred[sel])))
                if method == 'same_backbone':
                    parts = []
                    with torch.inference_mode():
                        for offset in range(0, len(ids), 32):
                            p, v = data.queries(ids[offset:offset+32])
                            a, b = model.physiology.forward_queried(p), model.vehicle.forward_queried(v)
                            valid = a.valid_mask | b.valid_mask
                            positions = torch.arange(96, device='cuda').expand_as(valid).masked_fill(~valid, -1).max(1).values
                            cat = torch.cat((a.sequence_embedding, b.sequence_embedding), -1)
                            parts.append(cat[torch.arange(len(cat), device='cuda'), positions].cpu().numpy())
                    branches[role] = np.concatenate(parts)
            if method == 'same_backbone':
                weight, bias = affine_readout(readouts[method])
                branch_weight = model.projection.weight.detach().cpu().numpy().T @ weight
                branch_bias = bias + float(model.projection.bias.detach().cpu().numpy() @ weight)
            del model, head
        save_rows(output/'role_metrics.csv', metrics)
        anchor = np.average(branches['fit'], axis=0, weights=data.weights[data.indices('fit')].cpu().numpy())
        branch_rows = []
        for role in ('fit', 'select', 'score'):
            ids = data.indices(role); z = branches[role]
            a, b = z[:, :64] @ branch_weight[:64], z[:, 64:] @ branch_weight[64:]
            pred = a+b+branch_bias
            np.testing.assert_allclose(pred, readouts['same_backbone'].predict(features['same_backbone'][role]), atol=1e-5, rtol=1e-5)
            alternatives = {'original': pred, 'physiology_at_fit_mean': b+float(anchor[:64] @ branch_weight[:64])+branch_bias,
                            'vehicle_at_fit_mean': a+float(anchor[64:] @ branch_weight[64:])+branch_bias}
            for subject in sorted(set(data.subjects[ids])):
                sel = data.subjects[ids] == subject
                for name, values in alternatives.items():
                    branch_rows.append(dict(role=role, subject=subject, intervention=name,
                                            physiology_contribution_mean=float(a[sel].mean()),
                                            vehicle_contribution_mean=float(b[sel].mean()),
                                            physiology_contribution_std=float(a[sel].std()),
                                            vehicle_contribution_std=float(b[sel].std()),
                                            **error_stats(data.y[ids][sel], values[sel])))
        save_rows(output/'branch_interventions.csv', branch_rows)
        stat_rows = []
        for role in ('fit', 'select', 'score'):
            ids = data.indices(role); p, v = data.historical_statistics(ids)
            for method, x in [('physiology_statistics', p), ('dual_statistics', np.concatenate((p, v), 1))]:
                weight, bias = affine_readout(readouts[method]); contribution = x*weight
                np.testing.assert_allclose(contribution.sum(1)+bias, readouts[method].predict(x), atol=1e-8, rtol=1e-8)
                for subject in sorted(set(data.subjects[ids])):
                    sel = data.subjects[ids] == subject
                    for f in range(x.shape[1]):
                        stat_rows.append(dict(method=method, role=role, subject=subject, feature=f,
                                              mean=float(x[sel, f].mean()), std=float(x[sel, f].std()),
                                              coefficient=float(weight[f]), mean_contribution=float(contribution[sel, f].mean()),
                                              scaler_center=float(readouts[method].named_steps['standardscaler'].mean_[f]),
                                              scaler_scale=float(readouts[method].named_steps['standardscaler'].scale_[f])))
        save_rows(output/'statistics_contributions.csv', stat_rows)
    changed = [p for p, h in frozen.items() if file_digest(p) != h]
    assert not changed
    result = dict(status='passed', protected_files=len(frozen), changed_sources=changed,
                  encoder_updates=0, new_readout_fits=0, frozen_prediction_replay=True,
                  branch_linear_decomposition_verified=True, diagnosis_scope='post_hoc_development_only')
    save_json(output/'verification.json', result)
    return result
