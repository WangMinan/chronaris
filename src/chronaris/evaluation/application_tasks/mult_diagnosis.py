"""Frozen interventions and source-native clipping diagnostics; no fitting."""
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS, FairResidualData
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def different_record_donors(rows, ids):
    """Preserve subject and approximate within-run progress, remove actual pairing."""
    donors = {}
    for subject in sorted({rows[i]['subject'] for i in ids}):
        groups = {}
        for i in ids:
            if rows[i]['subject'] == subject:
                groups.setdefault(rows[i]['record_id'], []).append(i)
        records = sorted(groups)
        if len(records) < 2:
            raise ValueError('need two records per subject')
        for j, key in enumerate(records):
            own = sorted(groups[key], key=lambda i: rows[i]['cutoff_s'])
            other = sorted(groups[records[(j+1) % len(records)]], key=lambda i: rows[i]['cutoff_s'])
            for k, i in enumerate(own):
                donors[i] = other[round(k*(len(other)-1)/max(1, len(own)-1))]
    return np.array([donors[i] for i in ids])


def intervene(physiology, variant, donor=None):
    if variant == 'intact':
        return physiology
    if variant == 'different_record':
        return replace(donor, timestamps_s=physiology.timestamps_s)
    if variant == 'zero_values':
        return replace(physiology, values=torch.zeros_like(physiology.values))
    if variant == 'missing':
        mask = torch.zeros_like(physiology.feature_mask)
        return replace(physiology, values=torch.zeros_like(physiology.values), feature_mask=mask,
                       modality_mask=mask.any(-1), observation_age_s=torch.full_like(physiology.values, torch.inf),
                       source_indices=torch.full_like(physiology.source_indices, -1))
    if variant == 'reverse_valid':
        values = physiology.values.clone()
        for b in range(len(values)):
            for f in range(values.shape[-1]):
                valid = physiology.feature_mask[b, :, f]
                values[b, valid, f] = values[b, valid, f].flip(0)
        return replace(physiology, values=values)
    raise ValueError(variant)


@torch.inference_mode()
def frozen_interventions(report, output):
    report, output = Path(report), Path(output)
    source = pd.read_csv(report/'predictions.csv.gz')
    snapshot = Path(json.loads((report/'qualification.json').read_text())['snapshot_path'])
    metrics, predictions, donors_out, replay = [], [], [], {}
    torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('GPU busy; no interference with other work')
        for spec in FOLDS:
            fold = spec['name']
            data = FairResidualData(snapshot, report/'qualification.json', spec)
            data.score_open = True
            ids = data.indices('score'); donor_ids = different_record_donors(data.rows, ids)
            receipt = json.loads((report/f'{fold}_mult_receipt.json').read_text())
            model = load_fair_model(receipt['checkpoint_path'], data)
            readout = joblib.load(snapshot.parent/f'{fold}_readouts.joblib')['mult']
            donors_out.extend(dict(fold=fold, sample_id=data.rows[i]['sample_id'], donor=data.rows[d]['sample_id'])
                              for i, d in zip(ids, donor_ids))
            for variant in ('intact', 'zero_values', 'missing', 'different_record', 'reverse_valid'):
                features, heads = [], []
                for off in range(0, len(ids), 32):
                    batch = ids[off:off+32]
                    p, v = data.queries(batch)
                    donor = data.queries(donor_ids[off:off+32])[0] if variant == 'different_record' else None
                    sequence, correction = model(intervene(p, variant, donor), v)
                    features.append(sequence[:, -1].cpu().numpy())
                    heads.append((data.anchor[batch, -1]+data.target_scale*correction[:, -1]).cpu().numpy())
                routes = dict(head=np.concatenate(heads), ridge=readout.predict(np.concatenate(features)))
                for route, values in routes.items():
                    if variant == 'intact':
                        prior = source[(source.fold == fold) & (source.method == 'mult_'+route)]
                        assert prior.sample_id.tolist() == [data.rows[i]['sample_id'] for i in ids]
                        np.testing.assert_allclose(values, prior.prediction, atol=1e-5, rtol=1e-5)
                        replay[fold+'_'+route] = float(np.max(np.abs(values-prior.prediction)))
                    for subject in spec['score']:
                        selected = data.subjects[ids] == subject
                        err = values[selected]-data.y[ids][selected]
                        metrics.append(dict(fold=fold, variant=variant, route=route, subject=subject,
                                            samples=int(selected.sum()), rmse=float(np.sqrt(np.mean(err**2))),
                                            mae=float(np.mean(np.abs(err)))))
                    predictions.extend(dict(fold=fold, variant=variant, route=route, sample_id=data.rows[i]['sample_id'],
                                            subject=data.rows[i]['subject'], truth=float(data.y[i]), prediction=float(p))
                                       for i, p in zip(ids, values))
                print(fold, variant, flush=True)
    save_rows(output/'intervention_metrics.csv', metrics)
    save_rows(output/'intervention_donors.csv', donors_out)
    pd.DataFrame(predictions).to_csv(output/'intervention_predictions.csv.gz', index=False)
    save_json(output/'frozen_replay.json', dict(maximum_deltas=replay, optimizer_updates=0, readout_fits=0))


def native_clip_audit(request):
    """Standalone remote function: source-only reads, final-query replay and loss."""
    # Imports stay inside so the exact function can run beside the remote raw files.
    from scipy.signal import butter, sosfilt, sosfilt_zi
    results, identities = [], {}
    groups = {}
    for row in request['windows']:
        groups.setdefault(row['record_id'], []).append(row)
    for record, windows in sorted(groups.items()):
        field = 0
        for token, columns in list(STREAM_COLUMNS.items())[:3]:
            path = next(Path(record).glob(f'*stream-{token}*_dat.csv'))
            sha = file_digest(path)
            assert sha == request['sources'][str(path)]
            frame = pd.read_csv(path, usecols=['time_dn', *columns])
            times = (frame.time_dn.to_numpy(float)-windows[0]['source_origin_native'])*86400
            for column in columns:
                raw = frame[column].to_numpy(float)
                if column == 'eda_hand_l_kOhms':
                    raw = np.divide(1000., raw, out=np.full_like(raw, np.nan), where=np.isfinite(raw)&(raw > 0))
                for row in windows:
                    end = row['cutoff_s']; start = end-30
                    keep = (times >= start) & (times < end)
                    t, x = times[keep]-start, raw[keep]
                    assert np.isfinite(x).all() and (np.diff(t) > 0).all()
                    initial = x[t < 2]
                    center = np.median(initial)
                    scale = max(1e-6, np.quantile(initial, .75)-np.quantile(initial, .25))
                    z = (x-center)/scale; clipped = np.clip(z, -20, 20)
                    fs = 1/np.median(np.diff(t[t < 1]))
                    sos = butter(2, PHYSIOLOGY_BANDS[field], btype='bandpass', fs=fs, output='sos')
                    a, _ = sosfilt(sos, z, zi=sosfilt_zi(sos)*z[0])
                    b, _ = sosfilt(sos, clipped, zi=sosfilt_zi(sos)*clipped[0])
                    tail = t >= 28
                    def feature(v, f):
                        return np.array([v[tail].mean(), np.log1p(v[tail].std()),
                                         np.log1p(np.sqrt(np.mean(f[tail]**2))),
                                         np.count_nonzero(np.diff(np.signbit(f[tail])))/(t[tail][-1]-t[tail][0])])
                    old, unbounded = feature(clipped, b), feature(z, a)
                    expected = np.array(row['final_features'][field*4:field*4+4])
                    np.testing.assert_allclose(old.astype(np.float32), expected, atol=1e-5, rtol=1e-5)
                    whole_iqr = np.quantile(x, .75)-np.quantile(x, .25)
                    std = z[tail].std(); clipped_std = clipped[tail].std()
                    rms = np.sqrt(np.mean(a[tail]**2)); clipped_rms = np.sqrt(np.mean(b[tail]**2))
                    results.append(dict(sample_id=row['sample_id'], subject=row['subject'], channel=field,
                                        scale=scale, whole_iqr_over_initial=whole_iqr/scale,
                                        clipped_fraction=float(np.mean(np.abs(z) > 20)),
                                        final_clipped_fraction=float(np.mean(np.abs(z[tail]) > 20)),
                                        final_raw_std=float(std), final_clipped_std=float(clipped_std),
                                        std_retained=float(clipped_std/std) if std > 1e-12 else 1.,
                                        band_rms_retained=float(clipped_rms/rms) if rms > 1e-12 else 1.,
                                        final_flattened=bool(std > 1e-6 and clipped_std < 1e-8),
                                        feature_replay_delta=float(np.max(np.abs(old.astype(np.float32)-expected))),
                                        mean_abs_change=float(abs(old[0]-unbounded[0])),
                                        log_std_abs_change=float(abs(old[1]-unbounded[1])),
                                        log_rms_abs_change=float(abs(old[2]-unbounded[2])),
                                        crossings_abs_change=float(abs(old[3]-unbounded[3]))))
                field += 1
            assert file_digest(path) == sha
            identities[str(path)] = sha
    return dict(rows=results, sources=identities, encoder_updates=0, readout_fits=0)
