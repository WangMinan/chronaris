"""Frozen interventions and source-native clipping diagnostics; no fitting."""
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from chronaris.dataset.cogpilot_performance import STREAM_COLUMNS, file_digest
from chronaris.dataset.cogpilot_history_features import PHYSIOLOGY_BANDS
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
                    ct = t[tail]
                    valid = len(ct) >= 50 and max(ct[0]-28, 30-ct[-1], np.diff(ct).max()) <= .25
                    assert row['final_mask'][field*4:field*4+4] == [bool(valid)]*4
                    replayed = old.astype(np.float32) if valid else np.zeros(4, np.float32)
                    np.testing.assert_allclose(replayed, expected, atol=1e-5, rtol=1e-5)
                    whole_iqr = np.quantile(x, .75)-np.quantile(x, .25)
                    std = z[tail].std(); clipped_std = clipped[tail].std()
                    rms = np.sqrt(np.mean(a[tail]**2)); clipped_rms = np.sqrt(np.mean(b[tail]**2))
                    results.append(dict(sample_id=row['sample_id'], subject=row['subject'], channel=field,
                                        final_feature_valid=bool(valid),
                                        scale=scale, whole_iqr_over_initial=whole_iqr/scale,
                                        clipped_fraction=float(np.mean(np.abs(z) > 20)),
                                        final_clipped_fraction=float(np.mean(np.abs(z[tail]) > 20)),
                                        final_raw_std=float(std), final_clipped_std=float(clipped_std),
                                        std_retained=float(clipped_std/std) if std > 1e-12 else 1.,
                                        band_rms_retained=float(clipped_rms/rms) if rms > 1e-12 else 1.,
                                        final_flattened=bool(std > 1e-6 and clipped_std < 1e-8),
                                        feature_replay_delta=float(np.max(np.abs(replayed-expected))),
                                        mean_abs_change=float(abs(old[0]-unbounded[0])),
                                        log_std_abs_change=float(abs(old[1]-unbounded[1])),
                                        log_rms_abs_change=float(abs(old[2]-unbounded[2])),
                                        crossings_abs_change=float(abs(old[3]-unbounded[3]))))
                field += 1
            assert file_digest(path) == sha
            identities[str(path)] = sha
    return dict(rows=results, sources=identities, encoder_updates=0, readout_fits=0)


def summarize_diagnosis(report, output):
    report, output = Path(report), Path(output)
    manifest = json.loads((output/'protected_sources.json').read_text())
    assert all(file_digest(p) == sha for p, sha in manifest.items())
    predictions = pd.read_csv(output/'intervention_predictions.csv.gz')
    recorded = pd.read_csv(output/'intervention_metrics.csv')
    delta = 0.
    for keys, group in predictions.groupby(['fold', 'variant', 'route', 'subject']):
        row = recorded[(recorded[['fold', 'variant', 'route', 'subject']] == keys).all(axis=1)].iloc[0]
        err = group.prediction-group.truth
        assert len(group) == row.samples and group.sample_id.nunique() == len(group)
        delta = max(delta, abs(np.sqrt(np.mean(err**2))-row.rmse), abs(np.abs(err).mean()-row.mae))
    assert delta < 1e-10
    recorded.groupby(['fold', 'variant', 'route'])[['rmse', 'mae']].mean().to_csv(output/'intervention_summary.csv')
    original = pd.read_csv(report/'subject_metrics.csv')
    table = original.pivot(index=['fold', 'subject'], columns='method', values='rmse')
    table['mult_over_aircraft_improvement'] = 1-table.mult_ridge/table.aircraft_ridge
    table['aircraft_minus_mult_rmse'] = table.aircraft_ridge-table.mult_ridge
    table['share_of_fold_rmse_drop'] = table.aircraft_minus_mult_rmse/table.groupby(level=0).aircraft_minus_mult_rmse.transform('sum')
    table.to_csv(output/'subject_comparison.csv')
    receipts = []
    for fold in FOLDS:
        for method in ('aircraft', 'contiformer', 'mult', 'candidate', 'control'):
            name = fold['name']+'_'+method
            receipt = json.loads((report/f'{name}_receipt.json').read_text())
            history = pd.read_csv(report/f'{name}_history.csv')
            best = history.loc[history.epoch == receipt['best_epoch']].iloc[0]
            receipts.append(dict(fold=fold['name'], method=method, parameters=receipt['parameter_count'],
                                 best_epoch=receipt['best_epoch'], normalized_training_mse=best.training_mse,
                                 selection_rmse=receipt['selection_rmse']))
    save_rows(output/'training_comparison.csv', receipts)
    native = pd.read_csv(output/'native_clip_windows.csv')
    assert len(native) == 5951*4 and not native[['sample_id', 'channel']].duplicated().any()
    old = pd.read_csv(report/'physiology_quality.csv')
    match = native.merge(old, on=['sample_id', 'channel'], validate='one_to_one', suffixes=('', '_old'))
    assert np.max(np.abs(match.clipped_fraction-match.clipped_fraction_old)) < 1e-12
    assert native.feature_replay_delta.max() <= 1e-5
    quality = []
    for (subject, channel), group in native.groupby(['subject', 'channel']):
        quality.append(dict(subject=subject, channel=channel, windows=len(group),
                            clipped_over_10=int((group.clipped_fraction > .1).sum()),
                            final_clipped_over_50=int((group.final_clipped_fraction > .5).sum()),
                            final_flattened=int(group.final_flattened.sum()),
                            std_below_half=int((group.std_retained < .5).sum()),
                            band_below_half=int((group.band_rms_retained < .5).sum()),
                            band_over_double=int((group.band_rms_retained > 2).sum()),
                            scale_ratio_median=group.whole_iqr_over_initial.median(),
                            scale_ratio_p95=group.whole_iqr_over_initial.quantile(.95)))
    save_rows(output/'clipping_by_subject.csv', quality)
    aggregate = []
    for channel, group in native.groupby('channel'):
        valid = group[group.final_feature_valid]
        aggregate.append(dict(channel=channel, windows=len(group), final_valid=len(valid),
                              initial_scale_floor=int((group.scale <= 1e-6).sum()),
                              clipped_over_10=int((group.clipped_fraction > .1).sum()),
                              final_flattened=int(valid.final_flattened.sum()),
                              std_below_half=int((valid.std_retained < .5).sum()),
                              band_below_half=int((valid.band_rms_retained < .5).sum()),
                              band_over_double=int((valid.band_rms_retained > 2).sum()),
                              whole_over_initial_iqr_median=group.whole_iqr_over_initial.median(),
                              whole_over_initial_iqr_p95=group.whole_iqr_over_initial.quantile(.95)))
    save_rows(output/'clipping_summary.csv', aggregate)
    # Descriptive slices only: overlapping windows do not provide independent tests.
    strata = native.groupby('sample_id').agg(high_clip=('clipped_fraction', lambda x: bool((x > .1).any())),
                                            flattened=('final_flattened', 'any'))
    source = pd.read_csv(report/'predictions.csv.gz')
    source = source[source.method.isin(['aircraft_ridge', 'mult_ridge', 'candidate_ridge'])]
    source = source.merge(strata, on='sample_id', validate='many_to_one')
    rows = []
    for (fold, subject, method, high), group in source.groupby(['fold', 'subject', 'method', 'high_clip']):
        rows.append(dict(fold=fold, subject=subject, method=method, high_clip=bool(high), samples=len(group),
                         rmse=float(np.sqrt(np.mean((group.prediction-group.truth)**2)))))
    save_rows(output/'clipping_error_slices.csv', rows)
    result = dict(status='passed', protected_files=len(manifest), changed_sources=[],
                  native_windows=5951, native_channel_windows=len(native),
                  native_sources=len(json.loads((output/'native_clip_receipt.json').read_text())['sources']),
                  max_native_replay_delta=float(native.feature_replay_delta.max()),
                  intervention_predictions=len(predictions), intervention_subject_metrics=len(recorded),
                  max_metric_recompute_delta=float(delta), encoder_updates=0, new_readout_fits=0)
    save_json(output/'verification.json', result)
    return result
