"""Versioned, strictly historical windows for published flight-error prediction.

This module can also execute read-only on the source host; it uses no repository
state and returns a compact observation snapshot with original timestamps.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


STREAM_COLUMNS = {
    'lslshimmereda': ('ppg_finger_mV', 'eda_hand_l_kOhms'),
    'lslshimmerresp': ('respiration_trace_mV',),
    'lslshimmerecg': ('ecg_projection_ll_ra_mV',),
    'lslxp11xpcac': (
        'aircraft_indicated_airspeed_kias', 'aircraft_pitch_deg',
        'aircraft_roll_deg', 'aircraft_agl_altitude_m',
        'aircraft_climb_rate_mps', 'aircraft_ils_deflection_gs',
        'aircraft_ils_deflection_h', 'aircraft_velocity_u_mps'),
}


def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def interval_mean(times, values, start, end, *, include_end):
    """Midpoint-cell quadrature using only observations inside the interval."""
    times, values = np.asarray(times), np.asarray(values)
    selected = (times > start) & ((times <= end) if include_end else (times < end))
    t, y = times[selected], values[selected]
    if len(t) < 2 or not np.isfinite(y).all():
        raise ValueError('target_points')
    if np.any(np.diff(t) <= 0):
        raise ValueError('duplicate_or_reversed_target_time')
    if max(t[0] - start, end - t[-1], np.diff(t).max()) > 1.0:
        raise ValueError('target_gap')
    edges = np.r_[start, (t[:-1] + t[1:]) / 2, end]
    return float(np.dot(np.diff(edges), y) / (end - start)), t, y


def history_queries(times, values, start, cutoff):
    """Retain only the observations needed by 96 causal featurewise queries."""
    q = np.linspace(start, cutoff, 96, endpoint=False)
    t, x = np.asarray(times), np.asarray(values)
    selected = (t >= start) & (t < cutoff)
    t, x = t[selected], x[selected]
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError('input_time_order')
    out = np.zeros((96, x.shape[1]), dtype=np.float32)
    stamps = np.full(out.shape, np.nan, dtype=np.float64)
    mask = np.zeros(out.shape, dtype=bool)
    for f in range(x.shape[1]):
        finite = np.isfinite(x[:, f])
        ft, fx = t[finite], x[finite, f]
        if not len(ft):
            raise ValueError('missing_input_field')
        idx = np.searchsorted(ft, q, side='right') - 1
        valid = idx >= 0
        positions = np.maximum(idx, 0)
        valid &= q - ft[positions] <= 2.0
        out[valid, f] = fx[positions[valid]]
        stamps[valid, f] = ft[positions[valid]] - start
        mask[:, f] = valid
    return out, stamps, mask


def extract_performance_windows(records):
    """Read the registered runs, leaving every source file untouched."""
    sources, inventory, windows, targets = {}, [], [], []
    values, stamps, masks = [], [], []
    for item in records:
        run = Path(item['record_id'])
        performance_path = Path(item['file'])
        paths = [next(run.glob(f'*stream-{token}*_dat.csv')) for token in STREAM_COLUMNS]
        before = {str(p): file_digest(p) for p in [performance_path, *paths]}
        perf = pd.read_csv(performance_path)
        origin = float(perf.time_dn.iloc[0])
        pt = (perf.time_dn.to_numpy(float) - origin) * 86400
        py = perf.total_error.to_numpy(float)
        components = perf[['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts']].to_numpy(float)
        if not np.isfinite(pt).all() or not np.isfinite(py).all() or np.any(np.diff(pt) <= 0):
            raise ValueError(f'invalid performance time/value: {run}')
        error = float(np.max(np.abs(np.abs(components).sum(axis=1) - py)))
        if error > 1e-10:
            raise ValueError(f'published component mismatch: {run}')
        streams, reversals, duplicate_counts = [], [], []
        for path, columns in zip(paths, STREAM_COLUMNS.values(), strict=True):
            df = pd.read_csv(path, usecols=['time_dn', *columns])
            t = (df.time_dn.to_numpy(float) - origin) * 86400
            x = df[list(columns)].to_numpy(float)
            if not np.isfinite(t).all():
                raise ValueError(f'nonfinite input time: {path}')
            local_reversals = []
            for k in np.flatnonzero(np.diff(t) < 0):
                local_reversals.append((float(t[k + 1]), float(t[k])))
            reversals.extend(local_reversals)
            unique, counts = np.unique(t, return_counts=True)
            duplicates = unique[counts > 1]
            if any(not any(lo <= point <= hi for lo, hi in local_reversals) for point in duplicates):
                raise ValueError(f'duplicate input time outside quarantined reversal: {path}')
            duplicate_counts.append(dict(file=str(path), duplicate_timestamps=len(duplicates)))
            if 'eda_hand_l_kOhms' in columns:
                f = columns.index('eda_hand_l_kOhms')
                valid = np.isfinite(x[:, f]) & (x[:, f] > 0)
                x[:, f] = np.divide(1000., x[:, f], out=np.full(len(x), np.nan), where=valid)
            streams.append((t, x))
        first = max(pt[0], *(t.min() for t, _ in streams))
        last = min(pt[-1], *(t.max() for t, _ in streams))
        count = max(0, int(np.floor((last - first - 35) / 5)) + 1)
        kept = 0
        for k in range(count):
            cutoff = first + 30 + 5 * k
            row = dict(sample_id=f"{item['subject']}::{run.parent.name}::{run.name}::performance{k:04d}",
                       subject=item['subject'], record_id=str(run), cutoff_s=cutoff,
                       source_origin_native=origin, accepted=False, reason='')
            try:
                if any(cutoff > lo and cutoff - 30 < hi for lo, hi in reversals):
                    raise ValueError('input_clock_reversal')
                y, future_t, future_y = interval_mean(pt, py, cutoff, cutoff + 5, include_end=True)
                persistence, past_t, past_y = interval_mean(pt, py, cutoff - 5, cutoff, include_end=False)
                queried = [history_queries(t, x, cutoff - 30, cutoff) for t, x in streams]
                x = np.concatenate([a[0] for a in queried], axis=1)
                t = np.concatenate([a[1] for a in queried], axis=1)
                m = np.concatenate([a[2] for a in queried], axis=1)
                if min(m[:, :4].any(axis=1).mean(), m[:, 4:].any(axis=1).mean()) < .9:
                    raise ValueError('input_coverage')
                row.update(accepted=True, target=y, persistence=persistence,
                           physiology_coverage=float(m[:, :4].any(axis=1).mean()),
                           vehicle_coverage=float(m[:, 4:].any(axis=1).mean()))
                values.append(x); stamps.append(t); masks.append(m)
                targets.append(dict(sample_id=row['sample_id'], future_t=(future_t-cutoff).tolist(),
                                    future_y=future_y.tolist(), past_t=(past_t-cutoff).tolist(),
                                    past_y=past_y.tolist()))
                kept += 1
            except ValueError as exc:
                row['reason'] = str(exc)
            windows.append(row)
        if any(file_digest(p) != h for p, h in before.items()):
            raise RuntimeError('source changed during extraction')
        sources.update(before)
        inventory.append(dict(record_id=str(run), subject=item['subject'], candidates=count,
                              accepted=kept, component_max_error=error, reversals=reversals,
                              duplicate_counts=duplicate_counts))
    return dict(values=np.asarray(values), observation_times=np.asarray(stamps), masks=np.asarray(masks),
                metadata=np.array(json.dumps(dict(sources=sources, records=inventory, windows=windows,
                                                 target_points=targets), allow_nan=False)))
