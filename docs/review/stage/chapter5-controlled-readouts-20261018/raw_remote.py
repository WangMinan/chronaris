"""Independent CSV audit, executed read-only on the machine holding raw recordings."""
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def digest_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@lru_cache(maxsize=8)
def table(path):
    return pd.read_csv(path)


@lru_cache(maxsize=None)
def file_info(path):
    p = Path(path)
    with p.open('rb') as f:
        h = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=path, size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=h)


@lru_cache(maxsize=8)
def events(vehicle_path, eda_path):
    v, e = table(vehicle_path), table(eda_path)
    t = v.time_dn.to_numpy(float)
    r = v.aircraft_roll_deg.to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(r)
    t, r = t[ok], r[ok]
    s = (t-t[0])*86400
    rate = abs(np.gradient(r, s))
    peaks = find_peaks(rate, height=np.percentile(rate, 85),
                       distance=max(1, int(15/np.median(np.diff(s)))))[0]
    et = e.time_dn.to_numpy(float)
    resistance = e.eda_hand_l_kOhms.to_numpy(float)
    g = np.full(len(e), np.nan)
    ok = np.isfinite(resistance) & (resistance > 0)
    g[ok] = 1000/resistance[ok]
    found = []
    for peak in peaks:
        event = t[peak]
        pre = np.isfinite(g) & (et >= event-2/86400) & (et < event)
        post = np.isfinite(g) & (et >= event) & (et < event+8/86400)
        if event+8/86400 > np.nanmax(et) or event-12/86400 < max(t[0], et[0]) or not pre.any() or not post.any():
            continue
        found.append((float(event), float(np.median(g[post])-np.median(g[pre])), int(pre.sum()), int(post.sum())))
    return found


def audit(row):
    sid = row['sample_id']
    paths = row['source_paths']
    clare = row['domain'] == 'clare'
    scale, duration = (1., 10.) if clare else (86400., 12. if '::event' in sid else 30.)
    extra = {}
    if clare:
        exp, win = map(int, re.search(r'experiment(\d+)_window(\d+)', sid).groups())
        label_path = str(Path(paths[0]).parents[2]/'Labels'/f"{row['subject_id']}.csv")
        label = float(table(label_path).iloc[win][f'level_{exp}'])
        start = float(table(paths[0]).Timestamp.iloc[0]) + win*10.
        targets = dict(workload_regression=label, workload_classification=int(label >= 7))
        extra['label_file'] = file_info(label_path)
        columns = [('TP9', 'AF7', 'AF8', 'TP10'), ('GSR Conductance CAL',), ('ECG LL-RA CAL',)]
        streams = [('physiology', 0), ('vehicle', 0), ('vehicle', 1)]
        hash_parts = [sid, start, duration, label, 'native_window_contract_v4']
    else:
        columns = [('ppg_finger_mV', 'eda_hand_l_kOhms'), ('respiration_trace_mV',),
                   ('ecg_projection_ll_ra_mV',), ('aircraft_indicated_airspeed_kias', 'aircraft_pitch_deg',
                   'aircraft_roll_deg', 'aircraft_agl_altitude_m', 'aircraft_climb_rate_mps',
                   'aircraft_ils_deflection_gs', 'aircraft_ils_deflection_h', 'aircraft_velocity_u_mps')]
        streams = [('physiology', 0), ('physiology', 2), ('physiology', 3), ('vehicle', 0)]
        if '::event' in sid:
            index = int(sid.rsplit('event', 1)[1])
            event, label, pre_n, post_n = events(paths[-1], paths[0])[index]
            start = event-12/86400
            targets = dict(event_response=label)
            extra.update(event_native=event, pre_count=pre_n, post_count=post_n)
            hash_parts = [sid, start, duration, label, 'native_window_contract_v4']
        else:
            index = int(sid.rsplit('window', 1)[1])
            origin = max(float(table(p).time_dn.min()) for p in paths)
            start = origin+(60+index*30)/86400
            targets = dict(difficulty=int(re.search(r'level-(\d+)', sid)[1])-1)
            hash_parts = [sid, start, duration, 'native_window_contract_v4']
    fields = []
    for path, cols, (stream, base) in zip(paths, columns, streams, strict=True):
        frame = table(path)
        t = frame['Timestamp' if clare else 'time_dn'].to_numpy(float)
        rel = (t-start)*scale
        in_window = (rel >= 0) & (rel < duration)
        selected_t = rel[in_window]
        for j, col in enumerate(cols):
            value = frame[col].to_numpy(float)[in_window]
            if col == 'eda_hand_l_kOhms':
                ok = np.isfinite(value) & (value > 0)
                value = np.divide(1000., value, out=np.full_like(value, np.nan), where=ok)
            ok = np.isfinite(value)
            # Cached sparse union keeps the last finite observation at a duplicate timestamp.
            series = pd.Series(value[ok], index=selected_t[ok])
            series = series[~series.index.duplicated(keep='last')].sort_index()
            times, values = series.index.to_numpy(float), series.to_numpy(np.float32)
            fields.append(dict(stream=stream, index=base+j, source_column=col, count=len(values),
                time_sha256=digest_array(times), value_sha256=digest_array(values),
                first_s=float(times[0]) if len(times) else None, last_s=float(times[-1]) if len(times) else None,
                raw_missing=int((~ok).sum()), duplicate_times=int(len(selected_t)-len(np.unique(selected_t))),
                time_reversals=int(np.sum(np.diff(t) < 0)), raw_start=float(np.nanmin(t)), raw_end=float(np.nanmax(t))))
    h = hashlib.sha256()
    sources = [file_info(p) for p in paths]
    for source in sources:
        for part in (source['path'], source['size'], source['mtime_ns'], source['sha256']):
            h.update(str(part).encode())
    for part in hash_parts:
        h.update(repr(part).encode())
    return dict(sample_id=sid, domain=row['domain'], targets=targets, start_native=start, duration_s=duration,
                time_tolerance_s=float(2*np.spacing(start)*scale), fields=fields, sources=sources,
                rebuilt_source_sample_hash=h.hexdigest(), **extra)


if __name__ == '__main__':
    request = json.load(sys.stdin)
    result = [audit(row) for row in request]
    print(json.dumps(result, allow_nan=False))
