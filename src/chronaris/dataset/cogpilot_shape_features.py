"""Causal native summaries before amplitude scaling; no waveform clipping."""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def shape_native_features(times, values, start, queries, band):
    keep = (times >= start) & (times < queries[-1])
    t, x = times[keep]-start, values[keep]
    q = queries-start
    out = np.zeros((len(q), 4), np.float64)
    mask = np.zeros_like(out, dtype=bool); age = np.zeros_like(out, dtype=np.float32)
    if len(t) < 50 or np.any(np.diff(t) <= 0) or not np.isfinite(x).all():
        raise ValueError('native source time/value failed')
    initial = x[t < 2]
    if len(initial) < 50:
        raise ValueError('initial support')
    first = t[t < 1]; fs = 1/np.median(np.diff(first))
    if fs < 100 or fs/2 <= band[1]:
        raise ValueError('native sampling rate')
    center = np.median(initial)
    scale = max(1e-6, np.quantile(initial, .75)-np.quantile(initial, .25))
    z = x-center
    sos = butter(2, band, btype='bandpass', output='sos', fs=fs)
    filtered, _ = sosfilt(sos, z, zi=sosfilt_zi(sos)*z[0])
    for j, end in enumerate(q):
        if end < 2:
            continue
        lo, hi = np.searchsorted(t, [end-2, end], side='left')
        ct, cx, cf = t[lo:hi], z[lo:hi], filtered[lo:hi]
        if len(ct) < 50 or max(ct[0]-(end-2), end-ct[-1], np.diff(ct).max()) > .25:
            continue
        out[j] = (cx.mean(), cx.std(), np.sqrt(np.mean(cf**2)),
                  np.count_nonzero(np.diff(np.signbit(cf)))/(ct[-1]-ct[0]))
        mask[j] = True; age[j] = end-ct[-1]
    return out, mask, age, dict(center=float(center), scale=float(scale), fs=float(fs))


def shape_scale_features(values, masks, scales, floors):
    effective = np.maximum(np.maximum(scales, np.asarray(floors)), 1e-6)
    result = values.copy()
    for channel in range(4):
        k = channel*4
        result[..., k] = np.arcsinh(values[..., k]/effective[:, channel, None])
        result[..., k+1:k+3] = np.log1p(values[..., k+1:k+3]/effective[:, channel, None, None])
    result = np.where(masks, result, 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('shape input not finite')
    return result


def shape_scale_floors(scales, fit_indices):
    floors = []
    for channel in range(4):
        vals = scales[fit_indices, channel]
        vals = vals[vals > 1e-6]
        if not len(vals):
            raise ValueError('no nondegenerate training scale')
        floors.append(float(np.quantile(vals, .1)))
    return floors
