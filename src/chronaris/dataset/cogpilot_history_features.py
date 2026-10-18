"""Causal native-rate physiology summaries and published performance geometry."""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


QUERY_SECONDS = np.arange(1, 97, dtype=float)*30/96
PHYSIOLOGY_BANDS = ((.5, 8.), (.05, 1.), (.05, 1.), (5., 40.))


def performance_row_identity(vehicle_times, performance_times):
    """Match the independently verified 15-significant-digit published time format."""
    canonical = np.array([float(format(t, '.15g')) for t in vehicle_times])
    if np.any(np.diff(canonical) <= 0):
        raise ValueError('published time format has ambiguous source rows')
    rows = np.searchsorted(canonical, performance_times)
    if np.any(rows >= len(canonical)) or not np.array_equal(canonical[rows], performance_times):
        raise ValueError('published time serialization does not match source')
    if len(np.unique(rows)) != len(rows):
        raise ValueError('performance rows repeat source identities')
    return rows


def geodetic_ned(latitude, longitude, height, origin):
    """WGS84 geodetic to local north/east/down, matching the source formula."""
    def ecef(lat, lon, h):
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
        radius = 6378137./np.sqrt(1-6.6943799901413165e-3*np.sin(lat)**2)
        return np.stack(((radius+h)*np.cos(lat)*np.cos(lon),
                         (radius+h)*np.cos(lat)*np.sin(lon),
                         (radius*(1-6.6943799901413165e-3)+h)*np.sin(lat)), axis=-1)
    lon0, lat0, h0 = origin
    delta = ecef(latitude, longitude, height)-ecef(lat0, lon0, h0)
    lat, lon = np.deg2rad(lat0), np.deg2rad(lon0)
    rotation = np.array([[-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                         [-np.sin(lon), np.cos(lon), 0.],
                         [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), -np.sin(lat)]])
    return delta @ rotation.T


def published_performance(vehicle):
    latitude = vehicle['aircraft_latitude_deg'].to_numpy(float)
    longitude = vehicle['aircraft_longitude_deg'].to_numpy(float)
    altitude = vehicle['aircraft_elevation_m'].to_numpy(float)
    a = geodetic_ned(latitude, longitude, altitude, (-71.2964, 42.4715, 37.4))
    slope = -3*(np.rad2deg(np.arcsin(np.clip(a[:, 2]/np.linalg.norm(a, axis=1), -1, 1)))+3)
    origin = (-71.26841, 42.46887, 37.4)
    b = geodetic_ned(latitude, longitude, altitude, origin)
    ref = geodetic_ned(42.4976, -71.5843, 37.4, origin)
    cosine = b[:, :2] @ ref[:2]/(np.linalg.norm(b[:, :2], axis=1)*np.linalg.norm(ref[:2]))
    localizer = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))*np.sign(vehicle['aircraft_ils_deflection_h'].to_numpy(float))
    passed = longitude > origin[0]
    slope[passed] = 3.; localizer[passed] = 3.
    speed = (vehicle['aircraft_indicated_airspeed_kias'].to_numpy(float)-115)/10
    components = np.round(np.clip(np.column_stack((slope, localizer, speed)), -3, 3), 4)
    return np.column_stack((components, np.abs(components).sum(1)))


def strict_history_queries(times, values, start, queries):
    out = np.zeros((len(queries), values.shape[1]), np.float32)
    mask = np.zeros_like(out, dtype=bool)
    age = np.zeros_like(out)
    for field in range(values.shape[1]):
        keep = (times >= start) & (times < queries[-1]) & np.isfinite(values[:, field])
        t, x = times[keep], values[keep, field]
        if len(t) < 2 or np.any(np.diff(t) <= 0):
            continue
        positions = np.searchsorted(t, queries, side='left')-1
        valid = positions >= 0
        idx = np.maximum(positions, 0)
        elapsed = queries-t[idx]
        valid &= (elapsed > 0) & (elapsed <= 2.)
        out[valid, field] = x[idx[valid]]
        age[valid, field] = elapsed[valid]
        mask[:, field] = valid
    return out, mask, age


def native_dynamic_features(times, values, start, queries, band):
    """Summarize preceding native samples; never decimate a raw waveform."""
    keep = (times >= start) & (times < queries[-1])
    t, x = times[keep]-start, values[keep]
    q = queries-start
    out = np.zeros((len(q), 4), np.float32); mask = np.zeros_like(out, dtype=bool)
    age = np.zeros_like(out)
    if len(t) < 50 or np.any(np.diff(t) <= 0):
        return out, mask, age, {'status': 'invalid_time'}
    initial = (t < 2.) & np.isfinite(x)
    first = t[t < 1.]
    if initial.sum() < 50 or len(first) < 2:
        return out, mask, age, {'status': 'initial_support'}
    fs = 1/np.median(np.diff(first))
    if fs < 100 or fs/2 <= band[1]:
        return out, mask, age, {'status': 'sampling_rate', 'fs': float(fs)}
    center = np.median(x[initial]); scale = max(1e-6, np.quantile(x[initial], .75)-np.quantile(x[initial], .25))
    normalized = (x-center)/scale
    finite = np.isfinite(normalized)
    z = np.clip(np.where(finite, normalized, 0.), -20, 20)
    sos = butter(2, band, btype='bandpass', output='sos', fs=fs)
    filtered, _ = sosfilt(sos, z, zi=sosfilt_zi(sos)*z[0])
    for j, end in enumerate(q):
        if end < 2.:
            continue
        lo, hi = np.searchsorted(t, [end-2, end], side='left')
        ct, cx, cf = t[lo:hi], z[lo:hi], filtered[lo:hi]
        if len(ct) < 50 or not finite[lo:hi].all():
            continue
        if max(ct[0]-(end-2), end-ct[-1], np.diff(ct).max()) > .25:
            continue
        out[j] = (cx.mean(), np.log1p(cx.std()), np.log1p(np.sqrt(np.mean(cf**2))),
                  np.count_nonzero(np.diff(np.signbit(cf)))/(ct[-1]-ct[0]))
        mask[j] = True; age[j] = end-ct[-1]
    return out, mask, age, dict(status='completed', fs=float(fs), center=float(center), scale=float(scale),
                               clipped_fraction=float(np.mean(np.abs(normalized[finite]) > 20)),
                               finite_fraction=float(finite.mean()))
