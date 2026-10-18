"""Read-only event-response reconstruction and past-only matched-time controls."""
import numpy as np
from scipy.signal import find_peaks


def response_at(times, conductance, event):
    """Use native Matlab dates so rounding matches the frozen label definition."""
    def interval(a, b):
        values = conductance[(times >= event + a / 86400) & (times < event + b / 86400)]
        return values[np.isfinite(values)]
    before, after = interval(-2, 0), interval(0, 8)
    if not len(before) or not len(after):
        return None
    first, second = interval(0, 4), interval(4, 8)
    return dict(response=float(np.median(after) - np.median(before)),
                baseline=float(np.median(before)), before_count=len(before), after_count=len(after),
                first_half=float(np.median(first)-np.median(before)) if len(first) else None,
                second_half=float(np.median(second)-np.median(before)) if len(second) else None)


def past_descriptor(vehicle_times, roll, speed, eda_times, conductance, event, origin):
    mask = (vehicle_times >= event-12/86400) & (vehicle_times < event)
    a, b = roll[mask], speed[mask]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    past = (eda_times >= event-12/86400) & (eda_times < event)
    if not len(a) or not len(b) or not np.any(past & np.isfinite(conductance)):
        return None
    return np.array([(event-origin)*86400, np.median(a), np.median(b),
                     np.sum(past & np.isfinite(conductance))/12], float)


def audit_record(vehicle, eda, registered):
    """Return every registered event, matched controls, and fixed timing perturbations."""
    vt = vehicle['time_dn'].to_numpy(float)
    roll = vehicle['aircraft_roll_deg'].to_numpy(float)
    speed = vehicle['aircraft_indicated_airspeed_kias'].to_numpy(float)
    valid = np.isfinite(vt) & np.isfinite(roll)
    t, r = vt[valid], roll[valid]
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError('event reconstruction requires strictly increasing roll times')
    seconds = (t-t[0])*86400
    rate = abs(np.gradient(r, seconds))
    peaks = find_peaks(rate, height=np.percentile(rate, 85),
                       distance=max(1, int(15/np.median(np.diff(seconds)))))[0]
    et = eda['time_dn'].to_numpy(float)
    resistance = eda['eda_hand_l_kOhms'].to_numpy(float)
    good = np.isfinite(resistance) & (resistance > 0)
    g = np.divide(1000., resistance, out=np.full_like(resistance, np.nan), where=good)
    if not np.isfinite(et).all():
        raise ValueError('invalid skin conductance timestamps')
    # Interval masks preserve the frozen label even when source rows run backwards; report reversals separately.
    original = []
    for j in peaks:
        event = t[j]
        result = response_at(et, g, event)
        if event+8/86400 <= np.max(et) and event-12/86400 >= max(t[0], et[0]) and result:
            original.append((event, result))
    if len(original) != len(registered):
        raise ValueError('raw event inventory differs from the frozen registry')
    origin = max(t[0], et[0]); end = min(t[-1], et[-1])
    candidates, descriptors = [], []
    for sec in np.arange(12., (end-origin)*86400-8., 2.):
        event = origin+sec/86400
        if len(peaks) and np.min(abs(t[peaks]-event))*86400 <= 10.:
            continue
        descriptor = past_descriptor(vt, roll, speed, et, g, event, origin)
        if descriptor is not None:
            candidates.append(float(event)); descriptors.append(descriptor)
    descriptors = np.asarray(descriptors)
    if len(candidates):
        scale = np.subtract(*np.percentile(descriptors, [75, 25], axis=0))
        scale = np.where(scale > 1e-9, scale, 1.)
    rows = []
    for index, (event, result) in enumerate(original):
        row = dict(registered[index], event_native=float(event), **result)
        row['event_start_error_s'] = abs(event-12/86400-row['start_native'])*86400
        row['target_matches_float32'] = float(np.float32(result['response'])) == row['cached_target']
        for delta in (-1., -.5, .5, 1.):
            shifted = response_at(et, g, event+delta/86400)
            row['shift_'+str(delta)] = shifted['response'] if shifted else None
        descriptor = past_descriptor(vt, roll, speed, et, g, event, origin)
        row.update(control_native=None, control_response=None, match_distance=None,
                   matched=False, candidate_count=len(candidates))
        if descriptor is not None and len(candidates):
            distance = np.mean(((descriptors-descriptor)/scale)**2, axis=1)
            j = int(np.argmin(distance)); control = response_at(et, g, candidates[j])
            row.update(control_native=candidates[j], match_distance=float(np.sqrt(distance[j])))
            if control:
                row.update(control_response=control['response'], matched=True)
        rows.append(row)
    return rows, dict(events=len(rows), candidates=len(candidates), raw_peaks=len(peaks),
                      eda_invalid=int((~good).sum()), eda_rows=len(et),
                      vehicle_rows=len(vt), origin_native=float(origin), end_native=float(end),
                      eda_time_reversals=int(np.sum(np.diff(et)<0)),
                      largest_reversal_s=float(max(0.,-np.min(np.diff(et))*86400)))
