"""Source-checked input upgrade for the fixed prospective performance windows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from chronaris.dataset.cogpilot_performance import STREAM_COLUMNS, file_digest, interval_mean
from chronaris.dataset.cogpilot_history_features import (
    QUERY_SECONDS, PHYSIOLOGY_BANDS, published_performance, strict_history_queries, native_dynamic_features,
    performance_row_identity,
)


def extract_fair_residual(request):
    sources, records, metadata, checks = {}, [], [], []
    arrays = {k: [] for k in ('physiology', 'physiology_mask', 'physiology_age',
                              'vehicle', 'vehicle_mask', 'vehicle_age', 'anchor')}
    for record in request['records']:
        run = Path(record['record_id'])
        paths = [next(run.glob(f'*stream-{token}*_dat.csv')) for token in STREAM_COLUMNS]
        perf_path = next(run.glob('*stream-lslxp11_feat-perfmetric*.csv'))
        before = {str(p): file_digest(p) for p in [*paths, perf_path]}
        if any(h != request['source_hashes'][p] for p, h in before.items()):
            raise ValueError('original source identity changed')
        origin = request['windows'][str(run)][0]['source_origin_native']
        perf = pd.read_csv(perf_path)
        pt = (perf.time_dn.to_numpy(float)-origin)*86400
        perf_values = perf[['glideslope_error_deg', 'localizer_error_deg', 'airspeed_error_kts', 'total_error']].to_numpy(float)
        aircraft = pd.read_csv(paths[-1])
        vt = (aircraft.time_dn.to_numpy(float)-origin)*86400
        reconstructed = published_performance(aircraft)
        exact_vehicle = pd.read_csv(paths[-1], usecols=['time_dn'], float_precision='round_trip').time_dn.to_numpy(float)
        exact_performance = pd.read_csv(perf_path, usecols=['time_dn'], float_precision='round_trip').time_dn.to_numpy(float)
        indices = performance_row_identity(exact_vehicle, exact_performance)
        errors = np.max(np.abs(reconstructed[indices]-perf_values), axis=0)
        checks.append(dict(record_id=str(run), rows=len(pt), time_max_delta_s=float(np.max(np.abs(vt[indices]-pt))),
                           canonical_time_max_delta_s=0., component_max_errors=errors.tolist()))
        if np.any(errors[:3] > .0002+1e-10) or errors[3] > .0006+1e-10:
            raise ValueError(f'performance reconstruction failed {run}: {errors}')
        physiology = []
        for path, columns in zip(paths[:3], list(STREAM_COLUMNS.values())[:3]):
            frame = pd.read_csv(path, usecols=['time_dn', *columns])
            t = (frame.time_dn.to_numpy(float)-origin)*86400
            for column in columns:
                x = frame[column].to_numpy(float)
                if column == 'eda_hand_l_kOhms':
                    x = np.divide(1000., x, out=np.full_like(x, np.nan), where=np.isfinite(x)&(x > 0))
                physiology.append((t, x))
        accepted = 0
        for old in request['windows'][str(run)]:
            end = old['cutoff_s']; start = end-30; queries = start+QUERY_SECONDS
            row = dict(old); row['accepted'] = False; row['reason'] = ''
            try:
                target, _, _ = interval_mean(pt, perf_values[:, 3], end, end+5, include_end=True)
                persistence, _, _ = interval_mean(pt, perf_values[:, 3], end-5, end, include_end=False)
                if max(abs(target-old['target']), abs(persistence-old['persistence'])) > 1e-10:
                    raise RuntimeError('original targets or persistence changed')
                v, vm, va = strict_history_queries(vt, aircraft[list(STREAM_COLUMNS['lslxp11xpcac'])].to_numpy(float), start, queries)
                h, hm, ha = strict_history_queries(pt, perf_values, start, queries)
                v, vm, va = np.c_[v, h], np.c_[vm, hm], np.c_[va, ha]
                anchor = np.zeros(96, np.float64)
                for j, q in enumerate(queries):
                    try:
                        anchor[j] = interval_mean(pt, perf_values[:, 3], max(start, q-5), q, include_end=False)[0]
                    except ValueError:
                        pass
                if abs(anchor[-1]-persistence) > 1e-10:
                    raise RuntimeError('last causal anchor mismatch')
                dyn = [native_dynamic_features(t, x, start, queries, band)
                       for (t, x), band in zip(physiology, PHYSIOLOGY_BANDS)]
                p, pm, pa = [np.concatenate([a[k] for a in dyn], axis=1) for k in range(3)]
                row['physiology_quality'] = [a[3] for a in dyn]
                row['minimum_phys_coverage'] = float(pm.mean(0).min())
                row['minimum_vehicle_coverage'] = float(vm.mean(0).min())
                if min(row['minimum_phys_coverage'], row['minimum_vehicle_coverage']) < .9:
                    raise ValueError('dynamic_or_history_coverage')
                row['accepted'] = True
                for name, value in zip(arrays, (p, pm, pa, v, vm, va, anchor)):
                    arrays[name].append(value)
                accepted += 1
            except ValueError as exc:
                row['reason'] = str(exc)
            metadata.append(row)
        if any(file_digest(p) != h for p, h in before.items()):
            raise RuntimeError('source changed during extraction')
        sources.update(before)
        records.append(dict(record_id=str(run), windows=len(request['windows'][str(run)]), accepted=accepted))
    return {**{k: np.asarray(v) for k, v in arrays.items()},
            'metadata': np.array(json.dumps(dict(windows=metadata, records=records, sources=sources,
                                                 reconstruction=checks), allow_nan=False))}
