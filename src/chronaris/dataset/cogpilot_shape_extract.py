"""Read-only native extraction with exact frozen sample roles and source checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import STREAM_COLUMNS, file_digest
from chronaris.dataset.cogpilot_history_features import QUERY_SECONDS, PHYSIOLOGY_BANDS
from chronaris.dataset.cogpilot_shape_features import shape_native_features


def extract_shape_native(request):
    sources = request['sources']
    assert all(file_digest(p) == h for p, h in sources.items())
    groups = {}
    for i, row in enumerate(request['windows']):
        groups.setdefault(row['record_id'], []).append((i, row))
    n = len(request['windows'])
    values = np.zeros((n, 96, 16)); masks = np.zeros_like(values, bool)
    ages = np.zeros_like(values, np.float32); scales = np.zeros((n, 4))
    checks = []; tested_subjects = set()
    for record, rows in sorted(groups.items()):
        field = 0
        for token, columns in list(STREAM_COLUMNS.items())[:3]:
            path = next(Path(record).glob(f'*stream-{token}*_dat.csv'))
            frame = pd.read_csv(path, usecols=['time_dn', *columns])
            times = (frame.time_dn.to_numpy(float)-rows[0][1]['source_origin_native'])*86400
            for column in columns:
                x = frame[column].to_numpy(float)
                if column == 'eda_hand_l_kOhms':
                    x = np.divide(1000., x, out=np.full_like(x, np.nan), where=np.isfinite(x)&(x > 0))
                for i, row in rows:
                    start = row['cutoff_s']-30; query = start+QUERY_SECONDS
                    out = shape_native_features(times, x, start, query, PHYSIOLOGY_BANDS[field])
                    values[i, :, field*4:field*4+4] = out[0]
                    masks[i, :, field*4:field*4+4] = out[1]
                    ages[i, :, field*4:field*4+4] = out[2]
                    scales[i, field] = out[3]['scale']
                    expected = row['physiology_quality'][field]
                    assert out[3]['scale'] == expected['scale'] and out[3]['center'] == expected['center']
                    if row['subject'] not in tested_subjects and i == rows[0][0]:
                        past = QUERY_SECONDS <= 15
                        changed = x.copy(); changed[times >= start+15] = 123456.
                        short = times < start+15
                        for variant, tx, xx in [('changed', times, changed), ('deleted', times[short], x[short])]:
                            actual = shape_native_features(tx, xx, start, query, PHYSIOLOGY_BANDS[field])
                            np.testing.assert_array_equal(actual[1][past], out[1][past])
                            np.testing.assert_allclose(actual[0][past], out[0][past], atol=1e-5, rtol=1e-5)
                            checks.append(dict(sample_id=row['sample_id'], field=field, variant=variant,
                                               max_delta=float(np.abs(actual[0][past]-out[0][past]).max())))
                field += 1
        tested_subjects.add(rows[0][1]['subject'])
    assert len(checks) == 56
    assert all(file_digest(p) == h for p, h in sources.items())
    return dict(values=values, masks=masks, ages=ages, scales=scales,
                metadata=np.array(json.dumps(dict(sample_ids=[r['sample_id'] for r in request['windows']],
                                                   sources=sources, prefix_checks=checks))))
