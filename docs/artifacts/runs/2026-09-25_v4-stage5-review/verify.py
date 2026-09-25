"""Verify the portable review bundle; no training files, CUDA or third-party imports."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def main():
    root = Path(__file__).resolve().parent
    read = lambda name: json.loads((root / name).read_text())
    for name, digest in read('bundle.sha256.json').items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    state, data = read('pipeline_state_snapshot.json'), read('stage5_results.json')
    assert state['status'] == 'failed' and state['current_stage'] == 'stage5_unit__12'
    assert not state['children'] and len(state['failures']) == 1
    units = data['units']
    assert [u['index'] for u in units] == list(range(12))
    assert sum(u['unit']['inherited'] for u in units) == 4
    assert all(u['unit']['fold_index'] == 0 for u in units)
    assert {u['unit']['seed'] for u in units} == {17, 29, 43}
    assert len(data['rows']) == 48 and len(data['consumer_details']) == 48
    assert data['completed_units'] == 12 and data['new_completed_units'] == 8
    assert data['total_units'] == 36 and not data['confirmation_opened']
    primary = []
    for consumer in data['consumer_details']:
        for summary in consumer['task_summary']:
            groups = defaultdict(list)
            for row in consumer['group_metrics']:
                if row['task'] == summary['task'] and row['status'] == 'completed':
                    groups[row['group_id']].append(row[summary['metric']])
            assert len(groups) == summary['group_count']
            assert math.isclose(mean(mean(v) for v in groups.values()), summary['value'], abs_tol=1e-12)
            if consumer['family'] == 'linear':
                primary.append(dict(unit_index=consumer['unit_index'], **consumer['unit'],
                                    route=consumer['route'], **summary))
    assert primary == data['rows']
    grouped = defaultdict(list)
    for row in data['rows']:
        grouped[(row['domain'], row['method'], row['route'], row['task'])].append(row['value'])
    assert len(grouped) == len(data['summary']) == 16
    for summary in data['summary']:
        values = grouped[tuple(summary[k] for k in ('domain', 'method', 'route', 'task'))]
        assert len(values) == summary['count'] == 3
        for name, fn in [('mean', mean), ('std', pstdev), ('minimum', min), ('maximum', max)]:
            assert math.isclose(fn(values), summary[name], abs_tol=1e-12)
    variance, failed = read('variance_diagnosis.json'), read('failed_unit.json')
    assert variance['shape'] == [578, 64, 96]
    assert variance['valid_cases'] == 573 and variance['invalid_cases'] == 5
    assert variance['threshold'] == 1e-7
    counts = [0] * 64
    for case in variance['failing_cases']:
        bad = [i for i, v in enumerate(case['temporal_std']) if not v > variance['threshold']]
        assert bad == case['bad_channels'] and 0 < len(bad) < 64
        for i in bad:
            counts[i] += 1
    assert len(variance['failing_cases']) == variance['cases_failing_any_channel'] == 36
    assert counts == variance['channel_failed_case_counts']
    assert all(c > 0 for c in counts) and variance['retained_channels'] == 0
    assert variance['cases_failing_all_channels'] == 0 and variance['pooled_nonconstant_dimensions'] == 64
    assert failed['unit']['fold_index'] == 1 and failed['unit']['seed'] == 17
    assert failed['not_counted_in_completed_units'] and not failed['task_guided_started']
    assert all(c['optimizer_updates'] == c['best_update'] == 300 for c in failed['pretraining_checkpoints'])
    assert 'MiniRocket train-only variance filter removed every channel' in (root/'failure_traceback.txt').read_text()
    source = read('source_inventory.json')
    assert source['verified_receipts'] == 24 and source['verified_completed_units'] == 12
    assert source['source_files_unchanged'] and not source['experiments_restarted']
    print('PASS: bundle hashes; 12 units (4 inherited + 8 new); 48 primary rows; '
          '48 grouped consumers; 16 seed summaries; zero-channel filter; failed unit excluded. '
          'Original training and source tensors are not independently reproduced.')


if __name__ == '__main__':
    main()
