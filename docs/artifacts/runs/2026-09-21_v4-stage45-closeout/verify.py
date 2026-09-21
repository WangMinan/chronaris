"""Recompute the archived Stage 4.5 decision using only Python's standard library."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def task_rows(result):
    return {(row['route'], item['task'], item['metric']): item['value']
            for row in result['results'] if 'training' in row
            for item in row['consumers']['components']['linear']['task_summary']['validation']}


def verify(root, local_sources=False):
    evidence = root/'evidence.json'
    assert sha(evidence) == (root/'evidence.sha256').read_text().split()[0], 'bundle hash changed'
    b = json.loads(evidence.read_text())
    state, report, plan = b['state'], b['report'], b['plan']
    assert state['status'] == 'stage45_completed' and not state['failures'] and not state['children']
    assert report['status'] == 'completed' and not report['configuration_frozen'] and not report['confirmation_opened']
    assert len(state['completed']) == b['verified_pipeline_receipts'] == 76
    units = {plan['screen_units'].index(row['unit']): row for row in report['screen_results']}
    assert set(units) == set(range(18)) and len(report['screen_results']) == 18
    assert all(row['status'] == 'completed' and not row.get('skipped') for row in units.values())
    assert len(b['references']) == 6 and len(b['diagnostics']) == 10
    assert len(report['review_results']) == 22 and all(row['skipped'] for row in report['review_results'])
    for key, source in b['sources'].items():
        parts = key.split('/')
        if parts[0] == 'screen':
            value = units[int(parts[1])]
        elif parts[0] == 'references':
            value = b['references']['/'.join(parts[1:])]
        elif parts[:2] == ['execution', 'qualifications']:
            value = b['execution']['qualifications']['/'.join(parts[2:])]
        else:
            value = b
            for part in parts:
                value = value[part]
        assert canonical(value) == source['content_sha256'], key
        if local_sources:
            assert sha(source['path']) == source['sha256'], source['path']
    reconstructed = {}
    for recipe, comparison in report['selection']['comparisons'].items():
        rows = [row for row in units.values() if row['unit']['method'] == 'chronaris'
                and row['unit']['recipe'] == recipe]
        assert len(rows) == 2 and {r['domain'] for r in rows} == {'clare', 'cogpilot'}
        improved, safe, changes = set(), True, []
        expected = {(x['domain'], x['route'], x['task'], x['metric']): x for x in comparison['rows']}
        actual_keys = set()
        for row in rows:
            reference = task_rows(b['references'][row['domain']+'/chronaris'])
            candidate = task_rows(row)
            assert reference.keys() == candidate.keys()
            for (route, task, metric), value in candidate.items():
                ref = reference[route, task, metric]
                delta = value-ref if metric == 'macro_f1' else (ref-value)/ref
                gain, floor = (.02, -.01) if metric == 'macro_f1' else (.05, -.05)
                safe &= delta >= floor
                if delta >= gain:
                    improved.add((row['domain'], route))
                key = row['domain'], route, task, metric
                actual_keys.add(key)
                recorded = expected[key]
                for field, number in [('value', value), ('reference', ref), ('improvement', delta)]:
                    assert math.isclose(recorded[field], number, abs_tol=1e-12, rel_tol=1e-12), (recipe, key, field)
                assert recorded['meaningful'] == (delta >= gain)
                changes.append(delta)
        assert actual_keys == expected.keys() and len(actual_keys) == 8
        assert comparison['safe'] == safe
        assert comparison['meaningful_domain_routes'] == len(improved)
        assert comparison['eligible'] == (safe and len(improved) >= 2)
        assert math.isclose(comparison['score'], sum(changes)/len(changes), abs_tol=1e-12)
        reconstructed[recipe] = comparison
    ranked = sorted((name for name, row in reconstructed.items() if row['eligible']),
                    key=lambda name: (-reconstructed[name]['score'], name))[:2]
    assert report['selection']['finalists'] == ranked == []
    # Re-aggregate per-field/per-group results, independent of the stored task summary.
    assert len(b['consumer_details']) == 96
    for key, detail in b['consumer_details'].items():
        validation = detail['validation']
        for task in validation['task_summary']:
            groups = defaultdict(list)
            for row in validation['group_metrics']:
                if row['task'] == task['task'] and row.get(task['metric']) is not None:
                    groups[row['group_id']].append(row[task['metric']])
            means = [sum(values)/len(values) for values in groups.values()]
            assert len(means) == task['group_count'], key
            assert math.isclose(sum(means)/len(means), task['value'], abs_tol=1e-10), key
        if local_sources:
            assert sha(detail['source_path']) == detail['source_sha256']
    ex = b['execution']
    assert ex['cuda_receipt']['status'] == 'completed' and ex['cuda_receipt']['exit_code'] == 0
    counts = ex['cuda_test_counts']
    assert counts['total']-counts['skipped'] == 697 and counts['skipped'] == 22 and counts['failed'] == 0
    for key in ['full_validation', 'actual_resume', 'first_joint', 'later_joint', 'extended_pretraining']:
        assert ex[key]['passed'], key
    assert not ex['first_joint_failed']['passed'] and not ex['old_output_failed']['passed']
    assert len(ex['qualifications']) == 6
    for check in ex['qualifications'].values():
        assert check['passed'] and check['independent_resume']['passed'] and check['consumer_impact']['passed']
    assert all(row['joint_micro_rows'] == 800 and any(row['max_gradient_groups'].values())
               for row in b['guided_gradient_audit'].values())
    print('PASS: 18 units, 6 references, 96 grouped consumer results, 5 rejected candidates, 22 skipped slots; execution evidence consistent.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-sources', action='store_true', help='Also hash original files on the research server')
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.local_sources)
