"""Read-only, full-validation device replay; retain strict failures and rankings."""
from collections import defaultdict
from pathlib import Path
import time

import joblib
import numpy as np
import torch

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write
from chronaris.evaluation.application_tasks.chapter5_native import unit_record, _inputs, _encode, compare_predictions
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_grouped_consumers import evaluate_native_consumers
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def prediction_changes(expected, actual):
    key = lambda r: (r['sample_id'], r['task'], r['field'])
    left, right = ({key(r): r for r in rows} for rows in (expected, actual))
    if len(left) != len(expected) or len(right) != len(actual) or left.keys() != right.keys():
        raise ValueError('replay prediction coverage changed')
    flips = classifications = 0
    maxima = dict(probability=0., regression=0.)
    for k, a in left.items():
        b = right[k]
        for field in ('target_valid', 'target', 'group_id', 'no_observation', 'sample_weight'):
            if a[field] != b[field]:
                raise ValueError(f'replay target/context changed: {k}/{field}')
        for field in ('prediction', 'probability'):
            x, y = a[field], b[field]
            if x is None or y is None:
                if x != y:
                    raise ValueError('replay availability changed')
                continue
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('replay shape or finite-value check failed')
            if field == 'probability':
                maxima['probability'] = max(maxima['probability'], float(np.max(np.abs(x-y))))
            elif a['probability'] is None:
                maxima['regression'] = max(maxima['regression'], float(np.max(np.abs(x-y))))
        if a['probability'] is not None:
            classifications += 1
            flips += int(a['prediction'] != b['prediction'])
    try:
        strict = compare_predictions(expected, actual)
    except ValueError as error:
        strict = dict(passed=False, error=str(error), atol=1e-5, rtol=1e-5)
    return dict(rows=len(left), classification_rows=classifications, class_flips=flips,
                max_probability_difference=maxima['probability'],
                max_regression_difference=maxima['regression'], strict=strict)


def ranking_changes(rows):
    panels = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in row['metrics']:
            if metric['before'] is not None and metric['after'] is not None:
                panels[(row['domain'], row['route'], row['family'], metric['task'], metric['metric'])][row['method']].append(metric)
    results = []
    for key, methods in sorted(panels.items()):
        means = {m: {side: float(np.mean([x[side] for x in values])) for side in ('before', 'after')}
                 for m, values in methods.items()}
        pairs = []
        names = sorted(means)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                before, after = (np.sign(means[a][s]-means[b][s]) for s in ('before', 'after'))
                if before != after:
                    pairs.append(dict(method_a=a, method_b=b, before_order=int(before), after_order=int(after)))
        results.append(dict(domain=key[0], route=key[1], family=key[2], task=key[3], metric=key[4],
                            means=means, changed_pairs=pairs, aggregation='equal_fold_seed_means_of_subject_means'))
    return results


def replay_unit(main_root, index, root):
    unit = unit_record(main_root, index)
    spec = unit['unit']
    binding = dict(main_models_sha256=sha(Path(main_root)/'main_models.json'), index=index,
                   source_code_sha256=v4_workflow_source_sha256(), role='validation', sample_limit=0,
                   device=torch.cuda.get_device_name())
    root = Path(root)
    result_path = root/'result.json'
    if result_path.exists():
        result = read(result_path)
        if result['binding'] != binding:
            raise ValueError('replay binding changed')
        verify(result['files'])
        return result
    files, rows = {}, []
    started = time.perf_counter()
    for row in unit['results']:
        evaluation = row.get('evaluation', row)
        if 'minirocket' not in evaluation['consumers']['components']:
            continue
        contract_path = Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
        verify({str(contract_path): evaluation['contract_file_sha256']})
        record = read(contract_path)
        verify(record['declaration']['evidence_files'])
        _, fold, targets, context, raw = _inputs(spec, record['contract'], 'validation', 0)
        output = _encode(spec, row, record['declaration'], record['contract'], raw, fold,
                         root/row['route']/'representation', 'validation')
        for family, component in evaluation['consumers']['components'].items():
            saved = joblib.load(component['model_path'])
            if (saved['protocol_sha256'] != evaluation['consumers']['protocol_sha256']
                or tuple(saved['consumer']['train_sample_ids']) != fold.train_sample_ids):
                raise ValueError('frozen consumer protocol or training role changed')
            before = read(component['result_path'])['evaluations']['validation']
            after = evaluate_native_consumers(saved['consumer'], output=output, targets=targets, context=context)
            delta = prediction_changes(before['prediction_rows'], after['prediction_rows'])
            old = {(m['task'], m['metric']): m for m in before['task_summary']}
            if old.keys() != {(m['task'], m['metric']) for m in after['task_summary']}:
                raise ValueError('replay metric coverage changed')
            metrics = [dict(task=m['task'], metric=m['metric'], before=old[m['task'], m['metric']]['value'],
                            after=m['value']) for m in after['task_summary']]
            target = root/row['route']/f'{family}.json'
            write(target, dict(before=before, after=after))
            files[str(target.resolve())] = sha(target)
            rows.append(dict(index=index, domain=spec['domain'], method=spec['method'], seed=spec['seed'],
                             fold_index=spec['fold_index'], route=row['route'], family=family,
                             samples=len(raw.sample_ids), metrics=metrics, **delta))
    return write(result_path, dict(status='completed', binding=binding, rows=rows, files=files,
                                   seconds=time.perf_counter()-started, confirmation_opened=False, downstream_refitted=False))


def run_replay(main_root, output_root):
    root = Path(output_root)
    formal = read(Path(main_root)/'formal_plan.json')
    indices = [i for i, u in enumerate(formal['units']) if u['domain'] in ('clare', 'cogpilot')
               and u['method'] in ('chronaris', 'physiology_only', 'vehicle_only', 'mult', 'contiformer', 'naive_time_sync')]
    torch.set_num_threads(1)
    rows = []
    with development_gpu_lock() as acquired:
        if not acquired:
            raise RuntimeError('GPU lock is held by another experiment')
        for index in indices:
            result = replay_unit(main_root, index, root/'units'/str(index))
            rows.extend(result['rows'])
            write(root/'progress.json', dict(completed_units=len({r['index'] for r in rows}),
                  total_units=len(indices), last_index=index, strict_failures=sum(not r['strict']['passed'] for r in rows),
                  class_flips=sum(r['class_flips'] for r in rows)))
            print(f'replayed unit {index}: {len(rows)} consumer routes', flush=True)
    return write(root/'summary.json', dict(status='completed', expected_units=indices, rows=rows,
        rankings=ranking_changes(rows), confirmation_opened=False, downstream_refitted=False,
        strict_passed=all(r['strict']['passed'] for r in rows), class_flips=sum(r['class_flips'] for r in rows)))
