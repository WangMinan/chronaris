"""Export verified fixed-reference results as portable, editable plotting tables."""
import csv
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def export_stage5_closeout(run_root, output_root):
    root, out = Path(run_root).resolve(), Path(output_root).resolve()
    state, report = _read(root/'pipeline_state.json'), _read(root/'stage5_report.json')
    if state['status'] != 'stage5_completed' or state['failures'] or report['completed_units'] != 36:
        raise ValueError('closeout requires the completed 36-unit fixed matrix')
    if report['confirmation_opened'] or report['formal_evaluation_ready']:
        raise ValueError('fixed development results cannot be relabeled as confirmation')
    files = {str(root/n): _sha(root/n) for n in ('pipeline_state.json', 'stage5_report.json', 'stage5_plan.json')}
    for receipt in state['completed'].values():
        files[receipt['path']] = receipt['sha256']
    metrics, groups, costs, training = [], [], [], []
    for index in range(36):
        path = root/'units'/f'{index}.json'; unit = _read(path); spec = unit['unit']
        files[str(path)] = _sha(path); files.update(unit['evidence_files'])
        if unit['status'] != 'completed' or len(unit['results']) != 2:
            raise ValueError('incomplete unit or representation routes')
        identity = dict(unit_index=index, domain=spec['domain'], method=spec['method'],
            fold_index=spec['fold_index'], seed=spec['seed'])
        costs.append(identity | dict(current_seconds=unit['total_seconds'],
            historical_seconds=unit.get('historical_total_seconds', 0.),
            cost_scope='consumer_revalidation' if unit.get('reused_training') else
                'partial_pretraining_inherited' if unit.get('preserved_pretraining_updates') else 'full_training_unit'))
        for row in unit['results']:
            key = identity | {'route': row['route']}
            training.append(key | dict(selected_checkpoint=row['training']['best_checkpoint_path'],
                completed_updates=row['training']['optimizer_updates'],
                new_updates=0 if unit.get('reused_training') else row['training'].get('new_optimizer_updates', row['training']['optimizer_updates'])))
            if set(row['consumers']['components']) != {'linear', 'minirocket'}:
                raise ValueError('primary or supplementary consumer missing')
            for family, component in row['consumers']['components'].items():
                for kind in ('model', 'result'):
                    files[component[kind+'_path']] = component[kind+'_sha256']
                evaluation = _read(component['result_path'])['evaluations']['validation']
                for item in evaluation['task_summary']:
                    metrics.append(key | dict(consumer=family, task=item['task'], metric=item['metric'],
                        value=item['value'], group_count=item['group_count']))
                for item in evaluation['group_metrics']:
                    metric = 'macro_f1' if 'macro_f1' in item else 'rmse'
                    groups.append(key | dict(consumer=family, task=item['task'], group_id=item['group_id'],
                        field=item['field'], status=item['status'], metric=metric, value=item.get(metric),
                        support=item['support'], no_observation_fraction=item.get('no_observation_fraction'),
                        p95_absolute_error=item.get('p95_absolute_error'),
                        top_five_squared_error_fraction=item.get('top_five_squared_error_fraction')))
    for path, digest in files.items():
        if _sha(path) != digest:
            raise ValueError(f'closeout source changed: {path}')
    if len(metrics) != 288 or len(training) != 72 or sum(x['new_updates'] for x in training) != 12900:
        raise ValueError('fixed matrix or inherited-update accounting changed')
    by_key = defaultdict(list)
    for row in metrics:
        by_key[tuple(row[k] for k in ('domain', 'method', 'route', 'consumer', 'task', 'metric'))].append(row['value'])
    summaries = []
    for key, values in sorted(by_key.items()):
        if len(values) != 9:
            raise ValueError('each fixed method/route requires three folds and three seeds')
        summaries.append(dict(zip(('domain', 'method', 'route', 'consumer', 'task', 'metric'), key)) |
            dict(settings=9, mean=mean(values), std_across_settings=pstdev(values), minimum=min(values), maximum=max(values)))
    for row in report['descriptive_summary']:
        exported = next(x for x in summaries if x['consumer']=='linear' and all(x[k]==row[k] for k in ('domain','method','route','task')))
        if any(abs(exported[k]-row[k]) > 1e-12 for k in ('mean','std_across_settings','minimum','maximum')):
            raise ValueError('portable table differs from the frozen primary report')
    out.mkdir(parents=True, exist_ok=True); data = out/'data'; data.mkdir(exist_ok=True)
    for name, rows in [('setting_metrics',metrics),('group_metrics',groups),('summary',summaries),('cost_units',costs),('training',training)]:
        _csv(data/f'{name}.csv', rows)
    for source, name in [(state, 'pipeline_state_snapshot.json'), (report, 'stage5_report.json')]:
        (out/name).write_text(json.dumps(source, ensure_ascii=False, indent=2)+'\n')
    provenance = dict(status='completed', run_root=str(root), completed_units=36, representation_routes=72,
        consumer_groups=144, metric_rows=len(metrics), group_metric_rows=len(groups), new_optimizer_updates=12900,
        revalidated_units=12, newly_completed_units=24,
        started_at=datetime.fromtimestamp(state['started_at_unix_s']).astimezone().isoformat(),
        completed_at=datetime.fromtimestamp(state['updated_at_unix_s']).astimezone().isoformat(),
        recovery_wall_hours=(state['updated_at_unix_s']-state['started_at_unix_s'])/3600,
        confirmation_opened=False, source_files=files,
        data_files={str(p.relative_to(out)): _sha(p) for p in sorted(data.glob('*.csv'))})
    (out/'evidence.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    return {k:v for k,v in provenance.items() if k not in ('source_files','data_files')}
