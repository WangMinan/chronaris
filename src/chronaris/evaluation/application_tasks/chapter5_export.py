"""Export frozen inner-validation detail and costs for independent local plotting."""
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
from statistics import mean, pstdev

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write


def csv_file(path, rows):
    rows=list(rows)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in row.items()} for row in rows)


def export_main_materials(main_root, output_root, audit_path):
    root,out=Path(main_root).resolve(),Path(output_root).resolve()
    audit=read(audit_path)
    if audit['status']!='completed' or audit['model_units']!=279 or audit['routes']!=477:
        raise ValueError('main material export requires complete asset verification')
    verify({str(root/name):audit['source_files'][str(root/name)] for name in ('main_models.json','formal_plan.json','pipeline_state.json')})
    state,models=read(root/'pipeline_state.json'),read(root/'main_models.json')
    existing=out/'evidence.json'
    binding=dict(main_models_sha256=sha(root/'main_models.json'),audit_sha256=sha(audit_path))
    if existing.exists() and read(existing)['binding']!=binding:
        raise ValueError('material sources changed; export a separate version')
    metrics,groups,costs,curves,training,attempts=[],[],[],[],[],[]
    sources={str(Path(audit_path).resolve()):sha(audit_path)}
    roots={root};prediction_count=0;consumer_count=0
    for index in range(279):
        receipt=state['completed'][f'stage5_formal_unit__{index}']
        verify({receipt['path']:models['unit_receipts'][receipt['path']]});sources[receipt['path']]=receipt['sha256']
        unit=read(receipt['path']);spec=unit['unit']
        identity={k:spec[k] for k in ('domain','method','fold_index','seed')}|{'unit_index':index}
        costs.append(identity|dict(current_seconds=unit['total_seconds'],historical_seconds=unit.get('historical_total_seconds',0.),
            accounted_seconds=unit['total_seconds']+unit.get('historical_total_seconds',0.),
            cost_scope='recorded_unit_cost_excludes_unattributed_failed_attempts',phase_seconds=unit.get('phase_seconds',{})))
        parent=unit.get('reuse',{}).get('parent_receipt')
        while parent:
            verify({parent['path']:parent['sha256']});sources[parent['path']]=parent['sha256']
            roots.add(Path(parent['path']).parents[1])
            parent=read(parent['path']).get('reuse',{}).get('parent_receipt')
        for row in unit['results']:
            key=identity|{'route':row['route'],'data_role':'inner_validation'}
            evaluation=row.get('evaluation',row)
            tr=row.get('training') or {}
            primary=tr.get('primary',tr)
            history=primary.get('history',primary.get('epoch_rows',[]))
            for position,item in enumerate(history):
                curves.append(key|{'row_index':position}|{k:v for k,v in item.items() if k not in ('sample_ids','batch_sample_ids')})
            contract_path=Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
            verify({str(contract_path):evaluation['contract_file_sha256']})
            sources[str(contract_path)]=evaluation['contract_file_sha256']
            record=read(contract_path);declaration=record['declaration']
            training.append(key|dict(checkpoint_sha256=declaration['checkpoint_sha256'],
                checkpoint_path=declaration['checkpoint_path'],feature_dim=declaration['feature_dim'],
                kind=declaration['kind'],supervision_stratum=evaluation['supervision_stratum'],
                optimizer_updates=primary.get('updates',primary.get('optimizer_updates',0)),
                best_update=primary.get('best_update'),history_rows=len(history),external_pretraining=declaration['external_pretraining']))
            for family,component in evaluation['consumers']['components'].items():
                consumer_count+=1
                path=component['result_path'];verify({path:component['result_sha256']});sources[path]=component['result_sha256']
                result=read(path)['evaluations']['validation'];meta=key|{'consumer':family}
                metrics.extend(meta|item for item in result['task_summary'])
                groups.extend(meta|item for item in result['group_metrics'])
                destination=out/'data/predictions'/f"{index:03d}_{row['route']}_{family}.csv.gz"
                destination.parent.mkdir(parents=True,exist_ok=True)
                predictions=result['prediction_rows'];prediction_count+=len(predictions)
                with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
                    with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as handle:
                        writer=csv.DictWriter(handle,fieldnames=list(meta)+list(predictions[0]))
                        writer.writeheader()
                        writer.writerows(meta|{k:json.dumps(v) if isinstance(v,list) else v for k,v in p.items()} for p in predictions)
                if result['vehicle_prediction_rows']:
                    csv_file(out/'data/vehicle_predictions'/f"{index:03d}_{row['route']}_{family}.csv",[meta|p for p in result['vehicle_prediction_rows']])
    for origin in sorted(roots):
        for path in sorted((origin/'attempt_costs').glob('*.json')):
            item=read(path);sources[str(path)]=sha(path)
            attempts.append(dict(origin_root=str(origin),attempt_file=path.name,**item,
                accounting='raw_attempt_record_do_not_add_successful_attempts_to_unit_cost'))
    by=defaultdict(list)
    keys=('domain','method','route','consumer','task','metric','error_units')
    for row in metrics:by[tuple(row[k] for k in keys)].append(row['value'])
    summaries=[]
    for key,values in sorted(by.items()):
        valid=[v for v in values if v is not None]
        summaries.append(dict(zip(keys,key))|dict(settings=len(values),available_settings=len(valid),
            mean=mean(valid) if valid else None,std=pstdev(valid) if valid else None,
            minimum=min(valid) if valid else None,maximum=max(valid) if valid else None,
            error_type='descriptive_population_std_across_fold_seed_settings',data_role='inner_validation'))
    for name,rows in [('setting_metrics',metrics),('group_metrics',groups),('cost_units',costs),
                      ('training_curves',curves),('training',training),('attempt_costs',attempts),('summary',summaries)]:
        csv_file(out/'data'/f'{name}.csv',rows)
    cost_groups=defaultdict(list)
    for row in costs:cost_groups[(row['domain'],row['method'])].append(row['accounted_seconds']/3600)
    csv_file(out/'data/cost_summary.csv',[dict(domain=k[0],method=k[1],units=len(v),mean_hours=mean(v),
        total_hours=sum(v),std_hours=pstdev(v)) for k,v in sorted(cost_groups.items())])
    return write(existing,dict(status='completed',binding=binding,source_files=sources,
        data_files={str(p.relative_to(out)):sha(p) for p in sorted((out/'data').rglob('*')) if p.is_file()},
        model_units=279,routes=477,consumer_groups=consumer_count,
        setting_metric_rows=len(metrics),group_metric_rows=len(groups),prediction_rows=prediction_count,
        training_curve_rows=len(curves),confirmation_opened=False,data_role='inner_validation',
        statistics='fixed fold/seed setting means and population standard deviations; descriptive only',
        failed_attempts_included_separately=True))
