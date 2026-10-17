"""Export complete declared results, preserving identities, failures and statistical units."""
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
from statistics import mean, pstdev
import subprocess
import sys
import tempfile

import numpy as np
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,verify,write
from chronaris.evaluation.application_tasks.chapter5_export import csv_file
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import simulation_unit_root


def compressed_csv(path,rows):
    rows=list(rows)
    if not rows:raise ValueError(f'empty prediction export: {path}')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
        with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def aggregate(rows,keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row['value'])
    result=[]
    for key,values in sorted(groups.items()):
        valid=[float(v) for v in values if v is not None]
        if not all(np.isfinite(valid)):raise ValueError('nonfinite summary values')
        result.append(dict(zip(keys,key))|dict(mean=mean(valid) if valid else None,
            std=pstdev(valid) if valid else None,settings=len(values),available_settings=len(valid)))
    return result


def archive_csv(source,destination,role='held_out'):
    with np.load(source,allow_pickle=False) as archive:
        ids=archive[f'{role}_sample_ids']
        arrays={k[len(role)+1:]:archive[k] for k in archive.files if k.startswith(role+'_') and k!=f'{role}_sample_ids'}
        if any(len(v)!=len(ids) for v in arrays.values()):raise ValueError('prediction archive sample count mismatch')
        compressed_csv(destination,[dict(sample_id=str(s),**{k:v[i].tolist() for k,v in arrays.items()}) for i,s in enumerate(ids)])


def export(root,plan,models):
    root=Path(root);out=root/'materials';out.mkdir(exist_ok=True)
    for path in Path(plan['plot_root']).iterdir():
        if path.is_file():shutil.copy2(path,out/path.name)
    data=out/'data';data.mkdir(exist_ok=True)
    metrics,groups,inventory,costs,training=[],[],[],[],[]
    sources={};predictions=0
    def source(path):
        path=Path(path);sources[str(path)]=sha(path);return read(path)
    source(root/'delivery_plan.json');source(root/'delivery_models.json')
    if (root/'execution_revision.json').exists():
        write(data/'protocol/execution_revision.json',source(root/'execution_revision.json'))
    if (root/'diagnostics_revision.json').exists():
        write(data/'protocol/diagnostics_revision.json',source(root/'diagnostics_revision.json'))
    if (root/'simulation_revision.json').exists():
        write(data/'protocol/simulation_revision.json',source(root/'simulation_revision.json'))
    for spec in plan['native']:
        receipt=source(spec['receipt']);unit=spec['unit'];result=source(root/'native'/spec['key']/'result.json')
        verify(result['files'])
        if result['status']!='completed' or result['downstream_refitted'] or not result['confirmation_opened']:
            raise ValueError('native result is not frozen outer inference')
        variant=unit.get('recipe_name','historical_reference' if unit['method']=='chronaris' else 'reference')
        meta=dict(domain=unit['domain'],method=unit['method'],variant=variant,fold_index=unit['fold_index'],seed=17,
                  unit_key=spec['key'],role='held_out')
        inventory.append(meta|dict(receipt=spec['receipt'],counts=unit['counts'],representation_routes=unit['routes']))
        costs.append(meta|dict(seconds=receipt.get('total_seconds',0),historical_seconds=receipt.get('historical_total_seconds',0),
            scope='receipt_recorded_training_cost_lower_bound_excludes_unattributed_failures'))
        for row in result['results']:
            if row['samples']!=unit['counts']['held_out']:raise ValueError('truncated outer windows')
            original=next(r for r in receipt['results'] if r['route']==row['route'])
            evaluation=original.get('evaluation',original)
            contract=source(Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json')
            training.append(meta|dict(route=row['route'],declaration=contract['declaration'],contract=contract['contract']))
            costs.append(meta|dict(route=row['route'],seconds=row['seconds'],scope='frozen_export_and_downstream_inference'))
            reference=root/'native'/spec['key']/row['route']/'representation/reference_executor/receipt.json'
            if reference.exists():
                execution=source(reference);verify(execution['files'])
                write(data/'reference_execution'/f"{spec['key']}_{row['route']}.json",execution)
                costs[-1].update(encoder_device=execution['device'],includes_remote_transfer=True,
                    encoder_seconds=execution['cost']['seconds'],encoder_peak_cuda_bytes=execution['cost']['peak_cuda_bytes'])
            for family,component in row['components'].items():
                record=source(component['path']);key=meta|dict(route=row['route'],consumer=family)
                metrics.extend(key|m for m in record['task_summary']);groups.extend(key|m for m in record['group_metrics'])
                compressed_csv(data/'native_predictions'/f"{spec['key']}_{row['route']}_{family}.csv.gz",record['prediction_rows'])
                predictions+=len(record['prediction_rows'])
                if record['vehicle_prediction_rows']:
                    compressed_csv(data/'vehicle_predictions'/f"{spec['key']}_{row['route']}_{family}.csv.gz",record['vehicle_prediction_rows'])
    csv_file(data/'native_metrics.csv',metrics);csv_file(data/'native_groups.csv',groups)
    csv_file(data/'native_summary.csv',aggregate(metrics,('domain','method','variant','route','consumer','task','metric','error_units','role')))
    csv_file(data/'training_contracts.csv',training);csv_file(data/'inventory.csv',inventory)
    csv_file(data/'costs.csv',costs)
    simulation,sim_groups,tails,encoding=[],[],[],[]
    simulation_count=0
    for i,unit in enumerate(models['records']):
        path=simulation_unit_root(root/'simulation',unit)/'confirmation_unit.json';result=source(path)
        if not result['completed']:raise ValueError('simulation unit incomplete')
        for route in unit['routes']:
            row=result['routes'][route]
            if len(row['conditions'])!=plan['pressure_conditions']:raise ValueError('pressure inventory incomplete')
            for condition,receipt in row['conditions'].items():
                verify(receipt['files']);record=source(receipt['result_path']);simulation_count+=1
                meta=dict(method=unit['method'],variant=unit['candidate_name'],route=route,seed=17,condition=condition,role='held_out')
                sim_groups.extend(meta|g for g in record['grouped']['profile_metrics'])
                tails.extend(meta|g for g in record['grouped']['regression_tails'])
                simulation.extend(meta|m for m in record['evaluation']['metric_rows'])
                encoding.append(meta|record['encoding_diagnostics'])
                archive_csv(record['evaluation']['prediction_path'],data/'simulation_predictions'/f'{i}_{route}_{condition}.csv.gz')
    csv_file(data/'simulation_metrics.csv',simulation);csv_file(data/'simulation_groups.csv',sim_groups)
    csv_file(data/'simulation_summary.csv',aggregate(sim_groups,('method','variant','route','consumer','task','metric','condition','role')))
    csv_file(data/'regression_tails.csv',tails);csv_file(data/'encoding_diagnostics.csv',encoding)
    sensitivity=source(root/'sensitivity/result.json');verify(sensitivity['files'])
    if {r['weight'] for r in sensitivity['rows']}!=set(plan['sensitivity_weights']) or len(sensitivity['rows'])!=8:
        raise ValueError('sensitivity grid incomplete')
    sensitivity_rows=[]
    for row in sensitivity['rows']:
        sensitivity_rows.extend(dict(weight=row['weight'],route=row['route'])|m for m in row['result']['metric_rows'])
        archive_csv(row['result']['model_manifest']['prediction_path'],data/'sensitivity_predictions'/f"{row['weight']}_{row['route']}.csv.gz",role='validation')
    csv_file(data/'sensitivity.csv',sensitivity_rows)
    diagnostics=source(root/'diagnostics/result.json');csv_file(data/'inference_costs.csv',diagnostics['timings'])
    csv_file(data/'inference_summary.csv',aggregate([dict(method=r['method'],variant=r['candidate'],route=r['route'],value=r['seconds'])
        for r in diagnostics['timings']],('method','variant','route')))
    csv_file(data/'causality_summary.csv',[dict(route=r['route'],cutoff_s=r['cutoff_s'],perturbation=p,max_delta=v)
        for r in diagnostics['causality'] for p,v in r['historical_max_delta'].items()])
    query=[]
    for record in diagnostics['causality']:
        query.extend(dict(method=record['method'],variant=record['candidate'],route=record['route'],atol=record['historical_atol'])|r for r in record['query_deltas'])
    csv_file(data/'causality_queries.csv',query)
    mechanisms=source(root/'mechanisms/mechanism_summary.json')
    if mechanisms['status']!='completed':raise ValueError('time mechanisms incomplete')
    timing=[]
    for i,row in enumerate(mechanisms['records']):
        verify(row['files']);meta={k:row[k] for k in ('unit','method','seed','route')}
        timing.extend(meta|dict(condition=condition)|m for condition,ms in row['condition_metrics'].items() for m in ms)
        with np.load(row['prediction_path'],allow_pickle=False) as archive:
            compressed_csv(data/'timing_predictions'/f'{i}.csv.gz',[dict(sample_id=str(s),profile_id=str(archive['profile_ids'][j]),
                condition=str(archive['conditions'][j]),truth=archive['truth'][j].tolist(),prediction=archive['predicted'][j].tolist())
                for j,s in enumerate(archive['sample_ids'])])
    csv_file(data/'timing_metrics.csv',timing)
    replay=[]
    for key in plan['replay']:replay.extend(dict(unit_key=key)|r for r in source(root/'replay'/key/'result.json')['rows'])
    csv_file(data/'device_replay.csv',replay)
    # Old failures, cross-device qualifications and interrupted command costs remain separately traceable.
    prior=Path(plan['subset_root']).parent
    historical=[]
    for directory in (prior/'commands',Path(plan['policy_path']).parent/'continuation'):
        for path in directory.glob('*.json'):
            historical.append(dict(path=str(path),sha256=sha(path),record=source(path)))
    csv_file(data/'historical_attempts.csv',historical)
    for path in (prior.parent/'replay/summary.json',prior.parent/'resume/qualification.json'):
        value=source(path);write(data/path.parent.name/(path.name),value)
    for filename in ('simulation_manifest.json','confirmation_generation_contract.json','confirmation_generation_audit.json'):
        value=source(root/'confirmation'/filename);write(data/'protocol'/filename,value)
    write(out/'protocol.json',dict(seed_policy=[17],public_folds=5,native_units=len(inventory),simulation_model_units=len(models['records']),
        public_aggregation='field_mean_then_subject_mean_then_equal_fold_mean',error_bars='population_SD_across_folds_or_simulation_profiles_not_CI',
        dingxin='single_record_chronological_blocks_descriptive_only',sensitivity_role='validation',
        time_mechanism='linear_decodability_not_direct_clock_identification',query_step_s=30/96,
        no_outer_model_selection=True,legacy_three_seed_scope_complete=False))
    write(out/'evidence.json',dict(status='generated',source_files=sources,native_prediction_rows=predictions,
        native_units=len(inventory),simulation_route_conditions=simulation_count,
        data_files={str(p.relative_to(out)):sha(p) for p in data.rglob('*') if p.is_file()}))
    subprocess.run([sys.executable,str(out/'plot.py')],check=True)
    # Verify away from source and raw experiment assets; plotting reads only shipped CSV/JSON.
    with tempfile.TemporaryDirectory(prefix='chronaris-chapter5-redraw-') as directory:
        copy=Path(directory)/'bundle';shutil.copytree(out,copy,ignore=shutil.ignore_patterns('figures'))
        subprocess.run([sys.executable,'-I',str(copy/'plot.py')],cwd=directory,check=True)
        subprocess.run([sys.executable,'-I',str(copy/'verify.py')],cwd=directory,check=True)
    write(out/'validation.json',dict(status='completed',standalone_redraw=True,all_declared_outputs_exported=True,
        visual_review='pending_manual_inspection_of_real_final_figures',git_delivery='pending_import_after_queue_completion'))
    return dict(status='completed',materials=str(out),files={str(p):sha(p) for p in out.rglob('*') if p.is_file()},
                next_step='inspect final figures and commit the complete generated bundle before declaring delivery complete')
