"""Read complete frozen units; retain subject/profile statistics and unfavorable outcomes."""
from pathlib import Path
import json

import numpy as np

from chronaris.evaluation.application_tasks.v4_confirmation_training import read_frozen_configuration
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_dingxin_data import build_dingxin_outer_consumer_inputs
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
from chronaris.evaluation.application_tasks.v4_grouped_statistics import paired_profile_statistics, paired_public_statistics
from chronaris.evaluation.application_tasks.v4_native_confirmation_cohort import build_native_confirmation_plan
from chronaris.evaluation.application_tasks.v4_native_result_audit import audit_native_consumer_result
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import simulation_unit_root
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

SIMULATION_METRICS = (
    ('macro_f1', 'workload_class_true', 'linear_class', (0,1,2)),
    ('rmse', 'workload_true', 'linear_regression', None),
    ('frame_macro_f1', 'state_true', 'causal_tcn_duration_state', (0,1,2,3,4)),
    ('segmental_f1_iou_0.50', 'state_true', 'causal_tcn_duration_state', None),
    ('boundary_f1_1s', 'state_true', 'causal_tcn_duration_state', None),
    ('boundary_detection_delay_s', 'state_true', 'causal_tcn_duration_state', None))


def _verified_file(path, files, digest=None):
    actual = sha256_file(path)
    if digest is not None and actual != digest:
        raise ValueError(f'final evidence changed: {path}')
    files[str(path)] = actual


def _prediction_archive(clean, files):
    manifest = clean['model_manifest']
    if 'held_out' not in manifest['evaluation_roles'] or manifest['consumer_fit_role'] != 'train':
        raise ValueError('formal statistics require frozen consumers and held-out predictions')
    path = manifest['prediction_path']; _verified_file(path, files, manifest['prediction_sha256'])
    for record in manifest['model_files'].values():
        _verified_file(record['path'], files, record['sha256'])
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files if key.startswith('held_out_')}


def collect_final_evidence(*, formal_root, mechanism_root, output_root, data_root, registry_path):
    formal, root = Path(formal_root), Path(output_root)
    freeze_path = formal/'frozen_configuration.json'; freeze_hash = sha256_file(freeze_path)
    frozen = read_frozen_configuration(freeze_path, freeze_hash)
    files = {str(freeze_path): freeze_hash}; native_rows = []; public_rows = []; native_units = 0
    pipeline=json.loads((formal.parent/'pipeline_state.json').read_text())
    receipts={}
    for stage in ('native_neural','native_nonparametric','simulation_evaluate','core_evaluate'):
        item=pipeline['completed'][stage]
        _verified_file(item['path'],files,item['sha256'])
        queue=json.loads(Path(item['path']).read_text())
        if queue['status']!='completed' or queue['freeze_sha256']!=freeze_hash:
            raise ValueError('final report requires completed frozen evaluation queues')
        receipts[stage]=queue['completed_receipts']
    cache = {}
    for unit in build_native_confirmation_plan(freeze_path, freeze_hash)['units']:
        domain, method, seed = unit['domain'], unit['method'], unit['seed']
        unit_root = formal/domain/method/unit['candidate_name']/f'fold{unit["fold_index"]+1:02d}'/f'seed{seed}'
        receipt_path = unit_root/'confirmation_unit.json'
        _verified_file(receipt_path, files, receipts['native_'+unit['backend']][str(unit_root.relative_to(formal))])
        receipt = json.loads(receipt_path.read_text())
        if not receipt['completed'] or receipt['freeze_sha256'] != freeze_hash:
            raise ValueError('native formal unit is incomplete')
        key = domain, unit['fold_index']
        if key not in cache:
            _, _, fold, _, _, _, _, data = load_development_inputs(domain, data_root, registry_path,
                fold_index=unit['fold_index'], subject_role='confirmation')
            if domain == 'dingxin':
                fitted = build_dingxin_outer_consumer_inputs(data, fold.fold_id)
                targets, definitions, context = (fitted[name] for name in ('targets','definitions','context'))
                consumer_fold=fitted['fold']
            else:
                targets, definitions, context = data.targets, data.task_definitions, native_consumer_context(domain, data, fold)
                consumer_fold=fold
            cache[key] = targets, definitions, context, fold, consumer_fold
        targets, definitions, context, encoder_fold, consumer_fold = cache[key]
        for route in unit['routes']:
            directory = unit_root/'evaluation'/('shared' if method == 'naive_time_sync' else route)
            state_path = directory/'run_state.json'; _verified_file(state_path, files)
            state = json.loads(state_path.read_text())
            if not state['completed'] or state['source']['engineering_only']:
                raise ValueError('native evidence is engineering-only or incomplete')
            if (state['source']['encoder_fold']!=encoder_fold.to_dict() or state['source']['consumer_fold']!=consumer_fold.to_dict()
                or state['source']['seed']!=seed or state['source']['method']!=method):
                raise ValueError('native evidence differs from the frozen unit fold, method or seed')
            outputs = {role: load_fusion_stream_batch(directory/'representations'/role) for role in ('train','validation','held_out')}
            if any(output.sample_ids!=getattr(consumer_fold,role+'_sample_ids') for role,output in outputs.items()):
                raise ValueError('native final representation roles changed')
            audit = audit_native_consumer_result(state['consumers'], outputs=outputs, targets=targets, definitions=definitions,
                context=context, label_used_for_encoder_training=route == 'task_guided' and method != 'naive_time_sync')
            files.update(audit['files'])
            for row in audit['metrics']:
                if row['role'] != 'held_out':
                    continue
                native_rows.append(row | dict(route=route))
                if domain != 'dingxin' and row['status'] == 'completed':
                    for metric in ('macro_f1','rmse'):
                        if row.get(metric) is not None:
                            public_rows.append(row | dict(route=route, metric=metric, value=row[metric]))
            native_units += 1
    public_statistics = []
    for method in ('physiology_only','vehicle_only','mult','contiformer','naive_time_sync'):
        public_statistics.extend(paired_public_statistics(public_rows, first_method='chronaris', second_method=method))
    simulation, pressure, archives = [], [], {}
    for family, directory, inventory in (
        ('main', formal, formal/'simulation_frozen_models.json'),
        ('ablation', formal/'core_ablations', formal/'core_ablations/ablation_frozen_models.json')):
        _verified_file(inventory, files); models = json.loads(inventory.read_text())
        for filename, digest in models['files'].items(): _verified_file(filename, files, digest)
        for unit in models['records']:
            unit_root=simulation_unit_root(directory,unit)
            path = unit_root/'confirmation_unit.json'
            _verified_file(path, files, receipts['simulation_evaluate' if family=='main' else 'core_evaluate'][str(unit_root.relative_to(directory))])
            state = json.loads(path.read_text())
            if not state['completed'] or state['freeze_sha256'] != freeze_hash:
                raise ValueError('simulation unit is incomplete')
            for route in unit['routes']:
                record = state['routes'][route]
                if 'shared_with' in record: record = state['routes'][record['shared_with']]
                if len(record['conditions']) != (38 if family == 'main' else 1):
                    raise ValueError('formal pressure or core ablation scope is incomplete')
                name = unit['method'] if family == 'main' else 'ablation:'+unit['ablation']
                archives[(route, name, unit['seed'])] = _prediction_archive(record['clean'], files)
                simulation.extend(row | dict(route=route, family=family, comparison_name=name)
                    for row in record['clean']['metric_rows'] if row['role']=='held_out')
                for condition, value in record['conditions'].items():
                    for filename, digest in value['files'].items(): _verified_file(filename, files, digest)
                    evaluated = json.loads(Path(value['result_path']).read_text())
                    pressure.append(dict(method=name, route=route, seed=unit['seed'], condition=condition,
                        grouped=evaluated['grouped'], encoding_diagnostics=evaluated['encoding_diagnostics']))
    registry = json.loads(Path('docs/requirements/thesis-v4-simulation-manifest.json').read_text())
    profiles = {sample: row['profile_id'] for row in registry['trajectories'] if row['role']=='confirmation' for sample in row['context_sample_ids']}
    simulation_statistics = []
    for route in ('self_supervised','task_guided'):
        reference = [archives[(route,'chronaris',seed)] for seed in (17,29,43)]
        for method in sorted({name for r,name,_ in archives if r==route and name!='chronaris'}):
            comparison = [archives[(route,method,seed)] for seed in (17,29,43)]
            ids = reference[0]['held_out_sample_ids'].tolist()
            if set(ids)!=set(profiles) or len(ids)!=len(profiles):
                raise ValueError('simulation statistics escaped the 512 registered confirmation contexts')
            for record in reference+comparison:
                if record['held_out_sample_ids'].tolist()!=ids:
                    raise ValueError('paired simulation sample ordering changed')
            for metric,target,prediction,classes in SIMULATION_METRICS:
                truth=reference[0]['held_out_'+target]
                if any(not np.array_equal(record['held_out_'+target],truth) for record in reference+comparison):
                    raise ValueError('paired simulation targets changed')
                result=paired_profile_statistics(truth=truth, first_predictions=[r['held_out_'+prediction] for r in reference],
                    second_predictions=[r['held_out_'+prediction] for r in comparison], profile_ids=[profiles[s] for s in ids],
                    metric=metric, classes=classes)
                simulation_statistics.append(dict(route=route,first_method='chronaris',second_method=method,**result))
    mechanism_path=Path(mechanism_root)/'mechanism_summary.json'; _verified_file(mechanism_path,files)
    mechanism=json.loads(mechanism_path.read_text())
    if mechanism['status']!='completed' or len(mechanism['records'])!=36:
        raise ValueError('separate time mechanisms are incomplete')
    for record in mechanism['records']:
        for filename,digest in record['files'].items(): _verified_file(filename,files,digest)
    mechanism_archives={}
    for record in mechanism['records']:
        with np.load(record['prediction_path'],allow_pickle=False) as archive:
            mechanism_archives[(record['route'],record['method'],record['seed'])]={key:archive[key].copy() for key in archive.files}
    mechanism_statistics=[]
    from chronaris.evaluation.application_tasks.v4_mechanism_evaluation import TARGETS
    for route in ('self_supervised','task_guided'):
        reference=[mechanism_archives[(route,'chronaris',seed)] for seed in (17,29,43)]
        for method in ('physiology_only','vehicle_only','mult','contiformer','naive_time_sync'):
            comparison=[mechanism_archives[(route,method,seed)] for seed in (17,29,43)]
            first=reference[0]
            if any(not np.array_equal(row['sample_ids'],first['sample_ids']) or not np.array_equal(row['truth'],first['truth'])
                   or not np.array_equal(row['profile_ids'],first['profile_ids']) for row in reference+comparison):
                raise ValueError('paired timing samples, targets or profiles changed')
            for index,target in enumerate(TARGETS):
                statistic=paired_profile_statistics(truth=first['truth'][:,index],metric='rmse',profile_ids=first['profile_ids'],
                    first_predictions=[row['predicted'][:,index] for row in reference],
                    second_predictions=[row['predicted'][:,index] for row in comparison])
                mechanism_statistics.append(dict(route=route,target=target,first_method='chronaris',second_method=method,**statistic))
    if native_units+36 != frozen['amended_total_evaluation_units']:
        raise ValueError('formal main table does not contain all frozen method/fold/seed/routes')
    cases=[]
    for route in ('self_supervised','task_guided'):
        for seed in (17,29,43):
            archive=archives[(route,'chronaris',seed)]
            ids=archive['held_out_sample_ids'].tolist()
            for sample in frozen['preselected_case_context_ids']:
                index=ids.index(sample)
                cases.append(dict(sample_id=sample,route=route,seed=seed,
                    workload_truth=float(archive['held_out_workload_true'][index]),
                    workload_prediction=float(archive['held_out_linear_regression'][index]),
                    class_truth=int(archive['held_out_workload_class_true'][index]),
                    class_prediction=int(archive['held_out_linear_class'][index])))
    training_updates={}
    for path in formal.parent.rglob('run_state.json'):
        state=json.loads(path.read_text())
        for route in ('self_supervised','task_guided'):
            training=state.get(route+'_training')
            if not training:continue
            _verified_file(path,files)
            checkpoint=str(Path(training['last_checkpoint_path']).resolve())
            training_updates[checkpoint]=dict(route=route,optimizer_updates=training['optimizer_updates'],
                head_warmup_updates=training.get('head_warmup_updates',0),joint_updates=training.get('joint_updates',0),
                training_elapsed_s=training['training_elapsed_s'],stage=path.relative_to(formal.parent).parts[0])
    result=dict(status='completed',freeze_sha256=freeze_hash,main_evaluation_units=native_units+36,files=files,
        native_subject_rows=native_rows,public_scalar_rows=public_rows,public_paired_statistics=public_statistics,
        simulation_rows=simulation,simulation_paired_statistics=simulation_statistics,pressure=pressure,
        mechanisms=mechanism['records'],mechanism_paired_statistics=mechanism_statistics,dingxin_inference='single_record_time_blocks_descriptive_only',
        preselected_cases=cases,training_update_ledger=training_updates,
        confirmation_feedback_to_selection=False)
    write_result(root/'evidence.json',result)
    from chronaris.evaluation.application_tasks.v4_final_reporting import write_final_report
    write_final_report(result,root)
    return dict(status='completed',main_evaluation_units=result['main_evaluation_units'],
        evidence_path=str(root/'evidence.json'),evidence_sha256=sha256_file(root/'evidence.json'))
