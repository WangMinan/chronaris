"""Fixed all-method training matrix; outer evaluation remains a separate frozen step."""
from dataclasses import asdict
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.common_downstream_contract import downstream_policy
from chronaris.evaluation.application_tasks.common_downstream_smoke import run_common_contract_smoke
from chronaris.evaluation.application_tasks.development_comparison import ASSETS, SENSOR_ASSETS, BASELINES, comparison_budget
from chronaris.evaluation.application_tasks.recent_model_smoke import run_recent_model_smoke
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

METHODS = {
    **{m: dict(routes=('self_supervised','task_guided'), kind='causal_sequence', external=False) for m in BASELINES if m!='naive_time_sync'},
    'naive_time_sync': dict(routes=('self_supervised',),kind='causal_sequence',external=False),
    'timecma': dict(routes=('summary_adapted',),kind='window_end',external=True),
    'chronos2': dict(routes=('frozen','history_adapted'),kind='window_end',external=True),
    'sensorllm_deepseek': dict(routes=('summary_adapted',),kind='window_end',external=True),
}


def units():
    return [dict(domain=domain, method=method, fold_index=fold, seed=seed)
        for domain in ('clare','cogpilot','dingxin')
        for fold in range(1 if domain=='dingxin' else 5) for seed in (17,29,43)
        for method in METHODS if method!='sensorllm_deepseek' or domain=='clare']


def groups():
    stages=['cuda_validation','stage5_formal_plan']
    stages += [f'stage5_formal_execution__{domain}__{mode}' for domain in ('clare','cogpilot')
               for mode in ('eager','graph','interrupt','resume','check')]
    stages += [f'stage5_formal_unit__{i}' for i in range(len(units()))]
    return [[(stage,'completed')] for stage in stages+['stage5_formal_models']]


def read(path):
    return json.loads(Path(path).read_text())


def verify(files):
    for path,digest in files.items():
        if sha256_file(path)!=digest:raise ValueError(f'formal training source changed: {path}')


def matrix(config):
    """Count from sealed manifests; no outer observations or outcome metrics opened."""
    registry=read(config['registry_path']); counts={}; sources={}; digests={}
    for domain in ('clare','cogpilot'):
        root=Path(config['confirmation_data_root'])/domain
        summary=read(root/'summary.json')
        if summary['status']!='completed' or summary['subject_role']!='confirmation' or summary['registry_sha256']!=sha256_file(config['registry_path']):
            raise ValueError('formal cohort preparation or registry changed')
        manifest=root/'sample_manifest.jsonl'
        if sha256_file(manifest)!=summary['sample_manifest_sha256']:raise ValueError('formal sample manifest changed')
        rows=[json.loads(line) for line in manifest.read_text().splitlines()]
        for name in ('summary.json','sample_manifest.jsonl','prepared_cache_index.json','task_targets.pt'):
            sources[str(root/name)]=sha256_file(root/name)
        digests[domain]=sources[str(root/'prepared_cache_index.json')]
        for fold,spec in enumerate(registry['domains'][domain]['folds']['confirmation']):
            roles={role:set(spec[role+'_subjects']) for role in ('train','validation','held_out')}
            if any(a&b for i,a in enumerate(roles.values()) for b in list(roles.values())[i+1:]):
                raise ValueError('formal subject roles overlap')
            counts[f'{domain}/{fold}']={role:sum(r['subject_id'] in ids for r in rows) for role,ids in roles.items()}
    from chronaris.evaluation.application_tasks.v4_dingxin_data import load_v4_dingxin_development
    from chronaris.evaluation.application_tasks.v4_dingxin_deduplicated import deduplicated_dingxin_data
    data,_=deduplicated_dingxin_data(load_v4_dingxin_development());fold=data.folds[0]
    digests['dingxin']=data.data_manifest_sha256
    counts['dingxin/0']={role:len(getattr(fold,role+'_sample_ids')) for role in ('train','validation','held_out')}
    configured=[]
    for unit in units():
        count=counts[f"{unit['domain']}/{unit['fold_index']}"]
        if not all(count.values()):raise ValueError('empty formal role')
        method=unit['method']; budget=comparison_budget(method,count['train']) | {'seed':unit['seed']}
        recipe=None
        if method in BASELINES and method!='naive_time_sync':
            candidate,pre,guided,arguments,_=training_recipe('stage4_reference',method=method,full=True,
                seed=unit['seed'],digest=digests[unit['domain']],train_count=count['train'],graph=method=='chronaris' and unit['domain']!='dingxin')
            recipe=dict(candidate=asdict(candidate),pretraining=asdict(pre),finetuning=asdict(guided),mechanisms=arguments)
        configured.append(unit | dict(routes=list(METHODS[method]['routes']),kind=METHODS[method]['kind'],
            consumers=['linear','minirocket'] if METHODS[method]['kind']=='causal_sequence' and unit['domain']!='dingxin' else ['linear'],
            counts=count,budget=budget,recipe=recipe,data_manifest_sha256=digests[unit['domain']]))
    return dict(units=configured,counts=counts,input_files=sources,model_units=len(configured),
        representation_routes=sum(len(u['routes']) for u in configured),confirmation_opened=False,
        budget_policy='unchanged reference and existing adapter budgets, at least one full training-role pass',
        methods=METHODS,consumer_policy=downstream_policy(),data_role='confirmation_train_and_inner_validation_only')


def plan(config):
    root=Path(config['root']);state=read(root/'pipeline_state.json')
    validation=_validation_evidence(state['completed']['cuda_validation']['path'])
    parent=Path(config['stage5_stability_root']);previous=read(parent/'pipeline_state.json')
    report=read(parent/'stage5_report.json')
    if previous['status']!='stage5_completed' or previous['failures'] or report['completed_units']!=36 or report['confirmation_opened']:
        raise ValueError('formal training requires complete fixed-reference development evidence')
    evidence={r['path']:r['sha256'] for r in previous['completed'].values()};verify(evidence)
    from chronaris.modeling.fusion_encoders.recent_models import verify_assets
    for path in (ASSETS,SENSOR_ASSETS):verify_assets(path);evidence[path]=sha256_file(path)
    configured=matrix(config)
    if config.get('stage5_formal_parent'):
        from chronaris.evaluation.application_tasks.stage5_formal_recovery import prepare_recovery
        configured['recovery'] = prepare_recovery(config, configured)
    configured['sensor_alignment_policy'] = 'train_observed_trends_v2; fixed update budget; all windows retained outside alignment'
    if (configured['model_units'],configured['representation_routes'])!=(279,477):
        raise ValueError('formal method coverage changed')
    frozen=seal_development_plan(dict(status='completed',format='chronaris.stage5_formal_training.v1',
        source_code_sha256=v4_workflow_source_sha256(),validation_files=validation,development_evidence=evidence,
        seed_policy=[17,29,43],selected_configuration='stage4_reference',candidate_search=False,
        outer_evaluation_implemented=False,stop_after='freeze_main_models_without_outer_evaluation',**configured))
    path=root/'formal_plan.json'
    if path.exists() and read(path)!=frozen:raise ValueError('formal plan changed; use a new root')
    return write_result(path,frozen)


def verified_plan(config):
    root=Path(config['root']);frozen=read(root/'formal_plan.json')
    receipt=read(root/'pipeline_state.json')['completed']['stage5_formal_plan'];verify({receipt['path']:receipt['sha256']})
    if (read(receipt['path'])!=frozen or frozen['source_code_sha256']!=v4_workflow_source_sha256()
        or seal_development_plan({k:v for k,v in frozen.items() if k!='plan_sha256'})!=frozen):
        raise ValueError('formal configuration or source changed')
    verify(frozen['input_files']);verify(frozen['development_evidence'])
    return frozen


def run_stage5_formal(stage,config):
    if stage=='stage5_formal_plan':return plan(config)
    root=Path(config['root']);frozen=verified_plan(config)
    if stage.startswith('stage5_formal_execution__'):
        from chronaris.evaluation.application_tasks.stage45_performance import performance_entry,compare_execution
        _,domain,mode=stage.split('__')
        if mode=='check':
            result=compare_execution(root/'performance'/domain/'stage4_reference',revised=True)
            if not result['passed']:raise ValueError('formal execution qualification failed')
            return write_result(root/'performance'/domain/'stage4_reference/check.json',result)
        return performance_entry(config,domain,'stage4_reference',mode)
    if stage=='stage5_formal_models':
        completed=read(root/'pipeline_state.json')['completed'];models={};consumers={};receipts={}
        for i in range(279):
            receipt=completed[f'stage5_formal_unit__{i}'];verify({receipt['path']:receipt['sha256']})
            unit=read(receipt['path']);verify(unit['model_files']);verify(unit['consumer_files'])
            models.update(unit['model_files']);consumers.update(unit['consumer_files']);receipts[receipt['path']]=receipt['sha256']
        if len(models)!=477:raise ValueError('formal main-model routes incomplete')
        return write_result(root/'main_models.json',dict(status='completed',main_models_frozen=True,
            plan_sha256=frozen['plan_sha256'],model_files=models,consumer_files=consumers,unit_receipts=receipts,
            confirmation_opened=False,core_models_frozen=False,
            next_step='freeze core mechanism models and validate outer evaluation before opening held-out outcomes'))
    index=int(stage.split('__')[1]);unit=frozen['units'][index]
    from chronaris.evaluation.application_tasks.stage5_formal_recovery import reuse_completed
    reused = reuse_completed(config, frozen, index)
    if reused is not None: return reused
    domain,method,fold,seed=(unit[k] for k in ('domain','method','fold_index','seed'))
    for domain_check in ('clare','cogpilot'):
        state=read(root/'pipeline_state.json');receipt=state['completed'][f'stage5_formal_execution__{domain_check}__check']
        verify({receipt['path']:receipt['sha256']})
        if not read(receipt['path'])['passed']:raise ValueError('formal execution qualification missing')
    kwargs=dict(domain=domain,output_root=root/'training'/str(index),full=True,seed=seed,fold_index=fold,
        subject_role='confirmation',data_root=config['confirmation_data_root'],registry_path=config['registry_path'])
    if method in BASELINES:
        result=run_common_contract_smoke(methods=(method,),recipe=None if method=='naive_time_sync' else 'stage4_reference',
            cuda_graph_recurrence=method=='chronaris' and domain!='dingxin',expected_recipe=unit['recipe'],**kwargs)
    else:
        result=run_recent_model_smoke(method=method,assets_path=SENSOR_ASSETS if method=='sensorllm_deepseek' else ASSETS,
            expected_budget=unit['budget'],**kwargs)
    if result['status']=='waiting_gpu':return result
    if result['scope']!='confirmation_training' or result['confirmation_opened'] or len(result['results'])!=len(unit['routes']):
        raise ValueError('formal unit scope or routes differ from the frozen plan')
    if result['sample_counts']!={k:unit['counts'][k] for k in ('train','validation')}:
        raise ValueError('formal unit used another cohort or fold')
    models={};consumers={}
    for row in result['results']:
        evaluation=row.get('evaluation',row)
        record=read(Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json')
        declaration=record['declaration'];models[declaration['checkpoint_path']]=declaration['checkpoint_sha256']
        for component in evaluation['consumers']['components'].values():
            for kind in ('model','result'):consumers[component[kind+'_path']]=component[kind+'_sha256']
        if (record['contract']['seed']!=seed or record['contract']['context'].get('cohort_role')!='confirmation'
            or record['contract']['data_manifest_sha256']!=unit['data_manifest_sha256']):
            raise ValueError('formal seed or cohort metadata differs')
    verify(models);verify(consumers)
    return write_result(root/'units'/f'{index}.json',result | dict(unit=unit,model_files=models,consumer_files=consumers,plan_sha256=frozen['plan_sha256']))
