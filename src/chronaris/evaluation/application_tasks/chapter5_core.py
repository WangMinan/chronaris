"""Bounded simulation reference, component removals and physics sensitivity queue."""
from copy import deepcopy
from pathlib import Path
import time

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write, audit_main_models
from chronaris.evaluation.application_tasks.v4_candidates import candidate_options, EXPANDED_SIMULATION_ROOT, validate_candidate_training_budget
from chronaris.evaluation.application_tasks.v4_confirmation_training import _train_confirmation_unit, CONFIRMATION_BUDGET
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256

REPLAY_UNITS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 135, 255)
SEEDS = (17, 29, 43)


def core_units():
    methods = ('chronaris','physiology_only','vehicle_only','mult','contiformer','naive_time_sync')
    configurations = []
    for method in methods:
        options = candidate_options(method,'reference') if method!='naive_time_sync' else {'name':'reference','nonparametric':True}
        configurations.append(dict(method=method, options=options, purpose='main_reference',
            routes=['self_supervised'] if method=='naive_time_sync' else ['self_supervised','task_guided']))
    base = candidate_options('chronaris','reference')
    for name, change in (
        ('no_continuous_evolution',{'variant':'no_continuous_evolution'}),
        ('no_physics_residual',{'physics_weight':0.}),
        ('no_explicit_shift',{'explicit_shift_enabled':False}),
        ('no_single_stream_bypass',{'variant':'no_single_stream_bypass'}),
        ('physics_weight_0025',{'physics_weight':.025}),
        ('physics_weight_0100',{'physics_weight':.1}),
    ):
        configurations.append(dict(method='chronaris',options=deepcopy(base)|change|{'name':name},
            purpose='sensitivity' if name.startswith('physics_weight') else 'ablation',
            routes=['self_supervised','task_guided']))
    return [dict(index=index, **unit, seed=seed) for index,(unit,seed) in enumerate(
        (unit,seed) for unit in configurations for seed in SEEDS)]


def groups():
    stages=['cuda_validation','chapter5_plan']
    stages += [f'chapter5_replay__{i}' for i in REPLAY_UNITS]
    stages += [f'chapter5_smoke__{i}' for i in (0,18,21,24,27)]
    stages += [f'chapter5_core__{i}' for i in range(len(core_units()))]
    return [[(stage,'completed')] for stage in stages+['chapter5_core_models']]


def plan(config):
    from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
    from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
    root = Path(config['root'])
    state = read(root/'pipeline_state.json')
    validation = _validation_evidence(state['completed']['cuda_validation']['path'])
    audit_main_models(config['chapter5_main_root'], root/'main_asset_audit.json')
    units=core_units()
    result=seal_development_plan(dict(status='completed',format='chronaris.chapter5_core_plan.v1',
        source_code_sha256=v4_workflow_source_sha256(), main_root=config['chapter5_main_root'],
        main_models_sha256=sha(Path(config['chapter5_main_root'])/'main_models.json'),
        asset_audit_sha256=sha(root/'main_asset_audit.json'),validation_files=validation,
        units=units,model_units=len(units),representation_routes=sum(len(u['routes']) for u in units),
        training_budget=CONFIRMATION_BUDGET,simulation_root=str(Path(EXPANDED_SIMULATION_ROOT).resolve()),
        replay_units=list(REPLAY_UNITS),replay_samples_per_route=4,replay_atol=1e-5,replay_rtol=1e-5,
        engineering_updates=dict(pretraining=2,head_warmup=2,joint=2),
        physics_sensitivity_weights=[0.,.025,.05,.1],sensitivity_role='development_validation_only_no_outer_selection',
        not_applicable=[dict(component=name,reason='not_enabled_in_the_fixed_reference') for name in
            ('independent_pair_loss','quality_gate','expanded_missingness')],
        supervision='self_supervised_and_summary_task_guided_separate',
        mechanism_scope='separate_simulation_configuration_not_public_main_model',
        observation_anchor_retained=True,confirmation_opened=False,stop_after='core_model_freeze',
        downstream_policy='native_main_consumers_reused_without_refitting',
        input_files=config['input_files']))
    path=root/'chapter5_plan.json'
    if path.exists() and read(path)!=result:
        raise ValueError('core plan changed; preserve this run and use a new root')
    return write(path,result)


def verified_plan(config):
    from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
    root=Path(config['root']); result=read(root/'chapter5_plan.json')
    receipt=read(root/'pipeline_state.json')['completed']['chapter5_plan']
    verify({receipt['path']:receipt['sha256']})
    if (read(receipt['path'])!=result or result['source_code_sha256']!=v4_workflow_source_sha256()
        or seal_development_plan({k:v for k,v in result.items() if k!='plan_sha256'})!=result
        or result['units']!=core_units() or result['training_budget']!=CONFIRMATION_BUDGET):
        raise ValueError('core plan or source changed')
    verify(result['input_files'])
    verify({str(root/'main_asset_audit.json'):result['asset_audit_sha256']})
    return result


def _train(unit, plan, root, *, smoke=False):
    import torch
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
    from chronaris.evaluation.application_tasks.v4_diagnostic_run import _require_diagnostic_device
    from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu')
        _require_diagnostic_device(unit['seed']);torch.set_num_threads(1)
        started=time.perf_counter()
        if unit['method']=='naive_time_sync':
            from chronaris.evaluation.application_tasks.v4_confirmation_training import _confirmation_training_inputs
            from chronaris.evaluation.application_tasks.v4_development_data import development_normalization
            from chronaris.evaluation.application_tasks.v4_naive_baseline import fit_v4_naive_encoder
            provider,schema,fold,_,digest,_,_,_=_confirmation_training_inputs('simulation',0,plan['simulation_root'],SIMULATION_REGISTRY,root)
            normalizer,_=development_normalization('simulation',provider,schema,fold,digest)
            checkpoint, result=fit_v4_naive_encoder(provider=provider,fold=fold,normalizer=normalizer,
                data_manifest_sha256=digest,output_root=root/'naive'/f"seed{unit['seed']}",seed=unit['seed'])
            files={str(checkpoint):sha(checkpoint),str(checkpoint.parent/'fit_manifest.json'):sha(checkpoint.parent/'fit_manifest.json')}
        else:
            result=_train_confirmation_unit(domain='simulation',fold_index=0,method=unit['method'],options=unit['options'],
                routes=unit['routes'],seed=unit['seed'],output_root=root,data_root=plan['simulation_root'],
                registry_path=SIMULATION_REGISTRY,freeze_sha256=plan['plan_sha256'],normalizer_root=root/'normalizers',
                engineering_smoke=smoke)
            files={}
            for route in unit['routes']:
                path=result[route+'_training']['best_checkpoint_path'];files[path]=sha(path)
                if not smoke:validate_candidate_training_budget(result,phase='review',route=route)
            if not result['completed'] or not all(v['passed'] for v in result['checkpoint_correctness'].values()):
                raise ValueError('core unit failed completion or causality validation')
        return dict(status='completed',unit=unit,engineering_smoke=smoke,training=result,files=files,
            total_seconds=time.perf_counter()-started,plan_sha256=plan['plan_sha256'],confirmation_opened=False)


def run_step(stage,config):
    root=Path(config['root'])
    if stage=='chapter5_plan':return plan(config)
    frozen=verified_plan(config)
    if stage.startswith('chapter5_replay__'):
        from chronaris.evaluation.application_tasks.chapter5_native import evaluate_main_unit
        index=int(stage.split('__')[1])
        if index not in REPLAY_UNITS:raise ValueError('unplanned replay unit')
        return evaluate_main_unit(main_root=frozen['main_root'],index=index,output_root=root/'replay'/str(index))
    if stage.startswith('chapter5_smoke__'):
        index=int(stage.split('__')[1]);unit=frozen['units'][index]
        result=_train(unit,frozen,root/'engineering'/str(index),smoke=True)
        if result['status']=='waiting_gpu':return result
        # A second process-stage invocation is also supported; completed trainers must reuse without updates.
        repeated=_train(unit,frozen,root/'engineering'/str(index),smoke=True)
        if repeated['status']=='waiting_gpu':return repeated
        if result['files']!=repeated['files']:raise ValueError('completed core training replay changed checkpoint bytes')
        result['completed_resume_identical']=True
        return write(root/'engineering'/f'{index}.json',result)
    if stage.startswith('chapter5_core__'):
        index=int(stage.split('__')[1]);unit=frozen['units'][index]
        result=_train(unit,frozen,root/'core_training')
        if result['status']=='waiting_gpu':return result
        return write(root/'core_units'/f'{index}.json',result)
    if stage=='chapter5_core_models':
        import torch
        state=read(root/'pipeline_state.json');files={};records=[]
        for unit in frozen['units']:
            receipt=state['completed'][f"chapter5_core__{unit['index']}"];verify({receipt['path']:receipt['sha256']})
            result=read(receipt['path']);verify(result['files'])
            if result['unit']!=unit or result['engineering_smoke'] or result['plan_sha256']!=frozen['plan_sha256']:
                raise ValueError('cannot freeze another configuration or engineering checkpoint')
            files.update(result['files']);files[receipt['path']]=receipt['sha256']
            if unit['method']!='naive_time_sync':
                for route in unit['routes']:
                    checkpoint=result['training'][route+'_training']['best_checkpoint_path']
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    if saved['training_status']!='completed' or saved['seed']!=unit['seed'] or saved['config']['device']!='cuda':
                        raise ValueError('core checkpoint is not a completed planned CUDA model')
            records.append(dict(unit=unit,files=result['files']))
        for key,receipt in state['completed'].items():
            if key.startswith(('chapter5_replay__','chapter5_smoke__')):
                verify({receipt['path']:receipt['sha256']});files[receipt['path']]=receipt['sha256']
                verify(read(receipt['path'])['files']);files.update(read(receipt['path'])['files'])
        return write(root/'core_models.json',dict(status='completed',core_models_frozen=True,
            source_code_sha256=v4_workflow_source_sha256(),main_models_sha256=frozen['main_models_sha256'],
            plan_sha256=frozen['plan_sha256'],files=files,records=records,confirmation_opened=False,
            next_step='open frozen outer evaluation and generate simulation confirmation after manual wakeup'))
    raise ValueError('unknown chapter 5 stage')
