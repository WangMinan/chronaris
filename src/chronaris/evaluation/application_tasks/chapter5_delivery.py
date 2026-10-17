"""Seed-17 writing delivery, executed against unchanged frozen training modules."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import time

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan

EXTRA = (24, 27, 30, 33)


def execution_files():
    code=[Path(__file__).parents[4]/'scripts/evaluation/application_tasks/run_chapter5_delivery.py',
          *Path(__file__).parent.glob('chapter5_delivery*.py')]
    code += [Path(__file__).with_name(n+'.py') for n in ('device_replay','v4_mechanism_data','v4_mechanism_evaluation',
        'v4_correctness','causality_recovery','v4_simulation_confirmation_evaluation')]
    code.append(Path(__file__).parents[2]/'modeling/fusion_encoders/sensorllm_adapter.py')
    return {str(p.resolve()):sha(p) for p in code}


def execution_binding(root, plan):
    path=Path(root)/'simulation_revision.json'
    if not path.exists():path=Path(root)/'diagnostics_revision.json'
    if not path.exists():path=Path(root)/'execution_revision.json'
    if not path.exists():
        if any(plan['input_files'].get(p)!=digest for p,digest in execution_files().items()):
            raise ValueError('delivery execution changed without a bound revision')
        return {}
    revision=read(path)
    if revision['plan_sha256']!=plan['plan_sha256'] or revision['execution_files']!=execution_files():
        raise ValueError('delivery execution revision differs from plan or running source')
    files=revision['execution_files']|revision['preserved_files'];verify(files)
    return files|{str(path.resolve()):sha(path)}


def helper(name):
    path = Path(__file__).with_name(name+'.py')
    spec = importlib.util.spec_from_file_location('_delivery_'+name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def immutable(path, value):
    if Path(path).exists() and read(path) != value:
        raise ValueError(f'delivery evidence changed; use a new root: {path}')
    return write(path, value)


def prepare(root, subset_root, policy_path, launcher, plot_root):
    from chronaris.evaluation.application_tasks.chapter5_subset import require_subset_freeze, verified_subset, _bind_native
    from chronaris.evaluation.application_tasks.chapter5_native import unit_record
    from chronaris.evaluation.application_tasks.chapter5_core import core_units
    from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
    root, subset_root = Path(root).resolve(), Path(subset_root).resolve()
    gate = require_subset_freeze(subset_root); subset = verified_subset(subset_root)
    if root == subset_root or subset_root in root.parents:
        raise ValueError('delivery needs a separate output root')
    files = dict(gate['files'])
    for p in (subset_root/'subset_models.json', subset_root/'subset_plan.json', Path(policy_path)):
        files[str(p)] = sha(p)
    files.update(read(policy_path)['files'])
    # Bind only delivery code: importing it never changes the frozen trainer fingerprint.
    files.update(execution_files())
    files.update({str(p.resolve()):sha(p) for p in Path(plot_root).rglob('*') if p.is_file()})
    state = read(Path(subset['main_root'])/'pipeline_state.json')
    formal = read(Path(subset['main_root'])/'formal_plan.json')
    native = [dict(key=f"reference_{r['index']}", index=r['index'], origin='reference', unit=r['unit'],
                   receipt=r['receipt']['path']) for r in subset['native_references']]
    for i, unit in enumerate(formal['units']):
        if unit['domain']=='dingxin' and unit['seed']==17:
            receipt = state['completed'][f'stage5_formal_unit__{i}']
            record = unit_record(subset['main_root'], i); _bind_native(files, receipt, record)
            native.append(dict(key=f'reference_{i}',index=i,origin='reference',unit=unit,receipt=receipt['path']))
    native += [dict(key=f'new_{i}',index=i,origin='new',unit=subset['units'][i],
                    receipt=str(subset_root/'units'/f'{i}.json')) for i in range(15)]
    assets = [str(Path('/mnt/e/chronaris-v4-results/2026-09-11-stage3-models')/n) for n in ('assets.json','assets_sensor.json')]
    replay = [r['key'] for r in native if r['origin']=='new' or (r['unit']['fold_index']==0 and
              r['unit']['method'] in ('timecma','chronos2')) or r['unit']['domain']=='dingxin' or r['unit']['method']=='sensorllm_deepseek']
    result = seal_development_plan(dict(format='chronaris.chapter5_draft_delivery.v1',root=str(root),
        subset_root=str(subset_root), policy_path=str(policy_path), main_root=subset['main_root'],
        source_code_sha256=v4_workflow_source_sha256(), input_files=files, assets=assets,
        native=native, replay=replay, extra_units=[core_units()[i] for i in EXTRA],
        simulation_root=subset['simulation_root'], data_root=subset['data_root'],registry_path=subset['registry_path'],
        plot_root=str(Path(plot_root).resolve()), seed_policy=[17], public_folds=5,
        sensitivity_weights=[0.,.025,.05,.1], sensitivity_role='validation',
        sensitivity_changes_reference=False, pressure_conditions=38,
        sensorllm_inference='qualified_original_RTX4090_frozen_encoder_remote_features_local_frozen_consumers',
        dingxin_chronos2_inference='qualified_original_RTX4090_frozen_encoder_remote_features_local_frozen_consumers',
        confirmation_opened=False, legacy_full_matrix_complete=False,
        statistics='subject-first then equal fold means; fold population SD, not seed uncertainty',
        completion='all declared results plus independently redrawable materials; failures stop queue'))
    verify(files)
    return immutable(root/'delivery_plan.json', result)


def validated_plan(root):
    from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
    plan = read(Path(root)/'delivery_plan.json')
    if (plan['root'] != str(Path(root).resolve()) or plan['source_code_sha256'] != v4_workflow_source_sha256()
        or seal_development_plan({k:v for k,v in plan.items() if k!='plan_sha256'}) != plan):
        raise ValueError('delivery plan or frozen implementation changed')
    verify(plan['input_files'])
    return plan


def plan_at(root):
    plan=validated_plan(root);execution_binding(root,plan)
    return plan


def revise(root, phase='sensitivity'):
    """Bind a stopped-stage repair without rewriting completed receipts or model freeze."""
    import fcntl
    if phase not in ('sensitivity','diagnostics','simulation:0'):raise ValueError('unsupported repair phase')
    root=Path(root)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=validated_plan(root);state=read(root/'state.json')
        completed=steps(plan)[:steps(plan).index(phase)]
        if (state['status']!='failed' or state['phase']!=phase or state['completed']!=completed
            or state['plan_sha256']!=plan['plan_sha256']
            or (root/phase/'result.json').exists()
            or ((root/'confirmation').exists() != (phase=='simulation:0'))
            or ((root/'delivery_models.json').exists() != (phase!='sensitivity'))
            or (phase=='simulation:0' and any((root/'simulation').rglob('confirmation_unit.json')))):
            raise ValueError('repair requires the stopped pre-'+phase+' queue')
        files={}
        for stage in completed:
            path=root/'steps'/(stage.replace(':','_')+'.json');receipt=read(path)
            if receipt['stage']!=stage or receipt['plan_sha256']!=plan['plan_sha256'] or receipt['exit_code']!=0:
                raise ValueError('completed step binding changed')
            verify(receipt['files']);files.update(receipt['files']);files[str(path)]=sha(path)
        if phase!='sensitivity':
            prior=root/('diagnostics_revision.json' if phase=='simulation:0' else 'execution_revision.json');revision=read(prior)
            if revision['plan_sha256']!=plan['plan_sha256']:raise ValueError('prior revision plan changed')
            verify(revision['preserved_files']|revision['execution_files'])
            files[str(prior)]=sha(prior)
        archive={'sensitivity':'target_guard','diagnostics':'diagnostics','simulation:0':'simulation_contract'}[phase]
        for name in ('state.json','controller.log',f'logs/{phase.replace(":","_")}.log'):
            original=root/name;backup=root/'recovery'/archive/Path(name).name
            backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists() and backup.read_bytes()!=original.read_bytes():raise ValueError('failure backup differs')
            backup.write_bytes(original.read_bytes());files[str(backup)]=sha(backup)
        destination={'sensitivity':'execution_revision.json','diagnostics':'diagnostics_revision.json',
            'simulation:0':'simulation_revision.json'}[phase]
        reason={'sensitivity':'target reference guard repair','diagnostics':'use existing non-neural inference adapter for fixed synchronization baseline',
            'simulation:0':'compare the existing structural schema fingerprint while retaining separate source file hashes'}[phase]
        return immutable(root/destination,dict(status='completed',plan_sha256=plan['plan_sha256'],
            reason=reason,preserved_stages=completed,preserved_files=files,execution_files=execution_files(),
            confirmation_opened=phase=='simulation:0'))


def check_extra(result, unit, plan):
    import torch
    from chronaris.evaluation.application_tasks.v4_candidates import validate_candidate_training_budget
    if (result['status']!='completed' or result['unit']!=unit or result['engineering_smoke']
        or result['confirmation_opened'] or result['plan_sha256']!=plan['plan_sha256']):
        raise ValueError('extra model is not a completed planned training unit')
    verify(result['files'])
    for route in unit['routes']:
        validate_candidate_training_budget(result['training'], phase='review', route=route)
        saved = torch.load(result['training'][route+'_training']['best_checkpoint_path'], map_location='cpu',weights_only=True)
        if saved['training_status']!='completed' or saved['seed']!=17:
            raise ValueError('extra checkpoint incomplete')
        if route=='self_supervised' and (
            saved['config']['physics_weight'] != unit['options'].get('physics_weight',.05)
            or saved['chronaris_variant'] != unit['options'].get('variant','full')
            or saved['chronaris_explicit_shift_enabled'] != unit['options'].get('explicit_shift_enabled',True)):
            raise ValueError('extra checkpoint does not implement its declared intervention')


def freeze(root, plan, *, execution_evidence=None):
    from chronaris.evaluation.application_tasks.chapter5_subset import require_subset_freeze
    from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import simulation_records
    import torch
    require_subset_freeze(plan['subset_root'])
    files = dict(plan['input_files'])|(execution_binding(root,plan) if execution_evidence is None else execution_evidence)
    models = simulation_records(plan['subset_root'])
    for key in plan['replay']:
        path=Path(root)/'replay'/key/'result.json'; result=read(path)
        if (result['status']!='completed' or not result['rows'] or result['confirmation_opened']
            or result['binding']['plan_sha256']!=plan['plan_sha256']):
            raise ValueError('missing internal replay qualification')
        spec=next(r for r in plan['native'] if r['key']==key)
        if spec['unit']['method'] in ('timecma','chronos2','sensorllm_deepseek') and any(
            not r['strict']['passed'] or r['class_flips'] for r in result['rows']):
            raise ValueError('recent-model inference qualification has unresolved differences')
        files[str(path)]=sha(path); files.update(result['files'])
    path=Path(root)/'sensitivity/result.json'; result=read(path)
    if (result['status']!='completed' or result['confirmation_opened'] or len(result['rows'])!=8
        or {r['weight'] for r in result['rows']}!=set(plan['sensitivity_weights'])):
        raise ValueError('development sensitivity grid incomplete')
    files[str(path)]=sha(path);files.update(result['files'])
    for manifest in plan['assets']:
        files.update(read(manifest)['files'])
    for unit in plan['extra_units']:
        path=Path(root)/'extra_units'/f"{unit['index']}.json"
        result=read(path); check_extra(result,unit,plan)
        files[str(path)]=sha(path); files.update(result['files'])
        if unit['purpose']=='sensitivity':
            continue  # Sensitivity stays on development data, never opens outer scores.
        checkpoints={r:result['training'][r+'_training']['best_checkpoint_path'] for r in unit['routes']}
        saved=torch.load(checkpoints['self_supervised'],map_location='cpu',weights_only=True)
        models['records'].append(dict(method=unit['method'],candidate_name=unit['options']['name'],seed=17,
            routes=unit['routes'],checkpoints=checkpoints,fold=saved['fold'],
            data_manifest_sha256=saved['config']['data_manifest_sha256'],training_source_sha256=saved['source_code_sha256']))
    verify(files)
    return immutable(Path(root)/'delivery_models.json',dict(status='frozen',plan_sha256=plan['plan_sha256'],
        files=files,**models,confirmation_opened=False,seed_policy=[17],legacy_core_models_frozen=False))


def require_freeze(root, plan):
    path=Path(root)/'delivery_models.json'
    if not path.exists():
        raise ValueError('delivery outer evaluation requires all extra models and restored assets')
    old=read(path)
    evidence=None
    if (Path(root)/'diagnostics_revision.json').exists():
        execution_binding(root,plan)
        prior=Path(root)/'execution_revision.json';revision=read(prior)
        evidence=revision['execution_files']|revision['preserved_files']|{str(prior.resolve()):sha(prior)}
    if freeze(root,plan,execution_evidence=evidence)!=old:
        raise ValueError('delivery freeze changed')
    return old


def steps(plan):
    return ['assets']+[f'engineering:{i}' for i in EXTRA]+[f'replay:{k}' for k in plan['replay']]+[f'train:{i}' for i in EXTRA]+['sensitivity','freeze','diagnostics','generate','timing-data']+[
        f'native:{r["key"]}' for r in plan['native']]+[f'simulation:{i}' for i in range(11)]+[
        'mechanisms','materials']


def native_record(plan, key):
    from chronaris.evaluation.application_tasks.chapter5_native import unit_record
    records = [r for r in plan['native'] if r['key']==key]
    if len(records)!=1: raise ValueError('native unit outside declared delivery')
    row=records[0]
    result=unit_record(plan['main_root'],row['index']) if row['origin']=='reference' else read(row['receipt'])
    if result['unit']!=row['unit']:raise ValueError('native receipt identity changed')
    if helper('chapter5_delivery_sensor').supports(row['unit']):
        from chronaris.evaluation.application_tasks import chapter5_native
        chapter5_native._encode=helper('chapter5_delivery_sensor').encode_remote
    return row,result


def worker(root, stage):
    import torch
    from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
    from chronaris.evaluation.application_tasks.chapter5_native import _evaluate_native_record
    root=Path(root); plan=plan_at(root); torch.set_num_threads(1)
    if stage not in steps(plan): raise ValueError('unplanned delivery step')
    if stage=='assets':
        return helper('chapter5_delivery_assets').restore_assets(plan['assets'])
    if stage.startswith(('train:','engineering:')):
        unit=next(u for u in plan['extra_units'] if u['index']==int(stage.split(':')[1]))
        if stage.startswith('engineering:'):
            helper('causality_recovery').install_policy(plan['policy_path'])
            from chronaris.evaluation.application_tasks.chapter5_core import _train
            destination=root/'engineering'/str(unit['index'])
            first=_train(unit,plan,destination,smoke=True)
            if first['status']=='waiting_gpu':return first
            second=_train(unit,plan,destination,smoke=True)
            if second['status']=='waiting_gpu':return second
            if first['files']!=second['files'] or not first['engineering_smoke']:
                raise ValueError('engineering completed-resume changed weights')
            return first|dict(completed_resume_identical=True)
        path=root/'extra_units'/f"{unit['index']}.json"
        if path.exists():
            result=read(path); check_extra(result,unit,plan); return result
        helper('causality_recovery').install_policy(plan['policy_path'])
        from chronaris.evaluation.application_tasks.chapter5_core import _train
        result=_train(unit,plan,root/'extra_training')
        if result['status']=='waiting_gpu':return result
        check_extra(result,unit,plan); return immutable(path,result)
    if stage.startswith('replay:'):
        row,unit=native_record(plan,stage.split(':')[1])
        with development_gpu_lock() as acquired:
            if not acquired:return dict(status='waiting_gpu')
            result=helper('device_replay').replay_record(unit,dict(index=row['index'],plan_sha256=plan['plan_sha256'],
                role='validation',sample_limit=0),root/'replay'/row['key'],data_root=plan['data_root'],registry_path=plan['registry_path'])
            if helper('chapter5_delivery_sensor').supports(row['unit']):
                result=helper('chapter5_delivery_sensor').bind_evidence(result,root/'replay'/row['key'])
            if row['unit']['method'] in ('timecma','chronos2','sensorllm_deepseek') and any(
                not r['strict']['passed'] or r['class_flips'] for r in result['rows']):
                raise ValueError('recent-model frozen inference qualification failed; inspect retained internal replay')
            return result
    if stage=='sensitivity':
        return helper('chapter5_delivery_diagnostics').sensitivity(root,plan)
    if stage=='freeze':
        # Replays validate sample identity and finite values. Strict numeric differences stay visible.
        for key in plan['replay']:
            replay=read(root/'replay'/key/'result.json'); verify(replay['files'])
            if replay['status']!='completed' or not replay['rows']:raise ValueError('missing replay qualification')
        return freeze(root,plan)
    models=require_freeze(root,plan); model_path=root/'delivery_models.json'
    binding=dict(delivery_models_sha256=sha(model_path),plan_sha256=plan['plan_sha256'],role='held_out',sample_limit=0)
    if stage=='diagnostics':
        return helper('chapter5_delivery_diagnostics').diagnostics(root,plan,models)
    if stage=='generate':
        from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import _generate_frozen_simulation
        return _generate_frozen_simulation(freeze_sha256=plan['plan_sha256'],model_freeze_sha256=sha(model_path),output_root=root/'confirmation')
    if stage=='timing-data':
        return helper('v4_mechanism_data').generate_frozen_timing_data(model_path=model_path,
            clean_root=plan['simulation_root'],output_root=root/'timing_data')
    if stage.startswith('native:'):
        row,unit=native_record(plan,stage.split(':')[1])
        result=_evaluate_native_record(unit,binding|dict(key=row['key']),root/'native'/row['key'],
            role='held_out',sample_limit=0,data_root=plan['data_root'],registry_path=plan['registry_path'])
        if helper('chapter5_delivery_sensor').supports(row['unit']) and result['status']=='completed':
            result=helper('chapter5_delivery_sensor').bind_evidence(result,root/'native'/row['key'])
        return result
    if stage.startswith('simulation:'):
        evaluate=helper('v4_simulation_confirmation_evaluation')._evaluate_simulation_unit
        with development_gpu_lock() as acquired:
            if not acquired:return dict(status='waiting_gpu')
            return evaluate(models=models,unit=models['records'][int(stage.split(':')[1])],
                freeze_sha256=plan['plan_sha256'],model_freeze_sha256=sha(model_path),output_root=root/'simulation',
                confirmation_root=root/'confirmation',include_pressure=True,
                target_initializations=helper('chapter5_delivery_diagnostics').reference_initializations(models['records'],17))
    if stage=='mechanisms':
        return helper('v4_mechanism_evaluation').evaluate_frozen_mechanisms(models=models,model_path=model_path,
            timing_root=root/'timing_data',confirmation_root=root/'confirmation',simulation_evaluation_root=root/'simulation',
            output_root=root/'mechanisms')
    return helper('chapter5_delivery_export').export(root,plan,models)


def run(root, launcher, workspace):
    """Sequential subprocesses free GPU memory; persisted receipts allow explicit restart."""
    import fcntl
    import signal
    def terminate(signum,frame):raise SystemExit(f'delivery interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,terminate)
    root=Path(root); plan=plan_at(root)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(status='running',pid=os.getpid(),phase='starting',completed=[],total=len(steps(plan)),
                   plan_sha256=plan['plan_sha256'],started_at=time.time())
        try:
            for stage in steps(plan):
                path=root/'steps'/(stage.replace(':','_')+'.json')
                if path.exists():
                    receipt=read(path)
                    if receipt['plan_sha256']!=plan['plan_sha256'] or receipt['exit_code']!=0:
                        raise ValueError('step receipt changed')
                    verify(receipt['files']); state['completed'].append(stage); continue
                log=root/'logs'/(stage.replace(':','_')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                result=root/'worker_results'/(stage.replace(':','_')+'.json')
                args=[sys.executable,str(launcher),'worker','--root',str(root),'--workspace',str(workspace),
                      '--stage',stage,'--result',str(result)]
                if stage.startswith(('engineering:','train:','replay:','native:','simulation:')) or stage in ('sensitivity','diagnostics','mechanisms'):
                    from chronaris.evaluation.application_tasks.v4_public_screen import GPU_LOCK_PATH
                    state.update(phase='waiting_gpu',next_stage=stage)
                    write(root/'state.json',state)
                    # Respect the other chapter's ownership; no polling and no process termination.
                    with open(GPU_LOCK_PATH,'a') as gpu_lock:
                        fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                    state.pop('next_stage',None)
                started=time.time(); state.update(phase=stage)
                with log.open('a') as handle:
                    child=subprocess.Popen(args,cwd=workspace,stdout=handle,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;write(root/'state.json',state)
                    try:code=child.wait()
                    except BaseException:
                        child.terminate();child.wait();raise
                state.pop('child_pid',None)
                if code:raise RuntimeError(f'{stage} exited {code}: {log}')
                value=read(result)
                if value.get('status') not in ('completed','frozen'):
                    raise ValueError(f'{stage} did not complete')
                files={str(result):sha(result)}|value.get('files',{})|execution_binding(root,plan)
                # Preserve every completed stage output without repeatedly writing large payloads.
                write(path,dict(plan_sha256=plan['plan_sha256'],stage=stage,exit_code=code,files=files,
                    started_at=started,seconds=time.time()-started,log=str(log),command=args))
                state['completed'].append(stage);write(root/'state.json',state)
            state.update(status='completed',phase='materials_generated_pending_visual_review_and_git_import',finished_at=time.time())
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}',finished_at=time.time())
            write(root/'state.json',state);raise
        write(root/'state.json',state)
    return state
