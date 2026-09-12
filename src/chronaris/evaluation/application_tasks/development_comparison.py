"""Complete first-fold comparison through the existing shared contracts and pipeline."""
import json
import math
from pathlib import Path
import time

from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs, run_common_contract_smoke
from chronaris.evaluation.application_tasks.recent_model_smoke import run_recent_model_smoke
from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

ASSETS = '/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets.json'
SENSOR_ASSETS = '/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets_sensor.json'
BASELINES = ('chronaris', 'physiology_only', 'vehicle_only', 'naive_time_sync', 'mult', 'contiformer')


def comparison_units():
    return [dict(domain=d, method=m) for d in ('clare', 'cogpilot', 'dingxin')
            for m in BASELINES+('timecma', 'chronos2')+ (('sensorllm_deepseek',) if d=='clare' else ())]


def comparison_groups():
    return [[('cuda_validation', 'completed')], [('comparison_plan', 'completed')]] + [
        [(f"comparison_unit__{u['domain']}__{u['method']}", 'completed')] for u in comparison_units()] + [
        [('comparison_costs', 'completed')]]


def comparison_budget(method, train_windows):
    if train_windows <= 0:
        raise ValueError('comparison requires training windows')
    if method in BASELINES:
        return dict(pretraining_updates=max(300, math.ceil(train_windows/16)) if method!='naive_time_sync' else 0,
            head_warmup_updates=50 if method!='naive_time_sync' else 0,
            joint_updates=max(200, math.ceil(train_windows/16)) if method!='naive_time_sync' else 0,
            effective_batch_size=16, seed=17)
    return dict(adaptation_updates=max(200, math.ceil(train_windows/(4 if method=='timecma' else 1))),
        history_alignment_updates=max(200, train_windows) if method=='sensorllm_deepseek' else 0,
        checkpoint_interval=25, replay_updates=1, seed=17, full_training_role=True)


def _common(config):
    return {k: config[k] for k in ('data_root', 'registry_path')}


def _plan(config):
    root = Path(config['root'])/'comparison'
    contracts, counts = {}, {}
    for domain in ('clare', 'cogpilot', 'dingxin'):
        _, _, fold, digest, targets, definitions, context, raw = contract_development_inputs(domain, **_common(config), full=True)
        contract = build_common_contract(domain=domain, fold=fold, observations=raw, targets=targets,
            definitions=definitions, context=context, data_manifest_sha256=digest, scope='development_comparison')
        path = root/domain/'data_contract.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('complete development contract changed')
        write_result(path, contract)
        contracts[domain] = dict(path=str(path), sha256=sha256_file(path), contract_sha256=contract['contract_sha256'])
        counts[domain] = {role: len(b.sample_ids) for role, b in raw.items()}
    plan = seal_development_plan(dict(format='chronaris.stage4_comparison.v1', status='completed',
        source_code_sha256=v4_workflow_source_sha256(), seed=17, fold_index=0, contracts=contracts, sample_counts=counts,
        units=[u | {'budget': comparison_budget(u['method'], counts[u['domain']]['train'])} for u in comparison_units()],
        expected_routes=43, confirmation_opened=False, early_stopping=False,
        numerical_policy='unchanged stage2 consumers; fixed Dingxin C=1/alpha=1; retain solver warnings and fallback',
        comparison_policy='compare matching contract, supervision stratum and actual training task set; disclose external pretraining',
        sensor_scope='CLARE only; DeepSeek Llama 8B variant; history alignment and classification adaptation',
        sequence_consumers='linear plus public MiniROCKET; window-end features linear only'))
    path = root/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('comparison plan changed; use a new root')
    if config.get('comparison_parent'):
        from chronaris.evaluation.application_tasks.comparison_reuse import prepare_parent_reuse
        prepare_parent_reuse(config, plan)
    return write_result(path, plan)


def _verify_plan(config):
    path = Path(config['root'])/'comparison/plan.json'
    plan = json.loads(path.read_text())
    unsealed = {k:v for k,v in plan.items() if k!='plan_sha256'}
    if seal_development_plan(unsealed) != plan or plan['source_code_sha256'] != v4_workflow_source_sha256():
        raise ValueError('comparison plan or source changed')
    for value in plan['contracts'].values():
        if sha256_file(value['path']) != value['sha256']:
            raise ValueError('comparison input contract changed')
    return plan


def _costs(config, plan):
    root = Path(config['root'])
    rows, sources = [], {}
    for unit in plan['units']:
        stage = f"comparison_unit__{unit['domain']}__{unit['method']}"
        state = json.loads((root/'pipeline_state.json').read_text())
        receipt = state['completed'][stage]
        if sha256_file(receipt['path']) != receipt['sha256']:
            raise ValueError('completed comparison receipt changed')
        result = json.loads(Path(receipt['path']).read_text())
        sources[receipt['path']] = receipt['sha256']
        for row in result['results']:
            evaluation = row.get('evaluation', row)
            components = evaluation['consumers']['components']
            for item in components.values():
                if any(sha256_file(item[k+'_path']) != item[k+'_sha256'] for k in ('model', 'result')):
                    raise ValueError('comparison consumer evidence changed')
            training = row['training'] or {}
            seconds = (training.get('first_update', {}).get('seconds', 0)+training.get('primary', {}).get('seconds', 0)
                       if 'primary' in training else training.get('training_elapsed_s', 0))
            rows.append(dict(domain=unit['domain'], method=unit['method'], route=row['route'],
                comparison_key=evaluation['comparison_key'], training_seconds=seconds,
                training_updates=training.get('primary', {}).get('updates', training.get('optimizer_updates', 0)),
                export_seconds=row.get('export_seconds', row.get('export_cost', {}).get('seconds')),
                downstream_fit_seconds=sum(v['fit_elapsed_s'] for v in components.values()),
                peak_cuda_bytes=row.get('peak_cuda_bytes', row.get('export_cost', {}).get('peak_cuda_bytes')),
                nonconstant_dimensions=row['nonconstant_dimensions'], consumers=components))
    if len(rows) != plan['expected_routes']:
        raise ValueError('incomplete development comparison routes')
    # ponytail: extrapolate identical units only; formal folds/budgets are not frozen in stage 4.
    parent_costs = []
    if config.get('comparison_parent'):
        from chronaris.evaluation.application_tasks.comparison_reuse import verify_reuse_inventory
        inventory = verify_reuse_inventory(config)
        parent_costs = [Path(p) for p in inventory['files'] if '/comparison/attempt_costs/' in p and p.endswith('.json')]
    unit_costs = []
    for unit in plan['units']:
        directory = root/'comparison/attempt_costs'
        paths = sorted(directory.glob(f"{unit['domain']}__{unit['method']}__*.json"))
        paths += [p for p in parent_costs if p.name.startswith(f"{unit['domain']}__{unit['method']}__")]
        attempts = [json.loads(p.read_text()) for p in paths]
        unit_costs.append(unit | dict(observed_attempt_seconds=sum(x['seconds'] for x in attempts),
            attempt_count=len(attempts), attempt_files=[str(p) for p in paths],
            formal_unit_count=None, formal_total_seconds=None))
    return write_result(root/'comparison/cost_report.json', dict(status='completed', scope='development_comparison',
        rows=rows, unit_costs=unit_costs, source_receipts=sources, plan_sha256=plan['plan_sha256'],
        confirmation_opened=False, remaining_stage4_units=0,
        extrapolation='For an unchanged unit, multiply observed attempt seconds by additional unit count; formal matrix awaits stage 5',
        diagnostic_only='single development fold, seed 17; no automatic model adoption'))


def run_comparison_step(stage, config):
    state_path = Path(config['root'])/'pipeline_state.json'
    completed = json.loads(state_path.read_text()).get('completed', {}) if state_path.exists() else {}
    validation = completed.get('cuda_validation', {}).get('path',
        str(Path(config['root'])/'cuda_validation/attempt_1/validation_receipt.json'))
    _validation_evidence(validation)
    if stage == 'comparison_plan':
        return _plan(config)
    plan = _verify_plan(config)
    if stage == 'comparison_costs':
        return _costs(config, plan)
    _, domain, method = stage.split('__')
    if dict(domain=domain, method=method) not in comparison_units():
        raise ValueError('unit not in fixed development comparison')
    root = Path(config['root'])/'comparison'
    started, status = time.perf_counter(), 'failed'
    try:
        result = None
        if config.get('comparison_parent'):
            from chronaris.evaluation.application_tasks.comparison_reuse import revalidate_completed_unit
            result = revalidate_completed_unit(stage, config, plan)
        if result is not None:
            pass
        elif method in BASELINES:
            result = run_common_contract_smoke(domain=domain, methods=(method,), full=True,
                output_root=root/'original'/method, **_common(config))
        else:
            result = run_recent_model_smoke(domain=domain, method=method, full=True,
                assets_path=SENSOR_ASSETS if method=='sensorllm_deepseek' else ASSETS,
                output_root=root/'recent', **_common(config))
        if result['status'] == 'completed' and result['contract_sha256'] != plan['contracts'][domain]['contract_sha256']:
            raise ValueError('unit used a different common contract')
        status = result['status']
        return result
    finally:
        write_result(root/'attempt_costs'/f'{domain}__{method}__{time.time_ns()}.json',
            dict(domain=domain, method=method, status=status, seconds=time.perf_counter()-started))
