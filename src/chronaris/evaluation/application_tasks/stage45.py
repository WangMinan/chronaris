"""Bounded development optimization using the existing pipeline and common contract."""
import json
from pathlib import Path
import time

from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs, run_common_contract_smoke
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract
from chronaris.evaluation.application_tasks.stage45_diagnostics import run_branch_probes, fidelity_trigger
from chronaris.evaluation.application_tasks.stage45_performance import performance_entry, compare_execution
from chronaris.evaluation.application_tasks.stage45_recipe import RECIPES
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock, seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

DOMAINS = ('clare', 'cogpilot')
BASELINES = {'clare': ('contiformer', 'physiology_only'), 'cogpilot': ('vehicle_only', 'mult')}
REVIEW_SETTINGS = ((0, 29), (1, 43))


def screen_units():
    return [dict(domain=d, method=m, recipe=r) for d in DOMAINS
        for m in ('chronaris',)+BASELINES[d]
        for r in (RECIPES[1:] if m == 'chronaris' else ('pretraining_lr', 'finetuning_lr'))]


def stage45_groups():
    stages = ['cuda_validation', 'stage45_plan']
    stages += [f'stage45_probes__{d}' for d in DOMAINS]
    for d in (*DOMAINS, 'dingxin'):
        for r in ('stage4_reference', 'thesis_reference'):
            stages += [f'stage45_perf__{d}__{r}__{m}' for m in ('eager', 'graph', 'interrupt', 'resume', 'check')]
    stages += ['stage45_screen__'+str(i) for i in range(len(screen_units()))]
    stages += ['stage45_select']
    # Two finalists at most; matching new-setting references and strong controls.
    stages += [f'stage45_review__{d}__{setting}__{slot}' for setting in range(2) for d in DOMAINS
               for slot in ('reference', 'baseline0', 'baseline1', 'finalist0', 'finalist1')]
    stages += [f'stage45_guard__{slot}' for slot in range(2)]
    stages += ['stage45_report']
    return [[(s, 'completed')] for s in stages]


def _read(path):
    return json.loads(Path(path).read_text())


def parent_result(config, domain, method):
    state = _read(Path(config['stage45_parent'])/'pipeline_state.json')
    receipt = state['completed'][f'comparison_unit__{domain}__{method}']
    if sha256_file(receipt['path']) != receipt['sha256']:
        raise ValueError('stage 4 receipt changed')
    result = _read(receipt['path'])
    for row in result['results']:
        for item in row['consumers']['components'].values():
            for kind in ('model', 'result'):
                if sha256_file(item[kind+'_path']) != item[kind+'_sha256']:
                    raise ValueError('stage 4 consumer changed')
    return result


def _inputs(config, domain, fold_index=0, full=True):
    return contract_development_inputs(domain, data_root=config['data_root'],
        registry_path=config['registry_path'], full=full, fold_index=fold_index)


def make_plan(config):
    acceptance = _read(config['stage45_acceptance'])
    if (acceptance['status'] != 'completed' or acceptance['model_units'] != 25
        or acceptance['representation_routes'] != 43 or acceptance['confirmation_opened']
        or Path(acceptance['source_root']).resolve() != Path(config['stage45_parent']).resolve()):
        raise ValueError('stage 4.5 requires the completed preserved stage 4 acceptance')
    for path, digest in acceptance['sources'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'preserved stage 4 source changed: {path}')
    contracts = {}
    for domain in (*DOMAINS, 'dingxin'):
        _, _, fold, digest, targets, definitions, context, observations = _inputs(config, domain)
        contract = build_common_contract(domain=domain, fold=fold, data_manifest_sha256=digest,
            targets=targets, definitions=definitions, context=context, observations=observations, scope='development_comparison')
        parent = parent_result(config, domain, 'chronaris')
        manifest = _read(parent['results'][0]['consumers']['manifest_path'])
        # The native consumer manifest binds the full role/target history via its sibling common contract.
        old_contract_path = Path(parent['results'][0]['consumers']['manifest_path']).parent.parent/'common_contract.json'
        old = _read(old_contract_path)['contract']
        def data_only(c):
            return {k:v for k,v in c.items() if k not in ('source_code_sha256', 'contract_sha256')}
        if data_only(contract) != data_only(old):
            raise ValueError('stage 4.5 altered the common data, task or consumer contract')
        from chronaris.evaluation.application_tasks.stage45_resume import verify_parent_contract
        verify_parent_contract(config, contract)
        contracts[f'{domain}/0/17'] = contract['contract_sha256']
        if domain in DOMAINS:
            for fold_index, seed in REVIEW_SETTINGS:
                if fold_index:
                    _, _, fold, digest, targets, definitions, context, observations = _inputs(config, domain, fold_index)
                reviewed = build_common_contract(domain=domain, fold=fold, data_manifest_sha256=digest,
                    targets=targets, definitions=definitions, context=context, observations=observations,
                    scope='development_comparison', seed=seed)
                verify_parent_contract(config, reviewed)
                contracts[f'{domain}/{fold_index}/{seed}'] = reviewed['contract_sha256']
    plan = seal_development_plan(dict(status='completed', format='chronaris.stage45.v1',
        source_code_sha256=v4_workflow_source_sha256(), contracts=contracts, screen_units=screen_units(),
        seed=17, fold_index=0, review_settings=REVIEW_SETTINGS, finalist_limit=2,
        budget_hours=config.get('stage45_budget_hours',72), model_adoption_automatic=False, confirmation_opened=False,
        selection='at least two meaningful domain-route gains; all other public tasks within fixed guardrails',
        fidelity_trigger='public projection F1 loss >=0.02 or RMSE increase >=5 percent',
        references='read-only stage4 checkpoints/results; changed recipes train from scratch in new roots',
        source_acceptance_sha256=sha256_file(config['stage45_acceptance'])))
    if config.get('stage45_resume_parent'):
        from chronaris.evaluation.application_tasks.stage45_resume import prepare_resume
        recovery = prepare_resume(config)
        plan = seal_development_plan({k:v for k,v in plan.items() if k != 'plan_sha256'} |
            {'recovery_evidence_sha256': recovery['evidence_sha256']})
    return write_result(Path(config['root'])/'stage45_plan.json', plan)


def metric_rows(result):
    rows = []
    for route in result.get('results', []):
        if 'route' not in route or 'training' not in route:
            continue
        for item in route['consumers']['components']['linear']['task_summary']['validation']:
            rows.append(dict(domain=result['domain'], route=route['route'], **item))
    return rows


def compare_candidate(candidate, reference):
    refs = {(r['domain'], r['route'], r['task']): r for r in reference}
    keys = {(r['domain'], r['route'], r['task']) for r in candidate}
    if not refs or len(refs) != len(reference) or len(keys) != len(candidate):
        raise ValueError('candidate/reference task coverage is empty or duplicated')
    rows, gains = [], set()
    safe = True
    if {(r['domain'], r['route'], r['task']) for r in candidate} != set(refs):
        raise ValueError('candidate task coverage differs from reference')
    for row in candidate:
        ref = refs[(row['domain'], row['route'], row['task'])]
        a, b = ref['value'], row['value']
        if a is None or b is None:
            safe = False
            continue
        improvement = b-a if row['metric'] == 'macro_f1' else (a-b)/a if a > 0 else (0. if b == 0 else -1.)
        threshold, floor = (.02, -.01) if row['metric'] == 'macro_f1' else (.05, -.05)
        safe &= improvement >= floor
        if improvement >= threshold:
            gains.add((row['domain'], row['route']))
        rows.append(row | dict(reference=a, improvement=improvement, meaningful=improvement >= threshold))
    return dict(eligible=safe and len(gains) >= 2, safe=safe, meaningful_domain_routes=len(gains), rows=rows,
                score=sum(r['improvement'] for r in rows)/len(rows) if rows else 0.)


def _result_path(config, phase, key):
    return Path(config['root'])/phase/(key+'.json')


def select_candidates(config):
    reference = [r for d in DOMAINS for r in metric_rows(parent_result(config, d, 'chronaris'))]
    comparisons = {}
    for recipe in RECIPES[1:]:
        records = [_read(_result_path(config, 'screen', str(i))) for i,u in enumerate(screen_units())
                   if u['method'] == 'chronaris' and u['recipe'] == recipe]
        if any(r.get('skipped') for r in records):
            comparisons[recipe] = dict(eligible=False, reason='prespecified fidelity trigger not met')
        else:
            comparisons[recipe] = compare_candidate([m for r in records for m in metric_rows(r)], reference)
    ranked = sorted((r for r,c in comparisons.items() if c['eligible']), key=lambda r: (-comparisons[r]['score'], r))
    return write_result(Path(config['root'])/'selection.json', dict(status='completed', finalists=ranked[:2],
        comparisons=comparisons, candidate_selection_only=True, confirmation_opened=False))


def _run_unit(config, unit, root):
    domain, method, recipe = (unit[k] for k in ('domain', 'method', 'recipe'))
    evidence = None
    if config.get('stage45_resume_evidence'):
        from chronaris.evaluation.application_tasks.stage45_resume import read_evidence
        evidence = read_evidence(config)
    accelerated = evidence is not None and evidence.get('format') == 'chronaris.stage45_execution_recovery.v2'
    if method == 'chronaris':
        execution = 'thesis_reference' if recipe == 'thesis_reference' else 'stage4_reference'
        check = _read(evidence['qualification_checks'][f'{domain}/{execution}'] if accelerated else
                      Path(config['root'])/'performance'/domain/execution/'check.json')
        graph = check['passed']
    else:
        graph = False
    fine_graph = True if accelerated and method == 'chronaris' else None
    if (not accelerated and config.get('stage45_resume_parent') and (domain,method,recipe,unit.get('fold_index',0),unit.get('seed',17))
        == ('clare','chronaris','thesis_reference',0,17)):
        from chronaris.evaluation.application_tasks.stage45_resume import read_evidence
        graph = read_evidence(config)['resume_pretraining_graph']
        fine_graph = False if graph else None
    pretraining_source = None
    if recipe == 'finetuning_lr' and unit.get('seed',17) == 17 and unit.get('fold_index',0) == 0:
        parent = parent_result(config, domain, method)
        pretraining_source = next(r['training']['best_checkpoint_path'] for r in parent['results'] if r['route']=='self_supervised')
        import torch
        if accelerated:
            from chronaris.evaluation.application_tasks.stage45_acceleration import migrated_reference
            pretraining_source = migrated_reference(config, pretraining_source, graph=method == 'chronaris')
        graph = bool(torch.load(pretraining_source, map_location='cpu', weights_only=True)['config']['cuda_graph_recurrence'])
    expected = _read(Path(config['root'])/'stage45_plan.json')['contracts'][
        f"{domain}/{unit.get('fold_index',0)}/{unit.get('seed',17)}"]
    started = time.perf_counter()
    result = run_common_contract_smoke(domain=domain, methods=(method,), recipe=recipe,
        seed=unit.get('seed',17), fold_index=unit.get('fold_index',0), full=True, output_root=root,
        cuda_graph_recurrence=graph, diagnostic_snapshots=method == 'chronaris', pretraining_source=pretraining_source,
        expected_contract_sha256=expected, finetuning_graph_recurrence=fine_graph,
        data_root=config['data_root'], registry_path=config['registry_path'])
    return result | dict(unit=unit, attempt_seconds=time.perf_counter()-started, graph_execution=graph)


def run_stage45(stage, config):
    root = Path(config['root'])
    if stage == 'stage45_plan':
        return make_plan(config)
    plan = _read(root/'stage45_plan.json')
    if (seal_development_plan({k:v for k,v in plan.items() if k != 'plan_sha256'}) != plan
        or plan['source_code_sha256'] != v4_workflow_source_sha256()):
        raise ValueError('stage 4.5 source changed')
    from chronaris.evaluation.application_tasks.stage45_resume import inherited_step
    inherited = inherited_step(stage, config)
    if inherited is not None:
        return inherited
    parts = stage.split('__')
    if parts[0] == 'stage45_probes':
        domain = parts[1]
        provider, _, fold, _, targets, definitions, context, _ = _inputs(config, domain)
        result, probes = parent_result(config, domain, 'chronaris'), []
        with development_gpu_lock() as acquired:
            if not acquired:
                return dict(status='waiting_gpu')
            for row in result['results']:
                probes.append(run_branch_probes(checkpoint=row['training']['best_checkpoint_path'], route=row['route'],
                    provider=provider, fold=fold, targets=targets, definitions=definitions, context=context,
                    output_root=root/'probes'/domain/row['route']))
        return write_result(root/'probes'/domain/'summary.json', dict(status='completed', probes=probes,
            trigger=fidelity_trigger(probes), confirmation_opened=False))
    if parts[0] == 'stage45_perf':
        _, domain, recipe, mode = parts
        if mode == 'check':
            directory = root/'performance'/domain/recipe
            return write_result(directory/'check.json', compare_execution(directory))
        return performance_entry(config, domain, recipe, mode)
    if parts[0] == 'stage45_screen':
        index = int(parts[1]); unit = screen_units()[index]
        enabled = any(_read(root/'probes'/d/'summary.json')['trigger']['enabled'] for d in DOMAINS)
        result = (dict(status='completed', skipped=True, reason='fidelity trigger not met', unit=unit)
            if unit['recipe'] == 'single_stream_fidelity' and not enabled else
            _run_unit(config, unit, root/'units'/'screen'/str(index)))
        return write_result(_result_path(config, 'screen', str(index)), result)
    if stage == 'stage45_select':
        return select_candidates(config)
    if parts[0] in ('stage45_review', 'stage45_guard'):
        finalists = _read(root/'selection.json')['finalists']
        if parts[0] == 'stage45_guard':
            slot = int(parts[1]); domain, fold, seed, method = 'dingxin', 0, 17, 'chronaris'
            recipe = finalists[slot] if slot < len(finalists) else None
        else:
            _, domain, setting, slot = parts
            fold, seed = REVIEW_SETTINGS[int(setting)]
            method, recipe = 'chronaris', 'stage4_reference'
            if slot.startswith('finalist'):
                index = int(slot[-1]); recipe = finalists[index] if index < len(finalists) else None
            elif slot.startswith('baseline'):
                method = BASELINES[domain][int(slot[-1])]
                # Same public recipe opportunity, selected before new folds/seeds are read.
                trials = [(u['recipe'], metric_rows(_read(_result_path(config, 'screen', str(i)))))
                    for i,u in enumerate(screen_units()) if (u['domain'],u['method']) == (domain,method)]
                reference = metric_rows(parent_result(config, domain, method))
                eligible = [(r,compare_candidate(rows,reference)) for r,rows in trials]
                eligible = [(r,c) for r,c in eligible if c['safe'] and c['score'] > 0]
                if eligible:
                    recipe = sorted(eligible,key=lambda item:(-item[1]['score'],item[0]))[0][0]
        if recipe is None or not finalists:
            result = dict(status='completed', skipped=True, reason='no eligible finalist for this slot', confirmation_opened=False)
        else:
            unit = dict(domain=domain, method=method, recipe=recipe, fold_index=fold, seed=seed)
            result = _run_unit(config, unit, root/'units'/stage)
        return write_result(_result_path(config, 'review', stage), result)
    if stage == 'stage45_report':
        selection = _read(root/'selection.json')
        records = [_read(p) for p in sorted((root/'review').glob('*.json'))]
        verification = {}
        for recipe in selection['finalists']:
            settings = []
            for fold,seed in REVIEW_SETTINGS:
                reference = [m for r in records if r.get('unit',{}).get('recipe') == 'stage4_reference'
                    and r['unit']['method'] == 'chronaris' and (r['unit']['fold_index'],r['unit']['seed']) == (fold,seed)
                    for m in metric_rows(r)]
                candidate = [m for r in records if r.get('unit',{}).get('recipe') == recipe
                    and r['unit']['method'] == 'chronaris' and r['unit']['domain'] in DOMAINS and (r['unit']['fold_index'],r['unit']['seed']) == (fold,seed)
                    for m in metric_rows(r)]
                settings.append(dict(fold_index=fold, seed=seed, **compare_candidate(candidate, reference)))
            verification[recipe] = settings
        return write_result(root/'stage45_report.json', dict(status='completed', selection=selection,
            development_verification=verification, review_results=records,
            screen_results=[_read(p) for p in sorted((root/'screen').glob('*.json'))],
            worker_attempt_costs=[_read(p) for p in sorted((root/'attempt_costs').glob('*.json'))],
            recovery=_read(root/'recovery.json') if (root/'recovery.json').exists() else None,
            confirmation_opened=False, configuration_frozen=False,
            next_action='manual research review before stage 5; preserve all improvements and regressions'))
    raise ValueError('unknown stage 4.5 step')
