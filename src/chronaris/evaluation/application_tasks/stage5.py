"""Fixed-reference stability replication; no candidate search or outer evaluation."""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, pstdev

from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs, run_common_contract_smoke
from chronaris.evaluation.application_tasks.stage45 import metric_rows
from chronaris.evaluation.application_tasks.stage45_recipe import training_recipe
from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import seal_development_plan
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

DOMAINS = {'clare': 'contiformer', 'cogpilot': 'vehicle_only'}
SEEDS = (17, 29, 43)


def units():
    return [dict(domain=d, method=m, fold_index=f, seed=s, inherited=f == 0 and s == 17)
            for f in range(3) for s in SEEDS for d, baseline in DOMAINS.items()
            for m in ('chronaris', baseline)]


def groups():
    stages = ['cuda_validation', 'stage5_plan']
    stages += [f'stage5_execution__{d}__{mode}' for d in DOMAINS
               for mode in ('eager', 'graph', 'interrupt', 'resume', 'check')]
    stages += [f'stage5_unit__{i}' for i in range(len(units()))]
    return [[(s, 'completed')] for s in stages + ['stage5_report']]


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    for p, digest in files.items():
        if sha256_file(p) != digest:
            raise ValueError(f'stage 5 evidence changed: {p}')


def data_contract(contract):
    return {k: v for k, v in contract.items() if k not in ('source_code_sha256', 'contract_sha256')}


def plan(config):
    root = Path(config['root'])
    receipt = read(root/'pipeline_state.json')['completed']['cuda_validation']
    verify_files({receipt['path']: receipt['sha256']})
    validation = _validation_evidence(receipt['path'])
    acceptance = read(config['stage5_acceptance'])
    if acceptance['status'] != 'completed' or acceptance['model_units'] != 25 or acceptance['confirmation_opened']:
        raise ValueError('stage 5 requires the completed development comparison')
    verify_files(acceptance['sources'])
    state = read(Path(acceptance['source_root'])/'pipeline_state.json')
    contracts, budgets, historical = {}, {}, {}
    for domain, baseline in DOMAINS.items():
        for fold_index in range(3):
            _, _, fold, digest, targets, definitions, context, observations = contract_development_inputs(
                domain, data_root=config['data_root'], registry_path=config['registry_path'], full=True, fold_index=fold_index)
            for seed in SEEDS:
                key = f'{domain}/{fold_index}/{seed}'
                contract = build_common_contract(domain=domain, fold=fold, observations=observations,
                    targets=targets, definitions=definitions, context=context, data_manifest_sha256=digest,
                    scope='development_comparison', seed=seed)
                path = root/'contracts'/f'{domain}_{fold_index}_{seed}.json'
                if path.exists() and read(path) != contract:
                    raise ValueError('stage 5 contract changed; use a new root')
                write_result(path, contract)
                contracts[key] = dict(path=str(path), sha256=sha256_file(path), contract_sha256=contract['contract_sha256'])
                for method in ('chronaris', baseline):
                    candidate, pre, guided, _, _ = training_recipe('stage4_reference', method=method, full=True,
                        seed=seed, digest=digest, train_count=len(fold.train_sample_ids))
                    budgets[key+'/'+method] = dict(candidate=asdict(candidate), pretraining=asdict(pre), finetuning=asdict(guided))
                    if fold_index or seed != 17:
                        continue
                    receipt = state['completed'][f'comparison_unit__{domain}__{method}']
                    if acceptance['sources'].get(receipt['path']) != receipt['sha256']:
                        raise ValueError('historical unit is outside the stage 4 acceptance')
                    previous = read(receipt['path'])
                    if previous['status'] != 'completed' or previous['confirmation_opened']:
                        raise ValueError('invalid historical development unit')
                    files = {receipt['path']: receipt['sha256']}
                    for row in previous['results']:
                        record_path = Path(row['consumers']['manifest_path']).parent.parent/'common_contract.json'
                        record = read(record_path)
                        if data_contract(record['contract']) != data_contract(contract):
                            raise ValueError('historical input, task or downstream policy differs')
                        declaration = record['declaration']
                        files.update(declaration['evidence_files'])
                        files[str(record_path)] = row['contract_file_sha256']
                        files[declaration['checkpoint_path']] = declaration['checkpoint_sha256']
                        for component in row['consumers']['components'].values():
                            for kind in ('model', 'result'):
                                files[component[kind+'_path']] = component[kind+'_sha256']
                    verify_files(files)
                    historical[domain+'/'+method] = dict(receipt=receipt, files=files,
                        original_source_code_sha256=previous['source_code_sha256'], new_encoder_updates=0)
    frozen = seal_development_plan(dict(status='completed', format='chronaris.stage5_stability.v1',
        source_code_sha256=v4_workflow_source_sha256(), validation_files=validation,
        acceptance_sha256=sha256_file(config['stage5_acceptance']), contracts=contracts, budgets=budgets,
        units=units(), historical=historical, model_units=36, inherited_units=4, new_units=32,
        representation_routes=72, candidate_search=False, confirmation_opened=False,
        configuration='stage4_reference; original loss selector; no stage45 changes adopted',
        completion_rule='all fixed units and source checks complete, irrespective of method ranking',
        statistics='paired within fold/seed; descriptive variability, not 9 independent subject cohorts'))
    path = root/'stage5_plan.json'
    if path.exists() and read(path) != frozen:
        raise ValueError('stage 5 plan changed; use a new root')
    return write_result(path, frozen)


def verified_plan(config):
    root = Path(config['root'])
    frozen = read(root/'stage5_plan.json')
    receipt = read(root/'pipeline_state.json')['completed']['stage5_plan']
    verify_files({receipt['path']: receipt['sha256']})
    if read(receipt['path']) != frozen:
        raise ValueError('stage 5 plan differs from its completion receipt')
    if (seal_development_plan({k: v for k, v in frozen.items() if k != 'plan_sha256'}) != frozen
            or frozen['source_code_sha256'] != v4_workflow_source_sha256()
            or frozen['units'] != units() or frozen['confirmation_opened']):
        raise ValueError('stage 5 plan or source changed')
    verify_files({v['path']: v['sha256'] for v in frozen['contracts'].values()})
    return frozen


def progress_report(root, frozen, *, complete=False):
    root = Path(root)
    rows, costs, sources = [], defaultdict(list), {}
    inherited = 0
    for index, unit in enumerate(frozen['units']):
        path = root/'units'/f'{index}.json'
        if not path.exists():
            continue
        result = read(path)
        if result['status'] != 'completed' or result['unit'] != unit:
            raise ValueError('invalid stage 5 completed unit')
        verify_files(result['evidence_files'])
        sources[str(path)] = sha256_file(path)
        inherited += int(unit['inherited'])
        if not unit['inherited']:
            costs[unit['domain']+'/'+unit['method']].append(result['total_seconds'])
        for row in metric_rows(result):
            rows.append(row | {k: unit[k] for k in ('method', 'fold_index', 'seed', 'inherited')})
    if complete and (len(sources) != 36 or len(rows) != 144):
        raise ValueError('stage 5 stability matrix incomplete')
    if complete:
        completed = read(root/'pipeline_state.json')['completed']
        for index in range(len(frozen['units'])):
            receipt = completed[f'stage5_unit__{index}']
            verify_files({receipt['path']: receipt['sha256']})
            if read(receipt['path']) != read(root/'units'/f'{index}.json'):
                raise ValueError('stage 5 unit differs from its completion receipt')
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['domain'], row['method'], row['route'], row['task'])].append(row['value'])
    summary = [dict(domain=k[0], method=k[1], route=k[2], task=k[3], settings=len(v),
                    mean=mean(v), std_across_settings=pstdev(v), minimum=min(v), maximum=max(v))
               for k, v in grouped.items()]
    return write_result(root/('stage5_report.json' if complete else 'progress_summary.json'),
        dict(status='completed' if complete else 'running', completed_units=len(sources), inherited_units=inherited,
             new_completed_units=len(sources)-inherited, total_units=36, rows=rows, descriptive_summary=summary,
             measured_new_unit_seconds=dict(costs), source_receipts=sources,
             plan_sha256=frozen['plan_sha256'], confirmation_opened=False, candidate_search=False,
             formal_evaluation_ready=False, next_step='complete all-method formal integration and model freeze'))


def run_stage5(stage, config):
    root = Path(config['root'])
    if stage == 'stage5_plan':
        return plan(config)
    frozen = verified_plan(config)
    if stage.startswith('stage5_execution__'):
        from chronaris.evaluation.application_tasks.stage45_performance import performance_entry, compare_execution
        _, domain, mode = stage.split('__')
        if mode == 'check':
            path = root/'performance'/domain/'stage4_reference'
            result = compare_execution(path, revised=True)
            write_result(path/'check.json', result)
            if not result['passed']:
                raise ValueError('stage 5 execution or independent resume failed; preserve evidence')
            return result
        return performance_entry(config, domain, 'stage4_reference', mode)
    if stage == 'stage5_report':
        for source in frozen['historical'].values():
            verify_files(source['files'])
        return progress_report(root, frozen, complete=True)
    if not stage.startswith('stage5_unit__'):
        raise ValueError('unknown stage 5 step')
    index = int(stage.split('__')[1])
    unit = frozen['units'][index]
    key = f"{unit['domain']}/{unit['fold_index']}/{unit['seed']}"
    if unit['inherited']:
        source = frozen['historical'][unit['domain']+'/'+unit['method']]
        verify_files(source['files'])
        result = read(source['receipt']['path'])
        files = source['files']
    else:
        check = root/'performance'/unit['domain']/'stage4_reference/check.json'
        state = read(root/'pipeline_state.json')
        receipt = state['completed'][f"stage5_execution__{unit['domain']}__check"]
        verify_files({receipt['path']: receipt['sha256']})
        if read(receipt['path']) != read(check) or not read(check)['passed']:
            raise ValueError('graph qualification changed')
        result = run_common_contract_smoke(domain=unit['domain'], methods=(unit['method'],),
            output_root=root/'training'/str(index), full=True, recipe='stage4_reference',
            fold_index=unit['fold_index'], seed=unit['seed'], cuda_graph_recurrence=unit['method'] == 'chronaris',
            expected_contract_sha256=frozen['contracts'][key]['contract_sha256'],
            data_root=config['data_root'], registry_path=config['registry_path'])
        if result['status'] == 'waiting_gpu':
            return result
        files = {str(check): sha256_file(check)}
        for row in result['results']:
            for checkpoint in ('best_checkpoint_path', 'last_checkpoint_path'):
                path = row['training'][checkpoint]
                files[path] = sha256_file(path)
            for component in row['consumers']['components'].values():
                for kind in ('model', 'result'):
                    files[component[kind+'_path']] = component[kind+'_sha256']
    saved = write_result(root/'units'/f'{index}.json', result | dict(unit=unit, evidence_files=files,
        historical_result_inherited=unit['inherited'], stage5_plan_sha256=frozen['plan_sha256']))
    progress_report(root, frozen)
    return saved
