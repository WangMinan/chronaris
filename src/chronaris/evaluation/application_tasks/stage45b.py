"""Bounded representation-retention development, reusing the v4 process runner."""
from collections import defaultdict
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.checkpoint_selection import selection_split, subset_targets
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs, run_common_contract_smoke
from chronaris.evaluation.application_tasks.stage45 import metric_rows, compare_candidate
from chronaris.evaluation.application_tasks.stage45_diagnostics import run_branch_probes
from chronaris.evaluation.application_tasks.stage45b_diagnostics import normalization_trigger, checkpoint_gradients
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

DOMAINS = ('clare', 'cogpilot')
BASELINES = {'clare': 'contiformer', 'cogpilot': 'vehicle_only'}
EVIDENCE = 'docs/artifacts/runs/2026-09-21_v4-stage45-closeout/evidence.json'


def units(*, review=False):
    return [dict(domain=d, method=BASELINES[d] if variant == 'baseline' else 'chronaris', variant=variant,
                 inner_index=i, seed=29 if review else 17)
            for d in DOMAINS for variant in (('reference', 'finalist', 'baseline') if review else
                                             ('reference', 'mechanisms', 'linear_projection', 'baseline'))
            for i in (range(2) if d == 'clare' else range(1))]


def groups():
    stages = ['stage45b_plan'] + ['stage45b_diagnostics__'+d for d in DOMAINS]
    stages += [f'stage45b_screen__{i}' for i in range(len(units()))]
    stages += ['stage45b_select']
    stages += [f'stage45b_review__{i}' for i in range(len(units(review=True)))]
    stages += ['stage45b_report']
    return [[(s, 'completed')] for s in stages]


def diagnostic_sources():
    evidence = json.loads(Path(EVIDENCE).read_text())
    probes, gradients = {}, {}
    for domain, index in (('clare', '0'), ('cogpilot', '9')):
        probes[domain] = [(r['route'], r['training']['best_checkpoint_path'])
            for r in evidence['references'][domain+'/chronaris']['results'] if 'training' in r]
        records = evidence['diagnostics'][index]['records']
        record = next(r for r in records if r['route'] == 'self_supervised')
        gradients[domain] = [('self_supervised', str(Path(record['last_checkpoint']).parent/f'update_{u:06d}.pt'))
                             for u in (50, 200)]
        gradients[domain] += [(r['route'], r['best_checkpoint']) for r in records]
    return probes, gradients


def plan(config):
    root = Path(config['root'])
    from chronaris.evaluation.application_tasks.v4_configuration_freeze import _validation_evidence
    validation_files = _validation_evidence(config['stage45b_validation'])
    splits, contracts = {}, {}
    for domain in DOMAINS:
        provider, _, fold, digest, targets, definitions, context, _ = contract_development_inputs(domain,
            data_root=config['data_root'], registry_path=config['registry_path'], full=True)
        for i in (range(2) if domain == 'clare' else range(1)):
            training, evaluation, support = selection_split(fold, targets, definitions, context, inner_index=i)
            splits[f'{domain}/{i}'] = dict(training=training.to_dict(), evaluation=evaluation.to_dict(), support=support)
            for seed in (17, 29):
                contract = build_common_contract(domain=domain, fold=evaluation,
                    observations={r:provider(getattr(evaluation, r+'_sample_ids')) for r in ('train','validation')},
                    targets=subset_targets(targets, evaluation.train_sample_ids+evaluation.validation_sample_ids),
                    definitions=definitions, context=context, data_manifest_sha256=digest, scope='development_comparison', seed=seed)
                contracts[f'{domain}/{i}/{seed}'] = contract['contract_sha256']
    probes, gradients = diagnostic_sources()
    paths = {path for source in (probes, gradients) for rows in source.values() for _, path in rows}
    # The source checkpoint for a guided model is also part of its immutable lineage.
    import torch
    for path in tuple(paths):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if 'source_checkpoint_path' in payload:
            paths.add(payload['source_checkpoint_path'])
    result = dict(status='completed', format='chronaris.stage45b.v1', source_code_sha256=v4_workflow_source_sha256(),
        validation_files=validation_files, splits=splits, contracts=contracts, screen=units(), review=units(review=True), sources={p: sha256_file(p) for p in sorted(paths)},
        probes=probes, gradients=gradients, maximum_screen_config_domain_pairs=8, maximum_screen_training_runs=12,
        budget_hours=config['stage45_budget_hours'], finalist_limit=1, confirmation_opened=False,
        review_policy='one eligible candidate plus paired reference and strong baseline at seed 29',
        reused_model_updates=0, regression_selection_scale='training_target_population_standard_deviation',
        selection='equal task mean of group-mean 1-F1 and train-normalized RMSE; earliest checkpoint wins ties')
    return write_result(root/'stage45b_plan.json', result)


def diagnostics(config, domain):
    root = Path(config['root']); frozen = json.loads((root/'stage45b_plan.json').read_text())
    provider, _, fold, _, targets, definitions, context, _ = contract_development_inputs(domain,
        data_root=config['data_root'], registry_path=config['registry_path'], full=True)
    probes, gradients = [], []
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        import torch
        torch.set_num_threads(1)
        for route, checkpoint in frozen['probes'][domain]:
            if sha256_file(checkpoint) != frozen['sources'][checkpoint]:
                raise ValueError('diagnostic checkpoint changed')
            probes.append(run_branch_probes(checkpoint=checkpoint, route=route, provider=provider, fold=fold,
                targets=targets, definitions=definitions, context=context, output_root=root/'diagnostics'/domain/route,
                extended=True))
        for index, (route, checkpoint) in enumerate(frozen['gradients'][domain]):
            if sha256_file(checkpoint) != frozen['sources'][checkpoint]:
                raise ValueError('gradient checkpoint changed')
            gradients.append(checkpoint_gradients(checkpoint=checkpoint, route=route, provider=provider, fold=fold,
                targets=targets, definitions=definitions,
                output_root=dict(path=str(root/'diagnostics'/domain/'gradients'/str(index)), context=context)))
    return write_result(root/'diagnostics'/domain/'summary.json', dict(status='completed', probes=probes,
        gradients=gradients, trigger=normalization_trigger(probes), confirmation_opened=False))


def run_unit(config, index, *, review=False):
    root = Path(config['root']); phase = 'review' if review else 'screen'
    unit = units(review=review)[index]
    variant = unit['variant']
    if review:
        selected = json.loads((root/'selection.json').read_text())['finalists']
        if not selected:
            return write_result(root/phase/f'{index}.json', dict(status='completed', skipped=True, reason='no eligible candidate'))
        if variant == 'finalist':
            variant = selected[0]
    if variant == 'linear_projection':
        enabled = any(json.loads((root/'diagnostics'/d/'summary.json').read_text())['trigger']['enabled'] for d in DOMAINS)
        if not enabled:
            return write_result(root/phase/f'{index}.json', dict(status='completed', skipped=True, reason='normalization hypothesis unsupported'))
    result = run_common_contract_smoke(domain=unit['domain'], output_root=root/'units'/phase/str(index),
        full=True, methods=(unit['method'],), recipe='thesis_reference' if variant == 'mechanisms' else 'stage4_reference',
        seed=unit['seed'], selection_inner_index=unit['inner_index'],
        expected_contract_sha256=json.loads((root/'stage45b_plan.json').read_text())['contracts'][f"{unit['domain']}/{unit['inner_index']}/{unit['seed']}"],
        private_projection_kind='linear' if variant == 'linear_projection' else 'layernorm_linear',
        cuda_graph_recurrence=unit['method'] == 'chronaris', data_root=config['data_root'], registry_path=config['registry_path'])
    saved = write_result(root/phase/f'{index}.json', result | {'unit': unit, 'effective_variant': variant})
    measured = [json.loads(p.read_text()) for p in (root/phase).glob('*.json')]
    costs = {d: [r['total_seconds'] for r in measured if r.get('domain') == d and 'total_seconds' in r] for d in DOMAINS}
    write_result(root/'cost_estimate.json', dict(measured_unit_seconds=costs, first_clare_complete=bool(costs['clare']),
        remaining_screen_runs=len(units())-len(list((root/'screen').glob('*.json'))),
        domain_mean_seconds={d:sum(v)/len(v) if v else None for d,v in costs.items()},
        estimate_only=True, unmeasured_domains_not_extrapolated=True, budget_hours=config['stage45_budget_hours']))
    return saved


def averaged_rows(root, variant, *, review=False):
    collected = defaultdict(list)
    for i, unit in enumerate(units(review=review)):
        if unit['variant'] != variant:
            continue
        result = json.loads((root/('review' if review else 'screen')/f'{i}.json').read_text())
        if result.get('skipped'):
            return []
        for row in metric_rows(result):
            collected[(row['domain'], row['route'], row['task'])].append(row)
    return [rows[0] | {'value': sum(r['value'] for r in rows)/len(rows), 'inner_training_replicas': len(rows),
        'aggregation_note': 'equal reciprocal training splits; development evaluation subjects are reused'}
        for rows in collected.values()]


def select(root):
    reference = averaged_rows(root, 'reference')
    comparisons = {}
    for name in ('mechanisms', 'linear_projection'):
        candidate = averaged_rows(root, name)
        comparisons[name] = compare_candidate(candidate, reference) if candidate else dict(eligible=False, reason='diagnostic trigger not met')
    ranked = sorted((n for n,c in comparisons.items() if c['eligible']), key=lambda n: (-comparisons[n]['score'], n))
    return write_result(root/'selection.json', dict(status='completed', finalists=ranked[:1], comparisons=comparisons,
        strong_reference_rows=averaged_rows(root, 'baseline'), confirmation_opened=False))


def run_stage45b(stage, config):
    root = Path(config['root']); parts = stage.split('__')
    if stage == 'stage45b_plan':
        return plan(config)
    if parts[0] == 'stage45b_diagnostics':
        return diagnostics(config, parts[1])
    if parts[0] in ('stage45b_screen', 'stage45b_review'):
        return run_unit(config, int(parts[1]), review=parts[0] == 'stage45b_review')
    if stage == 'stage45b_select':
        return select(root)
    if stage == 'stage45b_report':
        frozen = json.loads((root/'stage45b_plan.json').read_text())
        if any(sha256_file(p) != h for p,h in frozen['sources'].items()):
            raise ValueError('historical checkpoint integrity changed')
        selection = json.loads((root/'selection.json').read_text())
        review = {v: averaged_rows(root, v, review=True) for v in ('reference', 'finalist', 'baseline')}
        return write_result(root/'stage45b_report.json', dict(status='completed', selection=selection, review=review,
            review_comparison=compare_candidate(review['finalist'], review['reference']) if selection['finalists'] else None,
            historical_checkpoints_unchanged=True, confirmation_opened=False, model_adoption_automatic=False))
    raise ValueError('unknown stage 4.5-B step')
