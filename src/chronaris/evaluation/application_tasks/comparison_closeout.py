"""Read-only acceptance of a completed comparison; write only to a new report root."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from chronaris.evaluation.application_tasks.comparison_accounting import attempt_accounting
from chronaris.evaluation.application_tasks.development_comparison import comparison_units
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_summary(evaluation):
    """Recompute field-then-group means, preserving native versus normalized error units."""
    for summary in evaluation['task_summary']:
        rows = [x for x in evaluation['group_metrics']
                if x['task'] == summary['task'] and x['status'] == 'completed']
        groups = sorted({x['group_id'] for x in rows})
        value = mean(mean(x[summary['metric']] for x in rows if x['group_id'] == g) for g in groups) if groups else None
        require(summary['group_count'] == len(groups) and
                ((value is None and summary['value'] is None) or
                 (value is not None and math.isclose(value, summary['value'], rel_tol=1e-12, abs_tol=1e-12))),
                'grouped metric summary changed')


def learning_summary(row):
    training = row['training'] or {}
    recent = 'primary' in training
    if recent:
        history = training['primary']['history']
        require(training['resume_identical'] and
                training['primary']['state_sha256'] == training['replay']['state_sha256'] and
                history == training['replay']['history'], 'adapter resume mismatch')
        loss = [x['loss'] for x in history]
        gradients = [x['gradient_norm'] for x in history]
        updates = training['primary']['updates']
        require(len(history) == updates and all(x['parameters_with_gradient'] > 0 for x in history),
                'adapter update or gradient coverage missing')
        start, end = mean(loss[:min(20, len(loss))]), mean(loss[-min(20, len(loss)):])
    else:
        history = training.get('epoch_rows', [])
        gradients = [x.get('mean_gradient_norm_before_clip', x.get('gradient_norm_before_clip_last_step')) for x in history]
        gradients = [x for x in gradients if x is not None]
        loss = [x.get('public_selection_loss', x.get('validation_selection_loss')) for x in history]
        loss = [x for x in loss if x is not None]
        updates = training.get('optimizer_updates', 0)
        start, end = (loss[0], loss[-1]) if loss else (None, None)
        if training:
            require(training.get('status', training.get('training_status')) == 'completed' and row['resume_identical'], 'original training incomplete')
    require(all(math.isfinite(x) for x in loss+gradients), 'nonfinite learning history')
    if updates:
        require(gradients and max(gradients) > 0, 'no measured learning gradient')
    require(row['nonconstant_dimensions'] > 0, 'constant representations')
    return dict(updates=updates, best_update=training.get('best_update'),
        parameters_receiving_gradient=max((x['parameters_with_gradient'] for x in history), default=0) if recent else None,
        loss_kind='first_last_20_training_mean' if recent else 'first_last_validation_selection',
        initial_loss=start, final_loss=end, max_gradient=max(gradients) if gradients else None,
        nonconstant_dimensions=row['nonconstant_dimensions'],
        frozen_or_untrained=not updates, resume_verified=bool(training.get('resume_identical', row.get('resume_identical', True))))


def budget_scenarios(units):
    scenarios = []
    # ponytail: first-fold timings only; 30% is a planning allowance, not a measured confidence interval.
    for name, public_folds, reuse in [('development_three_folds_three_seeds', 3, True),
                                      ('confirmation_five_folds_three_seeds', 5, False)]:
        rows = []
        for unit in units:
            count = (1 if unit['domain'] == 'dingxin' else public_folds)*3
            remaining = count-int(reuse)
            rows.append(dict(domain=unit['domain'], method=unit['method'], tasks=unit['tasks'],
                routes=unit['routes'], folds=1 if unit['domain']=='dingxin' else public_folds,
                seeds=[17,29,43], total_units=count, reused_units=int(reuse), remaining_units=remaining,
                unit_seconds_estimate=unit['fresh_unit_seconds_estimate'],
                remaining_seconds_estimate=remaining*unit['fresh_unit_seconds_estimate'],
                budget=unit['budget']))
        total = sum(x['remaining_seconds_estimate'] for x in rows)
        scenarios.append(dict(name=name, status='proposal_not_frozen', rows=rows,
            total_units=sum(x['total_units'] for x in rows), remaining_units=sum(x['remaining_units'] for x in rows),
            remaining_routes=sum(x['remaining_units']*len(x['routes']) for x in rows),
            serial_hours_estimate=total/3600, hours_with_30_percent_allowance=total*1.3/3600,
            excludes='simulation, ablation, stress, new assets, changed roles or budgets; no launch authorization',
            reuse_condition='only identical source/data/protocol; confirmation reuses no development unit'))
    return scenarios


def closeout(source_root, output_root):
    root, out = Path(source_root).resolve(), Path(output_root).resolve()
    require(not out.exists() and out != root and root not in out.parents and out not in root.parents,
            'closeout requires a new, separate output root')
    sources = {}

    def checked(path, digest=None):
        path = Path(path).resolve()
        actual = sha256_file(path)
        require(digest is None or actual == digest, f'evidence changed: {path}')
        require(str(path) not in sources or sources[str(path)] == actual, f'evidence changed during audit: {path}')
        sources[str(path)] = actual
        return path

    def read(path, digest=None):
        return json.loads(checked(path, digest).read_text())

    state = read(root/'pipeline_state.json')
    require(state['status']=='comparison_completed' and not state['children'] and not state['failures'],
            'comparison is not successfully completed')
    for item in state['completed'].values():
        checked(item['path'], item['sha256'])
    plan = read(root/'comparison/plan.json')
    old_cost = read(root/'comparison/cost_report.json')
    cost_receipt = read(state['completed']['comparison_costs']['path'])
    require(old_cost == cost_receipt and old_cost['plan_sha256'] == plan['plan_sha256'], 'cost receipt mismatch')
    require(plan['seed']==17 and plan['fold_index']==0 and not plan['confirmation_opened'], 'unexpected development scope')
    require([{k:u[k] for k in ('domain','method')} for u in plan['units']] == comparison_units(), 'matrix changed')
    config = read(root/'pipeline_config.json')
    for p,h in config['input_files'].items():
        checked(p,h)
    launch = read(root/'launch.json')
    runtime = Path(launch['runtime'])
    digest = hashlib.sha256()
    for p in sorted((runtime/'src/chronaris').rglob('*.py'))+[runtime/'scripts/evaluation/application_tasks/run_thesis_v4.py']:
        digest.update(str(p.relative_to(runtime)).encode()); digest.update(p.read_bytes())
    require(digest.hexdigest()==plan['source_code_sha256'], 'frozen runtime source changed')
    migration = read(root/'comparison/execution_migration.json') if (root/'comparison/execution_migration.json').exists() else None
    if migration:
        for p,h in migration['parent_inventory'].items(): checked(p,h)
        for p,h in migration['bindings'].items(): checked(p,h)
    contracts = {d:read(x['path'],x['sha256']) for d,x in plan['contracts'].items()}
    metric_rows, learn, units, target_bindings, coverage = [], [], [], {}, []
    for spec, old_unit in zip(plan['units'], old_cost['unit_costs'], strict=True):
        domain, method = spec['domain'], spec['method']
        require((old_unit['domain'],old_unit['method'])==(domain,method), 'unit cost order changed')
        item = state['completed'][f'comparison_unit__{domain}__{method}']
        result = read(item['path'],item['sha256'])
        require(result['status']=='completed' and not result['confirmation_opened'], 'unit did not complete in development')
        contract = contracts[domain]; train = set(contract['roles']['train']); validation = set(contract['roles']['validation'])
        require(not train.intersection(validation), 'training and validation overlap')
        for row in result['results']:
            ev = row.get('evaluation',row)
            binding = read(Path(ev['consumers']['manifest_path']).parent.parent/'common_contract.json',ev['contract_file_sha256'])
            same = lambda d:{k:v for k,v in d.items() if k not in ('source_code_sha256','contract_sha256')}
            require(same(binding['contract'])==same(contract), 'method data contract differs')
            dec = binding['declaration']
            checked(dec['checkpoint_path'], dec['checkpoint_sha256'])
            for p,h in dec['evidence_files'].items(): checked(p,h)
            for k in ('encoder_fit_sample_ids','preprocessing_fit_sample_ids'):
                require(set(dec[k]) <= train, 'representation fitting escaped training role')
            require(dec['encoder_frozen'] and dec['task_heads_removed'], 'downstream used an unfrozen encoder or task head')
            require(dec['kind']!='window_end' or binding['families']==['linear'], 'window vector masquerades as a sequence')
            signal = learning_summary(row)
            if 'primary' in (row['training'] or {}):
                visited = {s for h in row['training']['primary']['history'] for s in h['sample_ids']}
                require(visited==train, 'adapter failed full training role coverage')
            for f in row.get('feature_files',{}).values(): checked(f['path'],f['sha256'])
            require(row.get('representation_reload_identical',True) and row.get('downstream_resume_identical',True), 'reload mismatch')
            learn.append(dict(domain=domain,method=method,route=row['route'],training_tasks=dec['training_tasks'],
                supervision=ev['supervision_stratum'],representation_kind=dec['kind'],external_pretraining=dec['external_pretraining']['source'],
                parameters=row.get('parameters',(row['training'] or {}).get('parameter_count')),
                feature_dim=dec['feature_dim'],**signal))
            manifest = read(ev['consumers']['manifest_path'])
            require(all(set(manifest['roles'][role]['sample_ids'])==set(ids) for role,ids in contract['roles'].items()), 'consumer roles changed')
            for family, component in ev['consumers']['components'].items():
                checked(component['model_path'],component['model_sha256'])
                data = read(component['result_path'],component['result_sha256'])
                require(all(set(x['train_sample_ids']) <= train for x in data['fit_rows']), 'consumer fitted validation data')
                val = data['evaluations']['validation']; check_summary(val)
                require(val['task_summary']==component['task_summary']['validation'], 'receipt metrics changed')
                predictions = val['prediction_rows']
                require({x['sample_id'] for x in predictions}==validation, 'validation window omitted')
                labels = sorted((x['sample_id'],x['task'],x['field'],x['target_valid'],x['target'],x['sample_weight']) for x in predictions)
                label_hash = hashlib.sha256(json.dumps(labels).encode()).hexdigest()
                require(domain not in target_bindings or label_hash==target_bindings[domain], 'methods used different evaluation targets')
                target_bindings[domain]=label_hash
                coverage.append(dict(domain=domain,method=method,route=row['route'],family=family,
                    sample_count=val['sample_count'],no_observation_fraction=val['no_observation_fraction'],
                    classification_targets={t['name']:dict(Counter(str(x['target']) for x in predictions if x['task']==t['name'] and x['target_valid']))
                        for t in contract['task_definitions'] if t['kind']=='classification'},
                    group_metrics=val['group_metrics']))
                for q in val['task_summary']:
                    require(q['value'] is not None and math.isfinite(q['value']), 'missing/nonfinite metric')
                    metric_rows.append(dict(domain=domain,method=method,route=row['route'],family=family,
                        supervision=ev['supervision_stratum'],training_tasks=dec['training_tasks'],**q))
        paths = old_unit['attempt_files']
        for p in paths: checked(p)
        accounting = attempt_accounting(spec,paths,migration)
        original = result.get('execution_inheritance',{}).get('original_total_seconds',result['total_seconds'])
        estimate = original
        estimate_note = 'observed complete unit, including validation/replay and loading; first fold only'
        if migration and (domain,method)==('cogpilot','chronaris'):
            ss = next(x for x in result['results'] if x['route']=='self_supervised')['training']
            delta = ss['training_elapsed_s']-migration['preserved_training_seconds']
            remaining = ss['optimizer_updates']-migration['preserved_optimizer_updates']
            require(delta>0 and remaining>0,'invalid migrated timing interval')
            estimate += delta/remaining*migration['preserved_optimizer_updates']
            estimate_note = 'restored unit wall time plus graph updates 201-300 average extrapolated to missing first 200; not a fresh-run measurement'
        units.append(spec | accounting | dict(routes=[x['route'] for x in result['results']],
            tasks=[x['name'] for x in contract['task_definitions']],fresh_unit_seconds_estimate=estimate,
            fresh_unit_estimate_note=estimate_note,observed_unit_seconds=original,
            language_cache=result.get('prompt_costs',{}),load_seconds=result.get('load_seconds')))
    require(len(learn)==plan['expected_routes']==43 and len(units)==25,'incomplete route matrix')
    scenarios = budget_scenarios(units)
    warning_counts = Counter()
    for path in sorted((root/'logs').glob('comparison_unit*.log')):
        log = checked(path).read_text()
        for marker in ('ConvergenceWarning', 'LinAlgWarning', 'ill-conditioned', 'fallback', 'Traceback', 'CUDA out of memory'):
            warning_counts[marker] += log.count(marker)
    report = dict(status='completed',scope='stage4_acceptance',confirmation_opened=False,
        completed_at=datetime.fromtimestamp(state['updated_at_unix_s']).astimezone().isoformat(),
        audited_at=datetime.now().astimezone().isoformat(),source_root=str(root),source_code_sha256=plan['source_code_sha256'],
        audit_source_sha256=sha256_file(__file__),sample_counts=plan['sample_counts'],
        model_units=len(units),representation_routes=len(learn),consumer_routes=len(coverage),
        historical_seconds_lower_bound=sum(x['historical_seconds_lower_bound'] for x in units),
        historical_cost_note='excludes external pretraining and separate setup/performance trials; interrupted parent tail remains unmeasured',
        cost_units=units,learning=learn,metrics=metric_rows,coverage=coverage,budget_scenarios=scenarios,
        log_marker_counts=dict(warning_counts),
        acceptance='computation, source/role/target checks, grouped summaries, learning diagnostics and budget proposals complete; no model adoption',
        sources=sources)
    # Fail before writing if any bound evidence was changed while auditing.
    for p,h in sources.items(): require(sha256_file(p)==h,f'evidence changed during closeout: {p}')
    write_result(out/'acceptance.json',report)
    return report
