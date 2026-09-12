"""Explicit source-preserving reuse for the stage-4 baseline dispatch repair."""
import hashlib
import json
from pathlib import Path
import time

import torch

from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.common_downstream_contract import build_common_contract, run_common_downstream
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.training import TrainedFusionAdapter
from chronaris.modeling.training.candidate_screen import _periodic_training_heartbeat
from chronaris.representation import load_fusion_stream_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def verify_dispatch_sources(frozen, project):
    """No blanket compatibility exemption for model, training or consumer changes."""
    frozen, project = Path(frozen), Path(project)
    old = {p.relative_to(frozen): p for p in (frozen/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(project): p for p in (project/'src/chronaris').rglob('*.py')}
    orchestration = {'development_comparison.py', 'v4_pipeline.py', 'comparison_reuse.py'}
    changes = []
    for name in sorted(old.keys() | new.keys()):
        before = old[name].read_text() if name in old else None
        after = new[name].read_text() if name in new else None
        if before == after:
            continue
        if str(name) == 'src/chronaris/evaluation/application_tasks/common_downstream_smoke.py':
            expected = before.replace("chronaris_fusion_kind='safe_lag')",
                "chronaris_fusion_kind='safe_lag' if method == 'chronaris' else 'multiscale')")
            if expected != after:
                raise ValueError('reuse requires exactly the reviewed method dispatch repair')
        elif name.parent != Path('src/chronaris/evaluation/application_tasks') or name.name not in orchestration:
            raise ValueError(f'unreviewed computational source change: {name}')
        changes.append(str(name))
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    if (frozen/entry).read_bytes() != (project/entry).read_bytes():
        raise ValueError('unreviewed training entry change')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((frozen/name).read_bytes())
    return digest.hexdigest(), changes, {str(p): sha256_file(p) for p in old.values()}


def prepare_parent_reuse(config, plan):
    parent, root = Path(config['comparison_parent']).resolve(), Path(config['root']).resolve()
    if parent == root or parent in root.parents or root in parent.parents:
        raise ValueError('reuse requires a separate run root')
    old_config = json.loads((parent/'pipeline_config.json').read_text())
    frozen = Path(old_config['registry_path']).parents[2]
    source, changes, files = verify_dispatch_sources(frozen, Path(__file__).parents[4])
    if source != old_config['source_code_sha256']:
        raise ValueError('parent source snapshot changed')
    for filename, digest in old_config['input_files'].items():
        if sha256_file(filename) != digest:
            raise ValueError('parent inputs changed')
        files[filename] = digest
    old_plan = json.loads((parent/'comparison/plan.json').read_text())
    ignored = {'contracts', 'source_code_sha256', 'plan_sha256'}
    if {k:v for k,v in old_plan.items() if k not in ignored} != {k:v for k,v in plan.items() if k not in ignored}:
        raise ValueError('reuse cannot change the development matrix or budget')
    for domain, item in plan['contracts'].items():
        before = json.loads(Path(old_plan['contracts'][domain]['path']).read_text())
        after = json.loads(Path(item['path']).read_text())
        excluded = {'source_code_sha256', 'contract_sha256'}
        if {k:v for k,v in before.items() if k not in excluded} != {k:v for k,v in after.items() if k not in excluded}:
            raise ValueError('reuse cannot change observations, targets, roles or downstream policy')
    # Bind the failed attempt too. The parent is never updated by this continuation.
    files.update({str(p): sha256_file(p) for p in parent.rglob('*') if p.is_file()})
    state = json.loads((parent/'pipeline_state.json').read_text())
    for receipt in state['completed'].values():
        if sha256_file(receipt['path']) != receipt['sha256']:
            raise ValueError('parent completion receipt changed')
    result = dict(status='completed', parent_root=str(parent), parent_source_code_sha256=source,
        source_code_sha256=v4_workflow_source_sha256(), reviewed_source_changes=changes, files=files,
        completed={k:v for k,v in state['completed'].items() if k.startswith('comparison_unit__')},
        parent_failures=state['failures'], plan_sha256=plan['plan_sha256'])
    path = root/'comparison/reuse_inventory.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('parent reuse inventory changed')
    return write_result(path, result)


def verify_reuse_inventory(config):
    path = Path(config['root'])/'comparison/reuse_inventory.json'
    inventory = json.loads(path.read_text())
    if inventory['source_code_sha256'] != v4_workflow_source_sha256():
        raise ValueError('reuse verification source changed')
    if any(sha256_file(p) != h for p,h in inventory['files'].items()):
        raise ValueError('parent evidence changed')
    return inventory


def revalidate_completed_unit(stage, config, plan):
    inventory = verify_reuse_inventory(config)
    if stage not in inventory['completed']:
        return None
    _, domain, method = stage.split('__')
    # The only completed unit before this repair; other migrations need their own evidence.
    if (domain, method) != ('clare', 'chronaris'):
        raise ValueError('this repair only reuses the completed CLARE Chronaris unit')
    receipt = inventory['completed'][stage]
    prior = json.loads(Path(receipt['path']).read_text())
    destination = Path(config['root'])/'comparison/revalidated'/domain/method
    started = time.perf_counter()
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        with _periodic_training_heartbeat('comparison_reuse', 30., root=destination) as progress:
            provider, _, fold, digest, targets, definitions, context, raw = contract_development_inputs(
                domain, data_root=config['data_root'], registry_path=config['registry_path'], full=True)
            kwargs = dict(fold=fold, observations=raw, targets=targets, definitions=definitions,
                context=context, data_manifest_sha256=digest)
            contract = build_common_contract(domain=domain, scope='development_comparison', **kwargs)
            if contract['contract_sha256'] != plan['contracts'][domain]['contract_sha256']:
                raise ValueError('current revalidation inputs changed')
            results, checks = [], []
            for row in prior['results']:
                route = row['route']; progress.update(phase='checkpoint_and_representation', route=route)
                old_evaluation = Path(row['consumers']['manifest_path']).parents[1]
                record_path = old_evaluation/'common_contract.json'
                if sha256_file(record_path) != row['contract_file_sha256']:
                    raise ValueError('parent representation declaration changed')
                record = json.loads(record_path.read_text()); declaration = dict(record['declaration'])
                checkpoint = declaration['checkpoint_path']
                encoder, normalizer, payload = load_frozen_application_encoder(checkpoint, route=route, fold=fold, device='cuda')
                expected_updates = 300 if route=='self_supervised' else 250
                if row['training']['optimizer_updates'] != expected_updates:
                    raise ValueError('parent unit did not finish its fixed training budget')
                if route == 'task_guided' and any(sum(r['task_valid_counts'][t.name] for r in payload['update_rows']
                        if r['stage']=='joint_adaptation') <= 0 for t in definitions):
                    raise ValueError('parent encoder task supervision missing')
                outputs = {role: load_fusion_stream_batch(old_evaluation.parent/'representations'/role) for role in raw}
                adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id,
                    checkpoint_sha256=declaration['checkpoint_sha256'])
                for role, output in outputs.items():
                    # All archives are hash-verified; additionally re-encode a fixed batch per role.
                    ids = output.sample_ids[:4]
                    fresh = adapter(provider(ids))
                    indices = [output.sample_ids.index(s) for s in ids]
                    for name in ('sequence_embedding', 'pooled_embedding', 'valid_mask', 'timestamps_s'):
                        if not torch.equal(getattr(fresh, name).cpu(), getattr(output, name)[indices].cpu()):
                            raise ValueError('checkpoint reload differs from preserved representation')
                    checks.append(dict(route=route, role=role, recomputed_windows=len(ids), bitwise_identical=True))
                declaration['evidence_files'] = declaration['evidence_files'] | {
                    str(Path(__file__).resolve()): sha256_file(__file__),
                    str(Path(config['root'])/'comparison/reuse_inventory.json'): sha256_file(Path(config['root'])/'comparison/reuse_inventory.json')}
                evaluation_root = destination/route/'evaluation'
                first = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration,
                    output_root=evaluation_root, families=record['families'])
                repeated = run_common_downstream(contract=contract, **kwargs, outputs=outputs, declaration=declaration,
                    output_root=evaluation_root, families=record['families'])
                if first != repeated:
                    raise ValueError('revalidated downstream resume differs')
                for family, component in first['consumers']['components'].items():
                    old_component = row['consumers']['components'][family]
                    if any(sha256_file(old_component[k+'_path']) != old_component[k+'_sha256'] for k in ('result','model')):
                        raise ValueError('original downstream artifact changed')
                    before, after = (json.loads(Path(p).read_text()) for p in (old_component['result_path'], component['result_path']))
                    if any(before[k] != after[k] for k in ('evaluations', 'fit_rows')):
                        raise ValueError('revalidated predictions, metrics or selected fits differ')
                results.append(row | first | {'reused_encoder_updates': expected_updates, 'extra_encoder_updates': 0})
                del adapter, encoder, payload
            verify_reuse_inventory(config)
            return write_result(destination/'summary.json', prior | dict(results=results,
                source_code_sha256=contract['source_code_sha256'], contract_sha256=contract['contract_sha256'],
                reuse=dict(parent_receipt=receipt, encoder_updates_preserved=550, extra_encoder_updates=0,
                    representation_checks=checks, all_parent_files_unchanged=len(inventory['files']),
                    revalidation_seconds=time.perf_counter()-started, original_total_seconds=prior['total_seconds'])))
