"""Bounded stage-5 consumer repair: preserve encoders, rebuild common consumers."""
from dataclasses import asdict
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


def read(path):
    return json.loads(Path(path).read_text())


def verify(files):
    for path, digest in files.items():
        if sha256_file(path) != digest:
            raise ValueError(f'stage 5 recovery source changed: {path}')


def same_training_contract(before, after):
    """Only the declared supplementary input policy and workflow source may change."""
    def normalized(value):
        value = {k: v for k, v in value.items() if k not in ('contract_sha256', 'source_code_sha256')}
        policy = dict(value['policy'])
        if policy['format'] == 'chronaris.common_downstream_policy.v2':
            if policy.pop('minirocket_input_policy') != 'all_finite_channels; local_constancy_diagnostic_only; observed_train_cases_fit':
                raise ValueError('unreviewed MiniROCKET input policy')
            policy['format'] = 'chronaris.common_downstream_policy.v1'
        return value | {'policy': policy}
    return normalized(before) == normalized(after)


def representation_root(prior, route):
    """Follow recorded stage-4 revalidation lineage, never guess a tensor path."""
    visited = set()
    while True:
        row = next(row for row in prior['results'] if row['route'] == route)
        root = Path(row['consumers']['manifest_path']).parents[2]/'representations'
        if (root/'train/representation_manifest.json').is_file():
            return root
        receipt = prior.get('reuse', {}).get('parent_receipt')
        if not receipt or receipt['path'] in visited:
            raise ValueError('preserved representations lack an explicit parent receipt')
        verify({receipt['path']: receipt['sha256']})
        visited.add(receipt['path'])
        prior = read(receipt['path'])


def prepare_recovery(config, contracts, budgets):
    parent, root = Path(config['stage5_parent']).resolve(), Path(config['root']).resolve()
    if parent == root or parent in root.parents or root in parent.parents:
        raise ValueError('stage 5 recovery requires a separate run root')
    previous = read(parent/'pipeline_config.json')
    frozen, project = Path(previous['registry_path']).parents[2], Path(__file__).parents[4]
    old = {p.relative_to(frozen): p for p in (frozen/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(project): p for p in (project/'src/chronaris').rglob('*.py')}
    allowed = {f'src/chronaris/evaluation/application_tasks/{name}.py' for name in (
        'application_consumers', 'common_downstream_contract', 'common_downstream_smoke',
        'v4_grouped_consumers', 'stage5', 'stage5_recovery', 'v4_pipeline')}
    changed = []
    for name in sorted(old.keys() | new.keys()):
        a = old[name].read_bytes() if name in old else None
        b = new[name].read_bytes() if name in new else None
        if a != b:
            if str(name) not in allowed:
                raise ValueError(f'unreviewed stage 5 computational change: {name}')
            changed.append(str(name))
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    if (frozen/entry).read_bytes() != (project/entry).read_bytes():
        raise ValueError('stage 5 training entry changed')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((frozen/name).read_bytes())
    if digest.hexdigest() != previous['source_code_sha256']:
        raise ValueError('parent frozen workflow changed')
    verify(previous['input_files'])
    state, plan = read(parent/'pipeline_state.json'), read(parent/'stage5_plan.json')
    completed = {k: v for k, v in state['completed'].items() if k.startswith('stage5_unit__')}
    if (state['status'] != 'failed' or state['children'] or state['current_stage'] != 'stage5_unit__12'
        or set(completed) != {f'stage5_unit__{i}' for i in range(12)}):
        raise ValueError('recovery is limited to the audited 12-unit stage 5 halt')
    if Path(f"/proc/{state['pid']}").exists():
        raise ValueError('parent controller must be stopped')
    from chronaris.evaluation.application_tasks.stage5 import units
    if plan['units'] != units() or plan['budgets'] != json.loads(json.dumps(budgets)) or plan['confirmation_opened']:
        raise ValueError('recovery cannot change matrix, training budget or data roles')
    for key, current in contracts.items():
        prior = plan['contracts'][key]
        verify({prior['path']: prior['sha256']})
        if not same_training_contract(read(prior['path']), read(current['path'])):
            raise ValueError('recovery inputs, targets or primary policy changed')
    files = {str(p): sha256_file(p) for p in parent.rglob('*') if p.is_file()}
    files.update({str(p): sha256_file(p) for p in old.values()})
    files.update(previous['input_files'])
    for receipt in state['completed'].values():
        verify({receipt['path']: receipt['sha256']})
        files[receipt['path']] = receipt['sha256']
    for receipt in completed.values():
        prior = read(receipt['path']); verify(prior['evidence_files']); files.update(prior['evidence_files'])
        for row in prior['results']:
            evaluation = Path(row['consumers']['manifest_path']).parents[1]
            record = read(evaluation/'common_contract.json')
            files.update(record['declaration']['evidence_files'])
            for p in evaluation.parent.rglob('*'):
                if p.is_file():
                    files[str(p)] = sha256_file(p)
            for p in representation_root(prior, row['route']).rglob('*'):
                if p.is_file():
                    files[str(p)] = sha256_file(p)
    partial = parent/'training/12/clare/chronaris/self_supervised/chronaris/C/best.pt'
    if not partial.is_file():
        raise ValueError('failed unit selected checkpoint missing')
    verify(files)
    inventory = dict(status='completed', parent_root=str(parent), parent_source_code_sha256=previous['source_code_sha256'],
        source_code_sha256=v4_workflow_source_sha256(), reviewed_source_changes=changed,
        files=files, completed=completed, partial_checkpoint=str(partial), partial_checkpoint_sha256=sha256_file(partial),
        parent_failures=state['failures'], encoder_training_changed=False, primary_policy_changed=False,
        supplementary_policy='all_finite_channels_v2', confirmation_opened=False)
    path = root/'recovery_inventory.json'
    if path.exists() and read(path) != inventory:
        raise ValueError('recovery inventory changed')
    write_result(path, inventory)
    return dict(path=str(path), sha256=sha256_file(path))


def recovery_inventory(config, frozen):
    receipt = frozen['recovery']; verify({receipt['path']: receipt['sha256']})
    inventory = read(receipt['path'])
    if inventory['source_code_sha256'] != v4_workflow_source_sha256():
        raise ValueError('recovery workflow changed')
    verify(inventory['files'])
    return inventory


def revalidate_completed(config, frozen, unit, prior, source_files, index):
    """Re-use frozen representations, then refit/replay identical primary consumers."""
    verify(source_files)
    destination = Path(config['root'])/'revalidated'/str(index)
    with development_gpu_lock() as acquired:
        if not acquired:
            return dict(status='waiting_gpu')
        torch.set_num_threads(1)
        started = time.perf_counter()
        with _periodic_training_heartbeat('stage5_revalidation', 30., root=destination) as progress:
            provider, _, fold, digest, targets, definitions, context, observations = contract_development_inputs(
                unit['domain'], data_root=config['data_root'], registry_path=config['registry_path'], full=True,
                fold_index=unit['fold_index'])
            contract = build_common_contract(domain=unit['domain'], fold=fold, observations=observations,
                targets=targets, definitions=definitions, context=context, data_manifest_sha256=digest,
                scope='development_comparison', seed=unit['seed'])
            key = f"{unit['domain']}/{unit['fold_index']}/{unit['seed']}"
            if contract['contract_sha256'] != frozen['contracts'][key]['contract_sha256']:
                raise ValueError('revalidation contract differs from plan')
            results, checks = [], []
            for row in prior['results']:
                route = row['route']; progress.update(route=route, phase='preserved_representation_replay')
                old_root = Path(row['consumers']['manifest_path']).parents[1]
                record_path = old_root/'common_contract.json'
                verify({str(record_path): row['contract_file_sha256']})
                record = read(record_path)
                if not same_training_contract(record['contract'], contract):
                    raise ValueError('preserved route input contract changed')
                declaration = record['declaration']
                encoder, normalizer, _ = load_frozen_application_encoder(declaration['checkpoint_path'],
                    route=route, fold=fold, device='cuda')
                preserved = representation_root(prior, route)
                outputs = {role: load_fusion_stream_batch(preserved/role) for role in observations}
                adapter = TrainedFusionAdapter(encoder=encoder, normalizer=normalizer, fold_id=fold.fold_id,
                    checkpoint_sha256=declaration['checkpoint_sha256'])
                for role, output in outputs.items():
                    ids = output.sample_ids[:4]; fresh = adapter(provider(ids))
                    for name in ('sequence_embedding', 'pooled_embedding', 'valid_mask', 'timestamps_s'):
                        if not torch.equal(getattr(fresh, name).cpu(), getattr(output, name)[:len(ids)].cpu()):
                            raise ValueError('preserved representation replay changed')
                    checks.append(dict(route=route, role=role, windows=len(ids), bitwise_identical=True))
                progress['phase'] = 'common_consumers'
                result = run_common_downstream(contract=contract, outputs=outputs, targets=targets, definitions=definitions,
                    context=context, observations=observations, fold=fold, data_manifest_sha256=digest,
                    declaration=declaration, output_root=destination/route/'evaluation', families=('linear', 'minirocket'))
                before = read(row['consumers']['components']['linear']['result_path'])
                after = read(result['consumers']['components']['linear']['result_path'])
                if any(before[k] != after[k] for k in ('fit_rows', 'evaluations')):
                    raise ValueError('revalidated primary predictions or parameter selection changed')
                results.append(row | result | dict(extra_encoder_updates=0, primary_replay_identical=True,
                    original_training_source=declaration['checkpoint_path']))
                del adapter, encoder
            verify(source_files)
            return prior | dict(results=results, source_code_sha256=contract['source_code_sha256'],
                contract_sha256=contract['contract_sha256'], total_seconds=time.perf_counter()-started,
                historical_total_seconds=prior['total_seconds'], reused_training=True, extra_encoder_updates=0,
                representation_replay=checks, supplementary_recomputed=True)


def validate_partial_pretraining(payload, recipe_configs, method):
    """Do not broaden the old fine-tuning candidate exception into arbitrary reuse."""
    candidate, pre, _, arguments, _ = recipe_configs
    from chronaris.modeling.training.candidate_checkpoint import candidate_source_code_sha256
    if (payload['source_code_sha256'] != candidate_source_code_sha256()
        or payload['method_name'] != method or payload['candidate_config'] != asdict(candidate)
        or payload['config'] != asdict(pre)
        or payload['optimizer_updates'] != pre.max_updates
        or payload['best_update'] != payload['optimizer_updates']
        or any(payload[k] != value for k, value in arguments.items())):
        raise ValueError('partial stage 5 pretraining differs from fixed selected reference')
