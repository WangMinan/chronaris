"""Explicit stage-4 execution migration, preserving parent results and training state."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from chronaris.evaluation.application_tasks.checkpoint_performance import compare_trials
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.training.candidate_checkpoint import (
    atomic_save_candidate, candidate_protocol_hash, candidate_source_code_sha256,
)
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

STATE_KEYS = ('encoder_state_dict', 'head_state_dict', 'explicit_time_shift_head_state_dict',
              'optimizer_state_dict', 'rng_state')


def migrate_payload(payload, *, parent_path, evidence_sha256):
    config = payload['config']
    if (payload['format'] != 'chronaris.common_pretraining_checkpoint.v2'
        or payload['method_name'] != 'chronaris' or payload['optimizer_updates'] != 200
        or payload['step_count'] != 200 or payload['source_code_sha256'] != candidate_source_code_sha256()
        or config['cuda_graph_recurrence'] or config['ode_method'] != 'euler'
        or config['max_ode_step_s'] is not None or config['batch_size'] != 4
        or config['effective_batch_size'] != 16 or config['max_updates'] != 300
        or config['seed'] != 17 or config['device'] != 'cuda'
        or payload['data_cursor']['samples_seen'] != 3200 or payload['data_cursor']['micro_batches_seen'] != 800):
        raise ValueError('execution migration requires the preserved stage4 update-200 state')
    state_hash = canonical_training_state_sha256(*(payload[k] for k in STATE_KEYS))
    if state_hash != payload['canonical_training_state_sha256']:
        raise ValueError('checkpoint training state changed')
    keys = ('source_data_sha256', 'source_code_sha256', 'method_name', 'config', 'augmentation_policy',
            'fold', 'normalizer', 'physiology_feature_names', 'vehicle_feature_names', 'vehicle_field_labels',
            'transfer_source', 'chronaris_fusion_kind', 'chronaris_variant', 'chronaris_lag_aware_weight',
            'chronaris_mechanism_enabled', 'chronaris_explicit_shift_enabled',
            'chronaris_explicit_shift_weight', 'chronaris_event_pair_weight')
    protocol = {k: payload[k] for k in keys}
    protocol.update(candidate=payload['candidate_config'], data_access_mode='lazy_batch_provider')
    if candidate_protocol_hash(**protocol) != payload['protocol_sha256']:
        raise ValueError('parent checkpoint protocol does not reproduce')
    migrated = deepcopy(payload)
    migrated['config']['cuda_graph_recurrence'] = True
    migrated['encoder_manifest']['backbone_config']['cuda_graph_recurrence'] = True
    migrated['protocol_sha256'] = candidate_protocol_hash(**(protocol | {'config': migrated['config']}))
    migrated['execution_migration'] = dict(parent_path=str(parent_path), parent_sha256=sha256_file(parent_path),
        parent_protocol_sha256=payload['protocol_sha256'], evidence_sha256=evidence_sha256,
        change='cuda_graph_recurrence_false_to_true', preserved_optimizer_updates=200)
    if canonical_training_state_sha256(*(migrated[k] for k in STATE_KEYS)) != state_hash:
        raise ValueError('migration changed model, optimizer or RNG')
    return migrated


def _verify_sources(frozen, current, expected_hash):
    old = {p.relative_to(frozen): p for p in (frozen/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(current): p for p in (current/'src/chronaris').rglob('*.py')}
    changes = []
    orchestration = {'development_comparison.py', 'v4_pipeline.py', 'execution_migration.py', 'checkpoint_performance.py'}
    for name in sorted(old.keys() | new.keys()):
        before = old[name].read_text() if name in old else None
        after = new[name].read_text() if name in new else None
        if before == after:
            continue
        if str(name) == 'src/chronaris/evaluation/application_tasks/common_downstream_smoke.py':
            expected = before.replace("full=False, methods=('naive_time_sync', 'chronaris')):",
                "full=False, methods=('naive_time_sync', 'chronaris'), cuda_graph_recurrence=False):")
            expected = expected.replace("        raise ValueError('unsupported common comparison method')\n",
                "        raise ValueError('unsupported common comparison method')\n"
                "    if cuda_graph_recurrence and (domain != 'cogpilot' or methods != ('chronaris',)):\n"
                "        raise ValueError('graph execution is validated only for CogPilot Chronaris')\n")
            expected = expected.replace("data_manifest_sha256=digest), chronaris_fusion_kind=",
                "data_manifest_sha256=digest, cuda_graph_recurrence=cuda_graph_recurrence), chronaris_fusion_kind=")
            if expected != after:
                raise ValueError('common entry change exceeds the execution-only dispatch')
        elif name.parent != Path('src/chronaris/evaluation/application_tasks') or name.name not in orchestration:
            raise ValueError(f'unreviewed computational source change: {name}')
        changes.append(str(name))
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    if (frozen/entry).read_bytes() != (current/entry).read_bytes():
        raise ValueError('training entry changed')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((frozen/name).read_bytes())
    if digest.hexdigest() != expected_hash:
        raise ValueError('parent frozen source changed')
    return changes


def verify_execution_migration(config):
    path = Path(config['root'])/'comparison/execution_migration.json'
    receipt = json.loads(path.read_text())
    if receipt['source_code_sha256'] != v4_workflow_source_sha256():
        raise ValueError('execution migration source changed')
    for name, digest in receipt['bindings'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'execution migration evidence changed: {name}')
    return receipt


def prepare_execution_migration(config, plan):
    root, parent = Path(config['root']).resolve(), Path(config['execution_parent']).resolve()
    if root == parent or root in parent.parents or parent in root.parents:
        raise ValueError('execution migration requires a separate run root')
    receipt_path = root/'comparison/execution_migration.json'
    if receipt_path.exists():
        return verify_execution_migration(config)
    started = time.perf_counter()
    evidence_path = Path(config['execution_evidence'])
    evidence = json.loads(evidence_path.read_text()); trial_root = evidence_path.parent
    bindings = {str(evidence_path): sha256_file(evidence_path)}
    for name, digest in evidence['source_artifacts_sha256'].items():
        if sha256_file(name) != digest:
            raise ValueError('performance trial artifact changed')
    trial_metadata = {name: json.loads((trial_root/name/'metadata.json').read_text()) for name in ('eager', 'graph')}
    for name, metadata in trial_metadata.items():
        if (metadata['graph'] != (name == 'graph') or metadata['threads'] != 1
            or metadata['source_code_sha256'] != candidate_source_code_sha256()
            or metadata['actual_batch'] != 4 or metadata['effective_batch'] != 16
            or [x['update'] for x in metadata['measurements']] != [201, 202]):
            raise ValueError('trial did not validate the proposed execution configuration')
    if (evidence['status'] != 'completed' or evidence['preserved_updates'] != 200
        or evidence['compute_speedup'] <= 1
        or not compare_trials(trial_root/'eager', trial_root/'graph')['passed']
        or not compare_trials(trial_root/'graph', trial_root/'graph_resume', exact=True)['passed']):
        raise ValueError('execution performance or equivalence was not validated')
    stop = json.loads((trial_root/'preserved/stop_receipt.json').read_text())
    if not stop.get('stopped') or stop['saved_updates'] != 200:
        raise ValueError('parent was not safely stopped')
    old_config = json.loads((parent/'pipeline_config.json').read_text())
    changes = _verify_sources(Path(old_config['registry_path']).parents[2], Path(__file__).parents[4], old_config['source_code_sha256'])
    state = json.loads((parent/'pipeline_state.json').read_text())
    if state.get('children') or state['current_stage'] != 'comparison_unit__cogpilot__chronaris':
        raise ValueError('parent is not at the preserved comparison boundary')
    for pid in (state['pid'], stop['worker_pid']):
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and str(parent).encode() in command.read_bytes():
            raise ValueError('parent process remains live')
    inventory = {p: item['sha256'] for p,item in json.loads((trial_root/'old_run_inventory.json').read_text())['files'].items()}
    inventory.update(json.loads((parent/'comparison/reuse_inventory.json').read_text())['files'])
    inventory.update(old_config['input_files'])
    for name, digest in inventory.items():
        if sha256_file(name) != digest:
            raise ValueError(f'preserved parent evidence changed: {name}')
    old_plan = json.loads((parent/'comparison/plan.json').read_text())
    ignored = {'contracts', 'source_code_sha256', 'plan_sha256', 'execution'}
    if {k:v for k,v in old_plan.items() if k not in ignored} != {k:v for k,v in plan.items() if k not in ignored}:
        raise ValueError('migration cannot change matrix, roles, objectives or budget')
    for domain,item in plan['contracts'].items():
        before = json.loads(Path(old_plan['contracts'][domain]['path']).read_text())
        after = json.loads(Path(item['path']).read_text())
        ignored_contract = {'source_code_sha256', 'contract_sha256'}
        if {k:v for k,v in before.items() if k not in ignored_contract} != {k:v for k,v in after.items() if k not in ignored_contract}:
            raise ValueError('migration changed a data or downstream contract')
    completed = {k:v for k,v in state['completed'].items() if k.startswith('comparison_unit__')}
    if len(completed) != 9 or any(not k.startswith('comparison_unit__clare__') for k in completed):
        raise ValueError('expected exactly nine preserved CLARE units')
    for item in completed.values():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('parent completion receipt changed')
        bindings[item['path']] = item['sha256']
    destination = root/'comparison/original/chronaris/cogpilot/chronaris/self_supervised/chronaris/C'
    migrated_files = {}
    for name in ('best.pt', 'last.pt'):
        source = trial_root/'preserved'/name
        original = parent/'comparison/original/chronaris/cogpilot/chronaris/self_supervised/chronaris/C'/name
        if sha256_file(source) != stop['checkpoint_copies_sha256'][str(original)] or sha256_file(source) != sha256_file(original):
            raise ValueError('preserved checkpoint no longer matches the parent')
        payload = torch.load(source, map_location='cpu', weights_only=True)
        if name == 'last.pt' and sha256_file(source) != json.loads((trial_root/'graph/metadata.json').read_text())['parent_sha256']:
            raise ValueError('performance trial used a different checkpoint')
        sealed = root/'migration/checkpoints'/name
        if sealed.exists() or (destination/name).exists():
            raise ValueError('partial migration exists; inspect it before retrying')
        sealed.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_candidate(sealed, migrate_payload(payload, parent_path=source, evidence_sha256=bindings[str(evidence_path)]))
        loaded = torch.load(sealed, map_location='cpu', weights_only=True)
        if canonical_training_state_sha256(*(loaded[k] for k in STATE_KEYS)) != payload['canonical_training_state_sha256']:
            raise ValueError('serialized migrated state differs')
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sealed, destination/name)
        bindings[str(sealed)] = sha256_file(sealed)
        migrated_files[name] = dict(path=str(sealed), sha256=bindings[str(sealed)],
            canonical_state_sha256=loaded['canonical_training_state_sha256'], protocol_sha256=loaded['protocol_sha256'])
    parent_costs = {p:h for p,h in inventory.items() if '/comparison/attempt_costs/' in p and p.endswith('.json')}
    bindings.update(parent_costs)
    for p in (parent/'pipeline_config.json', parent/'pipeline_state.json', trial_root/'preserved/stop_receipt.json'):
        bindings[str(p)] = sha256_file(p)
    return write_result(receipt_path, dict(status='completed', source_code_sha256=v4_workflow_source_sha256(),
        parent_root=str(parent), parent_source_code_sha256=old_config['source_code_sha256'],
        plan_sha256=plan['plan_sha256'], reviewed_source_changes=changes, completed=completed, bindings=bindings,
        verified_parent_files=len(inventory), parent_inventory=inventory, parent_attempt_costs=parent_costs,
        migrated_checkpoints=migrated_files, preserved_optimizer_updates=200, extra_optimizer_updates=0,
        preserved_training_seconds=payload['training_elapsed_s'], inherited_result_scope='unchanged CLARE computation; original per-route provenance retained',
        performance_trial_summary=str(evidence_path), migration_seconds=time.perf_counter()-started))


def inherited_execution_unit(stage, config, plan):
    migration = verify_execution_migration(config)
    if migration['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('execution migration plan changed')
    if stage not in migration['completed']:
        return None
    receipt = migration['completed'][stage]
    previous = json.loads(Path(receipt['path']).read_text())
    return previous | dict(source_code_sha256=plan['source_code_sha256'],
        contract_sha256=plan['contracts']['clare']['contract_sha256'],
        execution_inheritance=dict(parent_receipt=receipt, original_source_code_sha256=previous['source_code_sha256'],
            original_contract_sha256=previous['contract_sha256'], original_total_seconds=previous['total_seconds'],
            extra_encoder_updates=0, scope=migration['inherited_result_scope']))
