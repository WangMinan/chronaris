"""Preserve completed development evidence and resume one interrupted unit."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import torch

from chronaris.evaluation.application_tasks.common_downstream_contract import _digest
from chronaris.evaluation.application_tasks.execution_migration import STATE_KEYS
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.training.candidate_checkpoint import (
    atomic_save_candidate, candidate_protocol_hash, candidate_source_code_sha256)
from chronaris.modeling.training.rng import canonical_training_state_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def build_recovery_evidence(review_root):
    from chronaris.evaluation.application_tasks.checkpoint_performance import compare_trials
    root = Path(review_root).resolve()
    stop = json.loads((root/'stop_receipt.json').read_text())
    if not stop['stopped']:
        raise ValueError('recovery requires a stopped parent')
    checkpoint = Path(stop['checkpoint_copy'])
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if sha256_file(checkpoint) != sha256_file(stop['checkpoint']) or payload['optimizer_updates'] != stop['saved_updates']:
        raise ValueError('saved checkpoint copy or update count changed')
    numeric = compare_trials(root/'eager',root/'graph')
    resumed = compare_trials(root/'graph',root/'graph_resume',exact=True)
    write_result(root/'graph_comparison.json',numeric)
    write_result(root/'graph_resume_comparison.json',resumed)
    times = {}
    for name in ('eager','graph'):
        metadata = json.loads((root/name/'metadata.json').read_text())
        if (metadata['parent_sha256'] != sha256_file(checkpoint) or metadata['domain'] != 'clare'
            or metadata['source_code_sha256'] != candidate_source_code_sha256()
            or metadata['graph'] != (name == 'graph') or metadata['threads'] != 1
            or [r['update'] for r in metadata['measurements']] != [stop['saved_updates']+1,stop['saved_updates']+2]):
            raise ValueError('trial lineage, shape or measured updates differ')
        times[name] = sum(sum(m['forward_backward_s'] for m in row['micro_timings'])+row['optimizer_s']
                          for row in metadata['measurements'])
    speedup = times['eager']/times['graph']
    paths = [root/'stop_receipt.json',root/'parent_inventory.json',checkpoint,root/'benchmark_tool.py',
             root/'graph_comparison.json',root/'graph_resume_comparison.json']
    for name in ('eager','graph','graph_resume'):
        paths += list((root/name).glob('*.pt'))+[root/name/'metadata.json']
    return write_result(root/'recovery_evidence.json',dict(status='completed',parent_root=stop['old_root'],
        inventory_path=str(root/'parent_inventory.json'),worker_pid=stop['worker_pid'],saved_updates=stop['saved_updates'],
        uncheckpointed_updates_not_reused=stop['uncheckpointed_updates_not_reused'],fold=payload['fold'],
        candidate_source_sha256=candidate_source_code_sha256(),bindings={str(p):sha256_file(p) for p in paths},
        graph_comparison_path=str(root/'graph_comparison.json'),graph_resume_comparison_path=str(root/'graph_resume_comparison.json'),
        trial_roots={name:str(root/name) for name in ('eager','graph','graph_resume')},
        pretraining_compute_seconds=times,pretraining_compute_speedup=speedup,
        resume_pretraining_graph=numeric['passed'] and resumed['passed'] and speedup>1,
        historical_training_seconds_lower_bound=payload['training_elapsed_s'],
        finetuning_execution='ordinary; no inferred qualification from pretraining trial',confirmation_opened=False))


def read_evidence(config):
    path = Path(config['stage45_resume_evidence'])
    evidence = json.loads(path.read_text())
    if (evidence['status'] != 'completed' or evidence['confirmation_opened']
        or Path(evidence['parent_root']).resolve() != Path(config['stage45_resume_parent']).resolve()
        or evidence['candidate_source_sha256'] != candidate_source_code_sha256()):
        raise ValueError('stage 4.5 recovery evidence or computational source changed')
    for name, digest in evidence['bindings'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'recovery evidence changed: {name}')
    if evidence['resume_pretraining_graph']:
        for key in ('graph_comparison_path', 'graph_resume_comparison_path'):
            if not json.loads(Path(evidence[key]).read_text())['passed']:
                raise ValueError('graph migration requires both numerical and exact resume evidence')
        if evidence['pretraining_compute_speedup'] <= 1:
            raise ValueError('graph migration requires measured speedup')
    return evidence


def verify_execution_sources(parent):
    config = json.loads((parent/'pipeline_config.json').read_text())
    frozen = Path(config['registry_path']).parents[2]
    current = Path(__file__).parents[4]
    old = {p.relative_to(frozen):p for p in (frozen/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(current):p for p in (current/'src/chronaris').rglob('*.py')}
    allowed = {'stage45.py', 'stage45_resume.py', 'v4_pipeline.py', 'checkpoint_performance.py',
               'stage45_diagnostics.py', 'common_downstream_smoke.py'}
    changes = []
    for name in old.keys() | new.keys():
        if name in old and name in new and old[name].read_bytes() == new[name].read_bytes():
            continue
        if name.parent != Path('src/chronaris/evaluation/application_tasks') or name.name not in allowed:
            raise ValueError(f'unreviewed computational change: {name}')
        changes.append(str(name))
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    if (frozen/entry).read_bytes() != (current/entry).read_bytes():
        raise ValueError('training entry changed')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((frozen/name).read_bytes())
    if digest.hexdigest() != config['source_code_sha256']:
        raise ValueError('parent runtime source changed')
    return sorted(changes)


def verify_parent_contract(config, contract):
    if not config.get('stage45_resume_parent'):
        return
    plan = json.loads((Path(config['stage45_resume_parent'])/'stage45_plan.json').read_text())
    restored = {k:v for k,v in contract.items() if k != 'contract_sha256'}
    restored['source_code_sha256'] = plan['source_code_sha256']
    if _digest(restored) not in plan['contracts'].values():
        raise ValueError('recovery changed a frozen development data/target/role contract')


def execution_checkpoint(payload, *, parent_path, graph, evidence_sha256):
    """Change only the recorded execution choice; keep every training tensor and cursor."""
    if payload['source_code_sha256'] != candidate_source_code_sha256():
        raise ValueError('execution checkpoint requires unchanged computational source')
    before = canonical_training_state_sha256(*(payload[k] for k in STATE_KEYS))
    if before != payload['canonical_training_state_sha256']:
        raise ValueError('preserved optimizer/model/RNG digest differs')
    keys = ('source_data_sha256', 'source_code_sha256', 'method_name', 'config', 'augmentation_policy',
            'fold', 'normalizer', 'physiology_feature_names', 'vehicle_feature_names', 'vehicle_field_labels',
            'transfer_source', 'chronaris_fusion_kind', 'chronaris_variant', 'chronaris_lag_aware_weight',
            'chronaris_mechanism_enabled', 'chronaris_explicit_shift_enabled',
            'chronaris_explicit_shift_weight', 'chronaris_event_pair_weight')
    protocol = {k:payload[k] for k in keys}
    protocol.update(candidate=payload['candidate_config'], data_access_mode='lazy_batch_provider')
    if candidate_protocol_hash(**protocol) != payload['protocol_sha256']:
        raise ValueError('preserved protocol does not reproduce')
    result = deepcopy(payload)
    result['config']['cuda_graph_recurrence'] = graph
    result['encoder_manifest']['backbone_config']['cuda_graph_recurrence'] = graph
    result['protocol_sha256'] = candidate_protocol_hash(**(protocol | {'config':result['config']}))
    result.setdefault('execution_history', []).append(dict(parent_path=str(parent_path),
        parent_sha256=sha256_file(parent_path), evidence_sha256=evidence_sha256,
        from_graph=payload['config']['cuda_graph_recurrence'], to_graph=graph,
        preserved_updates=payload['optimizer_updates'], canonical_training_state_sha256=before))
    if canonical_training_state_sha256(*(result[k] for k in STATE_KEYS)) != before:
        raise ValueError('execution migration changed training state')
    return result


def prepare_resume(config):
    evidence = read_evidence(config)
    parent, root = Path(config['stage45_resume_parent']).resolve(), Path(config['root']).resolve()
    if root == parent or root in parent.parents or parent in root.parents:
        raise ValueError('recovery needs a separate run root')
    source_changes = verify_execution_sources(parent)
    old = json.loads((parent/'pipeline_state.json').read_text())
    if old.get('children') or old['current_stage'] != 'stage45_screen__0':
        raise ValueError('recovery parent is not stopped at the preserved unit')
    for pid in (old['pid'], evidence['worker_pid']):
        path = Path(f'/proc/{pid}/cmdline')
        if path.exists() and str(parent).encode() in path.read_bytes():
            raise ValueError('preserved parent still runs')
    inventory = json.loads(Path(evidence['inventory_path']).read_text())
    for name, item in inventory['files'].items():
        if sha256_file(name) != item['sha256']:
            raise ValueError(f'parent artifact changed: {name}')
    if evidence['resume_pretraining_graph']:
        from chronaris.evaluation.application_tasks.checkpoint_performance import compare_trials
        trials = {k:Path(v) for k,v in evidence['trial_roots'].items()}
        if (not compare_trials(trials['eager'],trials['graph'])['passed']
            or not compare_trials(trials['graph'],trials['graph_resume'],exact=True)['passed']):
            raise ValueError('checkpoint numerical evidence no longer reproduces')
    relative = Path('units/screen/0/clare/chronaris/self_supervised/chronaris/C')
    checkpoint_rows = []
    for source in sorted((parent/relative).glob('*.pt')):
        original = torch.load(source, map_location='cpu', weights_only=True)
        if original['config']['max_updates'] != 300 or original['fold'] != evidence['fold']:
            raise ValueError('saved pretraining budget or fold changed')
        target = root/relative/source.name
        payload = (execution_checkpoint(original, parent_path=source, graph=True,
            evidence_sha256=sha256_file(config['stage45_resume_evidence']))
            if evidence['resume_pretraining_graph'] else original)
        if target.exists():
            saved = torch.load(target, map_location='cpu', weights_only=True)
            if saved['canonical_training_state_sha256'] != payload['canonical_training_state_sha256']:
                raise ValueError('recovery destination already has a different checkpoint')
        elif evidence['resume_pretraining_graph']:
            atomic_save_candidate(target, payload)
        else:
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        checkpoint_rows.append(dict(source=str(source), source_sha256=sha256_file(source),
            target=str(target), initial_target_sha256=sha256_file(target),
            preserved_updates=payload['optimizer_updates'], state_sha256=payload['canonical_training_state_sha256']))
    if not any(Path(r['target']).name == 'last.pt' and r['preserved_updates'] == evidence['saved_updates'] for r in checkpoint_rows):
        raise ValueError('preserved last checkpoint missing')
    for source in list((parent/'probes').glob('*/summary.json')) + list((parent/'performance').glob('*/*/check.json')):
        target = root/source.relative_to(parent); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise ValueError('inherited diagnostic result changed')
        shutil.copyfile(source,target)
    record = dict(status='completed', parent_root=str(parent), source_code_sha256=v4_workflow_source_sha256(),
        reviewed_source_changes=source_changes,
        evidence_sha256=sha256_file(config['stage45_resume_evidence']), checkpoint_rows=checkpoint_rows,
        completed_parent_steps=len(old['completed']), inherited_diagnostic_steps=sum(
            s.startswith(('stage45_probes__','stage45_perf__')) for s in old['completed']),
        historical_training_seconds_lower_bound=evidence['historical_training_seconds_lower_bound'],
        uncheckpointed_updates_not_reused=evidence['uncheckpointed_updates_not_reused'], confirmation_opened=False)
    return write_result(root/'recovery.json',record)


def inherited_step(stage, config):
    if not config.get('stage45_resume_parent') or not stage.startswith(('stage45_probes__','stage45_perf__')):
        return None
    read_evidence(config)
    parent = Path(config['stage45_resume_parent'])
    state = json.loads((parent/'pipeline_state.json').read_text())
    receipt = state['completed'].get(stage)
    if receipt is None or sha256_file(receipt['path']) != receipt['sha256']:
        raise ValueError('completed parent diagnostic receipt is missing or changed')
    result = json.loads(Path(receipt['path']).read_text())
    if result['status'] != 'completed':
        raise ValueError('cannot inherit unfinished diagnostic')
    return result | dict(inherited_evidence=dict(parent_path=receipt['path'], parent_sha256=receipt['sha256'],
        interpretation='historical execution qualification preserved; no new training or qualification claim'))
