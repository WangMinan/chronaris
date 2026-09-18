"""Explicit evidence for the whole-stage execution revision and result reuse."""
import json
from pathlib import Path

import torch

from chronaris.evaluation.application_tasks.execution_equivalence import POLICY, compare_runtime
from chronaris.evaluation.application_tasks.checkpoint_performance import compare_trials
from chronaris.evaluation.application_tasks.stage45_resume import execution_checkpoint, read_evidence
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.modeling.training.candidate_checkpoint import atomic_save_candidate, candidate_source_code_sha256
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file

FORMAT = 'chronaris.stage45_execution_recovery.v2'


def migrated_reference(config, source, *, graph):
    evidence = read_evidence(config)
    if evidence.get('format') != FORMAT:
        raise ValueError('reference migration requires the whole-stage execution revision')
    digest = sha256_file(source)
    if evidence['reference_checkpoints'].get(str(source)) != digest:
        raise ValueError('reference pretraining checkpoint is not in the verified inventory')
    target = Path(config['root'])/'pretraining_reuse'/digest/'best.pt'
    payload = torch.load(source, map_location='cpu', weights_only=True)
    migrated = execution_checkpoint(payload, parent_path=source, graph=graph,
        evidence_sha256=sha256_file(config['stage45_resume_evidence']),
        reviewed_source_sha256=evidence['parent_candidate_source_sha256'])
    if target.exists():
        existing = torch.load(target, map_location='cpu', weights_only=True)
        if (existing['canonical_training_state_sha256'] != migrated['canonical_training_state_sha256']
            or existing['protocol_sha256'] != migrated['protocol_sha256']):
            raise ValueError('reference migration destination changed')
    else:
        atomic_save_candidate(target, migrated)
    return str(target)


def compare_checkpoint_trials(eager_root, graph_root):
    """Apply the revision to existing trace objects without overwriting old checks."""
    eager_root, graph_root = Path(eager_root), Path(graph_root)
    checks = {}
    states = sorted(eager_root.glob('state_*.pt'))
    if not states or [p.name for p in states] != sorted(p.name for p in graph_root.glob('state_*.pt')):
        raise ValueError('checkpoint trial coverage differs')
    for p in states:
        a, b = [torch.load(r/p.name, map_location='cpu', weights_only=True) for r in (eager_root,graph_root)]
        if a['parent_sha256'] != b['parent_sha256']:
            raise ValueError('checkpoint trial parent differs')
        row = {k:compare_runtime(a[k],b[k]) for k in ('encoder_state_dict','head_state_dict',
            'explicit_time_shift_head_state_dict','optimizer_state_dict','rng_state','data_cursor')}
        name = p.name.replace('state_', 'observations_')
        a, b = [torch.load(r/name, map_location='cpu', weights_only=True) for r in (eager_root,graph_root)]
        row.update({k:compare_runtime(a[k],b[k],representation=k=='outputs') for k in ('outputs','losses','gradients','batches')})
        checks[p.stem] = row
    return dict(passed=all(c['close'] for row in checks.values() for c in row.values()),
                execution_policy=POLICY, comparisons=checks)


def build_evidence(root, *, parent, qualification_root, resume_trial_root):
    """Seal measured checks and exact source/artifact inventories before launch."""
    root, parent, qualification_root = map(Path, (root,parent,qualification_root))
    stop = json.loads((root/'stop_receipt.json').read_text())
    if not stop['processes_exited'] or Path(stop['parent']) != parent:
        raise ValueError('whole-stage migration requires the preserved stopped parent')
    state = json.loads((parent/'pipeline_state.json').read_text())
    index = int(state['current_stage'].split('__')[1])
    from chronaris.evaluation.application_tasks.stage45 import screen_units, parent_result
    unit = screen_units()[index]
    checkpoint = parent/f'units/screen/{index}/{unit["domain"]}/chronaris/self_supervised/chronaris/C/last.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload['optimizer_updates'] != stop['checkpoint_updates'] or state['children']
        or sha256_file(checkpoint) != stop['manifest'][str(checkpoint)]['sha256']):
        raise ValueError('preserved checkpoint or stopped state differs')
    numeric = compare_checkpoint_trials(root/'eager',root/'graph')
    resumed = compare_trials(root/'graph',Path(resume_trial_root),exact=True)
    interrupted = Path(json.loads((Path(resume_trial_root)/'metadata.json').read_text())['resume_state']).parent
    first_resume = compare_trials(root/'graph',interrupted,exact=True)
    resumed['first_saved_update'] = first_resume
    resumed['passed'] &= first_resume['passed']
    write_result(root/'revised_graph_comparison.json',numeric)
    write_result(root/'revised_graph_resume_comparison.json',resumed)
    times = {name:sum(sum(m['forward_backward_s'] for m in row['micro_timings'])+row['optimizer_s']
        for row in json.loads((root/name/'metadata.json').read_text())['measurements']) for name in ('eager','graph')}
    qualifications = {f'{d}/{r}':str(qualification_root/d/r/'check.json')
        for d in ('clare','cogpilot','dingxin') for r in ('stage4_reference','thesis_reference')}
    for path in qualifications.values():
        check = json.loads(Path(path).read_text())
        if not check['passed'] or check['execution_policy'] != POLICY:
            raise ValueError(f'unqualified whole-stage execution: {path}')
    extended_path = root/'extended_comparison.json'
    extended = json.loads(extended_path.read_text())
    if (not numeric['passed'] or not resumed['passed'] or times['eager'] <= times['graph']
        or not extended['passed'] or extended['execution_policy'] != POLICY
        or extended['last_update'] < payload['optimizer_updates']+6):
        raise ValueError('preserved checkpoint execution or exact resume not qualified')
    old_config = json.loads((parent/'pipeline_config.json').read_text())
    frozen, current = Path(old_config['registry_path']).parents[2], Path(__file__).parents[4]
    old = {str(p.relative_to(frozen)):sha256_file(p) for p in (frozen/'src/chronaris').rglob('*.py')}
    new = {str(p.relative_to(current)):sha256_file(p) for p in (current/'src/chronaris').rglob('*.py')}
    changes = {p:dict(old=old.get(p),new=new.get(p)) for p in old.keys()|new.keys() if old.get(p)!=new.get(p)}
    inventory = {str(p):dict(sha256=sha256_file(p)) for p in parent.rglob('*') if p.is_file() and p.suffix in ('.pt','.json','.npz','.joblib')}
    inventory_path = root/'parent_inventory_v2.json'
    write_result(inventory_path,dict(files=inventory))
    references = {}
    for domain, methods in {'clare':('chronaris','contiformer','physiology_only'),
                            'cogpilot':('chronaris','vehicle_only','mult')}.items():
        for method in methods:
            result = parent_result(old_config, domain, method)
            p = next(r['training']['best_checkpoint_path'] for r in result['results'] if r['route']=='self_supervised')
            references[p] = sha256_file(p)
    paths = [root/'stop_receipt.json', inventory_path, root/'revised_graph_comparison.json',
             root/'revised_graph_resume_comparison.json', extended_path] + list(map(Path,qualifications.values()))
    paths += [p for directory in (root/'eager',root/'graph',Path(resume_trial_root)) for p in directory.glob('*.pt')]
    paths += [p for p in qualification_root.rglob('*') if p.is_file() and p.suffix in ('.json','.pt','.npz','.joblib')]
    for source in qualification_root.rglob('eager_source.json'):
        baseline = json.loads(source.read_text())['files']
        if any(sha256_file(p) != digest for p,digest in baseline.items()):
            raise ValueError('preserved qualification baseline changed')
        paths += list(map(Path,baseline))
    paths += list((root/'continuation').rglob('*.pt')) + list(interrupted.glob('*.pt'))
    evidence = dict(status='completed',format=FORMAT,parent_root=str(parent),inventory_path=str(inventory_path),
        worker_pid=stop['worker_pid'],saved_updates=payload['optimizer_updates'],unit_index=index,domain=unit['domain'],
        uncheckpointed_updates_not_reused=stop['unsaved_updates'],fold=payload['fold'],
        candidate_source_sha256=candidate_source_code_sha256(),parent_candidate_source_sha256=payload['source_code_sha256'],
        workflow_source_sha256=v4_workflow_source_sha256(),reviewed_source_changes=changes,execution_policy=POLICY,
        bindings={str(p):sha256_file(p) for p in paths},qualification_checks=qualifications,reference_checkpoints=references,
        graph_comparison_path=str(root/'revised_graph_comparison.json'),
        graph_resume_comparison_path=str(root/'revised_graph_resume_comparison.json'),
        extended_comparison_path=str(extended_path),
        pretraining_compute_seconds=times,pretraining_compute_speedup=times['eager']/times['graph'],resume_pretraining_graph=True,
        historical_training_seconds_lower_bound=payload['training_elapsed_s'],confirmation_opened=False)
    return write_result(root/'recovery_evidence_v2.json',evidence)
