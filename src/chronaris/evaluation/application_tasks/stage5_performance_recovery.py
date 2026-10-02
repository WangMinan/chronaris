"""Reuse completed formal results after a source-bound, validated execution change."""
import hashlib
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.stage5_recovery import read, verify
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def prepare_performance_recovery(config, configured):
    parent, root = Path(config['stage5_formal_parent']).resolve(), Path(config['root']).resolve()
    if parent == root or parent in root.parents or root in parent.parents:
        raise ValueError('performance recovery requires separate roots')
    evidence_path = Path(config['stage5_performance_evidence'])
    evidence = read(evidence_path)
    if (evidence['status'] != 'completed' or not evidence['passed']
        or evidence['source_code_sha256'] != v4_workflow_source_sha256()
        or evidence['parent_root'] != str(parent) or evidence['confirmation_opened']):
        raise ValueError('performance qualification is incomplete or belongs to another source')
    verify(evidence['files'])
    stop = read(evidence['stop_receipt'])
    if not stop['stopped'] or stop['parent_root'] != str(parent) or stop['active_optimizer_updates'] != 0:
        raise ValueError('performance recovery requires a stopped boundary without partial training')
    for pid in (stop['controller_pid'], *stop['children'].values()):
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists() and str(parent).encode() in command.read_bytes():
            raise ValueError('parent remains live')
    previous, state, plan = (read(parent/name) for name in ('pipeline_config.json','pipeline_state.json','formal_plan.json'))
    completed = {k:v for k,v in state['completed'].items() if k.startswith('stage5_formal_unit__')}
    if (state['status'] != 'failed' or state['children'] or len(completed) != stop['completed_units']
        or any(stop['completed_receipts'].get(v['path']) != v['sha256'] for v in completed.values())):
        raise ValueError('parent completion boundary changed')
    if set(completed) != {f'stage5_formal_unit__{i}' for i in range(stop['completed_units'])}:
        raise ValueError('parent completion matrix has a gap')
    if (plan['units'] != json.loads(json.dumps(configured['units']))
        or plan['consumer_policy'] != configured['consumer_policy'] or plan['confirmation_opened']):
        raise ValueError('performance recovery changed the research protocol')
    old_root, project = Path(previous['registry_path']).parents[2], Path(__file__).parents[4]
    old = {p.relative_to(old_root):p for p in (old_root/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(project):p for p in (project/'src/chronaris').rglob('*.py')}
    changes = [str(p) for p in sorted(old.keys() | new.keys())
        if (old[p].read_bytes() if p in old else None) != (new[p].read_bytes() if p in new else None)]
    if changes != evidence['reviewed_source_changes']:
        raise ValueError('source differences exceed the qualified execution revision')
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((old_root/name).read_bytes())
    if digest.hexdigest() != previous['source_code_sha256'] or (old_root/entry).read_bytes() != (project/entry).read_bytes():
        raise ValueError('parent frozen workflow changed')
    files = evidence['files'] | {str(evidence_path):sha256_file(evidence_path)}
    files.update({str(parent/name):sha256_file(parent/name) for name in ('pipeline_config.json','pipeline_state.json','formal_plan.json')})
    files.update(previous['input_files']); files.update(plan['input_files'])
    files.update({r['path']:r['sha256'] for r in state['completed'].values()})
    files.update({str(p):sha256_file(p) for p in old.values()}); files[str(old_root/entry)] = sha256_file(old_root/entry)
    if plan.get('recovery'):
        verify({plan['recovery']['path']:plan['recovery']['sha256']})
        inherited = read(plan['recovery']['path']); files.update(inherited['files'])
        files[plan['recovery']['path']] = plan['recovery']['sha256']
    unit_files = {}
    for i in range(len(completed)):
        receipt = completed[f'stage5_formal_unit__{i}']
        prior = read(receipt['path'])
        if (prior['status'] != 'completed' or prior['unit'] != plan['units'][i]
            or prior['plan_sha256'] != plan['plan_sha256'] or prior['confirmation_opened']):
            raise ValueError('completed formal provenance changed')
        used = {receipt['path']:receipt['sha256']} | prior['model_files'] | prior['consumer_files']
        for row in prior['results']:
            evaluation = row.get('evaluation', row)
            path = Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
            used[str(path)] = evaluation['contract_file_sha256']
            used.update(read(path)['declaration']['evidence_files'])
            for feature in row.get('feature_files', {}).values(): used[feature['path']] = feature['sha256']
        files.update(used); unit_files[str(i)] = sorted(used)
    verify(files)
    inventory = dict(status='completed', parent_root=str(parent), files=files, completed=completed,
        execution={}, unit_files=unit_files, reviewed_source_changes=changes, source_code_sha256=v4_workflow_source_sha256(),
        parent_source_code_sha256=previous['source_code_sha256'], parent_failures=state['failures'],
        failed_sensor_unit_restarts=False, completed_units_reused=len(completed),
        new_encoder_updates_in_reused_units=0, confirmation_opened=False)
    path = root/'recovery_inventory.json'
    if path.exists() and read(path) != inventory: raise ValueError('performance recovery inventory changed')
    write_result(path, inventory)
    return dict(path=str(path), sha256=sha256_file(path))
