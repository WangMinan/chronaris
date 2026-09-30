"""Preserve eight completed formal units across the bounded SensorLLM repair."""
import hashlib
import json
from pathlib import Path

from chronaris.evaluation.application_tasks.stage5_recovery import read, verify
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def prepare_recovery(config, configured):
    parent, root = Path(config['stage5_formal_parent']).resolve(), Path(config['root']).resolve()
    if parent == root or parent in root.parents or root in parent.parents:
        raise ValueError('formal recovery needs a separate root')
    previous = read(parent/'pipeline_config.json')
    old_root, project = Path(previous['registry_path']).parents[2], Path(__file__).parents[4]
    old = {p.relative_to(old_root): p for p in (old_root/'src/chronaris').rglob('*.py')}
    new = {p.relative_to(project): p for p in (project/'src/chronaris').rglob('*.py')}
    allowed = {'src/chronaris/modeling/fusion_encoders/sensorllm_adapter.py'} | {
        f'src/chronaris/evaluation/application_tasks/{name}.py' for name in
        ('stage5_formal', 'stage5_formal_recovery', 'v4_pipeline')}
    changed = [str(p) for p in sorted(old.keys() | new.keys())
        if (old[p].read_bytes() if p in old else None) != (new[p].read_bytes() if p in new else None)]
    if not set(changed) <= allowed:
        raise ValueError('unreviewed computation changed outside the SensorLLM repair')
    entry = Path('scripts/evaluation/application_tasks/run_thesis_v4.py')
    digest = hashlib.sha256()
    for name in sorted(old) + [entry]:
        digest.update(str(name).encode()); digest.update((old_root/name).read_bytes())
    if digest.hexdigest() != previous['source_code_sha256'] or (old_root/entry).read_bytes() != (project/entry).read_bytes():
        raise ValueError('parent frozen workflow changed')
    state, plan = read(parent/'pipeline_state.json'), read(parent/'formal_plan.json')
    completed = {k: v for k, v in state['completed'].items() if k.startswith('stage5_formal_unit__')}
    if (state['status'] != 'failed' or state['children'] or state['current_stage'] != 'stage5_formal_unit__8'
        or set(completed) != {f'stage5_formal_unit__{i}' for i in range(8)}):
        raise ValueError('recovery requires the preserved eight-unit SensorLLM halt')
    if (plan['units'] != json.loads(json.dumps(configured['units']))
        or plan['consumer_policy'] != configured['consumer_policy'] or plan['confirmation_opened']):
        raise ValueError('formal matrix, budget or consumer protocol changed')
    files = {str(parent/name): sha256_file(parent/name) for name in
        ('pipeline_state.json', 'pipeline_config.json', 'formal_plan.json')}
    files.update(previous['input_files']); files.update(plan['input_files'])
    files.update({r['path']: r['sha256'] for r in state['completed'].values()})
    files.update({str(p): sha256_file(p) for p in old.values()})
    files[str(old_root/entry)] = sha256_file(old_root/entry)
    for index in range(8):
        prior = read(completed[f'stage5_formal_unit__{index}']['path'])
        if (prior['status'] != 'completed' or prior['unit'] != plan['units'][index]
            or prior['source_code_sha256'] != previous['source_code_sha256']
            or prior['plan_sha256'] != plan['plan_sha256'] or prior['confirmation_opened']):
            raise ValueError('completed formal unit provenance changed')
        files.update(prior['model_files']); files.update(prior['consumer_files'])
        for row in prior['results']:
            evaluation = row.get('evaluation', row)
            path = Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json'
            files[str(path)] = evaluation['contract_file_sha256']
            files.update(read(path)['declaration']['evidence_files'])
            for feature in row.get('feature_files', {}).values(): files[feature['path']] = feature['sha256']
    partial = parent/'training/8/clare/sensorllm_deepseek/alignment/alignment.pt'
    files[str(partial)] = sha256_file(partial)
    verify(files)
    inventory = dict(status='completed', parent_root=str(parent), files=files, completed=completed,
        reviewed_source_changes=changed, source_code_sha256=v4_workflow_source_sha256(),
        parent_source_code_sha256=previous['source_code_sha256'], parent_failures=state['failures'],
        partial_checkpoint_preserved=str(partial), failed_sensor_unit_restarts=True,
        completed_units_reused=8, new_encoder_updates_in_reused_units=0, confirmation_opened=False)
    path = root/'recovery_inventory.json'
    if path.exists() and read(path) != inventory: raise ValueError('formal recovery inventory changed')
    write_result(path, inventory)
    return dict(path=str(path), sha256=sha256_file(path))


def reuse_completed(config, frozen, index):
    receipt = frozen.get('recovery')
    if receipt is None or index >= 8: return None
    verify({receipt['path']: receipt['sha256']})
    inventory = read(receipt['path'])
    if inventory['source_code_sha256'] != v4_workflow_source_sha256():
        raise ValueError('formal recovery source changed')
    verify(inventory['files'])
    source = inventory['completed'][f'stage5_formal_unit__{index}']
    prior = read(source['path'])
    if prior['unit'] != frozen['units'][index] or prior['unit']['method'] == 'sensorllm_deepseek':
        raise ValueError('cannot inherit changed SensorLLM or another matrix unit')
    return write_result(Path(config['root'])/'units'/f'{index}.json', prior | dict(
        plan_sha256=frozen['plan_sha256'], parent_plan_sha256=prior['plan_sha256'],
        reuse=dict(parent_receipt=source, source_code_sha256=prior['source_code_sha256'],
            verification_source_code_sha256=v4_workflow_source_sha256(), extra_encoder_updates=0),
        historical_total_seconds=prior['total_seconds'], total_seconds=0.))
