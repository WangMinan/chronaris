"""Traceable device-only execution copy and independent legacy checkpoint resume."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys


GUARD = Path('src/chronaris/evaluation/application_tasks/v4_diagnostic_run.py')
RELATIVE_UNIT = Path('simulation/chronaris/no_continuous_evolution/fold01/seed17')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return value


def compatible_guard(source):
    old = '"4090" not in torch.cuda.get_device_name()'
    if source.count(old) != 1:
        raise ValueError('unexpected legacy device guard')
    return source.replace(old, 'torch.cuda.get_device_name() not in ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4070 SUPER")').replace(
        'v4 experiments require RTX 4090 and an approved seed',
        'v4 experiments require RTX 4090 or RTX 4070 SUPER and an approved seed')


def verify_copy(receipt):
    for name, digest in receipt['files'].items():
        if _sha(name) != digest:
            raise ValueError(f'device execution evidence changed: {name}')
    old, new = (Path(receipt[k]) for k in ('legacy_workspace', 'workspace'))
    paths = {p.relative_to(old) for p in (old/'src/chronaris').rglob('*.py')}
    if paths != {p.relative_to(new) for p in (new/'src/chronaris').rglob('*.py')}:
        raise ValueError('device copy file inventory changed')
    for path in paths:
        expected = compatible_guard((old/path).read_text()) if path == GUARD else (old/path).read_text()
        if (new/path).read_text() != expected:
            raise ValueError(f'change beyond the device guard: {path}')


def prepare_copy(*, legacy_workspace, workspace, core_root, pause_receipt, root):
    import torch
    old, new, core, root = (Path(p).resolve() for p in (legacy_workspace, workspace, core_root, root))
    if new.exists() or root.exists():
        raise ValueError('preserve existing device recovery; use an empty destination')
    config, pause = _read(core/'pipeline_config.json'), _read(pause_receipt)
    if pause['parent_root'] != str(core) or not pause['stopped'] or pause['completed_units'] != 18:
        raise ValueError('unexpected legacy pause')
    files = {str(Path(pause_receipt).resolve()): _sha(pause_receipt)}
    for k, h in (('checkpoint', 'checkpoint_sha256'), ('preserved_checkpoint', 'preserved_sha256')):
        if _sha(pause[k]) != pause[h]:
            raise ValueError('paused checkpoint changed')
        files[pause[k]] = pause[h]
    saved = torch.load(pause['checkpoint'], map_location='cpu', weights_only=True)
    pre = torch.load(saved['source_checkpoint_path'], map_location='cpu', weights_only=True)
    if (saved['step_count'] != 125 or saved['stage_update_counts'] != dict(pretraining=0, head_warmup=50, joint_adaptation=75)
        or pre['total_optimizer_updates'] != 1500 or saved['data_cursor']['samples_seen'] != 125*32
        or saved['data_cursor']['micro_batches_seen'] != 125*8
        or _sha(saved['source_checkpoint_path']) != saved['source_checkpoint_sha256']):
        raise ValueError('paused optimizer, sampling cursor or initializer changed')
    files[saved['source_checkpoint_path']] = saved['source_checkpoint_sha256']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=old, text=True).strip()
    subprocess.run(['git', 'worktree', 'add', '--detach', str(new), commit], cwd=old, check=True)
    (new/'artifacts').symlink_to((old/'artifacts').resolve(), target_is_directory=True)
    (new/GUARD).write_text(compatible_guard((old/GUARD).read_text()))
    command = [sys.executable, '-c', 'from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256; print(v4_workflow_source_sha256())']
    sources = [subprocess.check_output(command, cwd=p, env=os.environ | {'PYTHONPATH': str(p/'src')}, text=True).strip()
               for p in (old, new)]
    if sources[0] != config['source_code_sha256']:
        raise ValueError('legacy workflow source changed')
    for p in (old, new):
        for path in (p/'src/chronaris').rglob('*.py'):
            files[str(path)] = _sha(path)
        path = p/'scripts/evaluation/application_tasks/run_thesis_v4.py'
        files[str(path)] = _sha(path)
    for name in ('chapter5_plan.json', 'pipeline_config.json', 'pipeline_state.json'):
        files[str(core/name)] = _sha(core/name)
    files.update(config['input_files'])
    for receipt in pause['completed_receipts'].values():
        files[receipt['path']] = receipt['sha256']
    source_unit = core/'core_training'/RELATIVE_UNIT
    for source in source_unit.rglob('*'):
        if source.is_file():
            files[str(source)] = _sha(source)
    receipt = dict(status='prepared', legacy_workspace=str(old), workspace=str(new), core_root=str(core),
        root=str(root), legacy_commit=commit, legacy_source_sha256=sources[0], source_code_sha256=sources[1],
        files=files, preserved_counts=dict(pretraining=1500, head_warmup=50, joint=75, remaining_joint=425),
        cursor=saved['data_cursor'], confirmation_opened=False)
    verify_copy(receipt)
    for mode in ('continuous', 'resumed'):
        target = root/mode/RELATIVE_UNIT
        shutil.copytree(source_unit, target)
        state = _read(target/'run_state.json')
        if state['source']['source_code_sha256'] != sources[0]:
            raise ValueError('legacy run-state source changed')
        state['source']['source_code_sha256'] = sources[1]
        _write(target/'run_state.json', state)
        if _sha(target/'task_guided/chronaris/last.pt') != pause['checkpoint_sha256']:
            raise ValueError('copied training state changed')
    return _write(root/'execution_copy.json', receipt)


def run_legacy(*, receipt_path, mode, index=18, stop_at=None):
    """Run under the compatibility workspace imports, with the unchanged trainer."""
    from chronaris.evaluation.application_tasks import application_finetuning as guided
    from chronaris.evaluation.application_tasks.chapter5_core import _train
    from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
    import torch
    receipt = _read(receipt_path); verify_copy(receipt)
    if v4_workflow_source_sha256() != receipt['source_code_sha256']:
        raise ValueError('worker did not import the compatible frozen source')
    if mode not in ('continuous', 'resumed') or index not in (18, 21):
        raise ValueError('unplanned device recovery work')
    plan = _read(Path(receipt['core_root'])/'chapter5_plan.json')
    root = Path(receipt['root'])/mode
    original_save = guided._atomic_save
    class CheckpointBoundary(Exception):
        pass
    def save(path, payload):
        original_save(path, payload)
        if (stop_at is not None and Path(path).name == 'last.pt' and payload['step_count'] == stop_at
            and not payload.get('pending_validation')):
            raise CheckpointBoundary()
    guided._atomic_save = save
    try:
        result = _train(plan['units'][index], plan, root)
    except CheckpointBoundary:
        path = root/RELATIVE_UNIT/'task_guided/chronaris/last.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        result = dict(status='checkpoint_boundary', checkpoint=str(path), checkpoint_sha256=_sha(path),
                      optimizer_updates=saved['step_count'], data_cursor=saved['data_cursor'], confirmation_opened=False)
    finally:
        guided._atomic_save = original_save
    result['execution_copy'] = dict(path=str(receipt_path), sha256=_sha(receipt_path), device=torch.cuda.get_device_name())
    verify_copy(receipt)
    return _write(root/f'unit_{index}_{stop_at or "completed"}.json', result)


def compare_resume(receipt_path):
    import torch
    from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values
    receipt = _read(receipt_path); verify_copy(receipt)
    paths = [Path(receipt['root'])/m/RELATIVE_UNIT/'task_guided/chronaris/last.pt' for m in ('continuous', 'resumed')]
    a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in paths)
    if a['step_count'] != 175 or b['step_count'] != 175:
        raise ValueError('resume comparison requires the preserved 125 to 175 interval')
    keys = ('model_state_dict', 'optimizer_state_dict', 'rng_state', 'data_cursor', 'update_rows', 'epoch_rows',
            'stage_update_counts', 'protocol_sha256', 'best_update')
    checks = {k: compare_values(a[k], b[k], atol=0., rtol=0.) for k in keys}
    preserved = Path(receipt['root'])/'qualification_checkpoints'
    preserved.mkdir(exist_ok=False)
    for mode, path in zip(('continuous', 'resumed'), paths):
        shutil.copyfile(path, preserved/f'{mode}.pt')
    return _write(Path(receipt['root'])/'qualification.json', dict(status='completed',
        passed=all(c['close'] for c in checks.values()), checks=checks, optimizer_updates=175,
        files={str(p): _sha(p) for p in preserved.glob('*.pt')}, execution_copy=dict(path=str(receipt_path), sha256=_sha(receipt_path)),
        confirmation_opened=False, interpretation='same-device independent resume from an unchanged 4090 checkpoint'))


def qualified_execution(path, *, legacy_workspace, core_root):
    receipt = _read(path); verify_copy(receipt)
    if receipt['legacy_workspace'] != str(Path(legacy_workspace).resolve()) or receipt['core_root'] != str(Path(core_root).resolve()):
        raise ValueError('device execution belongs to another legacy experiment')
    qualification_path = Path(receipt['root'])/'qualification.json'
    qualification = _read(qualification_path)
    if (not qualification['passed'] or qualification['confirmation_opened']
        or qualification['execution_copy'] != dict(path=str(path), sha256=_sha(path))):
        raise ValueError('device independent resume qualification failed')
    from chronaris.evaluation.application_tasks.chapter5_assets import verify
    verify(qualification['files'])
    return receipt, {str(path): _sha(path), str(qualification_path): _sha(qualification_path)} | qualification['files']
