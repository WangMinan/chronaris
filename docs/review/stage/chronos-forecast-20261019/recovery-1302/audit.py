"""Verify the interrupted unit without optimizer updates or score access."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import forecast_data, forecast_folds, predict_forecast
from chronaris.evaluation.application_tasks.chronos_forecast_results import load_selected
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.training.rng import canonical_training_state_sha256

report = Path(__file__).resolve().parent.parent
recovery = Path(__file__).resolve().parent
assets_root = Path('artifacts/application_evaluation/2026-10-19-chronos-forecast')
backup = recovery / 'before-restart'
backup.mkdir(exist_ok=False)
for path in report.iterdir():
    if path.is_file():
        shutil.copy2(path, backup / path.name)
asset_backup = assets_root / 'recovery-1302'
asset_backup.mkdir(exist_ok=False)
for path in assets_root.glob('*_unit.log'):
    shutil.copy2(path, asset_backup / path.name)
for path in (assets_root / 'training/fold02_aircraft').iterdir():
    if path.is_file():
        shutil.copy2(path, asset_backup / ('fold02_aircraft_' + path.name))
save_json(recovery / 'backup_manifest.json', {
    str(p.resolve()): file_digest(p) for root in (backup, asset_backup) for p in root.iterdir() if p.is_file()
})
protected = json.loads((report / 'protected_sources.json').read_text())
assert all(file_digest(p) == h for p, h in protected.items())
binding = json.loads((report / 'training_binding.json').read_text())
assert all(file_digest(p) == h for p, h in binding['source_files'].items())
for filename, key in [('PLAN.md', 'plan_sha256'), ('qualification.json', 'qualification_sha256'),
                      ('input_qualification.json', 'input_qualification_sha256')]:
    assert file_digest(report / filename) == binding[key]
assets = verify_assets(ASSETS)
assert file_digest(ASSETS) == binding['assets_sha256']
torch.set_num_threads(1)
checks = []
with development_gpu_lock() as acquired:
    assert acquired, 'GPU busy'
    for fold_index, view in [(0, 'aircraft'), (0, 'joint'), (1, 'aircraft')]:
        spec = forecast_folds()[fold_index]
        name = spec['name'] + '_' + view
        root = assets_root / 'training' / name
        state = torch.load(root / 'last.pt', map_location='cpu', weights_only=True)
        assert state['metadata']['binding'] == binding
        assert file_digest(root / 'best.pt') == state['best_sha256']
        data, x, mask, proof = forecast_data(spec, view)
        assert state['metadata']['proof'] == proof
        assert state['metadata']['normalizer'] == data.normalizer
        assert state['metadata']['target_scale'] == data.target_scale
        fit = data.indices('fit')
        assert state['metadata']['fit_sample_ids'] == [data.rows[i]['sample_id'] for i in fit]
        assert len(state['steps']) == state['updates']
        assert all(int(v['step']) == state['updates'] for v in state['optimizer']['state'].values())
        for tensor in state['model'].values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                assert torch.isfinite(tensor).all()
        for value in state['optimizer']['state'].values():
            assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
        assert {'torch_cpu', 'torch_cuda'} <= state['rng'].keys()
        check = dict(name=name, updates=state['updates'], epoch=state['epoch'], next_offset=state['next_offset'],
                     last_sha256=file_digest(root / 'last.pt'), best_sha256=state['best_sha256'])
        if fold_index == 0:
            receipt = json.loads((report / (name + '_receipt.json')).read_text())
            assert receipt['complete'] and receipt['updates'] == state['updates'] == 18760
            assert receipt['last_sha256'] == check['last_sha256']
            assert receipt['checkpoint_sha256'] == check['best_sha256']
            assert receipt['metadata'] == state['metadata']
            assert state['epoch'] == 6 and state['next_offset'] == 0
            model = load_selected(assets, receipt, 'adapted')
            ids = data.indices('select')
            predicted = predict_forecast(model, data, x, mask, ids, time.time() + 300)
            rmse = float(np.sqrt(np.mean((predicted - data.y[ids]) ** 2)))
            delta = abs(rmse - receipt['selection_rmse'])
            assert delta < 1e-5
            check.update(complete=True, replay_windows=len(ids), replay_rmse=rmse, replay_delta=delta)
            del model
        else:
            assert state['updates'] == 6000 and state['epoch'] == 2
            assert state['next_offset'] == state['seen'] == 8124
            order = np.random.default_rng(17 + state['epoch']).permutation(fit)
            assert state['orders'][str(state['epoch'])] == canonical_training_state_sha256([data.rows[i]['sample_id'] for i in order])
            assert check['last_sha256'] == file_digest(asset_backup / 'fold02_aircraft_last.pt')
            progress = json.loads((backup / 'training_progress.json').read_text())
            assert progress['stage'] == name and progress['updates'] >= state['updates']
            check.update(complete=False, unsaved_updates_at_least=progress['updates'] - state['updates'])
        checks.append(check)
        print(json.dumps(check), flush=True)
        del state, data, x, mask
        torch.cuda.empty_cache()
save_json(recovery / 'checkpoint_audit.json', dict(status='passed', checks=checks,
    protected_files=len(protected), changed_sources=[], optimizer_updates=0, outer_scored=False,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
