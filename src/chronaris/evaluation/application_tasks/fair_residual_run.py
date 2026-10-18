"""Two predeclared folds, cross-fitted anchors, matched correction controls."""
from pathlib import Path
import json
import subprocess
import time
import joblib
import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS, FairResidualData
from chronaris.evaluation.application_tasks.fair_residual_training import fit_fair_model, predict_fair, load_fair_model
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def fit_readout(data, features):
    a, b = data.indices('fit'), data.indices('select')
    return fit_regressor(features['fit'], data.y[a], features['select'], data.y[b],
                         alpha_values=(.1, 1., 10., 100., 1000.), scaler_with_mean=True,
                         train_sample_weight=data.weights[a].cpu().numpy(), validation_groups=data.subjects[b])


def run_fair_residual(snapshot, report, assets):
    report, assets = Path(report), Path(assets)
    if (report/'run_status.json').exists():
        raise FileExistsError('existing run cannot be overwritten')
    if json.loads((report/'real_prefix_check.json').read_text())['status'] != 'passed':
        raise ValueError('real historical prefix check did not pass')
    torch.set_num_threads(1)
    started = time.monotonic()
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('GPU is held by another experiment')
        if not torch.cuda.is_available():
            raise RuntimeError('registered neural training requires CUDA')
        binding = dict(snapshot_sha256=file_digest(snapshot), plan_sha256=file_digest(report/'PLAN.md'),
                       qualification_sha256=file_digest(report/'qualification.json'),
                       real_prefix_sha256=file_digest(report/'real_prefix_check.json'),
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       device=torch.cuda.get_device_name(), folds=FOLDS)
        save_json(report/'training_binding.json', binding)
        try:
            return _execute(Path(snapshot), report, assets, binding, started)
        except BaseException as exc:
            save_json(report/'run_status.json', dict(status='failed', error=repr(exc), elapsed_s=time.monotonic()-started))
            raise


def _execute(snapshot, report, assets, binding, started):
    root = assets/'training'; root.mkdir(exist_ok=False)
    folds, receipts, gates = {}, {}, []

    def train(method, data, name):
        save_json(report/'run_status.json', dict(status='running', stage=name, elapsed_s=time.monotonic()-started))
        model, receipt = fit_fair_model(method, data, root/name, report, name, binding, started+14400)
        receipts[name] = receipt
        if not receipt['complete']:
            raise TimeoutError(f'registered model budget incomplete: {name}')
        return model

    for spec in FOLDS:
        name = spec['name']; data = FairResidualData(snapshot, report/'qualification.json', spec)
        model = train('aircraft', data, name+'_aircraft')
        cache = {role: predict_fair(model, data, data.indices(role)) for role in ('fit', 'select')}
        a, b = data.indices('fit'), data.indices('select')
        constant = float(np.average(data.y[a], weights=data.weights[a].cpu().numpy()))
        mean_rmse = float(np.sqrt(np.mean((data.y[b]-constant)**2)))
        persistence_rmse = float(np.sqrt(np.mean((data.y[b]-[data.rows[i]['persistence'] for i in b])**2)))
        error = receipts[name+'_aircraft']['selection_rmse']
        gate = dict(fold=name, selection_rmse=error, mean_rmse=mean_rmse, persistence_rmse=persistence_rmse,
                    passed=error < mean_rmse and error <= 1.02*persistence_rmse)
        gates.append(gate)
        folds[name] = dict(data=data, cache={'aircraft': cache}, spec=spec)
        del model
        save_json(report/'base_gates.json', gates)
    if not all(g['passed'] for g in gates):
        result = dict(status='base_gate_failed', gates=gates, candidate_trained=False, score_opened=False,
                      completed_models=len(receipts), updates=sum(r['updates'] for r in receipts.values()),
                      elapsed_s=time.monotonic()-started)
        save_json(report/'run_status.json', result)
        return result

    oof_lineage = []
    for name, item in folds.items():
        data, spec = item['data'], item['spec']
        anchor = data.anchor.clone()
        filled = set()
        for held in spec['fit']:
            inner = dict(name=name+'_oof_'+held, fit=tuple(s for s in spec['fit'] if s != held),
                         select=spec['select'], score=(held,))
            inner_data = FairResidualData(snapshot, report/'qualification.json', inner)
            model = train('aircraft', inner_data, inner['name'])
            inner_data.score_open = True
            ids = inner_data.indices('score')
            _, pred = predict_fair(model, inner_data, ids)
            if filled & set(ids.tolist()):
                raise ValueError('OOF identity repeated')
            filled.update(ids.tolist())
            anchor[ids] = torch.tensor(pred, device=data.device)
            oof_lineage.append(dict(fold=name, held_subject=held, fit_subjects=inner['fit'], select_subjects=inner['select'],
                                    sample_ids=[data.rows[i]['sample_id'] for i in ids],
                                    checkpoint_sha256=receipts[inner['name']]['checkpoint_sha256']))
            del model, inner_data
        assert filled == set(data.indices('fit').tolist())
        select = data.indices('select')
        anchor[select] = torch.tensor(item['cache']['aircraft']['select'][1], device=data.device)
        residual_data = FairResidualData(snapshot, report/'qualification.json', spec)
        residual_data.set_anchor(anchor)
        np.savez_compressed(assets/f'{name}_crossfit.npz', anchor=anchor.cpu().numpy(),
                            fit_indices=data.indices('fit'), select_indices=select)
        item['residual_data'] = residual_data
        for method in ('contiformer', 'mult', 'control', 'candidate'):
            current = residual_data if method in ('control', 'candidate') else data
            base_hash = file_digest(root/(name+'_aircraft')/'best.pt')
            model = train(method, current, name+'_'+method)
            item['cache'][method] = {role: predict_fair(model, current, current.indices(role)) for role in ('fit', 'select')}
            assert file_digest(root/(name+'_aircraft')/'best.pt') == base_hash
            del model
        item['readouts'], item['choices'] = {}, {}
        for method in ('aircraft', 'contiformer', 'mult', 'control', 'candidate'):
            features = {role: item['cache'][method][role][0] for role in ('fit', 'select')}
            if method in ('control', 'candidate'):
                features = {role: np.c_[item['cache']['aircraft'][role][0], values] for role, values in features.items()}
            item['readouts'][method], item['choices'][method] = fit_readout(data, features)
        for method, physiology in [('vehicle_statistics', False), ('dual_statistics', True)]:
            features = {role: data.statistics(data.indices(role), physiology) for role in ('fit', 'select')}
            item['readouts'][method], item['choices'][method] = fit_readout(data, features)
        joblib.dump(item['readouts'], assets/f'{name}_readouts.joblib')
    save_json(report/'crossfit_lineage.json', oof_lineage)
    lock = dict(binding=binding, receipts=receipts, selected_ridge={name: item['choices'] for name,item in folds.items()},
                readout_sha256={name: file_digest(assets/f'{name}_readouts.joblib') for name in folds}, outer_scores_opened=False)
    save_json(report/'locked_models.json', lock)
    from chronaris.evaluation.application_tasks.fair_residual_results import score_fair_folds
    result = score_fair_folds(folds, root, report, assets, receipts)
    result.update(elapsed_s=time.monotonic()-started, completed_models=len(receipts),
                  updates=sum(r['updates'] for r in receipts.values()), binding=binding)
    save_json(report/'run_status.json', result)
    return result
