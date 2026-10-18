"""Independent arithmetic, source protection and fresh-process frozen replay."""
import csv
import gzip
import json
from pathlib import Path
import joblib
import numpy as np
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS, FairResidualData
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model, predict_fair
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def verify_fair_residual(report, replay=False):
    report = Path(report)
    protected = json.loads((report/'protected_sources.json').read_text())
    changed = [p for p,h in protected.items() if file_digest(p) != h]
    assert not changed, changed[:3]
    run = json.loads((report/'run_status.json').read_text())
    assert run['status'] in ('completed', 'base_gate_failed')
    binding = json.loads((report/'training_binding.json').read_text())
    assert file_digest(report/'PLAN.md') == binding['plan_sha256']
    assert file_digest(report/'qualification.json') == binding['qualification_sha256']
    assert file_digest(report/'real_prefix_check.json') == binding['real_prefix_sha256']
    qualification = json.loads((report/'qualification.json').read_text())
    snapshot = Path(qualification['snapshot_path']); assets = snapshot.parent
    assert file_digest(snapshot) == binding['snapshot_sha256'] == qualification['snapshot_sha256']
    receipts = {p.name.removesuffix('_receipt.json'): json.loads(p.read_text()) for p in report.glob('*_receipt.json')}
    updates = 0
    for name, r in receipts.items():
        assert r['complete'] and 20 <= r['epochs'] <= 100
        assert file_digest(r['checkpoint_path']) == r['checkpoint_sha256']
        if r['status'] == 'early_stopped':
            assert r['epochs']-r['best_epoch'] >= 15
        curves = list(csv.DictReader((report/f'{name}_history.csv').open()))
        assert len(curves) == r['epochs'] and int(curves[-1]['updates']) == r['updates']
        updates += r['updates']
    assert updates == run['updates'] and len(receipts) == run['completed_models']
    base_gates=json.loads((report/'base_gates.json').read_text())
    assert len(base_gates)==2
    for g in base_gates:
        assert g['passed']==(g['selection_rmse']<g['mean_rmse'] and g['selection_rmse']<=1.02*g['persistence_rmse'])
        assert abs(g['selection_rmse']-receipts[g['fold']+'_aircraft']['selection_rmse'])<1e-12
    check = dict(status='passed', run_status=run['status'], protected_files=len(protected), changed_sources=changed,
                 completed_models=len(receipts), optimizer_updates_in_original_run=updates,
                 optimizer_updates_in_verification=0)
    if run['status'] == 'completed':
        assert len(receipts) == 17
        assert all(g['passed'] for g in base_gates)
        lineage = json.loads((report/'crossfit_lineage.json').read_text())
        for spec in FOLDS:
            rows = [r for r in lineage if r['fold'] == spec['name']]
            assert {r['held_subject'] for r in rows} == set(spec['fit'])
            for r in rows:
                assert set(r['fit_subjects']) == set(spec['fit'])-{r['held_subject']}
                assert set(r['select_subjects']) == set(spec['select'])
                assert all(s.startswith(r['held_subject']+'::') for s in r['sample_ids'])
                receipt=receipts[spec['name']+'_oof_'+r['held_subject']]
                assert receipt['checkpoint_sha256']==r['checkpoint_sha256']
                state=torch.load(receipt['checkpoint_path'],map_location='cpu',weights_only=True)
                assert set(state['normalizer']['fit_subjects'])==set(r['fit_subjects'])
                assert state['fold']['score']==(r['held_subject'],)
        with gzip.open(report/'predictions.csv.gz', 'rt') as handle:
            predictions = list(csv.DictReader(handle))
        manifest={r['sample_id']:r for r in csv.DictReader((report/'sample_manifest.csv').open()) if r['accepted']=='True'}
        expected_methods={m+'_'+kind for m in ('aircraft','contiformer','mult','control','candidate') for kind in ('head','ridge')}
        expected_methods.update(('vehicle_statistics','dual_statistics','train_mean','train_median','persistence'))
        for spec in FOLDS:
            expected_samples={s for s,r in manifest.items() if r['subject'] in spec['score']}
            panel=[r for r in predictions if r['fold']==spec['name']]
            assert {r['method'] for r in panel}==expected_methods
            for method in expected_methods:
                rows=[r for r in panel if r['method']==method]
                assert len(rows)==len(expected_samples) and {r['sample_id'] for r in rows}==expected_samples
                assert all(float(r['truth'])==float(manifest[r['sample_id']]['target']) for r in rows)
        summaries = list(csv.DictReader((report/'summary.csv').open()))
        subjects = list(csv.DictReader((report/'subject_metrics.csv').open()))
        maximum = 0.
        for s in subjects:
            rows = [r for r in predictions if (r['fold'],r['method'],r['subject']) == (s['fold'],s['method'],s['subject'])]
            assert len(rows) == int(s['samples']) and len({r['sample_id'] for r in rows}) == len(rows)
            delta = np.array([float(r['prediction'])-float(r['truth']) for r in rows])
            rmse, mae = np.linalg.norm(delta)/np.sqrt(len(delta)), np.abs(delta).mean()
            maximum = max(maximum, abs(rmse-float(s['rmse'])), abs(mae-float(s['mae'])))
        for s in summaries:
            rows = [r for r in subjects if (r['fold'],r['method']) == (s['fold'],s['method'])]
            for metric in ('rmse', 'mae'):
                maximum = max(maximum, abs(np.mean([float(r[metric]) for r in rows])-float(s[metric])))
        assert maximum <= 1e-10
        gates = json.loads((report/'performance_gates.json').read_text())
        for spec in FOLDS:
            panel = {r['method']:r for r in summaries if r['fold'] == spec['name']}
            controls = {m:r for m,r in panel.items() if not m.startswith('candidate_')}
            strongest = min(controls, key=lambda m:float(controls[m]['rmse']))
            for method in ('candidate_ridge', 'candidate_head'):
                improvement = min(1-float(panel[method]['rmse'])/float(r['rmse']) for r in controls.values())
                wins = sum(next(float(r['rmse']) for r in subjects if r['fold']==spec['name'] and r['method']==method and r['subject']==s) <
                           next(float(r['rmse']) for r in subjects if r['fold']==spec['name'] and r['method']==strongest and r['subject']==s)
                           for s in spec['score'])
                mae_ok = all(float(panel[method]['mae']) <= 1.02*float(r['mae']) for r in controls.values())
                gate = next(r for r in gates if r['fold']==spec['name'] and r['candidate']==method)
                assert gate['passed'] == (improvement >= .02 and wins >= 2 and mae_ok)
        assert run['stable_success']==all(r['passed'] for r in gates)
        assert run['representation_success']==all(r['passed'] for r in gates if r['candidate']=='candidate_ridge')
        assert run['task_head_success']==all(r['passed'] for r in gates if r['candidate']=='candidate_head')
        check.update(predictions=len(predictions), subject_metrics=len(subjects), max_recompute_delta=float(maximum),
                     stable_success=run['stable_success'])
    if replay:
        check['replay'] = replay_fair(report, snapshot, assets, receipts, run['status'])
    save_json(report/'verification.json', check)
    return check


def replay_fair(report, snapshot, assets, receipts, status):
    torch.set_num_threads(1)
    if status == 'completed':
        with gzip.open(report/'predictions.csv.gz','rt') as f: prior = list(csv.DictReader(f))
        lock = json.loads((report/'locked_models.json').read_text())
    errors = {}
    with development_gpu_lock() as acquired:
        if not acquired:
            raise BlockingIOError('verification requires free GPU lock')
        for spec in FOLDS:
            name = spec['name'];data = FairResidualData(snapshot, report/'qualification.json', spec)
            model = load_fair_model(receipts[name+'_aircraft']['checkpoint_path'], data)
            ids = data.indices('select')
            _, pred = predict_fair(model,data,ids)
            rmse = float(np.sqrt(np.mean((pred[:,-1]-data.y[ids])**2)))
            assert abs(rmse-receipts[name+'_aircraft']['selection_rmse']) < 1e-8
            if status != 'completed':
                errors[name+'_selection'] = abs(rmse-receipts[name+'_aircraft']['selection_rmse']);continue
            data.score_open=True;ids=data.indices('score')
            base, anchor = predict_fair(model,data,ids);del model
            residual=FairResidualData(snapshot,report/'qualification.json',spec)
            with np.load(assets/f'{name}_crossfit.npz') as raw:
                residual.set_anchor(torch.tensor(raw['anchor'],device='cuda'))
            residual.score_open=True;residual.anchor[ids]=torch.tensor(anchor,device='cuda')
            path=assets/f'{name}_readouts.joblib';assert file_digest(path)==lock['readout_sha256'][name]
            readouts=joblib.load(path)
            generated={'aircraft_head':anchor[:,-1],'aircraft_ridge':readouts['aircraft'].predict(base)}
            for method in ('contiformer','mult','control','candidate'):
                current=residual if method in ('control','candidate') else data
                model=load_fair_model(receipts[name+'_'+method]['checkpoint_path'],current)
                features,pred=predict_fair(model,current,ids)
                if method in ('control','candidate'):features=np.c_[base,features]
                generated[method+'_head']=pred[:,-1];generated[method+'_ridge']=readouts[method].predict(features)
                del model
            for method, values in generated.items():
                rows=[r for r in prior if r['fold']==name and r['method']==method]
                assert [r['sample_id'] for r in rows]==[data.rows[i]['sample_id'] for i in ids]
                expected=np.array([float(r['prediction']) for r in rows])
                np.testing.assert_allclose(values,expected,atol=1e-5,rtol=1e-5)
                errors[name+'_'+method]=float(np.max(np.abs(values-expected)))
    return dict(status='passed',maximum_deltas=errors,optimizer_updates=0)
