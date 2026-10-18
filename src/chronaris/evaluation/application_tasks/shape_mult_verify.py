"""Independent arithmetic, binding checks and fresh-process scoring replay."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_shape_features import shape_scale_features,shape_scale_floors
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model,predict_fair
from chronaris.evaluation.application_tasks.shape_mult_run import ORIGINAL,VERSIONS,shape_data
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def same_tensor_bytes(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))


def verify_shape_mult(report,assets,replay=True):
    report,assets=Path(report),Path(assets)
    protected=json.loads((report/'protected_sources.json').read_text())
    assert all(file_digest(p)==h for p,h in protected.items())
    run=json.loads((report/'run_status.json').read_text());assert run['status']=='completed'
    binding=json.loads((report/'training_binding.json').read_text())
    assert file_digest(report/'PLAN.md')==binding['plan_sha256']
    assert file_digest(report/'qualification.json')==binding['qualification_sha256']
    for ref in binding.get('reference_reports',[]):
        assert file_digest(ref['predictions_path'])==ref['predictions_sha256']
    if binding.get('history_aux_weight',0.):
        path=report/'history_anchor_qualification.json'
        assert file_digest(path)==binding['history_anchor_qualification_sha256']
        q=json.loads(path.read_text());assert q['status']=='passed' and file_digest(q['mask_path'])==q['mask_sha256']
    locked=json.loads((report/('locked_with_baseline.json' if run['same_input_traditional_checked'] else 'locked_primary.json')).read_text())
    assert file_digest(locked['readout_path'])==locked['readout_sha256']
    original_path=json.loads((ORIGINAL/'qualification.json').read_text())['snapshot_path']
    with np.load(original_path,allow_pickle=False) as raw:old={k:raw[k] for k in raw.files}
    meta=json.loads(str(old['metadata']));subjects=np.array([r['subject'] for r in meta['windows']])
    with np.load(assets/'native.npz',allow_pickle=False) as raw:native={k:raw[k] for k in raw.files}
    for spec in FOLDS:
        q=json.loads((report/(spec['name']+'_qualification.json')).read_text())
        assert file_digest(q['snapshot_path'])==q['snapshot_sha256']
        fit=np.flatnonzero(np.isin(subjects,spec['fit']))
        assert q['floors']==shape_scale_floors(native['scales'],fit)
        assert q['fit_sample_ids']==[meta['windows'][i]['sample_id'] for i in fit]
        with np.load(q['snapshot_path'],allow_pickle=False) as raw:
            for key in old:
                if key!='physiology':np.testing.assert_array_equal(raw[key],old[key])
            np.testing.assert_array_equal(raw['physiology'],shape_scale_features(native['values'],native['masks'],native['scales'],q['floors']))
    receipts=locked['receipts'];updates=0
    for name,r in receipts.items():
        assert file_digest(r['checkpoint_path'])==r['checkpoint_sha256']
        assert r['complete'] and r['status'] in ('early_stopped','max_epochs')
        h=pd.read_csv(report/(name+'_history.csv'))
        assert len(h)==r['epochs'] and h.updates.iloc[-1]==r['updates']
        assert 20<=r['epochs']<=100
        if r['status']=='early_stopped':assert r['epochs']-r['best_epoch']>=15
        state=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
        assert json.loads(json.dumps(state['binding']))==binding
        assert state['normalizer']['fit_subjects']==list(state['fold']['fit'])
        updates+=r['updates']
    assert updates==run['updates'] and len(receipts)==run['completed_models']
    prediction=pd.read_csv(report/'predictions.csv.gz');pair=pd.read_csv(report/'pairing_predictions.csv.gz')
    people=pd.read_csv(report/'subject_metrics.csv');summary=pd.read_csv(report/'summary.csv')
    pair_people=pd.read_csv(report/'pairing_subject_metrics.csv');pair_summary=pd.read_csv(report/'pairing_summary.csv')
    maximum=0.
    for data,persons,totals in ((prediction,people,summary),(pair,pair_people,pair_summary)):
        assert not data[['fold','method','sample_id']].duplicated().any()
        for spec in FOLDS:
            expected={r['sample_id']:r['target'] for r in meta['windows'] if r['subject'] in spec['score']}
            for method,block in data[data.fold==spec['name']].groupby('method'):
                assert set(block.sample_id)==set(expected)
                np.testing.assert_allclose(block.truth,[expected[s] for s in block.sample_id],atol=1e-12,rtol=0)
        for row in persons.itertuples():
            block=data[(data.fold==row.fold)&(data.method==row.method)&(data.subject==row.subject)]
            assert len(block)==row.samples
            err=block.prediction.to_numpy()-block.truth.to_numpy()
            maximum=max(maximum,abs(np.linalg.norm(err)/np.sqrt(len(err))-row.rmse),abs(np.abs(err).mean()-row.mae))
        for row in totals.itertuples():
            block=persons[(persons.fold==row.fold)&(persons.method==row.method)]
            maximum=max(maximum,abs(block.rmse.mean()-row.rmse),abs(block.mae.mean()-row.mae))
    assert maximum<=1e-10
    gates=json.loads((report/'performance_gates.json').read_text());pair_gates=json.loads((report/'pairing_gates.json').read_text())
    for g in gates:
        panel=summary[summary.fold==g['fold']].set_index('method')
        ctrl=panel[~panel.index.str.startswith('candidate_')]
        strongest=ctrl.rmse.idxmin();own=panel.loc['candidate_'+g['route']]
        win=people[people.fold==g['fold']].pivot(index='subject',columns='method',values='rmse')
        gain=float((1-own.rmse/ctrl.rmse).min());wins=int((win['candidate_'+g['route']]<win[strongest]).sum())
        mae=bool((own.mae<=ctrl.mae*1.02).all())
        assert g['strongest']==strongest and abs(g['improvement']-gain)<1e-10
        assert g['passed']==(gain>=.02 and wins>=2 and mae)
    for g in pair_gates:
        p=pair_summary[pair_summary.fold==g['fold']].set_index('method').rmse
        baseline=float(summary[(summary.fold==g['fold'])&(summary.method=='candidate_ridge')].rmse.iloc[0])
        degradation=float(p[p.index.str.startswith('different_record')].mean()/baseline-1)
        assert abs(degradation-g['mean_pairing_degradation'])<1e-10
        assert g['passed']==(degradation>=.02 and baseline<=p['zero_values'])
    expected=all(g['passed'] for g in gates if g['route']=='ridge') and all(g['passed'] for g in pair_gates) and run['same_input_traditional_checked']
    assert run['stable_success']==expected
    original_weights={}
    for spec in FOLDS:
        name=spec['name']+'_original';new=torch.load(receipts[name]['checkpoint_path'],map_location='cpu',weights_only=True)
        before=json.loads((ORIGINAL/(spec['name']+'_mult_receipt.json')).read_text())
        previous=torch.load(before['checkpoint_path'],map_location='cpu',weights_only=True)
        assert new['state'].keys()==previous['state'].keys()
        original_weights[spec['name']]=all(same_tensor_bytes(new['state'][k],previous['state'][k]) for k in new['state'])
    if binding.get('selection_policy','task_head')=='task_head':assert all(original_weights.values())
    errors={}
    if replay:
        torch.set_num_threads(1)
        readouts=joblib.load(locked['readout_path'])
        with development_gpu_lock() as acquired:
            if not acquired:raise BlockingIOError('GPU busy')
            for spec in FOLDS:
                versions=list(VERSIONS)+(['repaired_contiformer'] if run['same_input_traditional_checked'] else [])
                for version in versions:
                    name=spec['name']+'_'+version;data=shape_data(report,assets,spec,version);data.score_open=True
                    ids=data.indices('score');model=load_fair_model(receipts[name]['checkpoint_path'],data)
                    features,head=predict_fair(model,data,ids)
                    generated={version+'_head':head[:,-1],version+'_ridge':readouts[name].predict(features)}
                    if version=='candidate':generated['shape_statistics']=readouts[spec['name']+'_shape_statistics'].predict(data.statistics(ids,True))
                    for method,values in generated.items():
                        oldp=prediction[(prediction.fold==spec['name'])&(prediction.method==method)]
                        assert oldp.sample_id.tolist()==[data.rows[i]['sample_id'] for i in ids]
                        np.testing.assert_allclose(values,oldp.prediction,atol=1e-5,rtol=1e-5)
                        errors[spec['name']+'_'+method]=float(np.max(np.abs(values-oldp.prediction)))
                    del model,data
    result=dict(status='passed',protected_files=len(protected),changed_sources=[],predictions=len(prediction),
                pairing_predictions=len(pair),subject_metrics=len(people)+len(pair_people),maximum_metric_delta=maximum,
                completed_models=len(receipts),original_mult_weights_identical=original_weights,
                replay=errors,optimizer_updates_in_verification=0,new_readout_fits=0,stable_success=run['stable_success'])
    assert all(file_digest(p)==h for p,h in protected.items())
    save_json(report/'verification.json',result)
    save_json(report/'asset_manifest.json',dict(root=str(assets.resolve()),files={str(p.relative_to(assets)):file_digest(p) for p in sorted(assets.rglob('*')) if p.is_file()}))
    return result
