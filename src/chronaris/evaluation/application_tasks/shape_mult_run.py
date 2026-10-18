"""Six primary fits only; conditional fair traditional baseline, no matrix expansion."""
import json
from pathlib import Path
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS,FairResidualData
from chronaris.evaluation.application_tasks.fair_residual_run import fit_readout
from chronaris.evaluation.application_tasks.fair_residual_training import fit_fair_model,predict_fair
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.shape_mult_results import score_shape_model,pairing_predictions,prediction_rows,write_shape_results
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


ORIGINAL=Path('docs/review/stage/fair-residual-20261019')
VERSIONS={'aircraft':'mult_aircraft','original':'mult','candidate':'mult'}


def shape_data(report,assets,spec,version):
    if version in ('candidate','repaired_contiformer'):
        return FairResidualData(assets/(spec['name']+'_inputs.npz'),report/(spec['name']+'_qualification.json'),spec)
    receipt=report/'original_qualification.json' if (report/'folds.json').exists() else ORIGINAL/'qualification.json'
    return FairResidualData(json.loads(receipt.read_text())['snapshot_path'],receipt,spec)


def run_shape_mult(report,assets,selection_policy='task_head',seed=17,history_aux_weight=0.,rest_calibration=False):
    report,assets=Path(report),Path(assets)
    if (report/'run_status.json').exists():raise FileExistsError('existing run retained')
    assert json.loads((report/'qualification.json').read_text())['status']=='passed'
    assert all(file_digest(p)==h for p,h in json.loads((report/'protected_sources.json').read_text()).items())
    if selection_policy not in ('task_head','common_readout'):raise ValueError(selection_policy)
    if selection_policy=='task_head' and seed!=17:raise ValueError('original trainer is fixed to seed17')
    if history_aux_weight not in (0.,.5) or (history_aux_weight and selection_policy!='common_readout'):
        raise ValueError('unregistered history supervision')
    torch.set_num_threads(1);start=time.monotonic()
    binding=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 plan_sha256=file_digest(report/'PLAN.md'),qualification_sha256=file_digest(report/'qualification.json'),
                 seed=seed,folds=FOLDS,selection_policy=selection_policy,history_aux_weight=history_aux_weight)
    if rest_calibration:
        prior=Path('docs/review/stage/eligible-fit-20261019')
        result=json.loads((prior/'run_status.json').read_text())
        assert result['status']=='completed' and json.loads((prior/'verification.json').read_text())['status']=='passed'
        assert not (result['representation_success'] and result['pairing_success'])
        assert selection_policy=='common_readout' and seed==17 and history_aux_weight==0.
        q=json.loads((report/'qualification.json').read_text())
        assert q['condition']=='preflight_unlabelled_rest_observation_available'
        binding.update(rest_calibration=True,versions={'candidate':'mult'},predecessor_sha256=file_digest(prior/'run_status.json'))
    if (report/'folds.json').exists() or rest_calibration:
        import datetime
        if (report/'folds.json').exists():
            binding['folds']=json.loads((report/'folds.json').read_text())
            binding['expanded_fit']=True
        binding['deadline']=datetime.datetime.fromisoformat('2026-10-19T09:30:00+08:00').timestamp()
        if time.time()>=binding['deadline']:raise TimeoutError('night deadline reached')
    if history_aux_weight:
        qualification=json.loads((report/'history_anchor_qualification.json').read_text())
        assert qualification['status']=='passed'
        assert file_digest(qualification['mask_path'])==qualification['mask_sha256']
        binding['history_anchor_qualification_sha256']=file_digest(report/'history_anchor_qualification.json')
    if (report/'reference_reports.json').exists():
        binding['reference_reports']=json.loads((report/'reference_reports.json').read_text())
        for ref in binding['reference_reports']:
            assert file_digest(ref['predictions_path'])==ref['predictions_sha256']
    save_json(report/'training_binding.json',binding)
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU unavailable')
        try:
            trainer=fit_fair_model
            if selection_policy=='common_readout':
                from chronaris.evaluation.application_tasks.readout_selected_training import fit_readout_selected
                trainer=fit_readout_selected
            return _execute(report,assets,binding,start,trainer)
        except BaseException as exc:
            save_json(report/'run_status.json',dict(status='budget_incomplete' if isinstance(exc,TimeoutError) else 'failed',
                                                   error=repr(exc),elapsed_s=time.monotonic()-start))
            raise


def _execute(report,assets,binding,start,trainer=fit_fair_model):
    receipts={};readouts={};choices={};datasets={};rows=[];pair_rows=[];donors=[]
    versions=binding.get('versions',VERSIONS)
    def train(spec,version,method):
        name=spec['name']+'_'+version
        data=shape_data(report,assets,spec,version)
        if binding.get('history_aux_weight',0.):
            q=json.loads((report/'history_anchor_qualification.json').read_text())
            assert q['sample_ids']==[r['sample_id'] for r in data.rows]
            with np.load(q['mask_path'],allow_pickle=False) as raw:
                data.history_anchor_mask=torch.tensor(raw['mask'],device=data.device)
        datasets[name]=data
        save_json(report/'run_status.json',dict(status='running',stage=name,elapsed_s=time.monotonic()-start))
        deadline=min(start+14400,time.monotonic()+binding['deadline']-time.time()) if 'deadline' in binding else start+14400
        model,r=trainer(method,data,assets/'training'/name,report,name,binding,deadline)
        receipts[name]=r
        if not r['complete']:raise TimeoutError('bounded training incomplete')
        features={role:predict_fair(model,data,data.indices(role))[0] for role in ('fit','select')}
        readouts[name],choices[name]=fit_readout(data,features)
        if version=='candidate':
            stats={role:data.statistics(data.indices(role),True) for role in ('fit','select')}
            readouts[spec['name']+'_shape_statistics'],choices[spec['name']+'_shape_statistics']=fit_readout(data,stats)
        del model
    def lock(name):
        path=assets/(name+'.joblib');joblib.dump(readouts,path)
        save_json(report/(name+'.json'),dict(receipts=receipts,selected_alpha=choices,readout_path=str(path.resolve()),
                                             readout_sha256=file_digest(path),binding=binding))
    for spec in binding['folds']:
        for version,method in versions.items():train(spec,version,method)
    lock('locked_primary')
    old=pd.read_csv(ORIGINAL/'predictions.csv.gz')
    old_methods=('persistence','train_mean','train_median','aircraft_head','aircraft_ridge',
                 'contiformer_head','contiformer_ridge','mult_head','mult_ridge','vehicle_statistics','dual_statistics')
    reference=old[old.method.isin(old_methods)].copy();reference['method']='frozen_'+reference.method
    rows.extend(reference.to_dict('records'))
    for ref in binding.get('reference_reports',[]):
        prior=pd.read_csv(ref['predictions_path'])
        prior=prior[prior.method.isin(ref['methods'])].copy()
        prior['method']=ref['prefix']+'_'+prior.method
        rows.extend(prior.to_dict('records'))
    replay={}
    for spec in binding['folds']:
        fold=spec['name']
        for version in versions:
            name=fold+'_'+version;data=datasets[name]
            scored=score_shape_model(data,receipts[name]['checkpoint_path'],readouts[name],version)
            rows+=scored
            if version=='original':
                for route in ('ridge','head'):
                    values=pd.DataFrame([r for r in scored if r['method']=='original_'+route])
                    before=old[(old.fold==fold)&(old.method=='mult_'+route)]
                    assert values.sample_id.tolist()==before.sample_id.tolist()
                    if binding.get('selection_policy','task_head')=='task_head' and not binding.get('expanded_fit'):
                        np.testing.assert_allclose(values.prediction,before.prediction,atol=1e-5,rtol=1e-5)
                    replay[fold+'_'+route]=float(np.abs(values.prediction.to_numpy()-before.prediction.to_numpy()).max())
            if version=='candidate':
                ids=data.indices('score')
                rows+=prediction_rows(data,ids,'shape_statistics',readouts[fold+'_shape_statistics'].predict(data.statistics(ids,True)))
                p,d=pairing_predictions(data,receipts[name]['checkpoint_path'],readouts[name]);pair_rows+=p;donors+=d
    save_json(report/'original_reproduction.json',replay)
    preliminary=write_shape_results(report,rows,pair_rows,donors)
    add_baseline=preliminary['representation_success'] and preliminary['pairing_success'] and not binding.get('expanded_fit')
    save_json(report/'conditional_baseline_decision.json',dict(run=add_baseline,primary_result=preliminary))
    if add_baseline:
        for spec in binding['folds']:train(spec,'repaired_contiformer','contiformer')
        lock('locked_with_baseline')
        for spec in binding['folds']:
            name=spec['name']+'_repaired_contiformer'
            rows+=score_shape_model(datasets[name],receipts[name]['checkpoint_path'],readouts[name],'repaired_contiformer')
    result=write_shape_results(report,rows,pair_rows,donors)
    result.update(status='completed',completed_models=len(receipts),updates=sum(r['updates'] for r in receipts.values()),
                  stable_success=result['representation_success'] and result['pairing_success'] and add_baseline,
                  same_input_traditional_checked=add_baseline,elapsed_s=time.monotonic()-start,
                  scope=f'two_previously_exposed_development_folds_seed{binding["seed"]}',binding=binding)
    save_json(report/'run_status.json',result)
    return result
