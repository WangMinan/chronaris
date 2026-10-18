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
    receipt=ORIGINAL/'qualification.json'
    return FairResidualData(json.loads(receipt.read_text())['snapshot_path'],receipt,spec)


def run_shape_mult(report,assets):
    report,assets=Path(report),Path(assets)
    if (report/'run_status.json').exists():raise FileExistsError('existing run retained')
    assert json.loads((report/'qualification.json').read_text())['status']=='passed'
    assert all(file_digest(p)==h for p,h in json.loads((report/'protected_sources.json').read_text()).items())
    torch.set_num_threads(1);start=time.monotonic()
    binding=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 plan_sha256=file_digest(report/'PLAN.md'),qualification_sha256=file_digest(report/'qualification.json'),
                 seed=17,folds=FOLDS)
    save_json(report/'training_binding.json',binding)
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU unavailable')
        try:
            return _execute(report,assets,binding,start)
        except BaseException as exc:
            save_json(report/'run_status.json',dict(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start))
            raise


def _execute(report,assets,binding,start):
    receipts={};readouts={};choices={};datasets={};rows=[];pair_rows=[];donors=[]
    def train(spec,version,method):
        name=spec['name']+'_'+version
        data=shape_data(report,assets,spec,version)
        datasets[name]=data
        save_json(report/'run_status.json',dict(status='running',stage=name,elapsed_s=time.monotonic()-start))
        model,r=fit_fair_model(method,data,assets/'training'/name,report,name,binding,start+14400)
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
    for spec in FOLDS:
        for version,method in VERSIONS.items():train(spec,version,method)
    lock('locked_primary')
    old=pd.read_csv(ORIGINAL/'predictions.csv.gz')
    old_methods=('persistence','train_mean','train_median','aircraft_head','aircraft_ridge',
                 'contiformer_head','contiformer_ridge','mult_head','mult_ridge','vehicle_statistics','dual_statistics')
    reference=old[old.method.isin(old_methods)].copy();reference['method']='frozen_'+reference.method
    rows.extend(reference.to_dict('records'))
    replay={}
    for spec in FOLDS:
        fold=spec['name']
        for version in VERSIONS:
            name=fold+'_'+version;data=datasets[name]
            scored=score_shape_model(data,receipts[name]['checkpoint_path'],readouts[name],version)
            rows+=scored
            if version=='original':
                for route in ('ridge','head'):
                    values=pd.DataFrame([r for r in scored if r['method']=='original_'+route])
                    before=old[(old.fold==fold)&(old.method=='mult_'+route)]
                    assert values.sample_id.tolist()==before.sample_id.tolist()
                    np.testing.assert_allclose(values.prediction,before.prediction,atol=1e-5,rtol=1e-5)
                    replay[fold+'_'+route]=float(np.abs(values.prediction.to_numpy()-before.prediction.to_numpy()).max())
            if version=='candidate':
                ids=data.indices('score')
                rows+=prediction_rows(data,ids,'shape_statistics',readouts[fold+'_shape_statistics'].predict(data.statistics(ids,True)))
                p,d=pairing_predictions(data,receipts[name]['checkpoint_path'],readouts[name]);pair_rows+=p;donors+=d
    save_json(report/'original_reproduction.json',replay)
    preliminary=write_shape_results(report,rows,pair_rows,donors)
    add_baseline=preliminary['representation_success'] and preliminary['pairing_success']
    save_json(report/'conditional_baseline_decision.json',dict(run=add_baseline,primary_result=preliminary))
    if add_baseline:
        for spec in FOLDS:train(spec,'repaired_contiformer','contiformer')
        lock('locked_with_baseline')
        for spec in FOLDS:
            name=spec['name']+'_repaired_contiformer'
            rows+=score_shape_model(datasets[name],receipts[name]['checkpoint_path'],readouts[name],'repaired_contiformer')
    result=write_shape_results(report,rows,pair_rows,donors)
    result.update(status='completed',completed_models=len(receipts),updates=sum(r['updates'] for r in receipts.values()),
                  stable_success=result['representation_success'] and result['pairing_success'] and add_baseline,
                  same_input_traditional_checked=add_baseline,elapsed_s=time.monotonic()-start,
                  scope='two_previously_exposed_development_folds_seed17',binding=binding)
    save_json(report/'run_status.json',result)
    return result
