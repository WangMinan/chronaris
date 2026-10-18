"""Frozen six-model readouts with the exact shared historical mean."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model,predict_fair
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.shape_mult_run import shape_data,VERSIONS
from chronaris.evaluation.application_tasks.shape_mult_results import prediction_rows,metric_tables,pairing_predictions
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock

PARENT=Path('docs/review/stage/eligible-fit-20261019')
STYLES=('raw64','augmented65','residual64')


def design(features,anchor,style):
    features=np.asarray(features,dtype=np.float64);anchor=np.asarray(anchor,dtype=np.float64)
    assert features.ndim==2 and len(features)==len(anchor) and np.isfinite(features).all() and np.isfinite(anchor).all()
    if style not in STYLES:raise ValueError(style)
    return np.column_stack((features,anchor)) if style=='augmented65' else features


def choose(cache,style):
    fit=design(cache['fit_features'],cache['fit_anchor'],style)
    val=design(cache['select_features'],cache['select_anchor'],style)
    a=cache['fit_y']-(cache['fit_anchor'] if style=='residual64' else 0)
    b=cache['select_y']-(cache['select_anchor'] if style=='residual64' else 0)
    return fit_regressor(fit,a,val,b,alpha_values=(.1,1.,10.,100.,1000.),scaler_with_mean=True,
                         train_sample_weight=cache['fit_weights'],validation_groups=cache['select_subjects'])


def prediction(model,features,anchor,style):
    return model.predict(design(features,anchor,style))+(anchor if style=='residual64' else 0)


class PairReadout:
    def __init__(self,model,anchor,style):self.model,self.anchor,self.style=model,anchor,style
    def predict(self,features):return prediction(self.model,features,self.anchor,self.style)


def gates_for(people,summary):
    gates=[]
    for fold,panel in summary.groupby('fold'):
        controls=panel[~panel.method.str.startswith(('candidate_','previous_candidate_'))]
        best=controls.loc[controls.rmse.idxmin()]
        pivot=people[people.fold==fold].pivot(index='subject',columns='method',values='rmse')
        for style in STYLES:
            own=panel[panel.method=='candidate_'+style].iloc[0]
            gain=float((1-own.rmse/controls.rmse).min());wins=int((pivot[own.method]<pivot[best.method]).sum())
            mae=bool((own.mae<=1.02*controls.mae).all())
            gates.append(dict(fold=fold,style=style,strongest=best.method,improvement=gain,subjects_improved=wins,
                              mae_passed=mae,passed=gain>=.02 and wins>=2 and mae))
    return gates


def parent_metadata():
    assert json.loads((PARENT/'run_status.json').read_text())['status']=='completed'
    assert json.loads((PARENT/'verification.json').read_text())['status']=='passed'
    root=Path(json.loads((PARENT/'asset_manifest.json').read_text())['root'])
    lock=json.loads((PARENT/'locked_primary.json').read_text())
    assert file_digest(lock['readout_path'])==lock['readout_sha256']
    return root,lock


def export_role(model,data,role):
    ids=data.indices(role);features=predict_fair(model,data,ids)[0]
    assert features.shape==(len(ids),64)
    return {role+'_features':features,role+'_anchor':data.anchor[ids,-1].cpu().numpy().astype(np.float64),
            role+'_y':data.y[ids],role+'_subjects':data.subjects[ids],
            role+'_ids':np.array([data.rows[i]['sample_id'] for i in ids]),
            role+'_weights':data.weights[ids].cpu().numpy()}


def run_observed_mean(report,assets):
    report,assets=Path(report),Path(assets);assets.mkdir(parents=True,exist_ok=False)
    assert not (report/'run_status.json').exists()
    protected=json.loads((report/'protected_sources.json').read_text())
    assert all(file_digest(p)==h for p,h in protected.items())
    root,parent=parent_metadata();start=time.monotonic();torch.set_num_threads(1)
    deadline=min(start+1200,start+1792373400-time.time())
    def check_time():
        if time.monotonic()>deadline:raise TimeoutError('readout audit budget')
    models={};choices={};caches={};items=[];rows=[];replays={}
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        for spec in parent['binding']['folds']:
            for version in VERSIONS:
                check_time();name=spec['name']+'_'+version;data=shape_data(PARENT,root,spec,version)
                receipt=parent['receipts'][name];assert file_digest(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
                model=load_fair_model(receipt['checkpoint_path'],data);cache={}
                for role in ('fit','select'):cache.update(export_role(model,data,role))
                caches[name]=cache
                for style in STYLES:models[name+'_'+style],choices[name+'_'+style]=choose(cache,style)
                path=assets/(name+'_selection.npz');np.savez_compressed(path,**cache)
                items.append(dict(name=name,fold=spec,version=version,checkpoint_path=receipt['checkpoint_path'],
                                  checkpoint_sha256=receipt['checkpoint_sha256'],cache_path=str(path.resolve()),cache_sha256=file_digest(path)))
                del model,data;torch.cuda.empty_cache();print(name,'readouts selected',flush=True)
        path=assets/'readouts.joblib';joblib.dump(models,path)
        save_json(report/'locked_readouts.json',dict(items=items,selected_alpha=choices,readout_path=str(path.resolve()),
                  readout_sha256=file_digest(path),plan_sha256=file_digest(report/'PLAN.md'),source_sha256=file_digest(__file__),outer_scored=False))
        previous=pd.read_csv(PARENT/'predictions.csv.gz',dtype={'sample_id':str,'subject':str})
        reference=previous.copy();reference['method']='previous_'+reference.method;rows+=reference.to_dict('records')
        old=joblib.load(parent['readout_path'])
        for item in items:
            check_time();name=item['name'];data=shape_data(PARENT,root,item['fold'],item['version']);data.score_open=True
            model=load_fair_model(item['checkpoint_path'],data);cache=export_role(model,data,'score')
            path=assets/(name+'_score.npz');np.savez_compressed(path,**cache)
            before=previous[(previous.fold==item['fold']['name'])&(previous.method==item['version']+'_ridge')]
            assert before.sample_id.tolist()==cache['score_ids'].tolist()
            actual=old[name].predict(cache['score_features']);np.testing.assert_allclose(actual,before.prediction,atol=1e-5,rtol=1e-5)
            replays[name]=float(np.max(np.abs(actual-before.prediction.to_numpy())))
            for style in STYLES:
                values=prediction(models[name+'_'+style],cache['score_features'],cache['score_anchor'],style)
                rows+=prediction_rows(data,data.indices('score'),item['version']+'_'+style,values)
            del model,data;torch.cuda.empty_cache()
        people,summary=metric_tables(rows);gates=gates_for(people,summary)
        pd.DataFrame(rows).to_csv(report/'predictions.csv.gz',index=False)
        people.to_csv(report/'subject_metrics.csv',index=False);summary.to_csv(report/'summary.csv',index=False)
        save_json(report/'performance_gates.json',gates);save_json(report/'parent_replay.json',replays)
        qualified=[s for s in STYLES if all(g['passed'] for g in gates if g['style']==s)]
        paired=[];donors=[];pairgates=[]
        for style in qualified:
            for item in (i for i in items if i['version']=='candidate'):
                check_time();data=shape_data(PARENT,root,item['fold'],'candidate');data.score_open=True;ids=data.indices('score')
                readout=PairReadout(models[item['name']+'_'+style],data.anchor[ids,-1].cpu().numpy().astype(np.float64),style)
                p,d=pairing_predictions(data,item['checkpoint_path'],readout)
                for row in p:row['method']=style+'_'+row['method']
                for row in d:row['style']=style
                paired+=p;donors+=d;del data
        if paired:
            pp,ps=metric_tables(paired);pd.DataFrame(paired).to_csv(report/'pairing_predictions.csv.gz',index=False)
            pp.to_csv(report/'pairing_subject_metrics.csv',index=False);ps.to_csv(report/'pairing_summary.csv',index=False)
            pd.DataFrame(donors).to_csv(report/'pairing_donors.csv',index=False)
            for style in qualified:
                for fold in summary.fold.unique():
                    base=float(summary[(summary.fold==fold)&(summary.method=='candidate_'+style)].rmse.iloc[0])
                    panel=ps[(ps.fold==fold)&ps.method.str.startswith(style+'_')]
                    shift=float(panel[panel.method.str.contains('different_record')].rmse.mean()/base-1)
                    zero=float(panel[panel.method==style+'_zero_values'].rmse.iloc[0])
                    pairgates.append(dict(fold=fold,style=style,degradation=shift,zero_rmse=zero,passed=shift>=.02 and base<=zero))
        assert all(file_digest(p)==h for p,h in protected.items());check_time()
        save_json(report/'pairing_gates.json',pairgates)
        result=dict(status='completed',encoder_updates=0,readout_selections=18,ridge_fits=90,qualified_styles=qualified,
                    paired_styles=[s for s in qualified if all(g['passed'] for g in pairgates if g['style']==s)],elapsed_s=time.monotonic()-start)
        save_json(report/'run_status.json',result);return result
