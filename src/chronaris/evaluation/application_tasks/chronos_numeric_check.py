"""Fixed-hyperparameter precision diagnosis; no encoder or model selection updates."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_information_readout import feature_rows,VIEWS,ROUTES
from chronaris.evaluation.application_tasks.chronos_information_qualification import information_data
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows
from chronaris.evaluation.application_tasks.shape_mult_results import prediction_rows,metric_tables


def check_chronos_precision(report,assets):
    report,assets=Path(report),Path(assets)
    if (report/'numeric_check.json').exists():raise FileExistsError('retained numeric check')
    status=json.loads((report/'encoding_status.json').read_text());assert status['status']=='completed'
    locked=json.loads((report/'locked_readouts.json').read_text());assert file_digest(locked['path'])==locked['sha256']
    for u in status['units']:
        for route in ROUTES:assert file_digest(u[route+'_path'])==u[route+'_sha256']
    original=pd.read_csv(report/'predictions.csv.gz');models={};selection=[]
    for spec in FOLDS:
        data,_,_=information_data(spec);a,b=data.indices('fit'),data.indices('select')
        for route in ROUTES:
            for view in VIEWS:
                name=spec['name']+'_'+route+'_'+view;alpha=locked['choices'][name]['alpha']
                x=feature_rows(status['units'],spec['name'],route,view,a).astype(np.float64)
                v=feature_rows(status['units'],spec['name'],route,view,b).astype(np.float64)
                model,chosen=fit_regressor(x,data.y[a],v,data.y[b],alpha_values=(alpha,),scaler_with_mean=True,
                                          train_sample_weight=data.weights[a].numpy(),validation_groups=data.subjects[b])
                assert chosen==alpha;models[name]=model
                selection.append(dict(name=name,alpha=alpha,rmse=float(np.sqrt(np.mean((model.predict(v)-data.y[b])**2))),
                                      original_rmse=locked['choices'][name]['selection_rmse']))
                print(name,'solved float64',flush=True);del x,v
    path=assets/'numeric_float64.joblib';joblib.dump(models,path)
    save_json(report/'numeric_locked.json',dict(path=str(path.resolve()),sha256=file_digest(path),
                                               original_readout_sha256=locked['sha256'],outer_opened=False,selection=selection))
    rows=[];diff=[]
    for spec in FOLDS:
        data,_,_=information_data(spec);data.score_open=True;ids=data.indices('score')
        for route in ROUTES:
            for view in VIEWS:
                name=spec['name']+'_'+route+'_'+view;method='chronos_'+route+'_'+view
                values=feature_rows(status['units'],spec['name'],route,view,ids).astype(np.float64)
                pred=models[name].predict(values);assert np.isfinite(pred).all()
                old=original[(original.fold==spec['name'])&(original.method==method)]
                assert old.sample_id.tolist()==[data.rows[i]['sample_id'] for i in ids]
                rows+=prediction_rows(data,ids,method,pred)
                diff.append(dict(fold=spec['name'],method=method,maximum_prediction_delta=float(np.abs(pred-old.prediction).max())))
    people,summary=metric_tables(rows);old=pd.read_csv(report/'summary.csv')
    for r in diff:
        a=float(summary[(summary.fold==r['fold'])&(summary.method==r['method'])].rmse.iloc[0])
        b=float(old[(old.fold==r['fold'])&(old.method==r['method'])].rmse.iloc[0])
        r.update(float64_rmse=a,original_rmse=b,relative_rmse_change=a/b-1,sensitive=abs(a/b-1)>.01)
    pd.DataFrame(rows).to_csv(report/'numeric_predictions.csv.gz',index=False)
    people.to_csv(report/'numeric_subject_metrics.csv',index=False);save_rows(report/'numeric_comparison.csv',diff)
    result=dict(status='completed',encoder_updates=0,readout_fits=16,fixed_original_alphas=True,
                sensitive_routes=[r for r in diff if r['sensitive']],maximum_relative_rmse_change=max(abs(r['relative_rmse_change']) for r in diff))
    save_json(report/'numeric_check.json',result);return result
