"""Locked native forecasts, same-condition controls, and conditional pairing checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import forecast_data,forecast_folds,VIEWS,predict_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.mult_diagnosis import different_record_donors
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows
from chronaris.evaluation.application_tasks.shape_mult_results import prediction_rows,metric_tables
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.chronos_forecast import load_forecast_model
from chronaris.modeling.fusion_encoders.recent_models import verify_assets


ROUTES=('frozen','adapted')


def forecast_gates(people,summary):
    gates=[]
    for fold,panel in summary.groupby('fold'):
        for route in ROUTES:
            name='native_'+route+'_joint';controls=panel[panel.method.str.startswith('reference_') |
                                                                (panel.method=='native_'+route+'_aircraft')]
            best=controls.loc[controls.rmse.idxmin()];own=panel[panel.method==name].iloc[0]
            pivot=people[people.fold==fold].pivot(index='subject',columns='method',values='rmse')
            gain=float((1-own.rmse/controls.rmse).min());wins=int((pivot[name]<pivot[best.method]).sum())
            mae=bool((own.mae<=1.02*controls.mae).all())
            gates.append(dict(fold=fold,route=route,strongest=best.method,improvement=gain,subjects_improved=wins,
                              mae_passed=mae,passed=gain>=.02 and wins>=2 and mae))
    return gates


def load_selected(assets,unit,route):
    model=load_forecast_model(assets).eval()
    if route=='adapted':
        assert file_digest(unit['checkpoint_path'])==unit['checkpoint_sha256']
        saved=torch.load(unit['checkpoint_path'],map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True);del saved
    return model


def score_chronos_forecast(report,root,deadline=NIGHT_DEADLINE):
    report,root=Path(report),Path(root)
    if (report/'performance_status.json').exists():raise FileExistsError('retained native forecast results')
    locked=json.loads((report/'locked_models.json').read_text());units=locked['units'];assert len(units)==4
    assert json.loads((report/'run_status.json').read_text())['status']=='completed'
    for u in units:
        assert file_digest(u['receipt_path'])==u['receipt_sha256'] and file_digest(u['checkpoint_path'])==u['checkpoint_sha256']
    protected=json.loads((report/'protected_sources.json').read_text());assert all(file_digest(p)==h for p,h in protected.items())
    references=json.loads((report/'reference_reports.json').read_text());rows=[]
    for reference in references:
        assert file_digest(reference['path'])==reference['sha256']
        before=pd.read_csv(reference['path'],dtype={'sample_id':str,'subject':str})
        if reference.get('methods'):before=before[before.method.isin(reference['methods'])]
        before=before.copy();before['method']=reference['prefix']+'_'+before.method;rows+=before.to_dict('records')
    assets=verify_assets(ASSETS);torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        for spec in forecast_folds():
            for view in VIEWS:
                data,x,mask,_=forecast_data(spec,view);data.score_open=True;ids=data.indices('score')
                unit=next(u for u in units if u['fold']==spec['name'] and u['view']==view)
                for route in ROUTES:
                    model=load_selected(assets,unit,route)
                    values=predict_forecast(model,data,x,mask,ids,deadline)
                    rows+=prediction_rows(data,ids,'native_'+route+'_'+view,values)
                    del model;torch.cuda.empty_cache();print(spec['name'],route,view,'scored',flush=True)
                del data,x,mask
        people,summary=metric_tables(rows);gates=forecast_gates(people,summary)
        pd.DataFrame(rows).to_csv(report/'predictions.csv.gz',index=False)
        people.to_csv(report/'subject_metrics.csv',index=False);summary.to_csv(report/'summary.csv',index=False)
        save_json(report/'performance_gates.json',gates)
        qualified=[r for r in ROUTES if all(g['passed'] for g in gates if g['route']==r)]
        pair_gates=forecast_pairing(report,assets,units,qualified,summary,deadline) if qualified else []
    assert all(file_digest(p)==h for p,h in protected.items());verify_assets(ASSETS)
    save_json(report/'pairing_gates.json',pair_gates)
    result=dict(status='completed',qualified_routes=qualified,
                paired_routes=[r for r in qualified if all(p['passed'] for p in pair_gates if p['route']==r)],
                predictions=len(rows),encoder_updates_in_scoring=0,reference_reports=references,
                locked_models_sha256=file_digest(report/'locked_models.json'),scoring_source_sha256=file_digest(__file__),route='native_task_forecast')
    save_json(report/'performance_status.json',result);return result


def forecast_pairing(report,assets,units,routes,summary,deadline):
    rows=[];donors=[];gates=[]
    for spec in forecast_folds():
        data,x,mask,_=forecast_data(spec,'joint');data.score_open=True;ids=data.indices('score')
        first=different_record_donors(data.rows,ids);lookup=dict(zip(ids,first));current=ids.copy();variants={'zero_values':None}
        for shift in (1,2,3):
            current=np.array([lookup[i] for i in current]);variants['different_record_'+str(shift)]=current.copy()
        unit=next(u for u in units if u['fold']==spec['name'] and u['view']=='joint')
        for route in routes:
            model=load_selected(assets,unit,route)
            for variant,mapping in variants.items():
                values=x.clone();valid=mask.clone()
                if mapping is None:values[ids,:,:16]=0
                else:
                    for columns in (slice(0,16),slice(28,44)):
                        values[ids,:,columns]=x[mapping,:,columns];valid[ids,:,columns]=mask[mapping,:,columns]
                    for i,j in zip(ids,mapping):
                        assert data.subjects[i]==data.subjects[j] and data.rows[i]['record_id']!=data.rows[j]['record_id']
                        donors.append(dict(fold=spec['name'],route=route,variant=variant,sample_id=data.rows[i]['sample_id'],donor=data.rows[j]['sample_id']))
                predicted=predict_forecast(model,data,values,valid,ids,deadline)
                rows+=prediction_rows(data,ids,route+'_'+variant,predicted);del values,valid
            del model;torch.cuda.empty_cache()
        del data,x,mask
    people,panel=metric_tables(rows);pd.DataFrame(rows).to_csv(report/'pairing_predictions.csv.gz',index=False)
    people.to_csv(report/'pairing_subject_metrics.csv',index=False);panel.to_csv(report/'pairing_summary.csv',index=False)
    save_rows(report/'pairing_donors.csv',donors)
    for spec in forecast_folds():
        for route in routes:
            base=float(summary[(summary.fold==spec['name'])&(summary.method=='native_'+route+'_joint')].rmse.iloc[0])
            local=panel[(panel.fold==spec['name'])&panel.method.str.startswith(route+'_')]
            shift=float(local[local.method.str.contains('different_record')].rmse.mean()/base-1)
            zero=float(local[local.method==route+'_zero_values'].rmse.iloc[0])
            gates.append(dict(fold=spec['name'],route=route,mean_degradation=shift,zero_rmse=zero,original_rmse=base,
                              passed=shift>=.02 and base<=zero))
    return gates
