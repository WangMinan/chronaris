"""Fit-only linear readouts for locked pretrained states, with all strong controls retained."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_information_qualification import information_data,VIEW_COLUMNS
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE,atomic_json,export_information
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_regressor
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.mult_diagnosis import different_record_donors
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows
from chronaris.evaluation.application_tasks.recent_model_training import load_encoder_checkpoint
from chronaris.evaluation.application_tasks.shape_mult_results import prediction_rows,metric_tables
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.chronos_information import ChronosInformationEncoder
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.training.rng import isolated_training_rng


ALPHAS=(.1,1.,10.,100.,1000.,10000.,100000.,1000000.)
VIEWS=('aircraft','physiology_history','independent','joint')
ROUTES=('frozen','adapted')


def feature_rows(units,fold,route,view,ids):
    if view=='independent':
        return np.concatenate([feature_rows(units,fold,route,v,ids) for v in ('physiology_history','aircraft')],axis=1)
    unit=next(u for u in units if u['fold']==fold and u['view']==view)
    return np.asarray(np.load(unit[route+'_path'],mmap_mode='r',allow_pickle=False)[ids])


def information_gates(people,summary):
    gates=[]
    for fold,panel in summary.groupby('fold'):
        for route in ROUTES:
            name='chronos_'+route+'_joint'
            controls=panel[panel.method.str.startswith('reference_') | panel.method.isin(
                ['chronos_'+route+'_'+view for view in VIEWS if view!='joint'])]
            strongest=controls.loc[controls.rmse.idxmin()]
            own=panel[panel.method==name].iloc[0]
            pivot=people[people.fold==fold].pivot(index='subject',columns='method',values='rmse')
            improvement=float((1-own.rmse/controls.rmse).min())
            wins=int((pivot[name]<pivot[strongest.method]).sum())
            mae=bool((own.mae<=controls.mae*1.02).all())
            gates.append(dict(fold=fold,route=route,strongest=strongest.method,rmse=own.rmse,
                              improvement=improvement,subjects_improved=wins,mae_passed=mae,
                              passed=improvement>=.02 and wins>=2 and mae))
    return gates


def score_chronos_information(report,root):
    report,root=Path(report),Path(root)
    if (report/'performance_status.json').exists():raise FileExistsError('retained performance result')
    status=json.loads((report/'encoding_status.json').read_text())
    assert status['status']=='completed' and len(status['units'])==6
    units=status['units'];protected=json.loads((report/'protected_sources.json').read_text())
    assert all(file_digest(p)==h for p,h in protected.items())
    for unit in units:
        for route in ROUTES:assert file_digest(unit[route+'_path'])==unit[route+'_sha256']
    binding=dict(encoding_status_sha256=file_digest(report/'encoding_status.json'),alphas=ALPHAS,
                 source_sha256=file_digest(__file__),outer_scores_opened=False)
    binding=json.loads(json.dumps(binding));models={};choices={};started=time.monotonic()
    partial=root/'readouts_partial.joblib';partial_meta=report/'readouts_partial.json'
    if partial_meta.exists():
        meta=json.loads(partial_meta.read_text());assert meta['binding']==binding
        assert file_digest(partial)==meta['sha256'];models=joblib.load(partial);choices=meta['choices']
    for spec in FOLDS:
        data,_,_=information_data(spec);a,b=data.indices('fit'),data.indices('select')
        for route in ROUTES:
            for view in VIEWS:
                name=spec['name']+'_'+route+'_'+view
                if name in models:continue
                if time.time()>=NIGHT_DEADLINE:raise TimeoutError('night budget saved before readout')
                x=feature_rows(units,spec['name'],route,view,a);v=feature_rows(units,spec['name'],route,view,b)
                model,alpha=fit_regressor(x,data.y[a],v,data.y[b],alpha_values=ALPHAS,scaler_with_mean=True,
                                          train_sample_weight=data.weights[a].numpy(),validation_groups=data.subjects[b])
                pred=model.predict(v);assert np.isfinite(pred).all()
                models[name]=model;choices[name]=dict(alpha=alpha,features=x.shape[1],
                    selection_rmse=float(np.sqrt(np.mean((pred-data.y[b])**2))))
                temp=partial.with_suffix('.tmp');joblib.dump(models,temp);temp.replace(partial)
                atomic_json(partial_meta,dict(binding=binding,sha256=file_digest(partial),choices=choices))
                atomic_json(report/'readout_status.json',dict(status='fitting',completed=len(models),total=16,stage=name))
                print(name,choices[name],flush=True);del x,v
    locked=root/'readouts.joblib';joblib.dump(models,locked)
    save_json(report/'locked_readouts.json',dict(binding=binding,choices=choices,path=str(locked.resolve()),
                                                sha256=file_digest(locked)))
    reference=pd.read_csv('docs/review/stage/history-supervision-20261019/predictions.csv.gz')
    reference['method']='reference_'+reference.method;rows=reference.to_dict('records')
    for spec in FOLDS:
        data,_,_=information_data(spec);data.score_open=True;ids=data.indices('score')
        for route in ROUTES:
            for view in VIEWS:
                x=feature_rows(units,spec['name'],route,view,ids)
                pred=models[spec['name']+'_'+route+'_'+view].predict(x)
                rows+=prediction_rows(data,ids,'chronos_'+route+'_'+view,pred)
    people,summary=metric_tables(rows);gates=information_gates(people,summary)
    pd.DataFrame(rows).to_csv(report/'predictions.csv.gz',index=False)
    people.to_csv(report/'subject_metrics.csv',index=False);summary.to_csv(report/'summary.csv',index=False)
    save_json(report/'performance_gates.json',gates)
    qualified=[route for route in ROUTES if all(g['passed'] for g in gates if g['route']==route)]
    pairing=pairing_diagnosis(report,root,units,models,qualified) if qualified else []
    save_json(report/'pairing_gates.json',pairing)
    result=dict(status='completed',qualified_routes=qualified,paired_routes=[r for r in qualified if all(
        p['passed'] for p in pairing if p['route']==r)],readouts=len(models),predictions=len(rows),
        elapsed_s=time.monotonic()-started,task_encoder_labels_used=False,criterion='development_information_only')
    save_json(report/'performance_status.json',result);return result


def pairing_diagnosis(report,root,units,models,routes):
    assets=verify_assets(json.loads((report/'qualification.json').read_text())['assets_path'])
    rows=[];donors=[];gates=[]
    original=pd.read_csv(report/'summary.csv')
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy for conditional pairing check')
        for spec in FOLDS:
            data,x,m=information_data(spec);data.score_open=True;ids=data.indices('score')
            one=different_record_donors(data.rows,ids);lookup=dict(zip(ids,one));current=ids.copy()
            mappings={'zero_values':None}
            for shift in (1,2,3):
                current=np.array([lookup[i] for i in current]);mappings['different_record_'+str(shift)]=current.copy()
            for route in routes:
                with isolated_training_rng(17):
                    encoder=ChronosInformationEncoder(assets,28).eval()
                    unit=next(u for u in units if u['fold']==spec['name'] and u['view']=='joint')
                    if route=='adapted':
                        saved=torch.load(unit['checkpoint_path'],map_location='cpu',weights_only=True)
                        load_encoder_checkpoint(encoder,saved['encoder']);del saved
                    for variant,mapping in mappings.items():
                        values=x[ids].clone();mask=m[ids].clone()
                        if mapping is None:values[:,:,:16]=0
                        else:
                            values[:,:,:16]=x[mapping,:,:16];mask[:,:,:16]=m[mapping,:,:16]
                            donors.extend(dict(fold=spec['name'],route=route,variant=variant,sample_id=data.rows[i]['sample_id'],
                                               donor=data.rows[j]['sample_id']) for i,j in zip(ids,mapping))
                        name=spec['name']+'_'+route+'_'+variant;path=root/(name+'.npy')
                        binding=dict(unit=unit,variant=variant,sample_ids=[data.rows[i]['sample_id'] for i in ids])
                        export_information(encoder,values,mask,path,binding,report/'pairing_status.json',name)
                        pred=models[spec['name']+'_'+route+'_joint'].predict(np.load(path,mmap_mode='r'))
                        rows+=prediction_rows(data,ids,route+'_'+variant,pred)
                    del encoder;torch.cuda.empty_cache()
    people,summary=metric_tables(rows)
    pd.DataFrame(rows).to_csv(report/'pairing_predictions.csv.gz',index=False);save_rows(report/'pairing_donors.csv',donors)
    people.to_csv(report/'pairing_subject_metrics.csv',index=False);summary.to_csv(report/'pairing_summary.csv',index=False)
    for spec in FOLDS:
        for route in routes:
            base=float(original[(original.fold==spec['name'])&(original.method=='chronos_'+route+'_joint')].rmse.iloc[0])
            panel=summary[(summary.fold==spec['name'])&summary.method.str.startswith(route+'_')]
            shifted=float(panel[panel.method.str.contains('different_record')].rmse.mean())
            zero=float(panel[panel.method==route+'_zero_values'].rmse.iloc[0]);degradation=shifted/base-1
            gates.append(dict(fold=spec['name'],route=route,mean_degradation=degradation,zero_rmse=zero,original_rmse=base,
                              passed=degradation>=.02 and base<=zero))
    return gates
