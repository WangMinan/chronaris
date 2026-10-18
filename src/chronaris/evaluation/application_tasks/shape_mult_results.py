"""Locked-model predictions, fixed pairing interventions and preregistered gates."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model, predict_fair
from chronaris.evaluation.application_tasks.mult_diagnosis import different_record_donors, intervene
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_rows, save_json


def prediction_rows(data, ids, method, values):
    if not np.isfinite(values).all():
        raise ValueError('nonfinite score')
    return [dict(fold=data.fold['name'], method=method, sample_id=data.rows[i]['sample_id'],
                 subject=str(data.subjects[i]), truth=float(data.y[i]), prediction=float(p)) for i,p in zip(ids,values)]


def score_shape_model(data, checkpoint, readout, version):
    data.score_open=True; ids=data.indices('score')
    model=load_fair_model(checkpoint,data)
    features,head=predict_fair(model,data,ids)
    rows=prediction_rows(data,ids,version+'_head',head[:,-1])
    rows+=prediction_rows(data,ids,version+'_ridge',readout.predict(features))
    return rows


@torch.inference_mode()
def pairing_predictions(data, checkpoint, readout):
    model=load_fair_model(checkpoint,data)
    data.score_open=True;ids=data.indices('score')
    first=different_record_donors(data.rows,ids);lookup=dict(zip(ids,first));current=ids.copy()
    variants={'zero_values':None}
    for offset in (1,2,3):
        current=np.array([lookup[i] for i in current]);variants[f'different_record_{offset}']=current.copy()
    rows=[];donors=[]
    for variant, donor_ids in variants.items():
        features=[]
        for off in range(0,len(ids),32):
            p,v=data.queries(ids[off:off+32])
            donor=data.queries(donor_ids[off:off+32])[0] if donor_ids is not None else None
            sequence,_=model(intervene(p,'different_record' if donor is not None else 'zero_values',donor),v)
            features.append(sequence[:,-1].cpu().numpy())
        rows+=prediction_rows(data,ids,variant,readout.predict(np.concatenate(features)))
        if donor_ids is not None:
            for i,j in zip(ids,donor_ids):
                assert data.rows[i]['record_id']!=data.rows[j]['record_id'] and data.subjects[i]==data.subjects[j]
                donors.append(dict(fold=data.fold['name'],variant=variant,sample_id=data.rows[i]['sample_id'],
                                   donor=data.rows[j]['sample_id']))
    return rows,donors


def metric_tables(rows):
    frame=pd.DataFrame(rows);people=[]
    assert not frame[['fold','method','sample_id']].duplicated().any()
    for (fold,method,person),group in frame.groupby(['fold','method','subject']):
        error=group.prediction-group.truth
        people.append(dict(fold=fold,method=method,subject=person,samples=len(group),
                           rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.abs(error).mean())))
    people=pd.DataFrame(people)
    summary=people.groupby(['fold','method'],as_index=False)[['rmse','mae']].mean()
    return people,summary


def shape_gates(people,summary,pair_summary):
    gates=[];pair_gates=[]
    for fold,panel in summary.groupby('fold'):
        controls=panel[~panel.method.str.startswith('candidate_')]
        strongest=controls.loc[controls.rmse.idxmin()]
        for route in ('ridge','head'):
            candidate=panel[panel.method=='candidate_'+route].iloc[0]
            own=people[(people.fold==fold)&(people.method==candidate.method)].set_index('subject')
            ref=people[(people.fold==fold)&(people.method==strongest.method)].set_index('subject')
            wins=int((own.rmse<ref.rmse).sum())
            gain=float((1-candidate.rmse/controls.rmse).min())
            mae_ok=bool((candidate.mae<=1.02*controls.mae).all())
            gates.append(dict(fold=fold,route=route,strongest=strongest.method,improvement=gain,
                              subjects_improved=wins,mae_passed=mae_ok,passed=gain>=.02 and wins>=2 and mae_ok))
        baseline=float(panel[panel.method=='candidate_ridge'].rmse.iloc[0])
        paired=pair_summary[pair_summary.fold==fold]
        degradation=float(paired[paired.method.str.startswith('different_record')].rmse.mean()/baseline-1)
        zero=float(paired[paired.method=='zero_values'].rmse.iloc[0])
        pair_gates.append(dict(fold=fold,mean_pairing_degradation=degradation,original_rmse=baseline,
                               zero_values_rmse=zero,passed=degradation>=.02 and baseline<=zero))
    return gates,pair_gates


def write_shape_results(report,rows,pair_rows,donors):
    report=Path(report)
    people,summary=metric_tables(rows);pair_people,pair_summary=metric_tables(pair_rows)
    gates,pair_gates=shape_gates(people,summary,pair_summary)
    pd.DataFrame(rows).to_csv(report/'predictions.csv.gz',index=False)
    pd.DataFrame(pair_rows).to_csv(report/'pairing_predictions.csv.gz',index=False)
    people.to_csv(report/'subject_metrics.csv',index=False);summary.to_csv(report/'summary.csv',index=False)
    pair_people.to_csv(report/'pairing_subject_metrics.csv',index=False);pair_summary.to_csv(report/'pairing_summary.csv',index=False)
    save_rows(report/'pairing_donors.csv',donors)
    save_json(report/'performance_gates.json',gates);save_json(report/'pairing_gates.json',pair_gates)
    return dict(representation_success=all(r['passed'] for r in gates if r['route']=='ridge'),
                task_head_success=all(r['passed'] for r in gates if r['route']=='head'),
                pairing_success=all(r['passed'] for r in pair_gates))
