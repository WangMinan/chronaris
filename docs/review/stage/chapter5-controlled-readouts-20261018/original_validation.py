"""Same original validation windows: fixed old heads, fixed ridge, simple baselines."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from common import HERE, REPO, INVENTORY, REGISTRY, save_csv, save_json, sha
from readouts import metric, fit
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development


def main():
    root=REPO/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'
    rows=[];subjects=[];sources={};missing=[]
    with INVENTORY.open() as f:
        units=[r for r in csv.DictReader(f) if r['domain'] in ('clare','cogpilot') and r['variant']=='thesis_reference']
    for domain in ('clare','cogpilot'):
        target=torch.load(root/domain/'task_targets.pt',weights_only=True)
        sources[str(root/domain/'task_targets.pt')]=sha(root/domain/'task_targets.pt')
        index={s:i for i,s in enumerate(target['sample_ids'])}
        data=load_prepared_public_development(domain,output_root=root,registry_path=REGISTRY,role='confirmation')
        history={}
        for unit in [u for u in units if u['domain']==domain]:
            receipt=json.loads(Path(unit['receipt']).read_text())
            for r in receipt['results']:
                path=Path(r['consumers']['components']['linear']['model_path'])
                sources[str(path)]=sha(path)
                old=joblib.load(path)['consumer']
                features={};ids={}
                for role in ('train','validation'):
                    feature_path=path.parents[2]/'representations'/role/'fusion_stream.npz'
                    sources[str(feature_path)]=sha(feature_path)
                    with np.load(feature_path) as z:
                        features[role]=z['pooled_embedding'];ids[role]=list(z['sample_ids'])
                assert ids['train']==list(old['train_sample_ids'])
                for task in old['definitions']:
                    classes=list(range(task.output_dim)) if task.kind=='classification' else None
                    selected={role:np.array([index[s] for s in ids[role]]) for role in ids}
                    masks={role:target['valid_masks'][task.name][idx].numpy().reshape(-1) for role,idx in selected.items()}
                    ys={role:target['values'][task.name][idx].numpy().reshape(-1)[masks[role]] for role,idx in selected.items()}
                    xs={role:features[role][masks[role]] for role in ids}
                    val_ids=np.array(ids['validation'])[masks['validation']]
                    groups=np.array([old['context']['groups'][s] for s in val_ids])
                    y=ys['validation'];pred=old['models'][(task.name,0)].predict(xs['validation'])
                    candidates={'frozen_full':pred}
                    if classes:
                        candidates['train_majority']=np.full(len(y),np.argmax(np.bincount(ys['train'].astype(int),minlength=len(classes))))
                    else:
                        candidates['train_mean']=np.full(len(y),np.mean(ys['train'].astype(float)))
                        candidates['train_median']=np.full(len(y),np.median(ys['train']))
                        candidates['full_fixed_alpha10000']=fit(xs['train'],ys['train'],10000.,None).predict(xs['validation'])
                        if domain=='cogpilot':
                            candidates['zero_change']=np.zeros(len(y))
                            for s in val_ids:
                                if s in history:continue
                                sample=data.dataset.load_sample(s)
                                t=sample.physiology_timestamps_s;v=sample.physiology_values[:,1];m=sample.physiology_feature_mask[:,1]
                                a=m&(t>=10)&(t<12);b=m&(t>=8)&(t<10)
                                history[s]=float(np.median(v[a])-np.median(v[b])) if a.any() and b.any() else np.nan
                                if not np.isfinite(history[s]):
                                    missing.append(dict(sample_id=s,fold=int(unit['fold_index']),route=r['route'],
                                        recent_points=int(a.sum()),earlier_points=int(b.sum())))
                            candidates['history_trend']=np.array([history[s] for s in val_ids])
                    for name,p in candidates.items():
                        meta=dict(domain=domain,task=task.name,route=r['route'],fold=int(unit['fold_index']),method=name)
                        if not np.isfinite(p).all():
                            rows.append(meta|dict(value='',subjects=len(set(groups)),status='missing_history_support'))
                            continue
                        values=[]
                        for g in sorted(set(groups)):
                            m=groups==g;value=metric(y[m],p[m],classes);values.append(value)
                            subjects.append(meta|dict(subject=g,value=value,prediction_std=float(np.std(p[m])),
                                target_std=float(np.std(y[m])),bias=float(np.mean(p[m]-y[m])),samples=int(m.sum())))
                        rows.append(meta|dict(value=float(np.mean(values)),subjects=len(values),status='completed'))
                print('original validation',domain,unit['fold_index'],r['route'],flush=True)
    assert all(sha(p)==h for p,h in sources.items())
    save_csv(HERE/'original_validation_folds.csv',rows);save_csv(HERE/'original_validation_subjects.csv',subjects)
    save_json(HERE/'original_validation_verification.json',dict(sources=sources,encoder_updates=0,
        original_consumers_unchanged=True,missing_history_support=missing,
        scope='same previously examined inner validation; fixed alpha is exploratory'))


if __name__=='__main__':main()
