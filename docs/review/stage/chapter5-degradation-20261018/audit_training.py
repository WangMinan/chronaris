"""Read training receipts and frozen linear consumers; no fitting or updates."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from audit_statistics import DATA, HERE, digest

OUT = HERE/'training'
SOURCES = {}


def register(path):
    path=Path(path)
    if str(path) not in SOURCES:SOURCES[str(path)]=digest(path)
    return path


def read(path):
    return json.loads(register(path).read_text())


def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def grouped(y,p,groups,classification,classes):
    return float(np.mean([f1_score(y[groups==g],p[groups==g],labels=classes,average='macro',zero_division=0)
        if classification else np.sqrt(np.mean((y[groups==g]-p[groups==g])**2)) for g in sorted(set(groups))]))


def main():
    OUT.mkdir(exist_ok=True)
    with register(DATA/'inventory.csv').open() as f:units=list(csv.DictReader(f))
    training,curves,validation,blocks,masked,fit_rows=[],[],[],[],[],[]
    verification=[];unavailable=[]
    for unit in units:
        if unit['domain'] not in ('clare','cogpilot') or unit['method'] not in ('chronaris','vehicle_only','physiology_only','mult','contiformer'):continue
        receipt=read(unit['receipt'])
        for r in receipt['results']:
            if not r.get('training'):continue
            meta={k:unit[k] for k in ('domain','method','variant','fold_index','seed','unit_key')};meta['route']=r['route']
            t=r['training'];recipe=receipt['unit'].get('recipe',{})
            training.append(meta|{k:t.get(k) for k in ('optimizer_updates','joint_updates','head_warmup_updates','best_update','parameter_count','training_elapsed_s','stopped_early')}|
                dict(counts=unit['counts'],budget=json.dumps(receipt['unit'].get('budget',{})),
                     recipe=json.dumps({k:v for k,v in recipe.items() if k in ('mechanism_arguments','observation_anchor_enabled','physics_relations')},sort_keys=True),
                     best_checkpoint=t['best_checkpoint_path']))
            for row in t.get('epoch_rows',[]):
                curve=meta|{k:v for k,v in row.items() if isinstance(v,(int,float,str,bool))}
                for k in ('train_losses','validation_losses'):
                    for term,value in row.get(k,{}).items():curve[k+'_'+term]=value
                for term in row.get('mechanism_validation',{}).get('terms',[]):
                    curve['mechanism_'+term['term_name']]=term.get('weighted_loss')
                curves.append(curve)
            evaluation=r.get('evaluation',r)
            for family,component in evaluation['consumers']['components'].items():
                result=read(component['result_path'])
                for s in result['evaluations']['validation']['task_summary']:
                    validation.append(meta|dict(consumer=family,**s))
                fit_rows.extend(meta|dict(consumer=family,**{k:v for k,v in row.items() if k!='train_sample_ids'},train_count=len(row['train_sample_ids'])) for row in result['fit_rows'])
            # Read frozen pooled features and original consumer, with the original fit untouched.
            component=evaluation['consumers']['components']['linear']
            model_path=register(component['model_path'])
            root=model_path.parents[2]
            if not all((root/'representations'/role/'fusion_stream.npz').exists() for role in ('train','validation')):
                unavailable.append(meta|dict(reason='original_training_representation_cache_not_present',root=str(root)))
                continue
            bundle=joblib.load(model_path)['consumer']
            features={};ids={}
            for role in ('train','validation'):
                path=register(root/'representations'/role/'fusion_stream.npz')
                with np.load(path,allow_pickle=False) as a:
                    features[role]=a['pooled_embedding'].astype(float);ids[role]=list(a['sample_ids'])
            assert tuple(ids['train'])==tuple(bundle['train_sample_ids'])
            result=read(component['result_path'])['evaluations']['validation']
            for definition in bundle['definitions']:
                task=definition.name;model=bundle['models'][(task,0)];scaler=model[0]
                by_id={row['sample_id']:row for row in result['prediction_rows'] if row['task']==task and row['field']==0}
                selected=np.array([bool(by_id[s]['target_valid']) for s in ids['validation']])
                y=np.array([by_id[s]['target'] for s,ok in zip(ids['validation'],selected) if ok])
                g=np.array([by_id[s]['group_id'] for s,ok in zip(ids['validation'],selected) if ok])
                x=features['validation'][selected];pred=model.predict(x)
                saved=np.array([by_id[s]['prediction'] for s,ok in zip(ids['validation'],selected) if ok])
                diff=float(np.max(np.abs(pred-saved)));assert diff<1e-5
                verification.append(meta|dict(task=task,replay_max_delta=diff))
                is_class=definition.kind=='classification';classes=list(range(definition.output_dim)) if is_class else None
                baseline=grouped(y,pred,g,is_class,classes)
                expected=next(row['value'] for row in result['task_summary'] if row['task']==task)
                assert np.isclose(baseline,expected,atol=1e-7)
                for name,idx in ([('all',slice(None)),('physiology',slice(0,24)),('vehicle',slice(24,48)),('cross',slice(48,64))]
                                  if unit['method']=='chronaris' else [('all',slice(None))]):
                    a=features['train'][:,idx];b=x[:,idx]
                    sv=np.linalg.svd(a-a.mean(0),compute_uv=False);energy=sv**2
                    effective=float(energy.sum()**2/(energy@energy)) if energy.sum()>0 else 0.
                    standardized=scaler.transform(x)[:,idx]
                    blocks.append(meta|dict(task=task,block=name,dimensions=a.shape[1],train_effective_rank=effective,
                        train_mean_norm=float(np.linalg.norm(a,axis=1).mean()),
                        validation_standardized_mean_norm=float(np.linalg.norm(standardized.mean(0))),
                        validation_standardized_rms=float(np.sqrt(np.mean(standardized**2)))))
                if unit['method']!='chronaris':continue
                for name,keep in [('all',slice(None)),('without_cross',slice(0,48)),('vehicle_only',slice(24,48)),
                                  ('physiology_only',slice(0,24)),('cross_only',slice(48,64))]:
                    counter=np.broadcast_to(scaler.mean_,x.shape).copy();counter[:,keep]=x[:,keep]
                    value=grouped(y,model.predict(counter),g,is_class,classes)
                    masked.append(meta|dict(task=task,mask=name,original=baseline,value=value,
                        benefit=(value-baseline)*(1 if is_class else -1),subjects=len(set(g)),
                        meaning='frozen_head_train_mean_intervention_no_refit_validation_only'))
        print('audited',unit['domain'],unit['method'],unit['variant'],unit['fold_index'],flush=True)
    comparable=defaultdict(list)
    for row in training:comparable[(row['domain'],row['fold_index'],row['route'])].append(row)
    for group in comparable.values():
        assert len({r['optimizer_updates'] for r in group})==1
        assert len({r['counts'] for r in group})==1
    for name,rows in [('training_units.csv',training),('loss_curves.csv',curves),('validation_metrics.csv',validation),
                      ('feature_blocks.csv',blocks),('frozen_head_interventions.csv',masked),('consumer_fit.csv',fit_rows),('replay.csv',verification),('cache_unavailable.csv',unavailable)]:write(name,rows)
    for path,expected in SOURCES.items():assert digest(Path(path))==expected
    (OUT/'verification.json').write_text(json.dumps(dict(sources=SOURCES,source_files_unchanged=True,
        encoder_updates=0,consumer_fits=0,roles=['train','validation'],replay_checks=len(verification),
        audited_training_routes=len(training),within_fold_route_budgets_and_counts_equal=True),indent=2)+'\n')
    print('completed',len(training),'training routes',len(verification),'consumer replay checks')


if __name__=='__main__':
    assert grouped(np.array([0.,2.]),np.array([0.,0.]),np.array(['a','b']),False,None)==1.
    main()
