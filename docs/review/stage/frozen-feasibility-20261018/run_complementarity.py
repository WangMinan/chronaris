"""Orchestrate the preregistered frozen-combination evaluation and preserve all predictions."""
import csv
import gzip
import json
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from audit_io import HERE,REVISED,REGISTRY,feature_path,sha,save_csv,save_json
from chronaris.evaluation.application_tasks.public_readout_revision import load_readout_revision
from chronaris.evaluation.application_tasks.frozen_complementarity import evaluate_frozen_combinations
from chronaris.evaluation.application_tasks.grouped_readout_repair import score


def main(domain):
    output=HERE/('complementarity_'+domain)
    if output.exists():raise FileExistsError('retain existing evaluation output')
    output.mkdir();start=time.monotonic()
    data=load_readout_revision(REVISED/domain)
    ids=np.array(data.targets.sample_ids);groups=np.array(data.dataset.group_ids)
    folds=json.loads(REGISTRY.read_text())['domains'][domain]['folds']['development']
    features={};sources={};metas={}
    for method in ('physiology_only','vehicle_only','chronaris'):
        for route in ('self_supervised','task_guided'):
            path=feature_path(domain,method,route);sources[str(path)]=sha(path)
            meta=json.loads(path.with_suffix('.json').read_text());metas[method,route]=meta
            assert meta['output_sha256']==sources[str(path)]
            assert meta['disjoint_subjects'] and not (set(meta['original_subjects']) & set(groups))
            # Corrected CogPilot exports bind their own hash via the boundary verification receipt.
            with np.load(path) as z:
                assert np.array_equal(z['sample_ids'],ids)
                features[method,route]=z['pooled'].astype(float)
    assert len({m['original_fold_id'] for m in metas.values()})==1
    metrics=[];subject_rows=[];prediction_rows=[];selection=[]
    for task,values in data.targets.values.items():
        y=values.numpy().reshape(-1).astype(float);valid=data.targets.valid_masks[task].numpy().reshape(-1)
        classes=[0,1,2,3] if task=='difficulty' else [0,1] if 'classification' in task else None
        bounds=(1.,9.) if domain=='clare' and classes is None else None
        for fold_id,fold in enumerate(folds):
            train=np.flatnonzero(valid & np.isin(groups,fold['train_subjects']))
            test=np.flatnonzero(valid & np.isin(groups,fold['validation_subjects']))
            for route in ('self_supervised','task_guided'):
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    outcomes,logs=evaluate_frozen_combinations(features['physiology_only',route],
                        features['vehicle_only',route],features['chronaris',route],y,groups,train,test,
                        classes=classes,bounds=bounds)
                meta=dict(domain=domain,task=task,fold=fold_id,route=route)
                selection.extend(meta|r for r in logs)
                constants={'majority':np.full(len(test),np.argmax(np.bincount(y[train].astype(int),minlength=len(classes))))} if classes else {
                    'mean':np.full(len(test),np.mean(y[train])),'median':np.full(len(test),np.median(y[train]))}
                if task=='event_response':constants['zero']=np.zeros(len(test))
                for method,pred in constants.items():outcomes[method,'constant']=dict(prediction=pred,parameter=None,weight=None)
                for (method,rule),result in outcomes.items():
                    pred=result['prediction'];values=[];assert np.isfinite(pred).all()
                    detail=meta|dict(method=method,rule=rule)
                    for subject in sorted(set(groups[test])):
                        mask=groups[test]==subject;t,p=y[test][mask],pred[mask]
                        value=score(t,p,classes);values.append(value)
                        subject_rows.append(detail|dict(subject=subject,samples=len(t),value=value,
                            bias=float(np.mean(p-t)),prediction_std=float(np.std(p)),target_std=float(np.std(t))))
                    metrics.append(detail|dict(value=float(np.mean(values)),parameter=json.dumps(result['parameter']),weight=result['weight']))
                    prediction_rows.extend(detail|dict(sample_id=str(s),subject=str(g),target=float(t),prediction=float(p))
                        for s,g,t,p in zip(ids[test],groups[test],y[test],pred,strict=True))
                if time.monotonic()-start>7200:raise TimeoutError('readout budget reached')
            print('completed',domain,task,fold_id,flush=True)
    for name,rows in [('folds',metrics),('subjects',subject_rows),('selection',selection)]:save_csv(output/(name+'.csv'),rows)
    with gzip.open(output/'predictions.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(prediction_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(prediction_rows)
    assert all(sha(p)==h for p,h in sources.items())
    save_json(output/'receipt.json',dict(status='completed',sources=sources,seconds=time.monotonic()-start,
        predictions=len(prediction_rows),subject_metrics=len(subject_rows),encoder_updates=0,new_gpu_exports=0,
        classification_candidates=3,regression_candidates=6,weight_candidates=5,
        encoder_normalizer_refitted=False,readout_scaler_train_only=True))


if __name__=='__main__':
    import sys
    main(sys.argv[1])
