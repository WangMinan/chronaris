"""Run the preregistered CPU-only readout comparisons on frozen representations."""
import argparse
import csv
import gzip
import json
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from repair_io import HERE,ASSETS,OLD_ASSETS,REGISTRY,METHODS,feature_path,sha,save_json,save_csv
from chronaris.evaluation.application_tasks.public_readout_revision import load_readout_revision
from chronaris.evaluation.application_tasks.grouped_readout_repair import nested_readout,score

C3=(.1,1.,10.)
C9=(.001,.003,.01,.03,.1,.3,1.,3.,10.)
ALPHAS=(.1,1.,10.,100.,1000.,10000.)
WEIGHTS=((1.,1.,1.),(.1,1.,.1),(1.,.1,.1))


def candidate_plan(method,classification):
    if not classification:
        return [dict(parameter=a) for a in ALPHAS],{'ridge':list(range(6))}
    if method=='chronaris':
        candidates=[dict(parameter=c,weights=w) for w in WEIGHTS for c in C3]
        return candidates,dict(block_selected=list(range(9)),uniform3=[0,1,2],
            favor_second=[3,4,5],favor_first=[6,7,8])
    return [dict(parameter=c) for c in C9],dict(matched9=list(range(9)),uniform3=[4,6,8])


def main(domain):
    assert json.loads((HERE/'preparation.json').read_text())['status']=='passed'
    output=HERE/domain
    if output.exists():raise FileExistsError('completed or partial experiment retained; do not overwrite')
    output.mkdir()
    start=time.monotonic();data=load_readout_revision(ASSETS/domain)
    targets=data.targets;ids=np.array(targets.sample_ids);groups=np.array(data.dataset.group_ids)
    folds=json.loads(REGISTRY.read_text())['domains'][domain]['folds']['development']
    metrics=[];subject_scores=[];predictions=[];selection=[];candidates_log=[]
    sources={str(ASSETS/domain/'revision_manifest.json'):sha(ASSETS/domain/'revision_manifest.json')}
    features={}
    for method in METHODS:
        for route in ('self_supervised','task_guided'):
            path=feature_path(domain,method,route);sources[str(path)]=sha(path)
            with np.load(path) as z:
                assert np.array_equal(z['sample_ids'],ids)
                features[method,route]=z['pooled'].astype(float)
    trend=None
    if domain=='cogpilot':
        path=OLD_ASSETS/'exclusive_endpoint/cogpilot__history.npz';sources[str(path)]=sha(path)
        with np.load(path) as z:
            assert np.array_equal(z['sample_ids'],ids);trend=z['trend']

    def record(meta,arm,indices,pred,parameter='',candidate=-1):
        y=targets.values[meta['task']].numpy().reshape(-1).astype(float)[indices]
        classes=[0,1,2,3] if meta['task']=='difficulty' else [0,1] if 'classification' in meta['task'] else None
        vals=[]
        for subject in sorted(set(groups[indices])):
            m=groups[indices]==subject;t=y[m];p=pred[m];value=score(t,p,classes);vals.append(value)
            correlation=float(np.corrcoef(t,p)[0,1]) if np.std(t)>1e-12 and np.std(p)>1e-12 else ''
            subject_scores.append(meta|dict(arm=arm,subject=str(subject),samples=int(m.sum()),value=value,
                bias=float(np.mean(p-t)),prediction_std=float(np.std(p)),target_std=float(np.std(t)),correlation=correlation))
        metrics.append(meta|dict(arm=arm,value=float(np.mean(vals)),parameter=parameter,candidate=candidate,
            samples=len(indices),subjects=len(vals),metric='macro_f1' if classes else 'rmse'))
        predictions.extend(meta|dict(arm=arm,sample_id=str(s),subject=str(g),target=float(t),prediction=float(p))
            for s,g,t,p in zip(ids[indices],groups[indices],y,pred,strict=True))

    for task,value in targets.values.items():
        y=value.numpy().reshape(-1).astype(float);valid=targets.valid_masks[task].numpy().reshape(-1)
        classes=[0,1,2,3] if task=='difficulty' else [0,1] if 'classification' in task else None
        for fold_index,fold in enumerate(folds):
            tr=np.flatnonzero(valid&np.isin(groups,fold['train_subjects']))
            te=np.flatnonzero(valid&np.isin(groups,fold['validation_subjects']))
            for route in ('self_supervised','task_guided'):
                meta=dict(domain=domain,task=task,fold=fold_index,route=route,method='constant')
                constants={'majority':np.full(len(te),np.argmax(np.bincount(y[tr].astype(int),minlength=len(classes))))} if classes else {
                    'mean':np.full(len(te),np.mean(y[tr])),'median':np.full(len(te),np.median(y[tr]))}
                if task=='event_response':
                    constants|={'zero':np.zeros(len(te)),'history':trend[te]}
                for name,pred in constants.items():
                    assert np.isfinite(pred).all();record(meta,name,te,pred)
                for method in METHODS:
                    meta=meta|{'method':method};x=features[method,route]
                    candidates,selections=candidate_plan(method,classes is not None)
                    bounds=(1.,9.) if domain=='clare' and classes is None else None
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        outcomes,inner=nested_readout(x,y,groups,tr,te,candidates,classes=classes,bounds=bounds,selections=selections)
                    selection.extend(meta|r for r in inner)
                    candidates_log.extend(meta|dict(index=i,candidate=json.dumps(c),selections=json.dumps(selections)) for i,c in enumerate(candidates))
                    for arm,result in outcomes.items():
                        i=result['candidate'];record(meta,arm,te,result['prediction'],json.dumps(candidates[i]),i)
                    if method=='chronaris' and classes:
                        for projection in ('second24','pca24'):
                            controls=[dict(parameter=c,projection=projection) for c in C3]
                            cm=meta|{'method':'chronaris_'+projection}
                            outcomes,inner=nested_readout(x,y,groups,tr,te,controls,classes=classes)
                            selection.extend(cm|r for r in inner)
                            candidates_log.extend(cm|dict(index=i,candidate=json.dumps(c),selections=json.dumps({'selected':[0,1,2]})) for i,c in enumerate(controls))
                            result=outcomes['selected'];i=result['candidate'];record(cm,'selected',te,result['prediction'],json.dumps(controls[i]),i)
                if time.monotonic()-start>3600:raise TimeoutError('CPU readout budget exceeded')
            print('completed',domain,task,fold_index,flush=True)
    for name,rows in [('folds',metrics),('subjects',subject_scores),('selection',selection),('candidates',candidates_log)]:
        save_csv(output/f'{name}.csv',rows)
    with gzip.open(output/'predictions.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions[0]),lineterminator='\n');writer.writeheader();writer.writerows(predictions)
    assert all(sha(p)==h for p,h in sources.items())
    save_json(output/'receipt.json',dict(status='completed',sources=sources,fold_scores=len(metrics),
        subject_scores=len(subject_scores),predictions=len(predictions),inner_fits=len(selection),
        seconds=time.monotonic()-start,encoder_updates=0,new_gpu_exports=0,
        classification_main_candidates_per_method=9,regression_candidates_per_method=6,
        task_manifest=dict(targets.manifest)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',choices=['clare','cogpilot']);main(p.parse_args().domain)
