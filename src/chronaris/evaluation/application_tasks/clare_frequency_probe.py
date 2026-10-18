"""Fit/selection-only feasibility for standard CLARE labels and native signal summaries."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows


CLARE_TRAIN=('1337','1419','1544','1688','1717','1892','1929','1933','1953','1981')
CLARE_FOLDS=(dict(name='fold01',fit=(*CLARE_TRAIN,'1674'),select=('1390',),score=('1624','1629')),
             dict(name='fold02',fit=(*CLARE_TRAIN,'1629'),select=('1624',),score=('1390','1674')))


def probe_clare_frequency(snapshot,report,assets):
    report,assets=Path(report),Path(assets)
    if (report/'qualification.json').exists():raise FileExistsError('qualification retained')
    with np.load(snapshot,allow_pickle=False) as z:x=z['features'];meta=json.loads(str(z['metadata']))
    rows=[r for r in meta['windows'] if r['accepted']];subjects=np.array([r['subject'] for r in rows]);y=np.array([r['target'] for r in rows])
    assert x.shape==(len(rows),36) and np.isfinite(x).all() and len({r['sample_id'] for r in rows})==len(rows)
    assert set(subjects)==set(CLARE_TRAIN)|{'1390','1624','1629','1674'}
    contract=Path('/mnt/e/chronaris-v4-results/2026-10-12-device-validation/workflow-thread1/subset/training/0/clare/chronaris/self_supervised/evaluation/common_contract.json')
    source=json.loads(contract.read_text())['contract'];assert source['fold']['fold_id']=='v4_clare_confirmation_fold01'
    assert set(CLARE_TRAIN)=={s.split('__')[0] for s in source['roles']['train']}
    old=pd.read_csv('docs/review/stage/chapter5-controlled-readouts-20261018/clare_predictions.csv.gz')
    old=old[old.task=='workload_regression'].drop_duplicates('sample_id').set_index('sample_id')
    checked=0
    for r in rows:
        if r['sample_id'] in old.index:assert r['score']==old.loc[r['sample_id'],'target'];checked+=1
    save_json(report/'source_hashes.json',meta['sources']);save_rows(report/'coverage.csv',meta['coverage'])
    save_rows(report/'sample_manifest.csv',meta['windows']);save_json(report/'folds.json',CLARE_FOLDS)
    frequencies=[]
    for row,quality in zip(rows,meta['quality']):
        for channel,q in enumerate(quality):frequencies.append(dict(sample_id=row['sample_id'],subject=row['subject'],channel=channel,**q))
    save_rows(report/'native_frequencies.csv',frequencies)
    gates=dict(coverage=all(r['passed'] for r in meta['coverage']),finite=True,source_identity=True,roles=True,labels=True)
    qualified=dict(status='passed' if all(gates.values()) else 'failed',gates=gates,snapshot_path=str(Path(snapshot).resolve()),
                   snapshot_sha256=file_digest(snapshot),windows=len(rows),source_files=len(meta['sources']),
                   old_development_labels_checked=checked,contract_path=str(contract),contract_sha256=file_digest(contract),
                   source_label_threshold=5,encoder_updates=0,outer_scored=False)
    save_json(report/'qualification.json',qualified)
    if qualified['status']!='passed':return qualified
    predictions=[];metrics=[];models={}
    views=dict(eeg=slice(0,24),peripheral=slice(24,36),dual=slice(0,36))
    for spec in CLARE_FOLDS:
        a=np.flatnonzero(np.isin(subjects,spec['fit']));b=np.flatnonzero(np.isin(subjects,spec['select']))
        assert not set(subjects[a])&set(spec['score']) and not set(subjects[b])&set(spec['score'])
        weights=np.array([len(a)/(len(spec['fit'])*sum(subjects[a]==s)) for s in subjects[a]])
        majority=int(np.argmax(np.bincount(y[a],weights=weights,minlength=2)))
        outputs=dict(majority=(np.full(len(b),majority),None))
        for view,cols in views.items():
            model,c=fit_classifier(x[a,cols],y[a],x[b,cols],y[b],c_values=(.01,.1,1.,10.,100.),random_state=17,
                                   scaler_with_mean=True,classification_labels=(0,1),train_sample_weight=weights,
                                   validation_groups=subjects[b])
            outputs[view]=(model.predict(x[b,cols]),c);models[spec['name']+'_'+view]=model
        for name,(pred,c) in outputs.items():
            metrics.append(dict(fold=spec['name'],method=name,subject=spec['select'][0],samples=len(b),C=c,
                                macro_f1=f1_score(y[b],pred,labels=(0,1),average='macro',zero_division=0),
                                balanced_accuracy=balanced_accuracy_score(y[b],pred)))
            predictions.extend(dict(fold=spec['name'],method=name,sample_id=rows[i]['sample_id'],truth=int(y[i]),prediction=int(v))
                               for i,v in zip(b,pred))
    path=assets/'selection_models.joblib';joblib.dump(models,path)
    save_rows(report/'selection_metrics.csv',metrics);save_rows(report/'selection_predictions.csv',predictions)
    decisions=[]
    for spec in CLARE_FOLDS:
        block=[r for r in metrics if r['fold']==spec['name']];base=next(r['macro_f1'] for r in block if r['method']=='majority')
        best=max(r['macro_f1'] for r in block if r['method']!='majority')
        decisions.append(dict(fold=spec['name'],maximum_observation_gain=best-base,passed=best-base>=.02))
    result=dict(status='completed',gates=decisions,task_feasibility=all(g['passed'] for g in decisions),
                encoder_updates=0,outer_scored=False,models_path=str(path.resolve()),models_sha256=file_digest(path))
    save_json(report/'feasibility.json',result);return result
