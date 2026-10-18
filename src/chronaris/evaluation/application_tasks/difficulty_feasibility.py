"""Source-verified classification scouting on fit/selection subjects only."""
import json
from pathlib import Path
import re
import shlex
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS,FairResidualData
from chronaris.evaluation.application_tasks.consumer_model_selection import fit_classifier
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows


def run_difficulty_feasibility(output):
    output=Path(output)
    if (output/'classification_feasibility.json').exists():raise FileExistsError('existing probe')
    source=json.loads(Path('docs/review/stage/chapter5-controlled-readouts-20261018/semantics_sources.json').read_text())
    assert file_digest(source['source_semantics_path'])==source['source_semantics_sha256']
    raw=json.loads(Path(source['source_semantics_path']).read_text())
    path='/home/wangminan/dataset/chronaris/physio_net/physionet.org/files/virtual-reality-piloting/1.0.0/dataPackage/task-ils/PerfMetrics.csv'
    code='import hashlib\nwith open('+repr(path)+',"rb") as h: print(hashlib.file_digest(h,"sha256").hexdigest())'
    command='exec /home/wangminan/env/anaconda3/envs/chronaris/bin/python -c '+shlex.quote(code)
    proc=subprocess.run(['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=10',
                         'wangminan@10.69.120.91',command],capture_output=True,text=True,timeout=120)
    if proc.returncode:raise RuntimeError(proc.stderr)
    assert proc.stdout.strip()==raw['perf_metrics_sha256']
    lookup={}
    for r in raw['perf_metrics']:
        key=(int(r['subject']),int(r['date']),int(r['run']))
        if key in lookup:raise ValueError('ambiguous source label')
        lookup[key]=int(r['difficulty'])-1
    report=Path('docs/review/stage/shape-mult-20261019')
    assets=Path('artifacts/application_evaluation/2026-10-19-shape-mult')
    predictions=[];scores=[];labels_out=[];coverage=[]
    for spec in FOLDS:
        q=report/(spec['name']+'_qualification.json')
        data=FairResidualData(assets/(spec['name']+'_inputs.npz'),q,spec,'cpu')
        labels=[];absolute=[]
        for row in data.rows:
            sid=row['sample_id'];key=(int(row['subject'].removeprefix('sub-cp')),
                                     int(re.search(r'ses-(\d+)',sid)[1]),int(re.search(r'run-(\d+)',sid)[1]))
            label=lookup[key];assert label==int(re.search(r'level-(\d+)',sid)[1])-1
            labels.append(label)
            quality=row['physiology_quality']
            absolute.append([*[np.arcsinh(r['center']) for r in quality],*[np.log1p(r['scale']) for r in quality]])
            if spec['name']=='fold01':labels_out.append(dict(sample_id=sid,subject=row['subject'],label=label,record_id=row['record_id']))
        labels=np.array(labels);absolute=np.array(absolute)
        for subject in sorted(set(data.subjects)):
            assert set(labels[data.subjects==subject])=={0,1,2,3}
            coverage.append(dict(fold=spec['name'],subject=subject,counts=np.bincount(labels[data.subjects==subject],minlength=4).tolist()))
        features={}
        for role in ('fit','select'):
            ids=data.indices(role);dual=data.statistics(ids,True);air=data.statistics(ids,False)
            phys=dual[:,:64];np.testing.assert_array_equal(dual[:,64:],air)
            features[role]=dict(aircraft=air,physiology=phys,dual=dual,
                                physiology_absolute=np.c_[phys,absolute[ids]],dual_absolute=np.c_[dual,absolute[ids]])
        a,b=data.indices('fit'),data.indices('select')
        values={}
        for method in features['fit']:
            model,c=fit_classifier(features['fit'][method],labels[a],features['select'][method],labels[b],
                                   c_values=(.01,.1,1.,10.,100.),random_state=17,scaler_with_mean=True,
                                   classification_labels=(0,1,2,3),train_sample_weight=data.weights[a].numpy(),
                                   validation_groups=data.subjects[b])
            values[method]=(model.predict(features['select'][method]),c)
        majority=int(np.argmax(np.bincount(labels[a],weights=data.weights[a].numpy(),minlength=4)))
        values['majority']=(np.full(len(b),majority),None)
        for method,(pred,c) in values.items():
            scores.append(dict(fold=spec['name'],method=method,subject=spec['select'][0],samples=len(b),C=c,
                               macro_f1=f1_score(labels[b],pred,labels=(0,1,2,3),average='macro',zero_division=0),
                               balanced_accuracy=balanced_accuracy_score(labels[b],pred)))
            predictions.extend(dict(fold=spec['name'],method=method,sample_id=data.rows[i]['sample_id'],
                                    truth=int(labels[i]),prediction=int(y)) for i,y in zip(b,pred))
        assert not data.score_open
    save_rows(output/'classification_selection_metrics.csv',scores)
    save_rows(output/'classification_labels.csv',labels_out)
    pd.DataFrame(predictions).to_csv(output/'classification_selection_predictions.csv.gz',index=False)
    result=dict(status='qualified',source_path=path,source_sha256=raw['perf_metrics_sha256'],windows=len(labels_out),
                coverage=coverage,encoder_updates=0,outer_score_opened=False,scope='fit_selection_only')
    save_json(output/'classification_feasibility.json',result)
    return result
