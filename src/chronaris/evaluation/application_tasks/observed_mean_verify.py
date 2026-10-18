"""Fresh frozen features and independent FP64 readout/metric verification."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.observed_mean_readout import PARENT,STYLES,parent_metadata,prediction,design,gates_for,PairReadout
from chronaris.evaluation.application_tasks.fair_residual_training import load_fair_model,predict_fair
from chronaris.evaluation.application_tasks.shape_mult_run import shape_data
from chronaris.evaluation.application_tasks.shape_mult_results import metric_tables,pairing_predictions
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock


def verify_observed_mean(report,assets):
    report,assets=Path(report),Path(assets);start=time.monotonic();torch.set_num_threads(1)
    status=json.loads((report/'run_status.json').read_text());assert status['status']=='completed' and status['encoder_updates']==0
    locked=json.loads((report/'locked_readouts.json').read_text());root,parent=parent_metadata()
    assert file_digest(report/'PLAN.md')==locked['plan_sha256']
    from chronaris.evaluation.application_tasks import observed_mean_readout
    assert file_digest(observed_mean_readout.__file__)==locked['source_sha256']
    assert file_digest(locked['readout_path'])==locked['readout_sha256']
    protected=json.loads((report/'protected_sources.json').read_text());assert all(file_digest(p)==h for p,h in protected.items())
    models=joblib.load(locked['readout_path']);rows=pd.read_csv(report/'predictions.csv.gz',dtype={'sample_id':str,'subject':str})
    people,summary=metric_tables(rows)
    for filename,table,keys in [('subject_metrics.csv',people,['fold','method','subject']),('summary.csv',summary,['fold','method'])]:
        before=pd.read_csv(report/filename);a=before.sort_values(keys);b=table.sort_values(keys)
        assert a[keys].values.tolist()==b[keys].values.tolist()
        np.testing.assert_allclose(a[['rmse','mae']],b[['rmse','mae']],atol=1e-10,rtol=0)
    gates=gates_for(people,summary);saved=json.loads((report/'performance_gates.json').read_text())
    for a,b in zip(gates,saved):
        assert {k:v for k,v in a.items() if k!='improvement'}=={k:v for k,v in b.items() if k!='improvement'}
        assert abs(a['improvement']-b['improvement'])<1e-10
    assert status['qualified_styles']==[s for s in STYLES if all(g['passed'] for g in gates if g['style']==s)]
    replay={};refit={}
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        for item in locked['items']:
            assert time.monotonic()-start<1200 and time.time()<1792373400
            assert file_digest(item['checkpoint_path'])==item['checkpoint_sha256'] and file_digest(item['cache_path'])==item['cache_sha256']
            name=item['name'];cache=dict(np.load(item['cache_path'],allow_pickle=False))
            cache.update(dict(np.load(assets/(name+'_score.npz'),allow_pickle=False)))
            data=shape_data(PARENT,root,item['fold'],item['version']);data.score_open=True
            model=load_fair_model(item['checkpoint_path'],data)
            for role in ('fit','select','score'):
                ids=data.indices(role)
                assert cache[role+'_ids'].tolist()==[data.rows[i]['sample_id'] for i in ids]
                np.testing.assert_array_equal(cache[role+'_y'],data.y[ids])
                np.testing.assert_array_equal(cache[role+'_subjects'],data.subjects[ids])
                np.testing.assert_array_equal(cache[role+'_weights'],data.weights[ids].cpu().numpy())
                np.testing.assert_array_equal(cache[role+'_anchor'],data.anchor[ids,-1].cpu().numpy())
                chosen=ids if role=='score' else ids[:3]
                features=predict_fair(model,data,chosen)[0];saved_features=cache[role+'_features'][:len(chosen)]
                np.testing.assert_allclose(features,saved_features,atol=1e-5,rtol=1e-5)
                replay[name+'_'+role]=float(np.max(np.abs(features-saved_features)))
            for style in STYLES:
                x=design(cache['fit_features'],cache['fit_anchor'],style)
                v=design(cache['select_features'],cache['select_anchor'],style)
                z=design(cache['score_features'],cache['score_anchor'],style)
                y=cache['fit_y']-(cache['fit_anchor'] if style=='residual64' else 0)
                target=cache['select_y'];groups=cache['select_subjects'];best=float('inf');selected=None;expected=None
                for alpha in (.1,1.,10.,100.,1000.):
                    scaler=StandardScaler().fit(x,sample_weight=cache['fit_weights'])
                    ridge=Ridge(alpha=alpha).fit(scaler.transform(x),y,sample_weight=cache['fit_weights'])
                    val=ridge.predict(scaler.transform(v))+(cache['select_anchor'] if style=='residual64' else 0)
                    error=float(np.mean([np.sqrt(np.mean((val[groups==s]-target[groups==s])**2)) for s in np.unique(groups)]))
                    if error<best:
                        best=error;selected=alpha
                        expected=ridge.predict(scaler.transform(z))+(cache['score_anchor'] if style=='residual64' else 0)
                assert selected==locked['selected_alpha'][name+'_'+style]
                actual=prediction(models[name+'_'+style],cache['score_features'],cache['score_anchor'],style)
                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                before=rows[(rows.fold==item['fold']['name'])&(rows.method==item['version']+'_'+style)]
                assert before.sample_id.tolist()==cache['score_ids'].tolist()
                np.testing.assert_array_equal(before.subject,cache['score_subjects'])
                np.testing.assert_allclose(before.truth,cache['score_y'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(before.prediction,actual,atol=1e-10,rtol=0)
                refit[name+'_'+style]=float(np.max(np.abs(actual-before.prediction.to_numpy())))
                if item['version']=='candidate' and style in status['qualified_styles']:
                    p,_=pairing_predictions(data,item['checkpoint_path'],PairReadout(models[name+'_'+style],cache['score_anchor'],style))
                    old=pd.read_csv(report/'pairing_predictions.csv.gz',dtype={'sample_id':str,'subject':str})
                    actual=pd.DataFrame(p);actual['method']=style+'_'+actual.method
                    before=old[(old.fold==item['fold']['name'])&old.method.str.startswith(style+'_')]
                    keys=['method','sample_id'];a=actual.sort_values(keys);b=before.sort_values(keys)
                    assert a[keys].values.tolist()==b[keys].values.tolist()
                    np.testing.assert_allclose(a.prediction,b.prediction,atol=1e-5,rtol=1e-5)
            del model,data;torch.cuda.empty_cache();print(name,'verified',flush=True)
    result=dict(status='passed',encoder_updates=0,independent_ridge_refits=90,protected_files=len(protected),
                feature_replay=replay,prediction_replay=refit,predictions=len(rows),elapsed_s=time.monotonic()-start)
    save_json(report/'verification.json',result);return result
