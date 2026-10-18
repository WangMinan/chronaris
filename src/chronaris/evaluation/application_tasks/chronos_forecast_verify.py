"""Independent forecast lineage, metric arithmetic and fresh checkpoint replay."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import forecast_data,forecast_folds,VIEWS,predict_forecast
from chronaris.evaluation.application_tasks.chronos_forecast_results import ROUTES,load_selected
from chronaris.evaluation.application_tasks.chronos_information_run import NIGHT_DEADLINE
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.shape_mult_verify import same_tensor_bytes
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.training.rng import canonical_training_state_sha256


def check_forecast_tables(prediction,people,summary,expected):
    assert not prediction[['fold','method','sample_id']].duplicated().any();maximum=0.
    for (fold,method),group in prediction.groupby(['fold','method']):
        truth=expected[fold];assert set(group.sample_id)==set(truth)
        np.testing.assert_allclose(group.truth,[truth[s][0] for s in group.sample_id],atol=1e-12,rtol=0)
        assert group.subject.tolist()==[truth[s][1] for s in group.sample_id]
    for r in people.itertuples():
        block=prediction[(prediction.fold==r.fold)&(prediction.method==r.method)&(prediction.subject==r.subject)]
        assert len(block)==r.samples;e=block.prediction-block.truth
        maximum=max(maximum,abs(np.linalg.norm(e)/np.sqrt(len(e))-r.rmse),abs(np.abs(e).mean()-r.mae))
    for r in summary.itertuples():
        block=people[(people.fold==r.fold)&(people.method==r.method)]
        maximum=max(maximum,abs(block.rmse.mean()-r.rmse),abs(block.mae.mean()-r.mae))
    assert maximum<1e-10;return maximum


def verify_chronos_forecast(report,root,deadline=NIGHT_DEADLINE):
    report,root=Path(report),Path(root);torch.set_num_threads(1)
    protected=json.loads((report/'protected_sources.json').read_text());assert all(file_digest(p)==h for p,h in protected.items())
    locked=json.loads((report/'locked_models.json').read_text());binding=locked['binding'];units=locked['units'];assert len(units)==4
    run=json.loads((report/'run_status.json').read_text());result=json.loads((report/'performance_status.json').read_text())
    assert run['status']==result['status']=='completed'
    assert file_digest(report/'locked_models.json')==result['locked_models_sha256']
    assert file_digest(Path(__file__).with_name('chronos_forecast_results.py'))==result['scoring_source_sha256']
    assert file_digest(report/'PLAN.md')==binding['plan_sha256']
    assert file_digest(report/'qualification.json')==binding['qualification_sha256']
    assert file_digest(report/'input_qualification.json')==binding['input_qualification_sha256']
    q=json.loads((report/'qualification.json').read_text())
    assert q['input_qualification_sha256']==binding['input_qualification_sha256']
    assert q['assets_sha256']==binding['assets_sha256']==file_digest(ASSETS)
    assert q['optimizer_updates']==0 and len(q['checks'])==4
    assert all(file_digest(p)==h for p,h in q['source_files'].items())
    assert all(file_digest(p)==h for p,h in binding['source_files'].items())
    for ref in result['reference_reports']:assert file_digest(ref['path'])==ref['sha256']
    assets=verify_assets(ASSETS);expected={};updates=0;exposures=0
    for spec in forecast_folds():
        for view in VIEWS:
            data,x,mask,proof=forecast_data(spec,view);fit=data.indices('fit')
            expected[spec['name']]={data.rows[i]['sample_id']:(data.y[i],str(data.subjects[i])) for i in data.roles['score']}
            u=next(u for u in units if u['fold']==spec['name'] and u['view']==view)
            assert file_digest(u['receipt_path'])==u['receipt_sha256'];r=json.loads(Path(u['receipt_path']).read_text())
            assert r['complete'] and 2<=r['epochs']<=5 and r['metadata']['binding']==binding
            assert r['metadata']['proof']==proof and r['metadata']['normalizer']==data.normalizer
            assert r['metadata']['fit_sample_ids']==[data.rows[i]['sample_id'] for i in fit]
            assert file_digest(r['checkpoint_path'])==r['checkpoint_sha256']==u['checkpoint_sha256']
            assert file_digest(r['last_path'])==r['last_sha256']
            best=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
            last=torch.load(r['last_path'],map_location='cpu',weights_only=True)
            assert best['metadata']==last['metadata']==r['metadata']
            assert best['epoch']==last['best_epoch']==r['best_epoch']
            assert best['updates']==r['best_epoch']*int(np.ceil(len(fit)/4))
            assert last['updates']==r['updates']==r['epochs']*int(np.ceil(len(fit)/4))
            assert last['epoch']==r['epochs']+1 and last['next_offset']==0 and last['best_sha256']==r['checkpoint_sha256']
            assert all(int(s['step'])==r['updates'] for s in last['optimizer']['state'].values())
            h=pd.read_csv(report/(u['name']+'_history.csv'),float_precision='round_trip');steps=pd.read_csv(report/(u['name']+'_steps.csv.gz'),float_precision='round_trip')
            assert len(h)==r['epochs']+1 and h.epoch.tolist()==list(range(r['epochs']+1))
            assert len(steps)==r['updates'] and int(steps['count'].sum())==r['windows_seen']==len(fit)*r['epochs']
            for key in ('epoch','offset','count','update'):assert steps[key].tolist()==[s[key] for s in last['steps']]
            np.testing.assert_allclose(steps.loss,[s['loss'] for s in last['steps']],atol=1e-10,rtol=1e-10)
            current=float(h.scaled_rmse.iloc[0]);chosen=0
            for row in h.iloc[1:].itertuples():
                order=np.random.default_rng(17+row.epoch).permutation(fit)
                assert last['orders'][str(row.epoch)]==canonical_training_state_sha256([data.rows[i]['sample_id'] for i in order])
                block=steps[steps.epoch==row.epoch];assert block.offset.tolist()==list(range(0,len(fit),4))
                assert block['count'].tolist()==[min(4,len(fit)-i) for i in range(0,len(fit),4)]
                if row.scaled_rmse<current-1e-6:current,chosen=row.scaled_rmse,row.epoch
                assert row.best_epoch==chosen
            assert chosen==r['best_epoch'] and abs(current*data.target_scale-r['selection_rmse'])<1e-10
            if r['status']=='early_stopped':assert r['epochs']-r['best_epoch']>=2
            else:assert r['status']=='max_epochs' and r['epochs']==5
            updates+=r['updates'];exposures+=r['windows_seen'];del data,x,mask,best,last
    assert updates==run['updates'] and exposures==run['windows_seen']
    prediction=pd.read_csv(report/'predictions.csv.gz',dtype={'sample_id':str,'subject':str});people=pd.read_csv(report/'subject_metrics.csv',dtype={'subject':str});summary=pd.read_csv(report/'summary.csv')
    maximum=check_forecast_tables(prediction,people,summary,expected)
    gates=json.loads((report/'performance_gates.json').read_text())
    for g in gates:
        panel=summary[summary.fold==g['fold']].set_index('method');route=g['route'];name='native_'+route+'_joint'
        controls=panel[panel.index.str.startswith('reference_')|(panel.index=='native_'+route+'_aircraft')]
        best=controls.rmse.idxmin();own=panel.loc[name]
        pivot=people[people.fold==g['fold']].pivot(index='subject',columns='method',values='rmse')
        gain=float((1-own.rmse/controls.rmse).min());wins=int((pivot[name]<pivot[best]).sum());mae=bool((own.mae<=controls.mae*1.02).all())
        assert best==g['strongest'] and abs(gain-g['improvement'])<1e-10 and wins==g['subjects_improved']
        assert g['passed']==(gain>=.02 and wins>=2 and mae)
    qualified=[route for route in ROUTES if all(g['passed'] for g in gates if g['route']==route)]
    assert qualified==result['qualified_routes'];errors={};zero_checks={}
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy for native replay')
        for spec in forecast_folds():
            for view in VIEWS:
                data,x,mask,_=forecast_data(spec,view);data.score_open=True;ids=data.indices('score');select=data.indices('select')
                unit=next(u for u in units if u['fold']==spec['name'] and u['view']==view)
                r=json.loads(Path(unit['receipt_path']).read_text())
                for route in ROUTES:
                    model=load_selected(assets,unit,route)
                    actual=predict_forecast(model,data,x,mask,ids,deadline);method='native_'+route+'_'+view
                    old=prediction[(prediction.fold==spec['name'])&(prediction.method==method)]
                    assert old.sample_id.tolist()==[data.rows[i]['sample_id'] for i in ids]
                    np.testing.assert_allclose(actual,old.prediction,atol=1e-5,rtol=1e-5)
                    errors[spec['name']+'_'+method]=float(np.max(np.abs(actual-old.prediction)))
                    validation=predict_forecast(model,data,x,mask,select,deadline)
                    if route=='frozen':
                        with np.load(Path(r['last_path']).parent/'zero_selection.npz') as z:
                            np.testing.assert_array_equal(z['ids'],select);np.testing.assert_allclose(validation,z['prediction'],atol=1e-5,rtol=1e-5)
                        if r['best_epoch']==0:
                            saved=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
                            assert all(same_tensor_bytes(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items())
                            zero_checks[unit['name']]=True;del saved
                    else:assert abs(np.sqrt(np.mean((validation-data.y[select])**2))-r['selection_rmse'])<1e-5
                    if view=='joint' and route in qualified:
                        _replay_pairs(report,model,data,x,mask,ids,route,deadline,errors)
                    del model;torch.cuda.empty_cache()
                del data,x,mask;save_json(report/'replay_progress.json',errors);print(spec['name'],view,'replayed',flush=True)
    pair_gates=json.loads((report/'pairing_gates.json').read_text())
    if qualified:
        pairs=pd.read_csv(report/'pairing_predictions.csv.gz',dtype={'sample_id':str,'subject':str});pp=pd.read_csv(report/'pairing_subject_metrics.csv',dtype={'subject':str});ps=pd.read_csv(report/'pairing_summary.csv')
        maximum=max(maximum,check_forecast_tables(pairs,pp,ps,expected))
        for g in pair_gates:
            block=ps[(ps.fold==g['fold'])&ps.method.str.startswith(g['route']+'_')]
            base=float(summary[(summary.fold==g['fold'])&(summary.method=='native_'+g['route']+'_joint')].rmse.iloc[0])
            shift=float(block[block.method.str.contains('different_record')].rmse.mean()/base-1)
            zero=float(block[block.method==g['route']+'_zero_values'].rmse.iloc[0])
            assert abs(shift-g['mean_degradation'])<1e-10 and g['passed']==(shift>=.02 and base<=zero)
    assert result['paired_routes']==[r for r in qualified if all(g['passed'] for g in pair_gates if g['route']==r)]
    assert all(file_digest(p)==h for p,h in protected.items());verify_assets(ASSETS)
    check=dict(status='passed',protected_files=len(protected),changed_sources=[],models=4,updates=updates,windows_seen=exposures,
               predictions=len(prediction),subject_metrics=len(people),maximum_metric_delta=maximum,replay=errors,
               selected_zero_states_identical=zero_checks,optimizer_updates_in_verification=0)
    save_json(report/'verification.json',check);return check


def _replay_pairs(report,model,data,x,mask,ids,route,deadline,errors):
    donors=pd.read_csv(report/'pairing_donors.csv',dtype={'sample_id':str,'donor':str});old=pd.read_csv(report/'pairing_predictions.csv.gz',dtype={'sample_id':str,'subject':str})
    lookup={r['sample_id']:i for i,r in enumerate(data.rows)}
    for variant in ('zero_values','different_record_1','different_record_2','different_record_3'):
        values=x.clone();valid=mask.clone()
        if variant=='zero_values':values[ids,:,:16]=0
        else:
            block=donors[(donors.fold==data.fold['name'])&(donors.route==route)&(donors.variant==variant)].set_index('sample_id')
            mapping=np.array([lookup[block.loc[data.rows[i]['sample_id'],'donor']] for i in ids])
            assert all(data.subjects[i]==data.subjects[j] and data.rows[i]['record_id']!=data.rows[j]['record_id'] for i,j in zip(ids,mapping))
            for cols in (slice(0,16),slice(28,32)):
                values[ids,:,cols]=x[mapping,:,cols];valid[ids,:,cols]=mask[mapping,:,cols]
        actual=predict_forecast(model,data,values,valid,ids,deadline)
        expected=old[(old.fold==data.fold['name'])&(old.method==route+'_'+variant)]
        assert expected.sample_id.tolist()==[data.rows[i]['sample_id'] for i in ids]
        np.testing.assert_allclose(actual,expected.prediction,atol=1e-5,rtol=1e-5)
        errors[data.fold['name']+'_'+route+'_'+variant]=float(np.max(np.abs(actual-expected.prediction)));del values,valid
