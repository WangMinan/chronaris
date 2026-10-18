"""Independent arithmetic, role/asset checks and fresh-process pretrained replay."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_information_qualification import information_data,VIEW_COLUMNS
from chronaris.evaluation.application_tasks.chronos_information_readout import VIEWS,ROUTES,feature_rows
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.recent_model_training import load_encoder_checkpoint
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.chronos_information import ChronosInformationEncoder
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.training.rng import isolated_training_rng,canonical_training_state_sha256


def verify_chronos_information(report,root,replay=True):
    report,root=Path(report),Path(root)
    protected=json.loads((report/'protected_sources.json').read_text())
    assert all(file_digest(p)==h for p,h in protected.items())
    encoded=json.loads((report/'encoding_status.json').read_text());assert encoded['status']=='completed'
    result=json.loads((report/'performance_status.json').read_text());assert result['status']=='completed'
    units=encoded['units'];assert len(units)==6
    assets=verify_assets(json.loads((report/'qualification.json').read_text())['assets_path'])
    locked=json.loads((report/'locked_readouts.json').read_text())
    assert file_digest(locked['path'])==locked['sha256']
    assert file_digest(report/'encoding_status.json')==locked['binding']['encoding_status_sha256']
    models=joblib.load(locked['path']);assert len(models)==16
    updates=0
    for spec in FOLDS:
        data,x,m=information_data(spec);fit=data.indices('fit');order=np.random.default_rng(17).permutation(fit)
        expected=[data.rows[i]['sample_id'] for i in order]
        for view,columns in VIEW_COLUMNS.items():
            unit=next(u for u in units if u['fold']==spec['name'] and u['view']==view)
            assert unit['fit_subjects']==list(spec['fit']) and unit['fit_sample_ids']==expected
            assert file_digest(unit['checkpoint_path'])==unit['checkpoint_sha256']
            saved=torch.load(unit['checkpoint_path'],map_location='cpu',weights_only=True)
            assert saved['update']==unit['updates']==len(expected)
            assert saved['metadata']['target_supervision']=='history_only' and not saved['heads']
            assert [r['sample_ids'][0] for r in saved['history']]==expected
            assert all(len(r['sample_ids'])==1 and not r['task_counts'] and np.isfinite(r['loss']) for r in saved['history'])
            digest=canonical_training_state_sha256(x[:,:,columns],m[:,:,columns],[r['sample_id'] for r in data.rows],data.normalizer)
            assert saved['metadata']['input_sha256']==digest
            for route in ROUTES:
                path=Path(unit[route+'_path']);assert file_digest(path)==unit[route+'_sha256']
                receipt=json.loads(path.with_suffix('.json').read_text())
                assert receipt['completed'] and receipt['rows']==len(data.rows)
                assert receipt['binding']['input_sha256']==digest
                matrix=np.load(path,mmap_mode='r',allow_pickle=False)
                assert matrix.shape==(len(data.rows),unit['features'])
            updates+=saved['update'];del saved
    assert updates==encoded['updates']
    prediction=pd.read_csv(report/'predictions.csv.gz');people=pd.read_csv(report/'subject_metrics.csv');summary=pd.read_csv(report/'summary.csv')
    assert not prediction[['fold','method','sample_id']].duplicated().any()
    maximum=0.
    for spec in FOLDS:
        data,_,_=information_data(spec);data.score_open=True;ids=data.indices('score')
        expected={data.rows[i]['sample_id']:data.y[i] for i in ids}
        for method,group in prediction[prediction.fold==spec['name']].groupby('method'):
            assert set(group.sample_id)==set(expected)
            np.testing.assert_allclose(group.truth,[expected[s] for s in group.sample_id],atol=1e-12,rtol=0)
    for row in people.itertuples():
        group=prediction[(prediction.fold==row.fold)&(prediction.method==row.method)&(prediction.subject==row.subject)]
        assert len(group)==row.samples
        err=group.prediction-group.truth
        maximum=max(maximum,abs(np.linalg.norm(err)/np.sqrt(len(err))-row.rmse),abs(np.abs(err).mean()-row.mae))
    for row in summary.itertuples():
        block=people[(people.fold==row.fold)&(people.method==row.method)]
        maximum=max(maximum,abs(block.rmse.mean()-row.rmse),abs(block.mae.mean()-row.mae))
    assert maximum<1e-10
    gates=json.loads((report/'performance_gates.json').read_text())
    for g in gates:
        panel=summary[summary.fold==g['fold']].set_index('method');route=g['route']
        controls=panel[panel.index.str.startswith('reference_') | panel.index.isin(
            ['chronos_'+route+'_'+v for v in VIEWS if v!='joint'])]
        own=panel.loc['chronos_'+route+'_joint'];best=controls.rmse.idxmin()
        pivot=people[people.fold==g['fold']].pivot(index='subject',columns='method',values='rmse')
        wins=int((pivot['chronos_'+route+'_joint']<pivot[best]).sum())
        gain=float((1-own.rmse/controls.rmse).min());mae=bool((own.mae<=1.02*controls.mae).all())
        assert g['strongest']==best and abs(g['improvement']-gain)<1e-10
        assert g['passed']==(gain>=.02 and wins>=2 and mae)
    errors={};torch.set_num_threads(1)
    if replay:
        with development_gpu_lock() as acquired:
            if not acquired:raise BlockingIOError('GPU busy for fresh replay')
            for spec in FOLDS:
                data,x,m=information_data(spec);data.score_open=True;ids=data.indices('score')
                for route in ROUTES:
                    vectors={}
                    for view,columns in VIEW_COLUMNS.items():
                        unit=next(u for u in units if u['fold']==spec['name'] and u['view']==view)
                        with isolated_training_rng(17):
                            encoder=ChronosInformationEncoder(assets,len(columns)).eval()
                            if route=='adapted':
                                saved=torch.load(unit['checkpoint_path'],map_location='cpu',weights_only=True)
                                load_encoder_checkpoint(encoder,saved['encoder']);del saved
                            pieces=[];reference=np.load(unit[route+'_path'],mmap_mode='r')
                            for off in range(0,len(ids),8):
                                batch=ids[off:off+8]
                                actual=encoder(x[batch][:,:,columns],m[batch][:,:,columns]).cpu().numpy()
                                np.testing.assert_allclose(actual,reference[batch],atol=1e-5,rtol=1e-5)
                                pieces.append(actual)
                            vectors[view]=np.concatenate(pieces)
                            key=spec['name']+'_'+route+'_'+view
                            errors[key+'_features']=float(np.max(np.abs(vectors[view]-reference[ids])))
                            del encoder;torch.cuda.empty_cache()
                    vectors['independent']=np.c_[vectors['physiology_history'],vectors['aircraft']]
                    for view,features in vectors.items():
                        values=models[spec['name']+'_'+route+'_'+view].predict(features)
                        old=prediction[(prediction.fold==spec['name'])&(prediction.method=='chronos_'+route+'_'+view)]
                        assert old.sample_id.tolist()==[data.rows[i]['sample_id'] for i in ids]
                        np.testing.assert_allclose(values,old.prediction,atol=1e-5,rtol=1e-5)
                        errors[spec['name']+'_'+route+'_'+view+'_prediction']=float(np.max(np.abs(values-old.prediction)))
                    save_json(report/'replay_progress.json',errors);print(spec['name'],route,'replayed',flush=True)
    if result['qualified_routes']:
        pairs=pd.read_csv(report/'pairing_predictions.csv.gz');pair_summary=pd.read_csv(report/'pairing_summary.csv')
        pair_gates=json.loads((report/'pairing_gates.json').read_text())
        assert len(pair_gates)==2*len(result['qualified_routes'])
        for g in pair_gates:
            p=pair_summary[(pair_summary.fold==g['fold'])&pair_summary.method.str.startswith(g['route']+'_')]
            base=float(summary[(summary.fold==g['fold'])&(summary.method=='chronos_'+g['route']+'_joint')].rmse.iloc[0])
            degradation=float(p[p.method.str.contains('different_record')].rmse.mean()/base-1)
            zero=float(p[p.method==g['route']+'_zero_values'].rmse.iloc[0])
            assert g['passed']==(degradation>=.02 and base<=zero)
        assert not pairs[['fold','method','sample_id']].duplicated().any()
    assert all(file_digest(p)==h for p,h in protected.items());verify_assets(json.loads((report/'qualification.json').read_text())['assets_path'])
    check=dict(status='passed',protected_files=len(protected),changed_sources=[],adaptation_units=len(units),updates=updates,
               predictions=len(prediction),subject_metrics=len(people),maximum_metric_delta=maximum,replay=errors,
               optimizer_updates_in_verification=0,readout_fits_in_verification=0)
    save_json(report/'verification.json',check)
    save_json(report/'asset_manifest.json',dict(root=str(root.resolve()),files={str(p.relative_to(root)):file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    return check
