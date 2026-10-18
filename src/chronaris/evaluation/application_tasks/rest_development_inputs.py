"""Fixed development-cohort resting reference; all original observations and roles stay bound."""
import json
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_shape_features import shape_scale_features,shape_scale_floors
from chronaris.evaluation.application_tasks.expanded_fit_inputs import load_arrays
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json


def resting_values(native,masks,local_centers,rest_centers,rest_scales,floors):
    values=native.copy()
    values[:,:,::4]+=np.asarray(local_centers)[:,None,:]-np.asarray(rest_centers)[:,None,:]
    return shape_scale_features(values,masks,rest_scales,floors)


def qualify_rest_development(source,report,assets):
    report,assets=Path(report),Path(assets);assets.mkdir(parents=True,exist_ok=True)
    if (report/'qualification.json').exists():raise FileExistsError('retained qualification')
    rest=json.loads(Path(source).read_text());lookup={r['subject']:r for r in rest['calibrations']}
    original=json.loads(Path('docs/review/stage/fair-residual-20261019/qualification.json').read_text())
    assert file_digest(original['snapshot_path'])==original['snapshot_sha256']
    old=load_arrays(original['snapshot_path']);meta=json.loads(str(old['metadata']));rows=meta['windows']
    native_path=Path('artifacts/application_evaluation/2026-10-19-shape-mult/native.npz');native=load_arrays(native_path)
    nmeta=json.loads(str(native['metadata']));assert nmeta['sample_ids']==[r['sample_id'] for r in rows]
    subjects=np.array([r['subject'] for r in rows]);unique=sorted(set(subjects))
    assert len(rows)==5951 and set(unique)=={'sub-cp'+s for s in ('003','004','008','011','019','022','025')}
    assert set(unique)<=set(lookup)
    for subject in unique:
        item=lookup[subject];assert item['accepted'] and len(item['center'])==len(item['scale'])==4
        earliest=min(r['source_origin_native']+(r['cutoff_s']-30)/86400 for r in rows if r['subject']==subject)
        assert item['end_native']<earliest and item['start_native']==item['end_native']-60/86400
        assert all(q['maximum_gap_s']<=.25 for q in item['quality'])
    local=np.array([[q['center'] for q in r['physiology_quality']] for r in rows])
    centers=np.array([lookup[s]['center'] for s in subjects]);scales=np.array([lookup[s]['scale'] for s in subjects])
    assert np.isfinite(local).all() and np.isfinite(centers).all() and np.isfinite(scales).all()
    person_scales=np.array([lookup[s]['scale'] for s in unique]);receipts=[]
    for spec in FOLDS:
        fit=np.flatnonzero(np.isin(unique,spec['fit']));floors=shape_scale_floors(person_scales,fit)
        values=resting_values(native['values'],native['masks'],local,centers,scales,floors)
        assert np.isfinite(values).all()
        unchanged_crossings=np.where(native['masks'][:,:,3::4],native['values'][:,:,3::4],0).astype(np.float32)
        np.testing.assert_array_equal(values[:,:,3::4],unchanged_crossings)
        path=assets/(spec['name']+'_inputs.npz');assert not path.exists()
        np.savez_compressed(path,**dict(old,physiology=values))
        gates=dict(preflight=True,calibration_coverage=True,original_identity=True,finite=True,fit_only_floor=True,history_isolation=True)
        receipt=dict(status='passed',gates=gates,snapshot_path=str(path.resolve()),snapshot_sha256=file_digest(path),
                     fit_subjects=list(spec['fit']),floor_fit_subjects=[unique[i] for i in fit],floors=floors,windows=len(rows))
        save_json(report/(spec['name']+'_qualification.json'),receipt);receipts.append(receipt)
    result=dict(status='passed',folds=receipts,windows=5951,subjects=unique,calibrations=[lookup[s] for s in unique],
                rest_source_path=str(Path(source).resolve()),rest_source_sha256=file_digest(source),
                original_snapshot_path=original['snapshot_path'],original_snapshot_sha256=original['snapshot_sha256'],
                native_path=str(native_path.resolve()),native_sha256=file_digest(native_path),encoder_updates=0,
                condition='preflight_unlabelled_rest_observation_available')
    save_json(report/'qualification.json',result);return result
