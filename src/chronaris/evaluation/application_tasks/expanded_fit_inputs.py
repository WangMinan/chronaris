"""Merge qualified fit-only participants without changing a development observation."""
import json
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_shape_features import shape_scale_features,shape_scale_floors
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def qualify_expanded_fit(additional,report,assets):
    report,assets=Path(report),Path(assets)
    if (report/'qualification.json').exists():raise FileExistsError('qualification already retained')
    extra=load_arrays(additional);meta=json.loads(str(extra['metadata']))
    save_json(report/'additional_qualification.json',meta)
    if meta['status']!='passed':raise ValueError('additional cohort qualification failed')
    expected={'sub-cp'+s for s in ('005','006','009','013','015','018','020','023','024','027','028','029','030','031','033','035','036','039')}
    if (report/'additional_subjects.json').exists():
        selected=set(json.loads((report/'additional_subjects.json').read_text()))
        assert selected<=expected and len(selected)>=12
        expected=selected
    assert {r['subject'] for r in meta['windows']}==expected
    assert all(c['qualified_records']>=8 and c['windows']>=100 for c in meta['coverage'])
    assert sum(r['accepted'] for r in meta['records'])>=.8*len(meta['records'])
    assert all(r['accepted'] for r in meta['windows'])
    assert all(np.std([r['target'] for r in meta['windows'] if r['subject']==s])>=.01 for s in expected)
    assert len(meta['prefix_checks'])==len(expected)*8 and max(r['max_delta'] for r in meta['prefix_checks'])<=1e-5
    oldq=json.loads(Path('docs/review/stage/fair-residual-20261019/qualification.json').read_text())
    assert file_digest(oldq['snapshot_path'])==oldq['snapshot_sha256']
    old=load_arrays(oldq['snapshot_path']);oldmeta=json.loads(str(old['metadata']))
    native_path=Path('artifacts/application_evaluation/2026-10-19-shape-mult/native.npz')
    native=load_arrays(native_path);nmeta=json.loads(str(native['metadata']))
    assert nmeta['sample_ids']==[r['sample_id'] for r in oldmeta['windows']]
    assert len(oldmeta['windows'])==5951
    assert not expected&{r['subject'] for r in oldmeta['windows']}
    arrays={k:np.concatenate([old[k],extra[k]]) for k in old if k!='metadata'}
    for k in arrays:np.testing.assert_array_equal(arrays[k][:5951],old[k])
    rows=oldmeta['windows']+meta['windows'];sources=dict(oldmeta['sources'],**meta['sources'])
    assert len({r['sample_id'] for r in rows})==len(rows)
    merged=dict(windows=rows,sources=sources,records=oldmeta['records']+meta['records'],
                reconstruction=oldmeta['reconstruction']+meta['reconstruction'])
    arrays['metadata']=np.array(json.dumps(merged,allow_nan=False))
    shapes={k:np.concatenate([native[k],extra['native_'+v]]) for k,v in
            (('values','values'),('masks','mask'),('ages','age'),('scales','scales'))}
    np.testing.assert_array_equal(shapes['masks'],arrays['physiology_mask'])
    np.testing.assert_array_equal(shapes['ages'],arrays['physiology_age'])
    np.testing.assert_array_equal(shapes['scales'],np.array([[q['scale'] for q in r['physiology_quality']] for r in rows]))
    assert all(np.isfinite(v).all() for k,v in arrays.items() if k!='metadata')
    assert all(np.isfinite(v).all() for v in shapes.values())
    native_output=assets/'native.npz'
    if native_output.exists():raise FileExistsError(native_output)
    np.savez_compressed(native_output,**shapes,metadata=np.array(json.dumps(dict(
        sample_ids=[r['sample_id'] for r in rows],sources=sources,
        prefix_checks=nmeta['prefix_checks']+meta['prefix_checks']))))
    for stream in ('physiology','vehicle'):
        mask=arrays[stream+'_mask'];age=arrays[stream+'_age']
        assert mask.mean(1).min()>=.9 and (age[mask]>0).all() and (age[mask]<=2).all()
    np.testing.assert_allclose(arrays['anchor'][:,-1],[r['persistence'] for r in rows],atol=1e-10,rtol=0)
    errors=np.array([r['component_max_errors'] for r in meta['reconstruction']])
    assert (errors[:,:3]<=.0002+1e-10).all() and (errors[:,3]<=.0006+1e-10).all()
    path=assets/'original_inputs.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,**arrays)
    gates=dict(original_identity=True,fit_roles=True,coverage=True,source_identity=True,formula=True,prefix=True,finite=True)
    original=dict(status='passed',gates=gates,snapshot_path=str(path.resolve()),snapshot_sha256=file_digest(path))
    save_json(report/'original_qualification.json',original)
    folds=[];receipts=[];subjects=np.array([r['subject'] for r in rows])
    for spec in FOLDS:
        fold=dict(spec,fit=sorted(set(spec['fit'])|expected));folds.append(fold)
        fit=np.flatnonzero(np.isin(subjects,fold['fit']));floors=shape_scale_floors(shapes['scales'],fit)
        values=shape_scale_features(shapes['values'],shapes['masks'],shapes['scales'],floors)
        assert np.isfinite(values).all()
        path=assets/(spec['name']+'_inputs.npz');np.savez_compressed(path,**dict(arrays,physiology=values))
        receipt=dict(status='passed',gates=gates,snapshot_path=str(path.resolve()),snapshot_sha256=file_digest(path),
                     fit_subjects=fold['fit'],fit_sample_ids=[rows[i]['sample_id'] for i in fit],floors=floors,windows=len(rows))
        save_json(report/(spec['name']+'_qualification.json'),receipt);receipts.append(receipt)
    save_json(report/'folds.json',folds);save_json(report/'source_hashes.json',sources)
    save_rows(report/'additional_subject_coverage.csv',meta['coverage'])
    save_rows(report/'sample_manifest.csv',[{k:r[k] for k in ('sample_id','subject','record_id','target','persistence')} for r in rows])
    result=dict(status='passed',gates=gates,folds=receipts,windows=len(rows),original_windows=5951,
                additional_sha256=file_digest(additional),native_sha256=file_digest(native_path),
                merged_native_path=str(native_output.resolve()),merged_native_sha256=file_digest(native_output),
                source_files=len(sources),encoder_updates=0)
    save_json(report/'qualification.json',result);return result
