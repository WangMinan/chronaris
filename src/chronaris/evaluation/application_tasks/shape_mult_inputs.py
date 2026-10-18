"""Fold-local shape inputs; all original sample roles and frozen observations stay bound."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.dataset.cogpilot_shape_features import shape_scale_features, shape_scale_floors
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json, save_rows


def qualify_shape_inputs(native_path, original_path, report, assets):
    report, assets = Path(report), Path(assets)
    if (report/'qualification.json').exists():
        raise FileExistsError('qualification retained')
    with np.load(original_path, allow_pickle=False) as raw:
        old = {k: raw[k] for k in raw.files}
    meta = json.loads(str(old['metadata'])); rows = meta['windows']
    with np.load(native_path, allow_pickle=False) as raw:
        native = {k: raw[k] for k in raw.files}
    nmeta = json.loads(str(native['metadata']))
    assert len(rows) == 5951 and all(r['accepted'] for r in rows)
    assert nmeta['sample_ids'] == [r['sample_id'] for r in rows]
    assert nmeta['sources'] == meta['sources'] and len(meta['sources']) == 420
    assert np.isfinite(native['values']).all() and np.isfinite(native['scales']).all()
    np.testing.assert_array_equal(native['masks'], old['physiology_mask'])
    np.testing.assert_array_equal(native['ages'], old['physiology_age'])
    expected_scales = np.array([[q['scale'] for q in r['physiology_quality']] for r in rows])
    np.testing.assert_array_equal(native['scales'], expected_scales)
    assert len(nmeta['prefix_checks']) == 56 and max(c['max_delta'] for c in nmeta['prefix_checks']) <= 1e-5
    flat = pd.read_csv('docs/review/stage/mult-diagnosis-20261019/native_clip_windows.csv')
    flat = flat[flat.final_flattened & flat.final_feature_valid]
    indices = {r['sample_id']: i for i,r in enumerate(rows)}
    subjects = np.array([r['subject'] for r in rows]); results = []; recovery = []
    for spec in FOLDS:
        fit = np.flatnonzero(np.isin(subjects, spec['fit']))
        floors = shape_scale_floors(native['scales'], fit)
        values = shape_scale_features(native['values'], native['masks'], native['scales'], floors)
        for r in flat.itertuples():
            i = indices[r.sample_id]; value = float(values[i,-1,r.channel*4+1])
            assert value > 0
            recovery.append(dict(fold=spec['name'], sample_id=r.sample_id, channel=r.channel, restored_log_std=value))
        arr = dict(old, physiology=values)
        path = assets/(spec['name']+'_inputs.npz')
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **arr)
        receipt = dict(status='passed', gates=dict(finite=True, original_identity=True, masks_ages=True,
                                                  fit_only_floor=True, prefix=True, flattened_recovered=True),
                       snapshot_path=str(path.resolve()), snapshot_sha256=file_digest(path),
                       fit_subjects=list(spec['fit']), fit_sample_ids=[rows[i]['sample_id'] for i in fit],
                       floors=floors, windows=len(rows), recovered_channel_windows=len(flat))
        save_json(report/(spec['name']+'_qualification.json'), receipt)
        results.append(receipt)
    receipt = dict(status='passed', folds=results, windows=len(rows), source_files=len(meta['sources']),
                   native_sha256=file_digest(native_path), original_sha256=file_digest(original_path),
                   real_prefix_checks=nmeta['prefix_checks'], new_training_updates=0)
    save_rows(report/'recovered_dynamics.csv', recovery)
    save_json(report/'qualification.json', receipt)
    return receipt
