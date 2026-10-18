"""Recompute all stored timing contrasts and reconstruct train-only constant references."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_io import HERE,MECHANISMS,MATERIALS,sha,save_csv,save_json
from chronaris.evaluation.application_tasks.frozen_timing_validation import validate_timing_arrays,timing_comparisons


def main():
    if (HERE/'timing_comparison.csv').exists():raise FileExistsError('retain existing timing analysis')
    state=json.loads((MECHANISMS/'mechanism_summary.json').read_text())
    root=MECHANISMS.parent/'timing_data'
    manifest=json.loads((root/'simulation_manifest.json').read_text())
    sources={str(root/'simulation_manifest.json'):sha(root/'simulation_manifest.json')}
    configs={r['scenario_id']:r for r in manifest['observation_scenarios']}
    expected=state['records'][0]['files']
    clean=[r for r in manifest['scenario_rows'] if r['split_id']=='v4_train' and r['scenario_id']=='clean_asynchronous']
    assert len(clean)==512 and len({r['profile_id'] for r in clean})==64
    train=[]
    for r in clean:
        path=Path(r['scenario_manifest_path']).with_name('ground_truth.npz');h=sha(path)
        assert h==r['ground_truth_sha256']==expected[str(path)]
        sources[str(path)]=h
        with np.load(path) as z:lag=float(z['realized_physiology_lag_s'][0])
        for name,config in configs.items():
            signed=config['physiology_clock_offset_s']-config['vehicle_clock_offset_s']
            train.append(dict(profile=r['profile_id'],trajectory=r['trajectory_id'],condition=name,
                clock_magnitude=abs(signed),signed_clock=signed,response_lag=lag+config['additional_physiology_lag_s']))
    # Verify the additive rule against every lag-condition oracle for one fixed training trajectory.
    first=sorted(clean,key=lambda r:r['trajectory_id'])[0]
    checks=[]
    for r in manifest['scenario_rows']:
        if r['trajectory_id']==first['trajectory_id']:
            p=Path(r['scenario_manifest_path']).with_name('ground_truth.npz');h=sha(p)
            assert h==r['ground_truth_sha256']==expected[str(p)];sources[str(p)]=h
            with np.load(p) as z:actual=float(z['realized_physiology_lag_s'][0])
            calculated=next(t['response_lag'] for t in train if t['trajectory']==r['trajectory_id'] and t['condition']==r['scenario_id'])
            assert abs(actual-calculated)<1e-12
            checks.append(dict(condition=r['scenario_id'],actual=actual,reconstructed=calculated))
    targets=['clock_magnitude','signed_clock','response_lag'];t=pd.DataFrame(train)[targets].to_numpy()
    constants={'zero':np.zeros(3),'train_mean':t.mean(axis=0),'train_median':np.median(t,axis=0)}
    all_rows=[];group_rows=[];conditions=[];replay=[];reference=None
    legacy=pd.read_csv(MATERIALS/'timing_metrics.csv')
    target_map=dict(zip(targets,['relative_clock_offset_magnitude_s','signed_relative_clock_offset_s','primary_physiology_response_lag_s']))
    for record in state['records']:
        path=Path(record['prediction_path']);h=sha(path);assert h==record['files'][str(path)]
        sources[str(path)]=h
        with np.load(path) as z:
            ids,profiles,cs,y,p=[z[k] for k in ('sample_ids','profile_ids','conditions','truth','predicted')]
        validate_timing_arrays(ids,profiles,cs,y,p)
        if reference is None:reference=(ids.copy(),profiles.copy(),cs.copy(),y.copy())
        else:assert all(np.array_equal(a,b) for a,b in zip(reference,(ids,profiles,cs,y)))
        assert set(cs)==set(configs)
        meta=dict(unit=record['unit'],method=record['method'],variant=record['unit'].split('/')[1],route=record['route'])
        rows,groups=timing_comparisons(y,p,cs,profiles,constants)
        all_rows.extend(meta|row for row in rows);group_rows.extend(meta|row for row in groups)
        for row in rows:
            if row['arm']=='model':
                old=legacy[(legacy.unit==record['unit'])&(legacy.route==record['route'])&(legacy.condition==row['condition'])&(legacy.target==target_map[row['target']])]
                assert len(old)==1
                delta=abs(row['rmse']-float(old.iloc[0].rmse));assert delta<1e-12
                replay.append(meta|dict(condition=row['condition'],target=row['target'],absolute_difference=delta))
    for name,c in configs.items():
        conditions.append(dict(condition=name,physiology_clock_offset_s=c['physiology_clock_offset_s'],
            vehicle_clock_offset_s=c['vehicle_clock_offset_s'],additional_lag_s=c['additional_physiology_lag_s'],
            physiology_jitter_std_ms=c['physiology_jitter_std_ms'],vehicle_jitter_std_ms=c['vehicle_jitter_std_ms'],
            clock_drift_ppm=c['physiology_clock_drift_ppm']))
    for name,rows in [('timing_comparison',all_rows),('timing_profiles',group_rows),('timing_conditions',conditions),
                      ('timing_replay',replay),('timing_train_targets',train),('timing_oracle_checks',checks)]:save_csv(HERE/(name+'.csv'),rows)
    assert all(sha(p)==h for p,h in sources.items())
    save_json(HERE/'timing_receipt.json',dict(status='completed',sources=sources,
        constants={k:v.tolist() for k,v in constants.items()},records=len(state['records']),conditions=len(configs),
        source_predictions=len(reference[0]),metric_replays=len(replay),training_trajectories=len(clean),
        training_targets_reconstruction='512 clean train oracles plus registered additive observation lag; four contexts per trajectory have equal weight',
        encoder_updates=0,probe_fits=0,direct_time_correction_output=False,
        drift_probe_predictions_available=False))
    print('timing complete',len(all_rows),len(replay),flush=True)


if __name__=='__main__':main()
