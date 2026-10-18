"""Independently recompute every new outer score and verify frozen source identities."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from audit_io import HERE,feature_path,sha,save_json


def main():
    protected=json.loads((HERE/'protected_sources.json').read_text())
    changed=[p for p,h in protected.items() if sha(p)!=h]
    assert not changed,changed
    errors=[];prediction_count=0;subject_count=0;fold_count=0;sources={}
    for domain in ('clare','cogpilot'):
        root=HERE/('complementarity_'+domain);p=pd.read_csv(root/'predictions.csv.gz')
        s=pd.read_csv(root/'subjects.csv');f=pd.read_csv(root/'folds.csv');prediction_count+=len(p)
        keys=['domain','task','fold','route','method','rule']
        assert not p.duplicated(keys+['sample_id']).any()
        rows=[]
        for group,d in p.groupby(keys+['subject']):
            task=group[1];labels=[0,1,2,3] if task=='difficulty' else [0,1] if 'classification' in task else None
            y,yp=d.target.to_numpy(),d.prediction.to_numpy()
            metric=float(f1_score(y,yp,labels=labels,average='macro',zero_division=0)) if labels else float(np.sqrt(np.mean((y-yp)**2)))
            rows.append(dict(zip(keys+['subject'],group))|{'computed':metric})
        scored=pd.DataFrame(rows);joined=s.merge(scored,on=keys+['subject'],validate='one_to_one')
        assert len(joined)==len(s);delta=max(abs(joined.value-joined.computed));assert delta<1e-10
        aggregate=scored.groupby(keys).computed.mean().reset_index()
        joined=f.merge(aggregate,on=keys,validate='one_to_one');assert len(joined)==len(f)
        delta=max(delta,max(abs(joined.value-joined.computed)));assert delta<1e-10
        errors.append(float(delta));subject_count+=len(s);fold_count+=len(f)
        for method in ('physiology_only','vehicle_only','chronaris'):
            for route in ('self_supervised','task_guided'):
                path=feature_path(domain,method,route);meta=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==meta['output_sha256'];checkpoint=meta['checkpoint']
                assert sha(checkpoint)==meta['checkpoint_sha256'];sources[checkpoint]=meta['checkpoint_sha256']
        receipt=json.loads((root/'receipt.json').read_text())
        assert receipt['encoder_updates']==receipt['new_gpu_exports']==0
    e=pd.read_csv(HERE/'event_targets.csv');assert e.target_matches_float32.all() and e.event_start_error_s.max()==0
    c=pd.read_csv(HERE/'clare_targets.csv');assert ((c.score>=5).astype(int)==c.standard_class).all()
    assert ((c.score>=7).astype(int)==c.legacy_class).all()
    assert (e.control_native-e.event_native).abs().min()*86400>10
    replay=pd.read_csv(HERE/'timing_replay.csv');assert replay.absolute_difference.max()<1e-12
    save_json(HERE/'verification.json',dict(status='passed',protected_sources=len(protected),changed_sources=changed,
        checkpoint_hashes=sources,new_outer_predictions=prediction_count,subject_metrics=subject_count,
        fold_metrics=fold_count,max_recomputed_score_error=max(errors),original_timing_metrics_recomputed=len(replay),
        raw_event_target_matches=len(e),raw_clare_windows=len(c),encoder_updates=0,new_gpu_exports=0,
        source_scope='registered files plus 12 bound encoder checkpoints; raw remote files independently hashed before and after reading'))
    print(json.dumps({k:v for k,v in json.loads((HERE/'verification.json').read_text()).items() if k!='checkpoint_hashes'},ensure_ascii=False))


if __name__=='__main__':main()
