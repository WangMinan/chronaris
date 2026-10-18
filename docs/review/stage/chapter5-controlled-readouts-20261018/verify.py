"""Verify originals, source identity, complete coverage and every saved metric; no fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import HERE, REPO, REGISTRY, PREP, manifest, sha, save_json
from readouts import metric


def main():
    sources={}
    old=HERE.parent/'chapter5-degradation-20261018'
    for relative in ('statistics/verification.json','training/verification.json','readouts/verification.json',
                     'statistics/constant_verification.json','revision/verification.json'):
        for name,h in json.loads((old/relative).read_text())['sources'].items():
            path=Path(name) if Path(name).is_absolute() else REPO/name
            if str(path) in sources:assert sources[str(path)]==h
            sources[str(path)]=h
    sources.update(json.loads((old/'forward/summary.json').read_text())['checkpoint_hashes'])
    prior_count=len(sources)
    export=json.loads((HERE/'export_verification.json').read_text())
    assert export['status']=='completed' and export['encoder_updates']==0 and len(export['records'])==22
    for r in export['records']:
        sources[r['checkpoint']]=r['checkpoint_sha256'];sources[r['receipt']]=r['receipt_sha256']
        assert sha(r['output'])==r['output_sha256'] and r['disjoint_subjects'] and r['encoder_updates']==0
        assert not set(r['original_subjects']) & set(r['development_subjects'])
    boundary=json.loads((HERE/'export_boundary_verification.json').read_text())
    assert boundary['status']=='completed' and len(boundary['records'])==11
    for r in boundary['records']:
        assert r['input_view']=='exclusive_event_endpoint_v1' and r['encoder_updates']==0
        assert sha(r['output'])==r['output_sha256'] and sha(r['checkpoint'])==r['checkpoint_sha256']
    assert json.loads((HERE/'boundary_control_verification.json').read_text())['classification_exact']
    raw=json.loads((HERE/'raw_verification.json').read_text())
    assert raw['status']=='passed' and raw['windows']==106 and raw['target_checks']==130 and raw['field_checks']==1128
    sources.update(raw['cache_files'])
    assert sha(raw['raw_response_path'])==raw['raw_response_sha256']
    original=json.loads((HERE/'original_validation_verification.json').read_text())
    sources.update(original['sources'])
    for name,h in sources.items():assert sha(name)==h,name
    registry=json.loads(REGISTRY.read_text())
    count=0;group_count=0
    for prefix in ('clare','cogpilot','cogpilot_boundary'):
        domain=prefix.split('_')[0]
        proof=json.loads((HERE/f'{prefix}_verification.json').read_text())
        assert sha(REGISTRY)==proof['registry_sha256']
        assert sha(PREP/domain/'task_targets.pt')==proof['targets_sha256']
        targets=torch.load(PREP/domain/'task_targets.pt',weights_only=True)
        all_ids=np.array(targets['sample_ids']);all_groups=np.array([r['subject_id'] for r in manifest(domain)])
        target_index={s:i for i,s in enumerate(all_ids)}
        pred=pd.read_csv(HERE/f'{prefix}_predictions.csv.gz',dtype={'subject':str})
        groups=pd.read_csv(HERE/f'{prefix}_subjects.csv',dtype={'subject':str})
        folds=pd.read_csv(HERE/f'{prefix}_folds.csv')
        selection=pd.read_csv(HERE/f'{prefix}_selection.csv',dtype={'subject':str})
        assert len(pred)==json.loads((HERE/f'{prefix}_verification.json').read_text())['predictions']
        assert not json.loads((HERE/f'{prefix}_verification.json').read_text())['unavailable']
        for key,part in pred.groupby(['task','route','fold','method','variant']):
            task,route,fold,method,variant=key
            assert not part.sample_id.duplicated().any()
            expected=registry['domains'][domain]['folds']['development'][fold]
            assert set(part.subject)==set(expected['validation_subjects'])
            mask=targets['valid_masks'][task].numpy().reshape(-1)&np.isin(all_groups,expected['validation_subjects'])
            assert set(part.sample_id)==set(all_ids[mask])
            positions=[target_index[s] for s in part.sample_id]
            truth=targets['values'][task].numpy().reshape(-1)[positions]
            assert np.allclose(part.target.to_numpy(),truth,atol=1e-15,rtol=1e-15)
            selected=selection[(selection.task==task)&(selection.route==route)&(selection.fold==fold)&
                               (selection.method==method)&(selection.variant==variant)]
            classes=list(range(4 if domain=='cogpilot' else 2)) if task in ('difficulty','workload_classification') else None
            if method!='constant':
                assert set(selected.subject)==set(expected['train_subjects'])
                vals=selected.groupby('parameter').value.mean().sort_index()
                chosen=vals.idxmax() if classes else vals.idxmin()
                saved=folds[(folds.task==task)&(folds.route==route)&(folds.fold==fold)&(folds.method==method)&(folds.variant==variant)]
                assert chosen==saved.parameter.iloc[0]
            for subject,sub in part.groupby('subject'):
                score=metric(sub.target.to_numpy(),sub.prediction.to_numpy(),classes)
                row=groups[(groups.task==task)&(groups.route==route)&(groups.fold==fold)&(groups.method==method)&
                           (groups.variant==variant)&(groups.subject==subject)]
                assert len(row)==1 and abs(score-row.value.iloc[0])<1e-10
                assert abs(np.std(sub.prediction)-row.prediction_std.iloc[0])<1e-10
                assert abs(np.mean(sub.prediction-sub.target)-row.bias.iloc[0])<1e-10
                group_count+=1
        count+=len(pred)
    save_json(HERE/'verification.json',dict(status='passed',prior_frozen_source_files_verified=prior_count,
        total_original_files_verified=len(sources),frozen_encoder_routes=22,additional_boundary_exports=11,encoder_updates=0,
        prediction_rows_recomputed=count,subject_metrics_recomputed=group_count,
        original_files_unchanged=True,outer_scoring_subjects_excluded_from_selection=True))
    files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()
           and '__pycache__' not in str(p) and p.name!='evidence.json'}
    save_json(HERE/'evidence.json',dict(files=files))
    print(json.dumps(json.loads((HERE/'verification.json').read_text()),indent=2))


if __name__=='__main__':main()
