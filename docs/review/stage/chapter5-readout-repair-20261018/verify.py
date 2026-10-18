"""Recompute every metric, selection, clip pair and immutable-source fingerprint."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from repair_io import HERE,OLD,REPO,ASSETS,REGISTRY,sha,save_json
from chronaris.evaluation.application_tasks.public_readout_revision import load_readout_revision


def main():
    for p,h in json.loads((HERE/'implementation_sources.json').read_text()).items():assert sha(REPO/p)==h,p
    source_hashes={}
    prior=OLD.parent/'chapter5-degradation-20261018'
    for name in ('statistics/verification.json','training/verification.json','readouts/verification.json',
                 'statistics/constant_verification.json','revision/verification.json'):
        for p,h in json.loads((prior/name).read_text())['sources'].items():
            source_hashes[str(Path(p) if Path(p).is_absolute() else REPO/p)]=h
    source_hashes.update(json.loads((prior/'forward/summary.json').read_text())['checkpoint_hashes'])
    source_hashes.update(json.loads((OLD/'raw_verification.json').read_text())['cache_files'])
    source_hashes.update(json.loads((OLD/'original_validation_verification.json').read_text())['sources'])
    for r in json.loads((OLD/'export_verification.json').read_text())['records']:
        source_hashes[r['checkpoint']]=r['checkpoint_sha256'];source_hashes[r['receipt']]=r['receipt_sha256']
    for p,h in source_hashes.items():assert sha(p)==h,p
    for p,h in json.loads((OLD/'evidence.json').read_text())['files'].items():assert sha(OLD/p)==h,p
    preparation=json.loads((HERE/'preparation.json').read_text())
    assert preparation['observations']==3318 and len(preparation['reused_representations'])==22
    for r in preparation['reused_representations']:
        assert sha(r['output'])==r['output_sha256'] and sha(r['checkpoint'])==r['checkpoint_sha256']
    total_pred=0;total_groups=0;old_checks=0;max_old_difference=0.
    for domain in ('clare','cogpilot'):
        root=HERE/domain;receipt=json.loads((root/'receipt.json').read_text())
        assert receipt['status']=='completed' and receipt['encoder_updates']==receipt['new_gpu_exports']==0
        assert all(sha(p)==h for p,h in receipt['sources'].items())
        data=load_readout_revision(ASSETS/domain)
        positions={s:i for i,s in enumerate(data.targets.sample_ids)}
        group_map=dict(zip(data.dataset.sample_ids,data.dataset.group_ids))
        pred=pd.read_csv(root/'predictions.csv.gz',dtype={'subject':str})
        groups=pd.read_csv(root/'subjects.csv',dtype={'subject':str})
        folds=pd.read_csv(root/'folds.csv')
        inner=pd.read_csv(root/'selection.csv',dtype={'subject':str})
        candidates=pd.read_csv(root/'candidates.csv')
        registry=json.loads(REGISTRY.read_text())['domains'][domain]['folds']['development']
        for key,part in pred.groupby(['task','route','fold','method','arm']):
            task,route,fold,method,arm=key;roles=registry[fold]
            assert not part.sample_id.duplicated().any()
            eligible={s for s,i in positions.items() if group_map[s] in roles['validation_subjects'] and bool(data.targets.valid_masks[task][i])}
            assert set(part.sample_id)==eligible
            assert set(part.subject)==set(roles['validation_subjects'])
            truth=np.array([float(data.targets.values[task][positions[s]]) for s in part.sample_id])
            np.testing.assert_allclose(part.target,truth,rtol=1e-15,atol=1e-15)
            classes=list(range(4)) if task=='difficulty' else [0,1] if 'classification' in task else None
            vals=[]
            for subject,g in part.groupby('subject'):
                val=float(f1_score(g.target,g.prediction,labels=classes,average='macro',zero_division=0)) if classes else float(np.sqrt(np.mean((g.target-g.prediction)**2)))
                r=groups[(groups.task==task)&(groups.route==route)&(groups.fold==fold)&(groups.method==method)&(groups.arm==arm)&(groups.subject==subject)]
                assert len(r)==1 and abs(r.value.iloc[0]-val)<1e-10
                assert abs(r.bias.iloc[0]-np.mean(g.prediction-g.target))<1e-10
                assert abs(r.prediction_std.iloc[0]-np.std(g.prediction))<1e-10
                vals.append(val);total_groups+=1
            r=folds[(folds.task==task)&(folds.route==route)&(folds.fold==fold)&(folds.method==method)&(folds.arm==arm)]
            assert len(r)==1 and abs(r.value.iloc[0]-np.mean(vals))<1e-10
            if method!='constant':
                p=inner[(inner.task==task)&(inner.route==route)&(inner.fold==fold)&(inner.method==method)]
                c=candidates[(candidates.task==task)&(candidates.route==route)&(candidates.fold==fold)&(candidates.method==method)]
                assert set(p.subject)==set(roles['train_subjects'])
                assert len(p)==len(c)*len(roles['train_subjects'])
                subset=json.loads(c.selections.iloc[0])[arm.removesuffix('_bounded_selection').removesuffix('_clipped')]
                metric='bounded_value' if arm.endswith('_bounded_selection') else 'value'
                means=p.groupby('candidate')[metric].mean().loc[subset]
                best=means.max() if classes else means.min()
                chosen=int(r.candidate.iloc[0])
                assert chosen in subset and abs(means.loc[chosen]-best)<1e-12
                if 'clipped' in arm or 'bounded_selection' in arm:
                    assert part.prediction.between(1,9).all()
                if arm=='ridge_clipped':
                    original=pred[(pred.task==task)&(pred.route==route)&(pred.fold==fold)&(pred.method==method)&(pred.arm=='ridge')].set_index('sample_id').loc[part.sample_id]
                    np.testing.assert_allclose(part.prediction,np.clip(original.prediction,1,9),atol=1e-15,rtol=1e-15)
        old=pd.read_csv(OLD/('clare_folds.csv' if domain=='clare' else 'cogpilot_boundary_folds.csv'))
        for r in folds.itertuples():
            if r.arm not in ('uniform3','ridge') or r.task=='workload_standard_classification':continue
            match=old[(old.task==r.task)&(old.method==r.method)&(old.route==r.route)&(old.fold==r.fold)&(old.variant=='full64')]
            assert len(match)==1
            difference=abs(r.value-match.value.iloc[0]);assert difference<1e-10
            max_old_difference=max(max_old_difference,difference);old_checks+=1
        total_pred+=len(pred)
    gates=pd.read_csv(HERE/'gates.csv')
    for gate in gates.itertuples():
        f=pd.read_csv(HERE/gate.domain/'folds.csv')
        f=f[(f.task==gate.task)&(f.route==gate.route)]
        value=f[(f.method=='chronaris')&(f.arm==gate.main_arm)].value.mean()
        classification='classification' in gate.task or gate.task=='difficulty'
        sign=1 if classification else -1
        constants=f[f.method=='constant'].groupby('arm').value.mean()
        assert bool(((value-constants)*sign>1e-12).all())==gate.mean_beats_all_simple
        arm='matched9' if classification else 'ridge_bounded_selection' if gate.domain=='clare' else 'ridge'
        singles=f[f.method.isin(['physiology_only','vehicle_only'])&(f.arm==arm)].groupby('method').value.mean()
        assert bool(((value-singles)*sign>1e-12).all())==gate.mean_beats_both_single
    assert len(gates[~gates.legacy_task])==8 and not gates[~gates.legacy_task].mean_joint_pass.any()
    result=dict(status='passed',original_files_verified=len(source_hashes),original_files_unchanged=True,previous_audit_bundle_unchanged=True,
        encoder_updates=0,new_gpu_exports=0,reused_frozen_routes=22,observations_verified=3318,
        predictions_recomputed=total_pred,subject_metrics_recomputed=total_groups,unchanged_control_scores=old_checks,
        max_control_difference=max_old_difference,primary_cells=8,primary_joint_passes=0)
    save_json(HERE/'verification.json',result)
    save_json(HERE/'evidence.json',{'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
        if p.is_file() and '__pycache__' not in str(p) and p.name!='evidence.json'}})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
