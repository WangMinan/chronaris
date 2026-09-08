"""Paired subject/profile uncertainty; folds and training seeds are not subjects."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from chronaris.evaluation.application_tasks.application_metrics import (
    paired_trajectory_statistic, _mean_segmental_f1, _mean_boundary_f1, _detection_delays)

REPETITIONS = 2000
SEEDS = (17, 29, 43)
LOWER_IS_BETTER = {'rmse', 'boundary_detection_delay_s'}


def paired_public_statistics(records, *, first_method, second_method, training_seeds=SEEDS, bootstrap_seed=17):
    """Accept one subject metric per method/seed, then average seeds within subject."""
    frame = pd.DataFrame(records)
    required = {'domain','route','consumer','task','metric','role','method','seed','fold','group_id','value'}
    if not required <= set(frame.columns) or not len(frame):
        raise ValueError('public paired statistics lack subject provenance')
    if (not set(frame.domain) <= {'cogpilot','clare'} or not set(frame.role) <= {'validation','held_out'}
        or not set(frame.route) <= {'self_supervised','task_guided'}
        or not set(frame.metric) <= {'macro_f1','rmse'} or first_method == second_method):
        raise ValueError('public statistics require approved domains, roles and scalar metrics')
    if not training_seeds or len(set(training_seeds))!=len(training_seeds) or not set(training_seeds)<=set(SEEDS):
        raise ValueError('invalid training seed set')
    selected=frame[frame.method.isin((first_method,second_method))]
    if set(selected.method)!={first_method,second_method} or selected[list(required)].isna().any().any() or not np.isfinite(selected.value).all():
        raise ValueError('paired subject values are missing or nonfinite')
    outputs=[]
    keys=['domain','route','consumer','task','metric','role']
    for key,group in selected.groupby(keys,sort=True):
        if (group.duplicated(['method','seed','group_id']).any()
            or group.groupby('group_id').fold.nunique().max()!=1 or set(group.seed)!=set(training_seeds)):
            raise ValueError('subject/seed observations overlap folds or are duplicated/incomplete')
        groups=tuple(sorted(group.group_id.unique()))
        expected=pd.MultiIndex.from_product([(first_method,second_method),training_seeds,groups],names=['method','seed','group_id'])
        values=group.set_index(['method','seed','group_id']).value.reindex(expected)
        if values.isna().any():
            raise ValueError('every paired subject needs the same methods and training seeds')
        values=values.to_numpy().reshape(2,len(training_seeds),len(groups))
        direction=-1 if key[4] in LOWER_IS_BETTER else 1
        statistic=paired_trajectory_statistic(direction*values[0].mean(axis=0),direction*values[1].mean(axis=0),
            seed=bootstrap_seed,bootstrap_repetitions=REPETITIONS)
        outputs.append(dict(zip(keys,key,strict=True)) | dict(first_method=first_method,second_method=second_method,
            independent_unit='subject',independent_unit_count=len(groups),training_seeds=list(training_seeds),
            first_value=float(values[0].mean()),second_value=float(values[1].mean()),
            gain_positive_favors_first=statistic.mean_difference,ci95=[statistic.bootstrap_lower,statistic.bootstrap_upper],
            per_seed_gain={str(seed):float(direction*(values[0,i]-values[1,i]).mean()) for i,seed in enumerate(training_seeds)},
            repetitions=REPETITIONS,bootstrap_seed=bootstrap_seed,aggregation='seed_mean_within_subject_then_subject_mean'))
    return outputs


def paired_profile_statistics(*, truth, first_predictions, second_predictions, profile_ids, metric,
                              classes=None, training_seeds=SEEDS, bootstrap_seed=17, query_step_s=30./96.):
    """Pool complete profile contents for each draw, evaluate seeds, then average seeds."""
    truth=np.asarray(truth)
    predictions=np.asarray([first_predictions,second_predictions])
    profiles=np.asarray(profile_ids)
    if (truth.ndim not in (1,2) or profiles.shape!=(len(truth),) or predictions.shape!=(2,len(training_seeds),*truth.shape)
        or not training_seeds or len(set(training_seeds))!=len(training_seeds) or not set(training_seeds)<=set(SEEDS)
        or not np.isfinite(truth).all() or not np.isfinite(predictions).all() or not np.isfinite(query_step_s) or query_step_s<=0):
        raise ValueError('profile statistics require finite aligned predictions with declared training seeds')
    if any(not isinstance(value,str) or not value for value in profiles.tolist()):
        raise ValueError('parameter profiles must have explicit nonempty identities')
    groups=tuple(sorted(set(profiles.tolist())))
    if len(groups)<2:
        raise ValueError('profile resampling requires at least two named parameter profiles')
    classification=metric in {'macro_f1','frame_macro_f1'}
    window_metric=metric in {'segmental_f1_iou_0.50','boundary_f1_1s'}
    if (metric not in {'macro_f1','rmse','frame_macro_f1','segmental_f1_iou_0.50','boundary_f1_1s','boundary_detection_delay_s'}
        or truth.ndim!=(1 if metric in {'macro_f1','rmse'} else 2)):
        raise ValueError('profile metric differs from the approved scalar/sequence target shape')
    if metric!='rmse':
        if (not np.equal(truth,np.floor(truth)).all() or not np.equal(predictions,np.floor(predictions)).all()):
            raise ValueError('classification and segmentation labels must be integers')
    if classification:
        if classes is None or len(classes)<2 or len(set(classes))!=len(classes) or not np.isin(truth,classes).all() or not np.isin(predictions,classes).all():
            raise ValueError('profile macro F1 requires the fixed complete class vocabulary')
    draws=np.random.default_rng(bootstrap_seed).integers(0,len(groups),size=(REPETITIONS,len(groups)))
    weights=np.zeros((REPETITIONS+1,len(groups)),dtype=np.int64)
    weights[0]=1
    np.add.at(weights,(np.arange(1,REPETITIONS+1)[:,None],draws),1)
    # Per-profile sums preserve every original window, including repeated profiles.
    numerator=np.zeros((2,len(training_seeds),len(groups),len(classes),len(classes))) if classification else np.zeros((2,len(training_seeds),len(groups)))
    denominator=np.zeros((2,len(training_seeds),len(groups)))
    for index,profile in enumerate(groups):
        selected=profiles==profile
        target=truth[selected]
        for method in range(2):
            for seed_index in range(len(training_seeds)):
                predicted=predictions[method,seed_index,selected]
                if classification:
                    numerator[method,seed_index,index]=confusion_matrix(target.ravel(),predicted.ravel(),labels=classes)
                elif metric=='rmse':
                    numerator[method,seed_index,index]=np.square(target.astype(float)-predicted).sum()
                    denominator[method,seed_index,index]=len(target)
                elif window_metric:
                    value=(_mean_segmental_f1(target,predicted,.5) if metric.startswith('segmental') else
                           _mean_boundary_f1(target,predicted,tolerance_s=1.,query_step_s=query_step_s))
                    numerator[method,seed_index,index]=value*len(target)
                    denominator[method,seed_index,index]=len(target)
                else:
                    delays=_detection_delays(target,predicted,query_step_s=query_step_s)
                    numerator[method,seed_index,index]=sum(delays)
                    denominator[method,seed_index,index]=len(delays)
    totals=np.einsum('rg,msg...->msr...',weights,numerator)
    if classification:
        twice_true_positive=2*np.diagonal(totals,axis1=-2,axis2=-1)
        class_totals=totals.sum(axis=-1)+totals.sum(axis=-2)
        estimates=np.divide(twice_true_positive,class_totals,out=np.zeros_like(twice_true_positive),where=class_totals>0).mean(axis=-1)
    else:
        counts=np.einsum('rg,msg->msr',weights,denominator)
        estimates=np.divide(totals,counts,out=np.full_like(totals,np.nan),where=counts>0)
        if metric=='rmse':estimates=np.sqrt(estimates)
    direction=-1 if metric in LOWER_IS_BETTER else 1
    gains=direction*(estimates[0]-estimates[1])
    complete=np.isfinite(gains).all(axis=0)
    averaged=gains.mean(axis=0)
    result=dict(metric=metric,independent_unit='parameter_profile',independent_unit_count=len(groups),
        profile_ids=list(groups),window_count=len(truth),windows_per_profile={p:int((profiles==p).sum()) for p in groups},
        training_seeds=list(training_seeds),repetitions=REPETITIONS,bootstrap_seed=bootstrap_seed,
        first_value=float(estimates[0,:,0].mean()) if np.isfinite(estimates[0,:,0]).all() else None,
        second_value=float(estimates[1,:,0].mean()) if np.isfinite(estimates[1,:,0]).all() else None,
        gain_positive_favors_first=float(averaged[0]) if complete[0] else None,
        per_seed_gain={str(seed):float(gains[i,0]) if np.isfinite(gains[i,0]) else None for i,seed in enumerate(training_seeds)},
        unavailable_bootstrap_draws=int((~complete[1:]).sum()),
        ci95=np.quantile(averaged[1:],(.025,.975)).tolist() if complete.all() else None,
        aggregation='pool_all_windows_of_sampled_profiles_then_seed_mean',all_windows_retained=True)
    if metric=='boundary_detection_delay_s':
        result['delay_semantics']='conditional_on_a_later_predicted_boundary'
        result['matched_boundary_counts']=denominator.sum(axis=-1).astype(int).tolist()
        result['true_boundary_count']=int((truth[:,1:]!=truth[:,:-1]).sum())
    return result


def summarize_initial_profile_statistics(*, output_root,
    runtime_root='artifacts/application_evaluation/2026-09-06_v4-learning-curves/simulation',
    registry_path='docs/requirements/thesis-v4-simulation-manifest.json'):
    """Source-checked seed-17 development application of the final grouped statistic."""
    import json
    from pathlib import Path
    from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True)
    registry=json.loads(Path(registry_path).read_text())
    samples={sample:row['profile_id'] for row in registry['trajectories'] if row['role']=='validation'
             for sample in row['context_sample_ids']}
    sources={str(registry_path):sha256_file(registry_path)}
    definitions=(('macro_f1','workload_class_true','linear_class',(0,1,2)),
        ('rmse','workload_true','linear_regression',None),
        ('frame_macro_f1','state_true','causal_tcn_duration_state',(0,1,2,3,4)),
        ('segmental_f1_iou_0.50','state_true','causal_tcn_duration_state',None),
        ('boundary_f1_1s','state_true','causal_tcn_duration_state',None),
        ('boundary_detection_delay_s','state_true','causal_tcn_duration_state',None))
    results=[]
    for route in ('self_supervised','task_guided'):
        archives={}
        for method in ('chronaris','physiology_only','vehicle_only','mult','contiformer'):
            directory=Path(runtime_root)/method/'consumers'/route/'500'/method
            path=directory/'consumer_manifest.json';manifest=json.loads(path.read_text())
            sources[str(path)]=sha256_file(path)
            if (manifest['label_used_for_encoder_training'] is not (route=='task_guided')
                or manifest['evaluation_roles']!=['validation']
                or manifest['fold_id']!='v4_simulation_g1_development_g2_confirmation'
                or sha256_file(manifest['prediction_path'])!=manifest['prediction_sha256']):
                raise ValueError('initial consumer prediction provenance changed')
            sources[manifest['prediction_path']]=manifest['prediction_sha256']
            with np.load(manifest['prediction_path'],allow_pickle=False) as archive:
                archives[method]={key:archive[key].copy() for key in archive.files}
        reference=archives['chronaris'];ids=reference['validation_sample_ids'].astype(str)
        if len(ids)!=256 or len(set(ids))!=len(ids) or set(ids)!=set(samples):
            raise ValueError('initial profile statistics require exactly the frozen 256 development windows')
        for method,archive in archives.items():
            if method=='chronaris':continue
            if not np.array_equal(archive['validation_sample_ids'],reference['validation_sample_ids']):
                raise ValueError('paired initial consumer sample order changed')
            for metric,target,prediction,classes in definitions:
                truth=reference['validation_'+target]
                if not np.array_equal(truth,archive['validation_'+target]):
                    raise ValueError('paired initial consumer truth changed')
                result=paired_profile_statistics(truth=truth,first_predictions=reference['validation_'+prediction][None],
                    second_predictions=archive['validation_'+prediction][None],profile_ids=[samples[s] for s in ids],
                    metric=metric,classes=classes,training_seeds=(17,))
                results.append(dict(route=route,first_method='chronaris',second_method=method,**result))
    result=dict(scope='retrospective_initial_training256_seed17_development_only',confirmation_opened=False,
                source_files=sources,source_code_sha256=sha256_file(__file__),comparisons=results)
    path=root/'profile_statistics.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
