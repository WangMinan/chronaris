"""Compare stored timing predictions without fitting or encoding new representations."""
import numpy as np


def validate_timing_arrays(ids,profiles,conditions,truth,predicted):
    if (truth.shape!=predicted.shape or truth.ndim!=2 or truth.shape[1]!=3
        or len(set(ids))!=len(ids) or not np.isfinite(truth).all() or not np.isfinite(predicted).all()
        or len(ids)!=len(profiles) or len(ids)!=len(conditions)):
        raise ValueError('invalid frozen timing predictions')


def timing_comparisons(truth,predicted,conditions,profiles,constants):
    """Return global and equal-profile errors separately, retaining every condition."""
    rows=[];group_rows=[]
    names=('clock_magnitude','signed_clock','response_lag')
    for condition in sorted(set(conditions)):
        mask=conditions==condition
        for j,target in enumerate(names):
            t=truth[mask,j];p=predicted[mask,j];g=profiles[mask]
            arms={'model':p,**{name:np.full(len(t),value[j]) for name,value in constants.items()}}
            for arm,values in arms.items():
                errors=[]
                for profile in sorted(set(g)):
                    q=g==profile;rmse=float(np.sqrt(np.mean((values[q]-t[q])**2)));errors.append(rmse)
                    group_rows.append(dict(condition=condition,target=target,arm=arm,profile=profile,
                        samples=int(q.sum()),rmse=rmse))
                rows.append(dict(condition=condition,target=target,arm=arm,samples=len(t),
                    profiles=len(set(g)),truth_min=float(t.min()),truth_max=float(t.max()),
                    truth_std=float(t.std()),prediction_std=float(values.std()),
                    rmse=float(np.sqrt(np.mean((values-t)**2))),profile_mean_rmse=float(np.mean(errors)),
                    mae=float(np.mean(abs(values-t)))))
    return rows,group_rows
