import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from chronaris.evaluation.application_tasks.application_metrics import paired_trajectory_statistic, segmentation_metrics
from chronaris.evaluation.application_tasks.v4_grouped_statistics import paired_public_statistics, paired_profile_statistics


def test_public_bootstrap_averages_seeds_within_subject_and_rejects_incomplete_pairs():
    rows=[]
    for method in ('first','second'):
        for seed in (17,29,43):
            for index,subject in enumerate(('a','b','c','d')):
                rows.append(dict(domain='clare',route='self_supervised',consumer='linear',task='load',metric='rmse',
                    role='held_out',method=method,seed=seed,fold=f'fold{index//2}',group_id=subject,
                    value=index+seed/100+(index/10 if method=='second' else 0)))
    actual=paired_public_statistics(rows,first_method='first',second_method='second')[0]
    expected=paired_trajectory_statistic(np.arange(4)/10,np.zeros(4))
    assert actual['independent_unit_count']==4
    assert actual['gain_positive_favors_first']==pytest.approx(expected.mean_difference)
    assert actual['ci95']==pytest.approx([expected.bootstrap_lower,expected.bootstrap_upper])
    for changed in (rows[:-1],rows+[rows[0]],pd.DataFrame(rows).assign(domain='dingxin').to_dict('records')):
        with pytest.raises(ValueError):paired_public_statistics(changed,first_method='first',second_method='second')
    changed=[dict(row) for row in rows];changed[0]['fold']='other'
    with pytest.raises(ValueError,match='folds'):paired_public_statistics(changed,first_method='first',second_method='second')


@pytest.mark.parametrize('metric',['macro_f1','rmse','frame_macro_f1','segmental_f1_iou_0.50','boundary_f1_1s','boundary_detection_delay_s'])
def test_profile_sufficient_statistics_match_full_repeated_window_resampling(metric):
    profiles=np.array(['a','a','a','b','c'])
    if metric in ('rmse','macro_f1'):
        truth=np.array([0,1,2,1,2]);first=np.array([0,2,2,0,1]);second=np.array([1,1,2,1,0])
    else:
        truth=np.tile([0,0,1,1,1,2,2,0],(5,1))
        first=np.roll(truth,1,axis=1);second=np.roll(truth,2,axis=1);first[0]=truth[0]
    result=paired_profile_statistics(truth=truth,first_predictions=first[None],second_predictions=second[None],
        profile_ids=profiles,metric=metric,classes=(0,1,2),training_seeds=(17,),query_step_s=.5)
    def score(pred,ids):
        if metric=='rmse':return np.sqrt(np.mean((truth[ids]-pred[ids])**2))
        if metric=='macro_f1':return f1_score(truth[ids],pred[ids],labels=(0,1,2),average='macro',zero_division=0)
        return segmentation_metrics(truth[ids],pred[ids],query_step_s=.5)[metric][0]
    direction=-1 if metric in ('rmse','boundary_detection_delay_s') else 1
    groups=np.unique(profiles)
    draws=np.random.default_rng(17).integers(0,3,size=(2000,3))
    gains=[]
    for draw in draws:
        ids=np.concatenate([np.flatnonzero(profiles==groups[i]) for i in draw])
        gains.append(direction*(score(first,ids)-score(second,ids)))
    assert result['independent_unit_count']==3 and result['window_count']==5
    assert result['windows_per_profile']=={'a':3,'b':1,'c':1}
    assert result['gain_positive_favors_first']==pytest.approx(direction*(score(first,np.arange(5))-score(second,np.arange(5))))
    assert result['ci95']==pytest.approx(np.quantile(gains,[.025,.975]),abs=1e-12)


def test_missing_delay_support_is_reported_without_dropping_resamples_or_windows():
    truth=np.tile([0,0,1,1],(4,1));first=truth[None];second=np.zeros_like(first)
    result=paired_profile_statistics(truth=truth,first_predictions=first,second_predictions=second,
        profile_ids=['a','a','b','b'],metric='boundary_detection_delay_s',training_seeds=(17,))
    assert result['window_count']==4 and result['true_boundary_count']==4
    assert result['matched_boundary_counts']==[[4],[0]]
    assert result['gain_positive_favors_first'] is None and result['ci95'] is None
    assert result['unavailable_bootstrap_draws']==2000


def test_repeated_training_seeds_do_not_inflate_profile_sample_size():
    truth=np.arange(5.)
    first=np.array([[0.,1.,1.,2.,2.]])
    second=np.zeros_like(first)
    common=dict(truth=truth,profile_ids=['a','a','a','b','c'],metric='rmse')
    single=paired_profile_statistics(**common,first_predictions=first,second_predictions=second,training_seeds=(17,))
    triple=paired_profile_statistics(**common,first_predictions=np.repeat(first,3,axis=0),second_predictions=np.repeat(second,3,axis=0))
    assert triple['independent_unit_count']==single['independent_unit_count']==3
    assert triple['ci95']==pytest.approx(single['ci95'])
    assert triple['gain_positive_favors_first']==pytest.approx(single['gain_positive_favors_first'])
