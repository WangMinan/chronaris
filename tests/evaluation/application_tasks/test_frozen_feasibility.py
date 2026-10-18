"""Checks for the inference-only audit's information boundaries and data semantics."""
import numpy as np
import pytest
from chronaris.evaluation.application_tasks.event_target_validation import response_at,past_descriptor
from chronaris.evaluation.application_tasks.frozen_complementarity import evaluate_frozen_combinations,transform_pair
from chronaris.evaluation.application_tasks.frozen_timing_validation import validate_timing_arrays,timing_comparisons


def test_response_native_endpoint_and_unordered_observations():
    origin=738193.;seconds=np.arange(-12.,9.,.25);times=origin+seconds/86400
    g=np.where(times<origin,2.,5.)
    result=response_at(times,g,origin)
    assert result['response']==3.
    assert result['baseline']==2.
    order=np.arange(len(g));order[8:10]=order[8:10][::-1]
    assert response_at(times[order],g[order],origin)==result


def test_match_descriptor_ignores_future_conductance():
    t=738193.+np.arange(-20.,20.)/86400;event=738193.
    roll=np.linspace(0,1,len(t));speed=np.full(len(t),150.);g=np.ones(len(t))
    before=past_descriptor(t,roll,speed,t,g,event,t[0])
    g[t>=event]=np.nan
    after=past_descriptor(t,roll,speed,t,g,event,t[0])
    np.testing.assert_array_equal(before,after)


def test_outer_labels_do_not_choose_readout_or_fusion_weights():
    rng=np.random.default_rng(17);n=160
    first=rng.normal(size=(n,64));second=rng.normal(size=(n,64));full=rng.normal(size=(n,64))
    groups=np.repeat(np.arange(4),40);y=np.tile([0.,1.],80)
    train=np.arange(120);test=np.arange(120,160)
    a,logs=evaluate_frozen_combinations(first,second,full,y,groups,train,test,classes=[0,1])
    y[test]=999
    b,other=evaluate_frozen_combinations(first,second,full,y,groups,train,test,classes=[0,1])
    assert logs==other
    for key in a:
        np.testing.assert_array_equal(a[key]['prediction'],b[key]['prediction'])
        assert a[key]['parameter']==b[key]['parameter'] and a[key]['weight']==b[key]['weight']
    with pytest.raises(ValueError,match='overlapping'):
        evaluate_frozen_combinations(first,second,full,y,groups,train,np.arange(110,130),classes=[0,1])


def test_pca_fit_uses_training_samples_only():
    rng=np.random.default_rng(2);train=rng.normal(size=(90,128));test=rng.normal(size=(4,128))
    a,b=transform_pair(train,test,64)
    c,d=transform_pair(train,np.concatenate([test,np.full((1,128),1e12)]),64)
    np.testing.assert_array_equal(a,c);np.testing.assert_allclose(b,d[:4],atol=1e-12)


def test_timing_nonzero_targets_and_duplicate_guard():
    ids=np.array(['a','b']);groups=np.array(['p','q']);conditions=np.array(['offset','offset'])
    truth=np.array([[3.,-3.,5.],[3.,3.,7.]])
    validate_timing_arrays(ids,groups,conditions,truth,truth)
    rows,_=timing_comparisons(truth,truth,conditions,groups,{'zero':np.zeros(3)})
    assert next(r['rmse'] for r in rows if r['target']=='signed_clock' and r['arm']=='zero')==3.
    assert next(r['rmse'] for r in rows if r['target']=='signed_clock' and r['arm']=='model')==0.
    with pytest.raises(ValueError):validate_timing_arrays(np.array(['a','a']),groups,conditions,truth,truth)
