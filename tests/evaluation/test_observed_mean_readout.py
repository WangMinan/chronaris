import numpy as np
from chronaris.evaluation.application_tasks.observed_mean_readout import choose,prediction,design

def test_exact_anchor_is_available_without_encoder_changes():
    rng=np.random.default_rng(7);cache={}
    for role,n in [('fit',200),('select',40)]:
        x=rng.normal(size=(n,3));anchor=rng.normal(size=n)*100
        cache.update({role+'_features':x,role+'_anchor':anchor,role+'_y':anchor+2*x[:,0],
                      role+'_weights':np.ones(n),role+'_subjects':np.repeat('person',n)})
    raw,_=choose(cache,'raw64');residual,_=choose(cache,'residual64');augmented,_=choose(cache,'augmented65')
    x,a=cache['select_features'],cache['select_anchor'];truth=cache['select_y']
    assert np.sqrt(np.mean((prediction(residual,x,a,'residual64')-truth)**2))<.01
    assert np.sqrt(np.mean((prediction(augmented,x,a,'augmented65')-truth)**2))<.1
    assert np.sqrt(np.mean((prediction(raw,x,a,'raw64')-truth)**2))>50
    assert design(x,a,'augmented65').shape==(40,4)
    np.testing.assert_array_equal(design(x,a,'residual64'),x)
