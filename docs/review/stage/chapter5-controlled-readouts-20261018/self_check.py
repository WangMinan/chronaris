"""Small adversarial checks for the audit's selection and transform boundaries."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from readouts import nested, transform, metric


def main():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(120,64));g=np.repeat(['a','b','c','d'],30)
    y=x[:,0]*.3+rng.normal(size=120)*.1
    tr=np.arange(90);te=np.arange(90,120)
    first=nested(x,y,g,tr,te,'pca16',None)
    changed=y.copy();changed[te]=1e9
    second=nested(x,changed,g,tr,te,'pca16',None)
    assert first[1:]==second[1:] and np.array_equal(first[0],second[0])
    # Changing scoring observations cannot alter inner selection either.
    altered=x.copy();altered[te]*=1000
    third=nested(altered,y,g,tr,te,'pca16',None)
    assert third[1:]==first[1:]
    a,b=transform(x[tr],x[te],'pca24')
    scaler=StandardScaler().fit(x[tr]);pca=PCA(24,svd_solver='full').fit(scaler.transform(x[tr]))
    assert np.array_equal(a,pca.transform(scaler.transform(x[tr])))
    assert np.array_equal(b,pca.transform(scaler.transform(x[te])))
    assert np.array_equal(transform(x[tr],x[te],'rp24_17')[0],transform(x[tr],x[te],'rp24_17')[0])
    assert not np.array_equal(transform(x[tr],x[te],'rp24_17')[0],transform(x[tr],x[te],'rp24_29')[0])
    labels=np.zeros(120);labels[:30]=1
    missing=nested(x,labels,g,tr,te,'full64',[0,1])
    assert missing[-1]=='inner_training_missing_class'
    assert metric(np.zeros(5),np.zeros(5),[0,1])==.5
    assert metric(np.array([1.,-1.]),np.zeros(2),None)==1.
    from dataclasses import make_dataclass
    from export_frozen import exclusive_sample
    fields=[s+'_'+f for s in ('physiology','vehicle') for f in ('timestamps_s','values','feature_mask')]
    Sample=make_dataclass('Sample',fields+['source_sample_hash'])
    start=738193.8;cutoff=((start+12/86400)-start)*86400
    t=np.array([0.,cutoff,cutoff+.01]);v=np.arange(3.)[:,None];mask=np.ones((3,1),bool)
    sample=Sample(t,v,mask,t,v,mask,'a'*64)
    row=dict(task_valid_mask={'event_response':True},record_time_origin_native=start,context_start_s=0.)
    result=exclusive_sample(sample,row)
    assert len(result.physiology_timestamps_s)==len(result.vehicle_timestamps_s)==1
    assert len(sample.physiology_timestamps_s)==3 and result.source_sample_hash!=sample.source_sample_hash
    assert exclusive_sample(sample,{'task_valid_mask':{'difficulty':True}}) is sample
    print('PASS: outer-label isolation, inner-only transform and selection, fixed projections, class-coverage rejection, metric controls')
    print('PASS: exclusive native endpoint, new observation identity, unchanged source and classification inputs')


if __name__=='__main__':main()
