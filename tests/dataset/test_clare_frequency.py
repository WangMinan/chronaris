import numpy as np
import pytest
from chronaris.dataset.clare_frequency import frequency_features


def test_native_frequency_survives_feature_extraction_and_future_is_excluded():
    t=np.arange(0,11,1/256);x=np.sin(2*np.pi*10*t)
    values,quality=frequency_features(t,x,0,10,'eeg')
    assert values[2]>.99 and quality['higher_than_query_nyquist']>.99
    changed=x.copy();changed[t>=10]=1e6
    assert frequency_features(t,changed,0,10,'eeg')==(values,quality)
    _,low=frequency_features(t,np.sin(2*np.pi*2*t),0,10,'eeg')
    assert low['higher_than_query_nyquist']<1e-10
    broken=t.copy();broken[100]=broken[99]
    with pytest.raises(ValueError,match='time_order'):frequency_features(broken,x,0,10,'eeg')


def test_missing_label_column_is_distinct_from_missing_signal(tmp_path):
    import pandas as pd,json
    from chronaris.dataset.clare_frequency import extract_clare_frequency
    (tmp_path/'Labels').mkdir();pd.DataFrame({'level_0':[1,5,9]}).to_csv(tmp_path/'Labels'/'a.csv',index=False)
    result=extract_clare_frequency(dict(root=str(tmp_path),subjects=['a']))
    meta=json.loads(str(result['metadata']))
    assert result['features'].shape==(0,36) and meta['status']=='failed'
    assert meta['coverage'][0]['legal']==3
    assert meta['records'][0]['missing_sources']
    assert all(r['reason']=='missing_label_column' for r in meta['records'][1:])
