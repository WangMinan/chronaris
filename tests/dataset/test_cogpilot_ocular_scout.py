import pandas as pd
from chronaris.dataset.cogpilot_ocular_scout import scout_ocular_sources


def test_named_placeholder_and_header_only_streams_are_not_observations(tmp_path):
    columns=['time_dn','validity_l','validity_r','pupil_diameter_l_mm','pupil_diameter_r_mm','eye_openness_l','eye_openness_r']
    run=tmp_path/'record';run.mkdir();path=run/'stream-lslhtcviveeye_dat.csv'
    pd.DataFrame([[0]*len(columns)],columns=columns).to_csv(path,index=False)
    r=scout_ocular_sources([dict(record_id=str(run),subject='a')])
    assert r['records'][0]['all_zero_single_row'] and not r['records'][0]['has_time_series']
    pd.DataFrame(columns=columns).to_csv(path,index=False)
    r=scout_ocular_sources([dict(record_id=str(run),subject='a')])
    assert r['records'][0]['rows']==0 and r['records'][0]['minimum_time'] is None
    assert not r['records'][0]['has_time_series'] and not r['field_validity_decoded']
    import json
    frame=pd.DataFrame({c:[1.]*60 for c in columns});frame['time_dn']=[i/120 for i in range(60)]
    frame.to_csv(path,index=False);r=scout_ocular_sources([dict(record_id=str(run),subject='a')])
    assert r['records'][0]['has_time_series'];json.dumps(r,allow_nan=False)
