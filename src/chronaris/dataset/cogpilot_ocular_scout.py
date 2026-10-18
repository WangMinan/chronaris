"""Availability only: raw validity flags are not interpreted as field qualification."""
from pathlib import Path
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest


def scout_ocular_sources(records):
    rows=[];sources={}
    for record in records:
        files=list(Path(record['record_id']).glob('*stream-lslhtcviveeye*_dat.csv'))
        row=dict(record,files=len(files),has_time_series=False)
        if len(files)!=1:
            row['reason']='missing_or_ambiguous_file';rows.append(row);continue
        path=files[0];digest=file_digest(path);sources[str(path)]=digest
        columns=['time_dn','validity_l','validity_r','pupil_diameter_l_mm','pupil_diameter_r_mm','eye_openness_l','eye_openness_r']
        frame=pd.read_csv(path,usecols=columns);time=frame.time_dn.to_numpy(float)
        finite_time=time[np.isfinite(time)]
        row.update(path=str(path),sha256=digest,rows=len(frame),finite_times=int(np.isfinite(time).sum()),
                   minimum_time=float(finite_time.min()) if len(finite_time) else None,
                   maximum_time=float(finite_time.max()) if len(finite_time) else None,
                   repeated_time=int((np.diff(time)==0).sum()),reversed_time=int((np.diff(time)<0).sum()))
        for column in columns[1:]:
            values=frame[column].to_numpy(float);finite=np.isfinite(values)
            row[column+'_finite']=int(finite.sum());row[column+'_positive']=int((values[finite]>0).sum())
            if column.startswith('validity'):row[column+'_counts']={str(k):int(v) for k,v in frame[column].value_counts().items()}
        row['has_time_series']=bool(len(frame)>=50 and np.isfinite(time).all() and np.all(np.diff(time)>0))
        row['all_zero_single_row']=len(frame)==1 and bool((frame.to_numpy(float)==0).all())
        assert file_digest(path)==digest;rows.append(row)
    return dict(records=rows,sources=sources,encoder_updates=0,field_validity_decoded=False)
