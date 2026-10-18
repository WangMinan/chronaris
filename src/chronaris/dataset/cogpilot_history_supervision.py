"""Resolve ambiguous zero historical means before any auxiliary supervision."""
from pathlib import Path
import numpy as np
import pandas as pd
from chronaris.dataset.cogpilot_performance import file_digest,interval_mean
from chronaris.dataset.cogpilot_history_features import QUERY_SECONDS


def audit_zero_anchors(request):
    grouped={};results=[];sources={}
    for row in request['rows']:grouped.setdefault(row['record_id'],[]).append(row)
    for record,rows in grouped.items():
        path=next(Path(record).glob('*stream-lslxp11_feat-perfmetric*.csv'))
        sha=file_digest(path);assert sha==request['sources'][str(path)]
        frame=pd.read_csv(path)
        times=(frame.time_dn.to_numpy(float)-rows[0]['source_origin_native'])*86400
        values=frame.total_error.to_numpy(float)
        for row in rows:
            query=row['cutoff_s']-30+QUERY_SECONDS[row['query_index']]
            valid=True;reason='true_zero'
            try:
                value,_,_=interval_mean(times,values,query-5,query,include_end=False)
                assert abs(value)<1e-10
            except ValueError as exc:
                valid=False;reason=str(exc)
            results.append(dict(sample_id=row['sample_id'],sample_index=row['sample_index'],
                                query_index=row['query_index'],valid=valid,reason=reason))
        assert file_digest(path)==sha;sources[str(path)]=sha
    return dict(status='passed',checks=results,sources=sources)
