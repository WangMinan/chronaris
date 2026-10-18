"""Inspect event endpoints in the original date coordinate, without rewriting windows."""
import json

import numpy as np

from common import HERE, PREP, REGISTRY, ASSETS, manifest, save_csv, save_json
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development


def main():
    known={r['sample_id']:r for r in json.loads((ASSETS/'raw_response.json').read_text()) if 'event_native' in r}
    data=load_prepared_public_development('cogpilot',output_root=PREP,registry_path=REGISTRY)
    rows=[]
    for row in manifest('cogpilot'):
        if not row['task_valid_mask']['event_response']:continue
        start=row['record_time_origin_native']+row['context_start_s']/86400
        event=start+12/86400
        if row['sample_id'] in known:
            assert start==known[row['sample_id']]['start_native'] and event==known[row['sample_id']]['event_native']
        cutoff=(event-start)*86400
        sample=data.dataset.load_sample(row['sample_id'])
        eda_t=sample.physiology_timestamps_s[sample.physiology_feature_mask[:,1]]
        v_t=sample.vehicle_timestamps_s[sample.vehicle_feature_mask.any(1)]
        rows.append(dict(sample_id=row['sample_id'],actual_event_relative_s=cutoff,
            nominal_endpoint_rounding_s=12-cutoff,eda_at_event=int(np.sum(eda_t==cutoff)),
            eda_after_event=int(np.sum(eda_t>cutoff)),vehicle_at_event=int(np.sum(v_t==cutoff)),
            vehicle_after_event=int(np.sum(v_t>cutoff))))
    save_csv(HERE/'event_boundaries.csv',rows)
    save_json(HERE/'event_boundary_summary.json',dict(windows=len(rows),
        windows_with_eda_at_event=sum(r['eda_at_event']>0 for r in rows),
        windows_with_eda_after_event=sum(r['eda_after_event']>0 for r in rows),
        windows_with_vehicle_at_event=sum(r['vehicle_at_event']>0 for r in rows),
        windows_with_vehicle_after_event=sum(r['vehicle_after_event']>0 for r in rows),
        raw_event_reconstruction_checks=len(known),maximum_rounding_s=max(r['nominal_endpoint_rounding_s'] for r in rows),
        source_windows_changed=False))
    print(json.dumps(json.loads((HERE/'event_boundary_summary.json').read_text())))


if __name__=='__main__':main()
