"""Prepare versioned task views and prove frozen feature reuse on every observation."""
import json
import numpy as np
from repair_io import HERE,OLD,ASSETS,PREP,REGISTRY,sha,save_json,save_csv,manifest
from export_frozen import exclusive_sample
from chronaris.evaluation.application_tasks.public_readout_revision import prepare_readout_revision,load_readout_revision,event_endpoint_s
from chronaris.evaluation.application_tasks.v4_public_data import load_prepared_public_development


def main():
    checks=[];reuse=[]
    for domain in ('clare','cogpilot'):
        root=ASSETS/domain
        if not root.exists():
            prepare_readout_revision(domain,source_root=PREP,registry_path=REGISTRY,output_root=ASSETS)
        revised=load_readout_revision(root)
        original=load_prepared_public_development(domain,output_root=PREP,registry_path=REGISTRY)
        rows=manifest(domain)
        for row in rows:
            sid=row['sample_id'];a=original.dataset.load_sample(sid);b=revised.dataset.load_sample(sid)
            reference=exclusive_sample(a,row) if domain=='cogpilot' else a
            assert b.source_sample_hash==reference.source_sample_hash
            for stream in ('physiology','vehicle'):
                for field in ('timestamps_s','values','feature_mask'):
                    key=stream+'_'+field
                    assert np.array_equal(getattr(b,key),getattr(reference,key)),(sid,key)
                if row['task_valid_mask'].get('event_response'):
                    assert np.all(getattr(b,stream+'_timestamps_s')<event_endpoint_s(row))
            checks.append(dict(domain=domain,sample_id=sid,original_hash=a.source_sample_hash,revised_hash=b.source_sample_hash,
                exact_preceding_view=True,physiology_points_before=len(a.physiology_timestamps_s),
                physiology_points_after=len(b.physiology_timestamps_s),vehicle_points_before=len(a.vehicle_timestamps_s),
                vehicle_points_after=len(b.vehicle_timestamps_s)))
        proof=json.loads((OLD/('export_boundary_verification.json' if domain=='cogpilot' else 'export_verification.json')).read_text())
        for r in proof['records']:
            if r['domain']!=domain:continue
            assert sha(r['output'])==r['output_sha256'] and sha(r['checkpoint'])==r['checkpoint_sha256']
            reuse.append(r|{'revision_manifest_sha256':sha(root/'revision_manifest.json')})
        save_json(HERE/f'{domain}_revision_manifest.json',json.loads((root/'revision_manifest.json').read_text()))
        print('prepared and checked',domain,len(rows),flush=True)
    save_csv(HERE/'observation_checks.csv',checks)
    save_json(HERE/'preparation.json',dict(status='passed',observations=len(checks),reused_representations=reuse,
        encoder_updates=0,new_encoder_exports=0,exact_preceding_view_all_windows=True))


if __name__=='__main__':main()
