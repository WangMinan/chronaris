"""Apply the preregistered complete-case participant rule, never a score-based filter."""
import json
from pathlib import Path
import numpy as np
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.expanded_fit_inputs import load_arrays,qualify_expanded_fit


def eligible_subjects(meta):
    subjects=sorted(c['subject'] for c in meta['coverage'] if c['qualified_records']>=8 and c['windows']>=100)
    records=[r for r in meta['records'] if r['subject'] in subjects]
    passed=len(subjects)>=12 and sum(r['accepted'] for r in records)>=.8*len(records)
    return subjects,passed


def qualify_eligible_fit(source,report,assets):
    report,assets=Path(report),Path(assets);assets.mkdir(parents=True,exist_ok=True)
    if (report/'cohort_decision.json').exists():raise FileExistsError('retained cohort decision')
    arrays=load_arrays(source);meta=json.loads(str(arrays['metadata']));subjects,passed=eligible_subjects(meta)
    decision=dict(status='passed' if passed else 'failed',source_path=str(Path(source).resolve()),source_sha256=file_digest(source),
                  included_subjects=subjects,excluded=[r for r in meta['coverage'] if r['subject'] not in subjects],
                  rule='all participants with >=8 records and >=100 windows; >=12 participants; >=80% cohort records')
    save_json(report/'cohort_decision.json',decision)
    if not passed:raise ValueError('eligible cohort qualification failed')
    ids=np.array([i for i,r in enumerate(meta['windows']) if r['subject'] in subjects])
    filtered={k:v[ids] for k,v in arrays.items() if k!='metadata'}
    updated=dict(meta,status='passed',windows=[meta['windows'][i] for i in ids],
                 coverage=[c for c in meta['coverage'] if c['subject'] in subjects],
                 records=[r for r in meta['records'] if r['subject'] in subjects],
                 prefix_checks=[c for c in meta['prefix_checks'] if c['sample_id'].split('::')[0] in subjects])
    accepted_records={r['record_id'] for r in updated['records'] if r['accepted']}
    updated['reconstruction']=[r for r in meta['reconstruction'] if r['record_id'] in accepted_records]
    filtered['metadata']=np.array(json.dumps(updated,allow_nan=False));output=assets/'additional.npz'
    if output.exists():raise FileExistsError(output)
    np.savez_compressed(output,**filtered)
    save_json(report/'additional_subjects.json',subjects)
    return qualify_expanded_fit(output,report,assets)
