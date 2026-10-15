from pathlib import Path
import json,math,torch
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_correctness import audit_checkpoint_causality

torch.set_num_threads(1)
r=Path('/mnt/e/chronaris-v4-results/2026-10-12-device-validation/resume/resumed/simulation/chronaris/no_continuous_evolution/fold01/seed17')
plan=json.loads(Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/chapter5_plan.json').read_text())
provider,_,fold,*_=load_development_inputs('simulation','artifacts/application_evaluation/2026-09-06_v4-public-development','docs/requirements/thesis-v4-public-subjects.json',simulation_root=plan['simulation_root'])
state=json.loads((r/'run_state.json').read_text());ids=fold.validation_sample_ids[::8][:32];assert len(ids)==32
rows=[]
for start in range(0,len(ids),4):
 batch=provider(ids[start:start+4])
 for route in ('self_supervised','task_guided'):
  path=state[route+'_training']['best_checkpoint_path']
  for cutoff in (7.5,14.84375,22.5):
   result=audit_checkpoint_causality(path,batch,cutoff_s=cutoff,allow_legacy_implementation=False)
   result.update(route=route,revised_atol=1e-5,revised_passed=all(math.isfinite(v) and v<=1e-5 for v in result['historical_max_delta'].values()))
   rows.append(result);print(len(rows),route,cutoff,result['historical_max_delta'],flush=True)
   Path('/mnt/e/chronaris-v4-results/2026-10-15-causality-diagnosis/sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
assert len(rows)==48 and all(r['revised_passed'] for r in rows)
