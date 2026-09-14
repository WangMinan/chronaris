import json
from pathlib import Path
import torch
from chronaris.evaluation.application_tasks.stage45 import make_plan, _run_unit
from chronaris.evaluation.application_tasks.checkpoint_performance import compare_values
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
from chronaris.modeling.training import candidate_screen
root=Path('/mnt/e/chronaris-v4-results/2026-09-14-stage45-pause-review')
config=json.loads(Path('/mnt/e/chronaris-v4-results/2026-09-14-stage45/pipeline_config.json').read_text())
config.update(root=str(root/'entry_validation'),stage45_resume_parent='/mnt/e/chronaris-v4-results/2026-09-14-stage45',stage45_resume_evidence=str(root/'recovery_evidence.json'),stage45_budget_hours=None,source_code_sha256=v4_workflow_source_sha256())
make_plan(config)
original=candidate_screen.pretext_micro_step
count=0
class DiagnosticStop(Exception):pass
checks={}
def checked(**kw):
 global count
 count+=1
 if count==1:
  assert kw['optimizer_updates']==76 and kw['epoch']==301
 if count==9:
  expected=torch.load(root/'graph/state_77.pt',map_location='cpu',weights_only=True)
  for key,module in [('encoder_state_dict',kw['encoder']),('head_state_dict',kw['heads']),('explicit_time_shift_head_state_dict',kw['shift_head'])]:
   checks[key]=compare_values(expected[key],{k:v.cpu() for k,v in module.state_dict().items()},atol=0,rtol=0)
  assert all(r['close'] for r in checks.values()), checks
  assert kw['optimizer_updates']==78 and kw['epoch']==309
  raise DiagnosticStop()
 return original(**kw)
candidate_screen.pretext_micro_step=checked
try:
 _run_unit(config,dict(domain='clare',method='chronaris',recipe='thesis_reference'),Path(config['root'])/'units/screen/0')
except DiagnosticStop:
 result=dict(status='completed',resumed_update=75,executed_updates=[76,77],exact_trial_state_checks=checks,original_budget=300,diagnostic_stop=True,confirmation_opened=False)
 (root/'entry_validation.json').write_text(json.dumps(result,indent=2))
 print(result)
else:raise AssertionError('bounded diagnostic did not stop')
