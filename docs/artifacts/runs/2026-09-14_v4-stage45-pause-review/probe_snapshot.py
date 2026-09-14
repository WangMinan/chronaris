import json
from pathlib import Path
from chronaris.evaluation.application_tasks.stage45 import _inputs
from chronaris.evaluation.application_tasks.stage45_diagnostics import run_branch_probes
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
root=Path('/mnt/e/chronaris-v4-results/2026-09-14-stage45-pause-review')
config=json.loads(Path('/mnt/e/chronaris-v4-results/2026-09-14-stage45/pipeline_config.json').read_text())
provider,_,fold,_,targets,definitions,context,_=_inputs(config,'clare')
with development_gpu_lock() as acquired:
 assert acquired
 r=run_branch_probes(checkpoint=root/'preserved/units/screen/0/clare/chronaris/self_supervised/chronaris/C/update_000050.pt',route='self_supervised',provider=provider,fold=fold,targets=targets,definitions=definitions,context=context,output_root=root/'snapshot50',allow_diagnostic_snapshot=True)
 print(json.dumps({k:v['components']['linear']['task_summary']['validation'] for k,v in r['branches'].items()},indent=2))
