import importlib.util,sys
from pathlib import Path
sys.path.insert(0,'/home/wangminan/projects/chronaris-runtime-local-validation-20261012/src')
import torch
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
root=Path('/mnt/e/chronaris-v4-results/2026-10-16-chapter5-chronos-recovery')
source=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('repaired_diagnostics',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
models=read(root/'delivery_models.json');models['records']=[r for r in models['records'] if r['method']=='naive_time_sync']
assert len(models['records'])==1
torch.set_num_threads(1)
result=module.diagnostics(Path(__file__).parent/'naive_preflight',read(root/'delivery_plan.json'),models)
assert result['status']=='completed' and len(result['timings'])==10 and all(t['device']=='cpu' and t['max_allocated_bytes']==0 for t in result['timings'])
write(Path(__file__).parent/'preflight.json',dict(status='completed',source_sha256=sha(source),timing_count=len(result['timings']),optimizer_updates=0,confirmation_opened=False,model_freeze_sha256=sha(root/'delivery_models.json'),result=result))
print('real fixed-synchronization warmup and ten repeats passed',flush=True)
