import importlib.util,sys
from pathlib import Path
sys.path.insert(0,'/home/wangminan/projects/chronaris-runtime-local-validation-20261012/src')
import torch
from chronaris.evaluation.application_tasks.chapter5_assets import read,write,sha,verify
from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import simulation_records
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets,_require_development_checkpoint
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY
source=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('repaired_diagnostics',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path('/mnt/e/chronaris-v4-results/2026-10-16-chapter5-chronos-recovery');plan=read(root/'delivery_plan.json');torch.set_num_threads(1)
models=simulation_records(plan['subset_root']);paths=m.reference_initializations(models['records'],17)
data,fold=load_v4_simulation_development(simulation_root=plan['simulation_root'],registry_path=SIMULATION_REGISTRY)
assert not set(data.batch.sample_ids)&set(fold.held_out_sample_ids)
targets=build_guarded_application_consumer_targets(data,completed_pretraining_checkpoints=paths,smoke_only=False)
selected=[]
for name,weight in [('no_physics_residual',0.),('reference',.05)]:
 r=next(r for r in models['records'] if r['candidate_name']==name and r['method']=='chronaris');selected.append((weight,r['checkpoints']))
for index,weight in [(30,.025),(33,.1)]:
 result=read(root/'extra_units'/f'{index}.json');verify(result['files'])
 selected.append((weight,{r:result['training'][r+'_training']['best_checkpoint_path'] for r in ('self_supervised','task_guided')}))
checks=[]
for weight,checkpoints in selected:
 _require_development_checkpoint([checkpoints['self_supervised']],data)
 for route,checkpoint in checkpoints.items():
  encoder,normalizer,saved=load_frozen_application_encoder(checkpoint,route=route,fold=fold,device='cpu')
  assert saved['training_status']=='completed'
  del encoder,normalizer,saved
  checks.append(dict(weight=weight,route=route,checkpoint=checkpoint,sha256=sha(checkpoint)))
record=dict(status='completed',diagnostics_sha256=sha(source),source_path=str(source),target_samples=len(targets.sample_ids),roles={k:len(v) for k,v in data.role_sample_ids.items()},reference_checkpoints={p:sha(p) for p in paths},sensitivity_checkpoints=checks,thresholds=targets.manifest['workload_thresholds_train_only'],confirmation_opened=False,optimizer_updates=0)
write(Path(__file__).parent/'preflight.json',record);print({k:v for k,v in record.items() if k not in ('reference_checkpoints','sensitivity_checkpoints')},flush=True)
