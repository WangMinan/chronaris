from pathlib import Path
import hashlib,json
import torch
from chronaris.evaluation.application_tasks.common_downstream_smoke import contract_development_inputs
from chronaris.evaluation.application_tasks.checkpoint_selection import selection_split
from chronaris.evaluation.application_tasks.stage45b_diagnostics import checkpoint_gradients
from chronaris.evaluation.application_tasks.v4_pipeline_steps import write_result
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256

torch.set_num_threads(1)
root=Path('/mnt/e/chronaris-v4-results/2026-09-22-stage45c-analysis');parent=Path('/mnt/e/chronaris-v4-results/2026-09-21-stage45b')
records=[]
for domain,indices in [('clare',(0,1)),('cogpilot',(8,))]:
 p,_,fold,_,targets,definitions,context,_=contract_development_inputs(domain,data_root='artifacts/application_evaluation/2026-09-06_v4-public-development',registry_path='docs/requirements/thesis-v4-public-subjects.json',full=True)
 index={s:i for i,s in enumerate(targets.sample_ids)}
 for unit in indices:
  result=json.loads((parent/'screen'/f'{unit}.json').read_text()); training,_,_=selection_split(fold,targets,definitions,context,inner_index=result['unit']['inner_index'])
  batches=[]
  for group in sorted({context['groups'][s] for s in training.train_sample_ids}):
   for task in definitions:
    eligible=sorted((s for s in training.train_sample_ids if context['groups'][s]==group and bool(targets.valid_masks[task.name][index[s]].any())),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    if len(eligible)<8:raise ValueError('insufficient task-supported training windows')
    for offset in (0,4):
     batch=tuple(eligible[offset:offset+4])
     if batch not in batches:batches.append(batch)
  checkpoint=next(r['training']['best_checkpoint_path'] for r in result['results'] if r.get('route')=='task_guided' and 'training' in r)
  write_result(root/'progress.json',dict(unit=unit,batch_count=len(batches),status='running'))
  measured=checkpoint_gradients(checkpoint=checkpoint,route='task_guided',provider=p,fold=training,targets=targets,definitions=definitions,output_root={'path':str(root/str(unit)),'context':context},sample_batches=batches)
  records.append(dict(unit=unit,domain=domain,result=measured))
  print('COMPLETED',unit,len(batches),flush=True)
write_result(root/'summary.json',dict(status='completed',source_code_sha256=v4_workflow_source_sha256(),records=records,optimizer_updates=0))
