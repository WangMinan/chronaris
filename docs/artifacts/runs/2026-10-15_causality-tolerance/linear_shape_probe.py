from dataclasses import replace
from pathlib import Path
import json,torch
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations
from chronaris.modeling.training import load_common_pretraining_checkpoint
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.fusion_encoders.alignment_bridge import build_alignment_batch_from_observations

torch.set_num_threads(1)
r=Path('/mnt/e/chronaris-v4-results/2026-10-12-device-validation/resume/resumed/simulation/chronaris/no_continuous_evolution/fold01/seed17')
p=json.loads(Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/chapter5_plan.json').read_text())
provider,schema,fold,*_=load_development_inputs('simulation','artifacts/application_evaluation/2026-09-06_v4-public-development','docs/requirements/thesis-v4-public-subjects.json',simulation_root=p['simulation_root'])
raw=provider(fold.validation_sample_ids[:4]);cutoff=14.84375
checkpoint=json.loads((r/'run_state.json').read_text())['self_supervised_training']['best_checkpoint_path']
e,_,n,_=load_common_pretraining_checkpoint(checkpoint,device='cuda',allow_legacy_implementation=False);e.eval()
batches=[build_alignment_batch_from_observations(move_observation_batch(n.transform(b),device='cuda'),physiology_feature_names=schema.physiology_feature_names,vehicle_feature_names=schema.vehicle_feature_names) for b in (raw,remove_future_observations(raw,cutoff_s=cutoff))]
results=[]
with torch.inference_mode():
 for name in ('physiology','vehicle'):
  streams=[getattr(b,name) for b in batches];masks=[s.mask & (s.offsets_s<=cutoff) for s in streams]
  values=[torch.cat((s.values,s.feature_valid_mask.to(s.values.dtype)),dim=-1).flatten(0,1) for s in streams]
  row=dict(stream=name,shapes=[list(v.shape) for v in values],layers=[])
  for i,layer in enumerate(getattr(e.backbone.continuous_backbone,name+'_stream').encoder.network):
   values=[layer(v) for v in values]
   history=[v[m.flatten()] for v,m in zip(values,masks)]
   row['layers'].append(dict(index=i,type=type(layer).__name__,historical_max_delta=float((history[0]-history[1]).abs().max())))
  results.append(row)
print(results)
Path('/mnt/e/chronaris-v4-results/2026-10-15-causality-diagnosis/linear-shape.json').write_text(json.dumps(results,indent=2)+'\n')
