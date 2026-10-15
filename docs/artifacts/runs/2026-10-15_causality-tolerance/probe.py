from dataclasses import replace,fields
from pathlib import Path
import json,torch
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_correctness import remove_future_observations,audit_checkpoint_causality
from chronaris.evaluation.application_tasks.deep_baseline_adapter_audit import perturb_future_observations
from chronaris.modeling.training import load_common_pretraining_checkpoint
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.modeling.fusion_encoders.alignment_bridge import build_alignment_batch_from_observations

torch.set_num_threads(1)
root=Path('/mnt/e/chronaris-v4-results/2026-10-12-device-validation/resume/resumed/simulation/chronaris/no_continuous_evolution/fold01/seed17')
plan=json.loads(Path('/mnt/e/chronaris-v4-results/2026-10-11-chapter5-reload/chapter5_plan.json').read_text())
provider,schema,fold,*_=load_development_inputs('simulation','artifacts/application_evaluation/2026-09-06_v4-public-development','docs/requirements/thesis-v4-public-subjects.json',simulation_root=plan['simulation_root'])
raw=provider(fold.validation_sample_ids[:4]);cutoff=float(raw.query_timestamps_s[:,-1].min())/2
state=json.loads((root/'run_state.json').read_text());results=[]
for route in ('self_supervised','task_guided'):
 checkpoint=state[route+'_training']['best_checkpoint_path'];stored=torch.load(checkpoint,map_location='cpu',weights_only=True)
 print('ORIGINAL',route,audit_checkpoint_causality(checkpoint,raw,cutoff_s=cutoff,allow_legacy_implementation=False),flush=True)
 for device,dtype,graph in [('cuda',torch.float32,True),('cpu',torch.float32,False),('cuda',torch.float32,False),('cuda',torch.float64,False),('cpu',torch.float64,False)]:
  encoder,_,normalizer,_=load_common_pretraining_checkpoint(stored['source_checkpoint_path'] if route=='task_guided' else checkpoint,device=device,allow_legacy_implementation=False)
  if route=='task_guided':encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in stored['model_state_dict'].items() if k.startswith('encoder.')})
  encoder.eval().to(dtype=dtype)
  from chronaris.models.alignment.prototype import SingleStreamODERNNPrototype
  for m in encoder.modules():
   if isinstance(m,SingleStreamODERNNPrototype):m.config=replace(m.config,cuda_graph_recurrence=graph)
  def encode(b):
   b=move_observation_batch(normalizer.transform(b),device=device)
   b=replace(b,**{f.name:getattr(b,f.name).to(dtype=dtype) for f in fields(b) if isinstance(getattr(b,f.name),torch.Tensor) and getattr(b,f.name).is_floating_point()})
   aligned=build_alignment_batch_from_observations(b,physiology_feature_names=schema.physiology_feature_names,vehicle_feature_names=schema.vehicle_feature_names)
   observations={};hooks=[]
   for name in ('physiology','vehicle'):
    stream=getattr(aligned,name);mask=stream.mask & (stream.offsets_s<=cutoff)
    observations['input_'+name]=stream.values[mask].detach().clone()
    observations['mask_'+name]=stream.feature_valid_mask[mask].to(dtype=dtype)
    module=getattr(encoder.backbone.continuous_backbone,name+'_stream').encoder
    def capture(module,args,out,name=name,mask=mask):observations['embedding_'+name]=out[mask].detach().clone()
    hooks.append(module.register_forward_hook(capture))
   with torch.inference_mode(): z=encoder(b)
   for h in hooks:h.remove()
   a=z.auxiliary['alignment_output']
   return dict(sequence=z.sequence_embedding,physiology=a.physiology.reference_hidden_states,vehicle=a.vehicle.reference_hidden_states,fusion=z.auxiliary['fusion_output'].sequence_embedding,**observations),z.modality_available_mask
  baseline,mask=encode(raw);past=(raw.query_timestamps_s<=cutoff).to(device)
  for name,b in [('repeat',raw),('future_values',perturb_future_observations(raw,cutoff_s=cutoff)),('future_missing',remove_future_observations(raw,cutoff_s=cutoff))]:
   changed,m=encode(b)
   delta={k:float(((v-changed[k]) if k.startswith(('input_','mask_','embedding_')) else (v[past]-changed[k][past])).abs().max()) for k,v in baseline.items()}
   result=dict(route=route,device=device,dtype=str(dtype),graph=graph,alteration=name,deltas=delta,mask_equal=torch.equal(mask[past],m[past]))
   results.append(result);print(result,flush=True)
  if dtype==torch.float64 and device=='cuda':
   fusion=encoder.backbone.causal_fusion;fusion.config=replace(fusion.config,use_causal_mask=False)
   a,_=encode(raw);b,_=encode(perturb_future_observations(raw,cutoff_s=cutoff))
   result=dict(route=route,negative_control='causal_mask_disabled',device=device,dtype=str(dtype),delta=float((a['sequence'][past]-b['sequence'][past]).abs().max()))
   results.append(result);print(result,flush=True)
  del encoder
 Path('/mnt/e/chronaris-v4-results/2026-10-15-causality-diagnosis/precision.json').write_text(json.dumps(results,indent=2)+'\n')
