import io,json,shutil,sys,time
from pathlib import Path
import torch
from chronaris.evaluation.application_tasks.development_comparison import SENSOR_ASSETS
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder,alignment_support
from chronaris.modeling.training.rng import isolated_training_rng,canonical_training_state_sha256
from chronaris.evaluation.application_tasks.v4_development_data import v4_workflow_source_sha256
mode=sys.argv[1];root=Path('/mnt/e/chronaris-v4-results/2026-09-30-stage5-sensor-probes');bundle=Path('/home/wangminan/projects/chronaris/docs/artifacts/runs/2026-09-30_v4-stage5-sensor-recovery')
data=torch.load(root/'inputs.pt',map_location='cpu',weights_only=True);indices=[431,432,433,1454]
x,m=data['inputs']['train'];x=x[indices];m=m[indices];ids=tuple(data['train_ids'][i] for i in indices)
source=v4_workflow_source_sha256();binding=dict(scope='real_training_windows_repair_probe_no_outer',source_code_sha256=source,sample_ids=list(ids))
eligible,_,support=alignment_support(x,m,data['names'],ids);assert eligible==[0,2]
with development_gpu_lock() as acquired:
 assert acquired
 torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.95)
 with isolated_training_rng(17):
  assets=verify_assets(SENSOR_ASSETS);print('assets verified; loading encoder',flush=True)
  encoder=SensorLLMWindowEncoder(assets,x.shape[-1]);encoder.channel_names=data['names']
  arguments=dict(values=x,mask=m,names=data['names'],train_ids=ids,binding=binding)
  if mode=='primary':
   encoder.align_history(**arguments,root=root/'primary',stop_after=1)
   (root/'resume').mkdir(exist_ok=True);shutil.copyfile(root/'primary/alignment.pt',root/'resume/alignment.pt')
   receipt=encoder.align_history(**arguments,root=root/'primary',stop_after=4)
  else:receipt=encoder.align_history(**arguments,root=root/'resume',stop_after=4)
  expected=canonical_training_state_sha256(encoder.checkpoint_state())
  assert [r['sample_id'] for r in receipt['updates']]==[ids[0],ids[2],ids[0],ids[2]]
  encoder.eval()
  with torch.no_grad():
   zero=encoder(x[1:2],m[1:2]);assert torch.isfinite(zero).all() and torch.count_nonzero(zero)==0
   feature=encoder(x[:1],m[:1]);assert torch.isfinite(feature).all() and torch.count_nonzero(feature)>0
  # The unchanged classification route must still train on the empty window.
  encoder.train();head=torch.nn.Linear(encoder.feature_dim,2).cuda();loss=torch.nn.functional.cross_entropy(head(encoder(x[1:2],m[1:2])),torch.tensor([0],device='cuda'));loss.backward()
  assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
  result=dict(status='completed',mode=mode,source_code_sha256=source,updates=receipt['updates'],support=receipt['support'],state_sha256=expected,zero_observation_classification_backward_finite=True,zero_observation_features_zero=True,observed_features_sha256=canonical_training_state_sha256(feature.cpu()),pid=__import__('os').getpid())
  if mode=='replay':
   prior=json.loads((bundle/'probe_primary.json').read_text())
   assert prior['pid']!=result['pid']
   for key in ('updates','support','state_sha256','observed_features_sha256'):assert prior[key]==result[key],key
   result['independent_process_resume_identical']=True
  assert source==v4_workflow_source_sha256()
  (bundle/f'probe_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
  print('PASS',mode,'alignment, real missing windows, classification backward, feature reload',flush=True)
