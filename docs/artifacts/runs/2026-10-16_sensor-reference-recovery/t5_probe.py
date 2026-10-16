import sys,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--layers',required=True);a=p.parse_args()
sys.path.insert(0,'/home/wangminan/projects/chronaris-stage3-upstream/SensorLLM')
import torch
from sensorllm.model.chronos_model import ChronosPipeline
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.evaluation.application_tasks.chapter5_assets import write
root=Path(a.out);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
with development_gpu_lock() as acquired:
 assert acquired
 model=ChronosPipeline.from_pretrained('/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/weights/chronos_t5_large',device_map='cuda',tc='StanNormalizeUniformBins',torch_dtype=torch.bfloat16,local_files_only=True).model
 model.eval().requires_grad_(False);z=torch.load(a.layers,map_location='cpu',weights_only=True)
 tensors={};handles=[]
 def hook(name):
  def capture(m,args,out):
   v=out[0] if isinstance(out,tuple) else out
   if isinstance(v,torch.Tensor):tensors[name]=v.detach().cpu().clone()
  return capture
 for name,mod in model.model.encoder.named_modules():
  if name and ('block.0.' in name or name.startswith('embed_tokens')):handles.append(mod.register_forward_hook(hook(name)))
 with torch.inference_mode():result=model.encode(input_ids=z['ts_ids'].cuda(),attention_mask=z['ts_mask'].cuda())
 tensors['result']=result.cpu();torch.save(tensors,root/'t5_layers.pt')
 print('against reference',float((result.cpu()-z['pt_encoder_backbone.model.encoder']).abs().max()),flush=True)
 for h in handles:h.remove()
 for size in (1,8,16):
  with torch.inference_mode():
   if size==1:out=torch.cat([model.encode(input_ids=z['ts_ids'][i:i+1].cuda(),attention_mask=z['ts_mask'][i:i+1].cuda()) for i in range(6)])
   else:
    ids=torch.cat((z['ts_ids'],z['ts_ids'][:1].expand(size-6,-1))).cuda();mask=torch.cat((z['ts_mask'],z['ts_mask'][:1].expand(size-6,-1))).cuda();out=model.encode(input_ids=ids,attention_mask=mask)[:6]
  torch.save(out.cpu(),root/f'batch{size}.pt');print('batch',size,'reference delta',float((out.cpu()-z['pt_encoder_backbone.model.encoder']).abs().max()),flush=True)
