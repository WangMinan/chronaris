import argparse,sys,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workspace',required=True);p.add_argument('--offload',action='store_true');a=p.parse_args();sys.path.insert(0,str(Path(a.workspace)/'src'))
import torch,numpy as np
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
from chronaris.evaluation.application_tasks.chapter5_native import unit_record,_inputs
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import historical_channels
from chronaris.modeling.fusion_encoders.sensorllm_adapter import SensorLLMWindowEncoder
from chronaris.modeling.training.rng import isolated_training_rng
from chronaris.representation import TrainOnlyRobustNormalizer
root=Path(a.out);root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
with development_gpu_lock() as acquired:
 assert acquired,'GPU busy'
 u=unit_record('/mnt/e/chronaris-v4-results/2026-10-02-stage5-performance-recovery',8);row=u['results'][0];e=row.get('evaluation',row)
 c=read(Path(e['consumers']['manifest_path']).parents[1]/'common_contract.json');cp=c['declaration']['checkpoint_path'];saved=torch.load(cp,map_location='cpu',weights_only=True)
 _,fold,targets,context,raw=_inputs(u['unit'],c['contract'],'validation',4)
 values,mask=historical_channels(TrainOnlyRobustNormalizer.from_manifest(saved['normalizer']).transform(raw))
 assets=read('/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets_sensor.json')
 with isolated_training_rng(17):enc=SensorLLMWindowEncoder(assets,len(saved['channel_names']),**({'inference_offload':True} if a.offload else {}))
 enc.channel_names=tuple(saved['channel_names']);enc.load_checkpoint_state(saved['encoder']);enc.eval().requires_grad_(False)
 tensors={'input_values':values,'input_mask':mask,'rope':enc.core.rotary_emb.inv_freq.detach().cpu()}
 x=values[0].T.masked_fill(~mask[0].T,torch.nan);ts_ids,ts_mask,_=enc.ts_tokenizer.context_input_transform(x);tensors.update(ts_ids=ts_ids,ts_mask=ts_mask,centers=enc.ts_tokenizer.centers)
 def hook(name):
  def capture(module,args,output):
   v=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0] if isinstance(output,tuple) else output
   tensors[name]=v.detach().cpu().clone()
  return capture
 handles=[]
 for name,mod in enc.core.named_modules():
  if name in ('embed_tokens','pt_encoder_backbone.model.encoder','ts_proj','norm') or (name.startswith('layers.') and name.count('.')==1):handles.append(mod.register_forward_hook(hook(name)))
 with torch.inference_mode():actual=enc(values[:1],mask[:1]).cpu()
 with np.load(Path(cp).parent/'validation_features.npz') as f:expected=torch.from_numpy(f['pooled_embedding'][:1].copy())
 tensors.update(actual=actual,expected=expected)
 torch.save(tensors,root/'layers.pt')
 records={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()) for k,v in tensors.items()}
 write(root/'summary.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,offload=a.offload,checkpoint_sha256=sha(cp),sample_ids=list(raw.sample_ids),max_delta=float((actual-expected).abs().max()),records=records))
 print('saved',root,'delta',float((actual-expected).abs().max()),flush=True)
