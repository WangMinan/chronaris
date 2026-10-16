import sys,json,gc
from pathlib import Path
sys.path.insert(0,'/home/wangminan/projects/chronaris-runtime-chapter5-reload-20261011/src')
import torch
from chronaris.evaluation.application_tasks.chapter5_assets import read,write,sha
from chronaris.evaluation.application_tasks.recent_model_smoke import _export
from chronaris.evaluation.application_tasks.recent_model_training import load_encoder_checkpoint
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import Chronos2WindowEncoder,verify_assets
from chronaris.modeling.training.rng import isolated_training_rng
root=Path(sys.argv[1]);torch.set_num_threads(1);records=[]
with development_gpu_lock() as acquired:
 assert acquired,'GPU busy'
 assets=verify_assets('/mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets.json')
 for job in read(root/'jobs.json'):
  assert sha(job['checkpoint'])==job['checkpoint_sha256']
  saved=torch.load(job['checkpoint'],map_location='cpu',weights_only=True)
  data=torch.load(root/(job['route']+'.pt'),weights_only=True)
  with isolated_training_rng(saved['seed']):enc=Chronos2WindowEncoder(assets,len(saved['channel_names']))
  load_encoder_checkpoint(enc,saved['encoder'])
  out,cost=_export(enc,{'validation':(data['values'],data['mask'])},{'validation':None});actual=out['validation']
  torch.save(actual,root/(job['route']+'_remote.pt'))
  records.append(dict(route=job['route'],original_delta=float((actual-data['expected']).abs().max()),local_delta=float((actual-data['actual']).abs().max()),cost=cost,device=torch.cuda.get_device_name(),torch=torch.__version__))
  del enc,saved;gc.collect();torch.cuda.empty_cache()
write(root/'remote_control.json',records);print(json.dumps(records,indent=2))
