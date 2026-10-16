import sys,json
from pathlib import Path
sys.path.insert(0,'/home/wangminan/projects/chronaris-runtime-local-validation-20261012/src')
import torch,numpy as np,joblib
from chronaris.evaluation.application_tasks.chapter5_assets import read,write,sha
from chronaris.evaluation.application_tasks.chapter5_native import unit_record,_inputs,_encode
from chronaris.evaluation.application_tasks.device_replay import prediction_changes
from chronaris.evaluation.application_tasks.v4_grouped_consumers import evaluate_native_consumers
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import historical_channels
from chronaris.representation import TrainOnlyRobustNormalizer
from chronaris.representation.window_features import WindowFeatureBatch
root=Path(__file__).parent;torch.set_num_threads(1)
u=unit_record('/mnt/e/chronaris-v4-results/2026-10-02-stage5-performance-recovery',262)
report=[];jobs=[]
with development_gpu_lock() as acquired:
 assert acquired,'GPU busy'
 for row in u['results']:
  e=row.get('evaluation',row);record=read(Path(e['consumers']['manifest_path']).parents[1]/'common_contract.json');c,d=record['contract'],record['declaration']
  _,fold,targets,context,raw=_inputs(u['unit'],c,'validation',0)
  saved=torch.load(d['checkpoint_path'],map_location='cpu',weights_only=True)
  values,mask=historical_channels(TrainOnlyRobustNormalizer.from_manifest(saved['normalizer']).transform(raw))
  actual=_encode(u['unit'],row,d,c,raw,fold,root/row['route'],'validation')
  with np.load(root/(row['route']+'_original.npz')) as f:
   expected=torch.from_numpy(f['pooled_embedding'].copy())
   assert tuple(f['sample_ids'])==raw.sample_ids
  control=WindowFeatureBatch(raw.sample_ids,raw.context_durations_s,expected,mask.any((1,2)),u['unit']['method'],fold.fold_id,sha(d['checkpoint_path']),raw.source_sample_hashes)
  component=e['consumers']['components']['linear'];bundle=joblib.load(component['model_path'])['consumer'];before=read(component['result_path'])['evaluations']['validation']['prediction_rows']
  delta=prediction_changes(before,evaluate_native_consumers(bundle,output=control,targets=targets,context=context)['prediction_rows'])
  torch.save(dict(values=values,mask=mask,expected=expected,actual=actual.pooled_embedding),root/(row['route']+'.pt'))
  report.append(dict(route=row['route'],feature_max_delta=float((actual.pooled_embedding-expected).abs().max()),saved_feature_control=delta))
  jobs.append(dict(route=row['route'],checkpoint=d['checkpoint_path'],checkpoint_sha256=sha(d['checkpoint_path'])))
write(root/'local_control.json',report);write(root/'jobs.json',jobs);print(json.dumps(report,indent=2))
