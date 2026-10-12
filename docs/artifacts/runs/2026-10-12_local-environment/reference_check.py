from dataclasses import replace
import json
from pathlib import Path
import joblib
import torch
from chronaris.representation import load_fusion_stream_batch
from chronaris.evaluation.application_tasks.chapter5_native import unit_record, _inputs, compare_predictions
from chronaris.evaluation.application_tasks.v4_grouped_consumers import evaluate_native_consumers

root=Path('/home/wangminan/chronaris-migration-20261012')
torch.set_num_threads(1)
unit=unit_record('/mnt/e/chronaris-v4-results/2026-10-02-stage5-performance-recovery',0)
row=unit['results'][0];evaluation=row.get('evaluation',row)
record=json.loads((Path(evaluation['consumers']['manifest_path']).parents[1]/'common_contract.json').read_text())
_,fold,targets,context,raw=_inputs(unit['unit'],record['contract'],'validation',4)
full=load_fusion_stream_batch(root/'reference-representation')
indices=[full.sample_ids.index(x) for x in raw.sample_ids]
ref=replace(full,sample_ids=raw.sample_ids,source_sample_hashes=tuple(full.source_sample_hashes[i] for i in indices),**{k:getattr(full,k)[indices] for k in ['timestamps_s','sequence_embedding','valid_mask','pooled_embedding']})
current=load_fusion_stream_batch(root/'replay-diagnostic/clare/self_supervised/validation')
result={'domain':'clare','route':'self_supervised','samples':4,'sequence_max_absolute_difference':(ref.sequence_embedding-current.sequence_embedding).abs().max().item(),'pooled_max_absolute_difference':(ref.pooled_embedding-current.pooled_embedding).abs().max().item(),'fixed_representation_downstream':{}}
for family,component in evaluation['consumers']['components'].items():
 bundle=joblib.load(component['model_path'])['consumer']
 actual=evaluate_native_consumers(bundle,output=ref,targets=targets,context=context)['prediction_rows']
 expected=json.loads(Path(component['result_path']).read_text())['evaluations']['validation']['prediction_rows']
 expected=[x for x in expected if x['sample_id'] in set(raw.sample_ids)]
 try: result['fixed_representation_downstream'][family]=compare_predictions(expected,actual)
 except ValueError as e: result['fixed_representation_downstream'][family]={'passed':False,'error':str(e)}
(root/'reference-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
