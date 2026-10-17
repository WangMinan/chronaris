import importlib.util,sys,time,gc
from pathlib import Path
sys.path.insert(0,'/home/wangminan/projects/chronaris-runtime-local-validation-20261012/src')
import torch
from chronaris.evaluation.application_tasks.chapter5_assets import read,sha,write
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development
from chronaris.evaluation.application_tasks.v4_simulation_confirmation_data import load_simulation_confirmation
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY
from chronaris.simulation.aviation_dual_stream.config import locked_stress_observation_scenarios
source=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('repaired_simulation',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path('/mnt/e/chronaris-v4-results/2026-10-16-chapter5-chronos-recovery');out=Path(__file__).parent
plan=read(root/'delivery_plan.json');models=read(root/'delivery_models.json');torch.set_num_threads(1)
data,fold=load_v4_simulation_development(simulation_root=plan['simulation_root'],registry_path=SIMULATION_REGISTRY)
assert all(unit['fold']==fold.to_dict() for unit in models['records'])
conditions=['clean_asynchronous']+[s.scenario_id for s in locked_stress_observation_scenarios()]+['physiology_missing','vehicle_missing']
assert len(conditions)==len(set(conditions))==38
rows=[];started=time.time();freeze_sha=sha(root/'delivery_models.json')
for condition in conditions:
 held=load_simulation_confirmation(root/'confirmation',model_freeze_sha256=freeze_sha,condition=condition)
 module._require_confirmation_contract(held,fold,data.schema)
 assert len(held.batch.sample_ids)==512
 rows.append(dict(condition=condition,samples=len(held.batch.sample_ids),exact_order=True,schema_sha256=held.schema.schema_sha256,source_manifest_sha256=held.schema.source_manifest_sha256,source_sample_hash_count=len(held.batch.source_sample_hashes)))
 write(out/'progress.json',dict(completed=len(rows),total=38,last_condition=condition,seconds=time.time()-started))
 print('qualified',len(rows),condition,flush=True)
 del held;gc.collect()
assert sha(root/'delivery_models.json')==freeze_sha
write(out/'preflight.json',dict(status='completed',source_sha256=sha(source),model_freeze_sha256=freeze_sha,development_schema_sha256=data.schema.schema_sha256,development_source_manifest_sha256=data.schema.source_manifest_sha256,conditions=rows,optimizer_updates=0,model_predictions_generated=False,seconds=time.time()-started))
