"""Development-only sensitivity, causality and timed inference for the frozen delivery."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import time
import torch

from chronaris.evaluation.application_tasks.chapter5_assets import read, sha, verify, write
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.evaluation.application_tasks.v4_simulation_data import load_v4_simulation_development
from chronaris.evaluation.application_tasks.v4_simulation_confirmation import SIMULATION_REGISTRY
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder, export_loaded_application_encoder
from chronaris.representation import select_observation_batch


def reference_initializations(records, seed):
    methods={'chronaris','physiology_only','vehicle_only','mult','contiformer'}
    selected=[r for r in records if r['seed']==seed and r['method'] in methods and r['candidate_name']=='reference']
    if len(selected)!=5 or {r['method'] for r in selected}!=methods:
        raise ValueError('target access requires all five distinct frozen reference methods')
    return [r['checkpoints']['self_supervised'] for r in selected]


def sensitivity(root, plan):
    from chronaris.evaluation.application_tasks.application_consumer_smoke_data import build_guarded_application_consumer_targets
    from chronaris.evaluation.application_tasks.application_consumer_runtime import ApplicationConsumerProtocol, run_application_method_consumers
    from chronaris.evaluation.application_tasks.application_consumers import LinearConsumerConfig, MiniRocketConsumerConfig, TCNConsumerConfig
    from chronaris.evaluation.application_tasks.chapter5_subset_evaluation import simulation_records
    records=simulation_records(plan['subset_root'])['records']
    reference=next(r for r in records if r['method']=='chronaris' and r['candidate_name']=='reference')
    zero=next(r for r in records if r['candidate_name']=='no_physics_residual')
    selected=[(0.,zero['checkpoints']),(.05,reference['checkpoints'])]
    for i,weight in ((30,.025),(33,.1)):
        result=read(Path(root)/'extra_units'/f'{i}.json');verify(result['files'])
        selected.append((weight,{r:result['training'][r+'_training']['best_checkpoint_path'] for r in ('self_supervised','task_guided')}))
    rows,files=[],{}
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu')
        data,fold=load_v4_simulation_development(simulation_root=plan['simulation_root'],registry_path=SIMULATION_REGISTRY)
        targets=build_guarded_application_consumer_targets(data,
            completed_pretraining_checkpoints=reference_initializations(records,17),smoke_only=False)
        for weight,checkpoints in sorted(selected):
            for route,checkpoint in checkpoints.items():
                destination=Path(root)/'sensitivity'/str(weight)/route
                receipt=destination/'result.json'
                if receipt.exists():
                    record=read(receipt);verify(record['files']);rows.append(record);files[str(receipt)]=sha(receipt);continue
                encoder,normalizer,_=load_frozen_application_encoder(checkpoint,route=route,fold=fold,device='cuda')
                outputs=export_loaded_application_encoder(encoder=encoder,normalizer=normalizer,checkpoint=checkpoint,
                    provider=lambda ids:select_observation_batch(data.batch,ids),fold=fold,root=destination/'representations',
                    export_roles=('train','validation'),export_prefix='sensitivity',label_used_for_encoder_training=route=='task_guided')
                result=run_application_method_consumers(method_name='chronaris',outputs=outputs,targets=targets,
                    output_root=destination/'consumers',fold_id=fold.fold_id,protocol=ApplicationConsumerProtocol(
                        linear=LinearConsumerConfig(random_state=17,tune_on_validation=True),
                        minirocket=MiniRocketConsumerConfig(random_state=17,tune_on_validation=True),
                        tcn=TCNConsumerConfig(epochs=40,patience=6,seed=17,device='cuda'),label_used_for_encoder_training=route=='task_guided'))
                evidence={str(p):sha(p) for p in destination.rglob('*') if p.is_file() and p.name!='heartbeat.json'}
                evidence[str(checkpoint)]=sha(checkpoint)
                record=dict(weight=weight,route=route,role='validation',result=asdict(result),files=evidence)
                write(receipt,record);rows.append(record);files[str(receipt)]=sha(receipt)
                del encoder;torch.cuda.empty_cache()
    return write(Path(root)/'sensitivity/result.json',dict(status='completed',rows=rows,files=files,
        configuration_selection=False,confirmation_opened=False,weights=plan['sensitivity_weights']))


def diagnostics(root,plan,models):
    from chronaris.modeling.training import TrainedFusionAdapter
    from chronaris.evaluation.application_tasks.v4_naive_baseline import load_v4_naive_encoder
    spec=importlib.util.spec_from_file_location('delivery_auditor',Path(__file__).with_name('v4_correctness.py'))
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    rows,timings=[],[]
    with development_gpu_lock() as acquired:
        if not acquired:return dict(status='waiting_gpu')
        data,fold=load_v4_simulation_development(simulation_root=plan['simulation_root'],registry_path=SIMULATION_REGISTRY)
        ids=fold.validation_sample_ids
        positions=torch.linspace(0,len(ids)-1,32).round().long().tolist()
        batch=select_observation_batch(data.batch,tuple(ids[i] for i in positions))
        for unit in models['records']:
            for route,checkpoint in unit['checkpoints'].items():
                identity=dict(method=unit['method'],candidate=unit['candidate_name'],route=route,seed=17)
                if unit['method']=='chronaris' and unit['candidate_name']=='reference':
                    for cutoff in (5.,15.,25.):
                        audit=auditor.audit_checkpoint_causality(checkpoint,batch,device='cuda',cutoff_s=cutoff,include_query_deltas=True)
                        if not audit['passed']:raise ValueError('full reference failed qualified causality audit')
                        rows.append(identity|audit)
                if unit['method']=='naive_time_sync':
                    encoder,normalizer,_=load_v4_naive_encoder(checkpoint,fold=fold,data_manifest_sha256=unit['data_manifest_sha256'],
                        expected_training_source_sha256=unit['training_source_sha256'])
                else:encoder,normalizer,_=load_frozen_application_encoder(checkpoint,route=route,fold=fold,device='cuda')
                adapter=TrainedFusionAdapter(encoder=encoder,normalizer=normalizer,fold_id=fold.fold_id,checkpoint_sha256=sha(checkpoint))
                timed_batch=select_observation_batch(batch,batch.sample_ids[:4])
                for _ in range(2):adapter(timed_batch)
                for repeat in range(10):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                    adapter(timed_batch);torch.cuda.synchronize()
                    timings.append(identity|dict(repeat=repeat,batch_size=4,seconds=time.perf_counter()-started,
                        max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved(),
                        device=torch.cuda.get_device_name(),scope='four_validation_windows_end_to_end_adapter_excludes_downstream'))
                del encoder,adapter;torch.cuda.empty_cache()
    path=Path(root)/'diagnostics/result.json'
    return write(path,dict(status='completed',causality=rows,timings=timings,confirmation_opened=False,files={}))
