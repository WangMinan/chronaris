"""Bounded frozen-checkpoint forward/gradient diagnostics; no optimizer steps."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audit_statistics import DATA, HERE, REPO, digest
from chronaris.evaluation.application_tasks.application_finetuning_export import load_frozen_application_encoder
from chronaris.evaluation.application_tasks.stage45_diagnostics import branch_features
from chronaris.evaluation.application_tasks.stage45b_diagnostics import checkpoint_gradients
from chronaris.evaluation.application_tasks.v4_development_data import load_development_inputs
from chronaris.evaluation.application_tasks.v4_grouped_consumers import native_consumer_context
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.single_stream import move_observation_batch
from chronaris.representation.contracts import pool_exported_sequence
from chronaris.models.alignment.cuda_recurrence import ordinary_recurrence

OUT=HERE/'forward'
ROOT=REPO/'artifacts/application_evaluation/2026-09-08_v4-public-confirmation-prepared'
REGISTRY=REPO/'docs/requirements/thesis-v4-public-subjects.json'


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with (DATA/'inventory.csv').open() as f:inventory=list(csv.DictReader(f))
    checkpoints={};records=[];gradients=[];logs=[]
    with development_gpu_lock() as acquired:
        if not acquired:raise RuntimeError('GPU lock unavailable; no fallback or preemption')
        for domain in ('clare','cogpilot'):
            assert (ROOT/domain/'prepared_cache_index.json').is_file()
            provider,_,fold,_,_,targets,definitions,data=load_development_inputs(domain,ROOT,REGISTRY,fold_index=0,subject_role='confirmation')
            context=native_consumer_context(domain,data,fold)
            index={s:i for i,s in enumerate(targets.sample_ids)}
            batches=[]
            # Two train subjects, two task-supported batches each; selection ignores outcomes.
            for group in sorted({context['groups'][s] for s in fold.train_sample_ids})[:2]:
                for task in definitions:
                    eligible=sorted((s for s in fold.train_sample_ids if context['groups'][s]==group and bool(targets.valid_masks[task.name][index[s]].any())),key=lambda s:hashlib.sha256(s.encode()).hexdigest())
                    for offset in (0,4):
                        batch=tuple(eligible[offset:offset+4]);assert len(batch)==4
                        if batch not in batches:batches.append(batch)
            ids=tuple(sorted(fold.validation_sample_ids,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:16])
            for unit in inventory:
                if unit['domain']!=domain or unit['method']!='chronaris' or unit['fold_index']!='0':continue
                receipt=json.loads(Path(unit['receipt']).read_text())
                for r in receipt['results']:
                    route=r['route'];path=Path(r['training']['best_checkpoint_path']);checkpoints[str(path)]=digest(path)
                    encoder,normalizer,payload=load_frozen_application_encoder(path,route=route,fold=fold,device='cuda')
                    encoder.eval();parts={};gates=[];attention=[]
                    with torch.inference_mode(), ordinary_recurrence(encoder,True):
                        for offset in range(0,len(ids),4):
                            output=encoder(move_observation_batch(normalizer.transform(provider(ids[offset:offset+4])),device='cuda'))
                            fusion=output.auxiliary['fusion_output']
                            valid=fusion.scale_available_mask.any(-1)&output.auxiliary['alignment_output'].physiology.reference_valid_mask
                            gates.extend(fusion.cross_gate[valid].flatten().cpu().tolist())
                            for a,mask in zip(fusion.attention_weights,fusion.lag_masks):
                                entropy=-(a.clamp_min(1e-30).log()*a).sum(-1);n=mask.sum(-1);keep=n>1
                                attention.extend((entropy[keep]/n[keep].float().log()).cpu().tolist())
                            for name,(sequence,valid) in branch_features(output,encoder).items():
                                parts.setdefault(name,[]).append(pool_exported_sequence(sequence.cpu(),valid.cpu()).numpy())
                    stats={}
                    for name,values in parts.items():
                        a=np.concatenate(values);sv=np.linalg.svd(a-a.mean(0),compute_uv=False);e=sv**2
                        stats[name]=dict(dim=a.shape[1],effective_rank=float(e.sum()**2/(e@e)) if e.sum()>0 else 0.,mean_norm=float(np.linalg.norm(a,axis=1).mean()))
                    records.append(dict(domain=domain,variant=unit['variant'],route=route,checkpoint=str(path),
                        sample_ids=ids,role='validation',gate_quantiles=np.quantile(gates,[0,.1,.5,.9,1]).tolist(),
                        attention_entropy_over_uniform_mean=float(np.mean(attention)),branches=stats,
                        physics_relations=len((encoder.backbone.config.physics_calibration or {}).get('relations',[]))))
                    del encoder,output;torch.cuda.empty_cache()
                    if unit['variant']=='thesis_reference' and route=='task_guided':
                        gradients.append(dict(domain=domain,result=checkpoint_gradients(checkpoint=path,route=route,
                            provider=provider,fold=fold,targets=targets,definitions=definitions,
                            output_root={'path':str(OUT/domain),'context':context},sample_batches=batches)))
                    # Last checkpoint stores the whole actual trajectory, including after selected best.
                    last=Path(r['training']['last_checkpoint_path'])
                    if not last.is_file():
                        logs.append(dict(domain=domain,variant=unit['variant'],route=route,status='last_checkpoint_not_present',path=str(last)))
                        print('checked',domain,unit['variant'],route,flush=True)
                        continue
                    checkpoints[str(last)]=digest(last)
                    saved=torch.load(last,map_location='cpu',weights_only=True)
                    if route=='task_guided':
                        rows=[x for x in saved['update_rows'] if x['stage']=='joint_adaptation']
                        terms={}
                        for row in rows:
                            for term in row.get('mechanism_terms',[]):terms.setdefault(term['term_name'],[]).append(term.get('weighted_loss') or 0.)
                        logs.append(dict(domain=domain,variant=unit['variant'],route=route,
                            best_update=saved['best_update'],updates=saved['optimizer_updates'],
                            mean_task_loss=float(np.mean([x['task_loss'] for x in rows])),
                            mean_public_weighted=float(np.mean([x['public_loss']*x['public_weight'] for x in rows])),
                            mean_mechanism_loss=float(np.mean([x['mechanism_loss'] for x in rows])),
                            weighted_term_means={k:float(np.mean(v)) for k,v in terms.items()},
                            task_heads_standardize_input=False,downstream_standardizes_input=True))
                    del saved
                    print('checked',domain,unit['variant'],route,flush=True)
    for path,expected in checkpoints.items():assert digest(Path(path))==expected
    result=dict(records=records,gradients=gradients,training_logs=logs,checkpoint_hashes=checkpoints,
                checkpoint_files_unchanged=True,optimizer_updates=0,consumer_fits=0,
                sampling='fixed first fold, hash-selected 16 validation windows; training-only task-supported gradient batches',
                gradient_scope='local per-task weighted gradient diagnosis, not exact accumulated optimizer-update replay')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
