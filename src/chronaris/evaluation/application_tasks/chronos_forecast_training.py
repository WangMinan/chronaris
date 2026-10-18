"""Bounded full-parameter native forecast fitting with exact minibatch resume."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import predict_forecast
from chronaris.evaluation.application_tasks.chronos_information_run import atomic_json
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json,save_rows
from chronaris.evaluation.application_tasks.performance_pilot_training import atomic_checkpoint
from chronaris.modeling.fusion_encoders.chronos_forecast import load_forecast_model,forecast_point
from chronaris.modeling.training.rng import isolated_training_rng,capture_rng_state,restore_rng_state,canonical_training_state_sha256


def fit_forecast(assets,data,x,mask,proof,root,report,name,binding,deadline):
    root,report=Path(root),Path(report);root.mkdir(parents=True,exist_ok=True)
    receipt_path=report/(name+'_receipt.json');fit=data.indices('fit');select=data.indices('select')
    metadata=dict(binding=binding,fold=data.fold,proof=proof,normalizer=data.normalizer,target_scale=data.target_scale,
                  fit_sample_ids=[data.rows[i]['sample_id'] for i in fit])
    metadata=json.loads(json.dumps(metadata));last=root/'last.pt';best_path=root/'best.pt'
    if receipt_path.exists():
        r=json.loads(receipt_path.read_text())
        if r['complete']:
            assert r['metadata']==metadata and file_digest(r['checkpoint_path'])==r['checkpoint_sha256']
            assert file_digest(r['last_path'])==r['last_sha256'];return r
    started=time.monotonic();previous_elapsed=0.
    with isolated_training_rng(17):
        model=load_forecast_model(assets)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=.01)
        center=data.normalizer['vehicle']['center'][11];scale=data.normalizer['vehicle']['scale'][11]
        epoch=1;offset=0;updates=0;loss_sum=0.;seen=0;best=float('inf');best_epoch=0;history=[];steps=[];orders={};best_sha=None
        if last.exists():
            saved=torch.load(last,map_location='cpu',weights_only=True);assert saved['metadata']==metadata
            assert file_digest(best_path)==saved['best_sha256']
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            epoch,offset,updates=saved['epoch'],saved['next_offset'],saved['updates']
            loss_sum,seen=saved['loss_sum'],saved['seen'];best,best_epoch=saved['best'],saved['best_epoch']
            history,steps,orders=saved['history'],saved['steps'],saved['orders'];best_sha=saved['best_sha256']
            previous_elapsed=saved.get('elapsed_s',0.)
            restore_rng_state(saved['rng']);del saved
        else:
            initial=predict_forecast(model,data,x,mask,select,deadline)
            rmse=float(np.sqrt(np.mean((initial-data.y[select])**2)));best=rmse/data.target_scale
            history.append(dict(epoch=0,updates=0,selection_rmse=rmse,scaled_rmse=best,best_epoch=0,training_mse=None))
            atomic_checkpoint(best_path,dict(model=model.state_dict(),epoch=0,updates=0,metadata=metadata))
            best_sha=file_digest(best_path)
            np.savez_compressed(root/'zero_selection.npz',prediction=initial,ids=select)
        def checkpoint():
            atomic_checkpoint(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng_state(),
                metadata=metadata,epoch=epoch,next_offset=offset,updates=updates,loss_sum=loss_sum,seen=seen,
                best=best,best_epoch=best_epoch,best_sha256=best_sha,history=history,steps=steps,orders=orders,
                elapsed_s=previous_elapsed+time.monotonic()-started))
            save_rows(report/(name+'_history.csv'),history)
            pd.DataFrame(steps).to_csv(report/(name+'_steps.csv.gz'),index=False)
        status='budget_incomplete';last_progress=0.
        if history[-1]['epoch']>=2 and history[-1]['epoch']-best_epoch>=2:status='early_stopped'
        elif history[-1]['epoch']>=5:status='max_epochs'
        try:
            while epoch<=5 and status=='budget_incomplete':
                order=np.random.default_rng(17+epoch).permutation(fit)
                digest=canonical_training_state_sha256([data.rows[i]['sample_id'] for i in order])
                if str(epoch) in orders:assert orders[str(epoch)]==digest
                orders[str(epoch)]=digest;model.train()
                while offset<len(order):
                    if time.time()>=deadline:raise TimeoutError('native adaptation deadline')
                    ids=order[offset:offset+4];optimizer.zero_grad(set_to_none=True)
                    pred=forecast_point(model,x[ids],mask[ids],center,scale)
                    target=torch.as_tensor(data.y[ids],device=pred.device,dtype=pred.dtype)
                    weight=data.weights[ids].to(pred.device)
                    loss=(((pred-target)/data.target_scale).square()*weight).mean()
                    if not torch.isfinite(loss):raise ValueError('nonfinite native adaptation loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();updates+=1;value=float(loss.detach());loss_sum+=value*len(ids);seen+=len(ids)
                    steps.append(dict(epoch=epoch,offset=offset,count=len(ids),update=updates,loss=value,gradient_norm=float(norm)))
                    offset+=len(ids)
                    if updates%1000==0:checkpoint()
                    if time.monotonic()-last_progress>=10:
                        atomic_json(report/'training_progress.json',dict(status='running',stage=name,epoch=epoch,updates=updates,
                            samples_in_epoch=offset,total_samples=len(order),elapsed_s=time.monotonic()-started))
                        last_progress=time.monotonic()
                prediction=predict_forecast(model,data,x,mask,select,deadline)
                rmse=float(np.sqrt(np.mean((prediction-data.y[select])**2)));score=rmse/data.target_scale
                if score<best-1e-6:
                    best,best_epoch=score,epoch
                    atomic_checkpoint(best_path,dict(model=model.state_dict(),epoch=epoch,updates=updates,metadata=metadata))
                    best_sha=file_digest(best_path)
                assert seen==len(fit)
                history.append(dict(epoch=epoch,updates=updates,selection_rmse=rmse,scaled_rmse=score,best_epoch=best_epoch,
                                    training_mse=loss_sum/seen))
                print(name,json.dumps(history[-1]),flush=True)
                stopped=epoch>=2 and epoch-best_epoch>=2
                if stopped:status='early_stopped'
                elif epoch==5:status='max_epochs'
                epoch+=1;offset=0;loss_sum=0.;seen=0;checkpoint()
                if status!='budget_incomplete':break
        except TimeoutError:
            checkpoint()
        except BaseException:
            checkpoint();raise
        receipt=dict(name=name,status=status,complete=status!='budget_incomplete',epochs=history[-1]['epoch'],
                     best_epoch=best_epoch,updates=updates,windows_seen=sum(s['count'] for s in steps),selection_rmse=best*data.target_scale,
                     checkpoint_path=str(best_path.resolve()),checkpoint_sha256=best_sha,last_path=str(last.resolve()),last_sha256=file_digest(last),
                     parameters=sum(p.numel() for p in model.parameters()),elapsed_s=previous_elapsed+time.monotonic()-started,metadata=metadata)
        save_json(receipt_path,receipt)
        del model,optimizer;torch.cuda.empty_cache()
        return receipt
