"""Same supervised trajectory, checkpoint selection by the declared common readout."""
from pathlib import Path
import time
import json
import numpy as np
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_run import fit_readout
from chronaris.evaluation.application_tasks.fair_residual_training import predict_fair,check_history,load_fair_model
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_rows,save_json
from chronaris.evaluation.application_tasks.performance_pilot_training import atomic_checkpoint
from chronaris.modeling.fusion_encoders.fair_residual import FairResidualModel
from chronaris.modeling.training.rng import isolated_training_rng,capture_rng_state


def fit_readout_selected(method,data,root,report,name,binding,deadline):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();fit=data.indices('fit');select=data.indices('select');seed=binding['seed']
    with isolated_training_rng(seed):
        model=FairResidualModel(method).to(data.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        check_history(model,data)
        history=[];best=float('inf');best_epoch=0;updates=0;status='budget_incomplete'
        for epoch in range(1,101):
            if time.monotonic()>=min(deadline,start+3600):break
            model.train();losses=[]
            order=np.random.default_rng(seed+epoch).permutation(fit)
            for offset in range(0,len(order),32):
                ids=order[offset:offset+32];optimizer.zero_grad(set_to_none=True)
                _,correction=model(*data.queries(ids))
                loss=(((correction[:,-1]-data.targets[ids])**2)*data.weights[ids]).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite readout-selected loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;losses.append((float(loss.detach()),len(ids)))
            train_features,_=predict_fair(model,data,fit)
            selected_features,head=predict_fair(model,data,select)
            readout,alpha=fit_readout(data,dict(fit=train_features,select=selected_features))
            pred=readout.predict(selected_features)
            rmse=float(np.sqrt(np.mean((pred-data.y[select])**2)));score=rmse/data.target_scale
            improved=score<best-1e-6
            if improved:best,best_epoch=score,epoch
            history.append(dict(model=name,epoch=epoch,updates=updates,
                                training_mse=sum(v*n for v,n in losses)/len(fit),selection_rmse=rmse,
                                selection_head_rmse=float(np.sqrt(np.mean((head[:,-1]-data.y[select])**2))),
                                selected_alpha=alpha,selection_scaled_rmse=score,best_epoch=best_epoch,
                                elapsed_s=time.monotonic()-start))
            payload=dict(format='chronaris.fair_residual.v1',method=method,name=name,seed=seed,epoch=epoch,
                         best_epoch=best_epoch,updates=updates,target_scale=data.target_scale,state=model.state_dict(),
                         optimizer=optimizer.state_dict(),rng=capture_rng_state(),normalizer=data.normalizer,
                         fold=data.fold,binding=binding)
            atomic_checkpoint(root/'last.pt',payload)
            if improved:atomic_checkpoint(root/'best.pt',payload)
            save_rows(Path(report)/f'{name}_history.csv',history);print(json.dumps(history[-1]),flush=True)
            if epoch>=20 and epoch-best_epoch>=15:status='early_stopped';break
            if epoch==100:status='max_epochs'
        if not history:raise TimeoutError('no complete epoch')
        restored=load_fair_model(root/'best.pt',data)
        features={role:predict_fair(restored,data,data.indices(role))[0] for role in ('fit','select')}
        readout,alpha=fit_readout(data,features)
        actual=float(np.sqrt(np.mean((readout.predict(features['select'])-data.y[select])**2)))
        assert abs(actual/data.target_scale-best)<1e-8
        receipt=dict(name=name,method=method,status=status,complete=status!='budget_incomplete',
                     epochs=history[-1]['epoch'],best_epoch=best_epoch,updates=updates,selection_rmse=actual,
                     checkpoint_path=str((root/'best.pt').resolve()),checkpoint_sha256=file_digest(root/'best.pt'),
                     parameter_count=sum(x.numel() for x in restored.parameters()),elapsed_s=time.monotonic()-start,
                     history_check=check_history(restored,data),selection_policy='common_readout',selected_alpha=alpha)
        save_json(Path(report)/f'{name}_receipt.json',receipt)
        del model,optimizer
        return restored,receipt
