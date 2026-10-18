"""Input and zero-update hardware checks before native supervised adaptation."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.chronos_forecast_inputs import forecast_data,forecast_folds,VIEWS
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.chronos_forecast import load_forecast_model,forecast_point
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.training.rng import isolated_training_rng


def qualify_forecast_inputs(report):
    report=Path(report)
    if (report/'input_qualification.json').exists():raise FileExistsError('retained native input qualification')
    proofs=[]
    for spec in forecast_folds():
        for view in VIEWS:
            data,x,mask,proof=forecast_data(spec,view)
            proofs.append(dict(fold=spec['name'],fit_subjects=spec['fit'],fit_windows=len(data.indices('fit')),
                               windows=len(data.rows),**proof))
            del data,x,mask
    result=dict(status='passed',proofs=proofs,optimizer_updates=0,outer_scored=False)
    save_json(report/'input_qualification.json',result);return result


def qualify_forecast_hardware(report):
    report=Path(report)
    if (report/'qualification.json').exists():raise FileExistsError('retained hardware qualification')
    inputs=json.loads((report/'input_qualification.json').read_text());assert inputs['status']=='passed'
    assets=verify_assets(ASSETS);checks=[];torch.set_num_threads(1)
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        for spec in forecast_folds():
            for view in VIEWS:
                data,x,mask,proof=forecast_data(spec,view)
                prior=next(p for p in inputs['proofs'] if p['fold']==spec['name'] and p['view']==view)
                assert all(prior[k]==v for k,v in proof.items())
                ids=data.indices('fit')[:4];center,scale=proof['center'],proof['scale']
                with isolated_training_rng(17):
                    model=load_forecast_model(assets).eval();torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        expected=forecast_point(model,x[ids],mask[ids],center,scale)
                        changed=x[ids].clone();changed[1:]=changed[1:].flip(1)*17+3
                        actual=forecast_point(model,changed,mask[ids],center,scale)
                        torch.testing.assert_close(expected[0],actual[0],atol=1e-5,rtol=1e-5)
                    model.train();started=time.monotonic()
                    pred=forecast_point(model,x[ids],mask[ids],center,scale)
                    target=torch.tensor(data.y[ids],dtype=pred.dtype,device=pred.device)
                    loss=(((pred-target)/data.target_scale).square()*data.weights[ids].to(pred.device)).mean()
                    assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    assert model.output_patch_embedding.output_layer.weight.grad is not None
                    torch.cuda.synchronize();seconds=time.monotonic()-started
                    parameters=sum(p.numel() for p in model.parameters());peak=torch.cuda.max_memory_allocated()
                    model.zero_grad(set_to_none=True);del model,pred;torch.cuda.empty_cache()
                    fresh=load_forecast_model(assets).eval()
                    with torch.no_grad():replay=forecast_point(fresh,x[ids],mask[ids],center,scale)
                    torch.testing.assert_close(expected,replay,atol=1e-5,rtol=1e-5)
                    checks.append(dict(fold=spec['name'],view=view,channels=proof['channels'],parameters=parameters,
                                       gradient_norm=float(norm),loss=float(loss.detach()),forward_backward_seconds=seconds,
                                       peak_allocated_bytes=peak,batch_isolation_delta=float((expected[0]-actual[0]).abs()),
                                       fresh_reload_delta=float((expected-replay).abs().max())))
                    del fresh,loss;torch.cuda.empty_cache()
                del data,x,mask;print(spec['name'],view,'qualified',flush=True)
    verify_assets(ASSETS)
    code=[Path(__file__),Path(__file__).with_name('chronos_forecast_inputs.py'),Path(__file__).with_name('fair_residual_inputs.py'),
          Path(__file__).parents[2]/'modeling/fusion_encoders/chronos_forecast.py',Path(__file__).parents[2]/'modeling/training/rng.py']
    result=dict(status='passed',checks=checks,input_qualification_sha256=file_digest(report/'input_qualification.json'),
                assets_path=ASSETS,assets_sha256=file_digest(ASSETS),source_files={str(p.resolve()):file_digest(p) for p in code},
                optimizer_updates=0,device=torch.cuda.get_device_name())
    save_json(report/'qualification.json',result);return result
