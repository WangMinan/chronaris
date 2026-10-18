"""Zero-update real-input qualification for the pinned pretrained representation."""
import json
from pathlib import Path
import numpy as np
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.development_comparison import ASSETS
from chronaris.evaluation.application_tasks.fair_residual_inputs import FOLDS,FairResidualData
from chronaris.evaluation.application_tasks.performance_pilot_audit import save_json
from chronaris.evaluation.application_tasks.v4_public_screen import development_gpu_lock
from chronaris.modeling.fusion_encoders.recent_models import verify_assets
from chronaris.modeling.fusion_encoders.chronos_information import ChronosInformationEncoder
from chronaris.modeling.training.rng import isolated_training_rng


VIEW_COLUMNS=dict(aircraft=list(range(16,28)),physiology_history=list(range(16))+list(range(24,28)),joint=list(range(28)))


def information_data(spec):
    root=Path('docs/review/stage/shape-mult-20261019')
    q=root/(spec['name']+'_qualification.json')
    data=FairResidualData(json.loads(q.read_text())['snapshot_path'],q,spec,'cpu')
    p,pm,_=data.streams['physiology'];v,vm,_=data.streams['vehicle']
    return data,torch.cat((p,v),-1),torch.cat((pm,vm),-1)


def qualify_chronos_information(report):
    report=Path(report)
    if (report/'qualification.json').exists():raise FileExistsError('retained qualification')
    assets=verify_assets(ASSETS);torch.set_num_threads(1);checks=[]
    with development_gpu_lock() as acquired:
        if not acquired:raise BlockingIOError('GPU busy')
        for spec in FOLDS:
            data,x,m=information_data(spec);ids=data.indices('fit')[:2]
            for view,columns in VIEW_COLUMNS.items():
                values=x[ids][:,:,columns];mask=m[ids][:,:,columns]
                with isolated_training_rng(17):
                    encoder=ChronosInformationEncoder(assets,len(columns)).eval()
                    original=encoder(values,mask)
                    changed=values.clone();changed[1]+=123
                    actual=encoder(changed,mask)
                    np.testing.assert_allclose(original[0],actual[0],atol=1e-5,rtol=1e-5)
                    encoder.train();encoder.requires_grad_(True)
                    loss=encoder.history_loss(values[:1],mask[:1]);assert torch.isfinite(loss)
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True)
                    assert any(p.grad is not None for p in encoder.parameters())
                    encoder.zero_grad(set_to_none=True);encoder.eval()
                    del encoder;torch.cuda.empty_cache()
                    fresh=ChronosInformationEncoder(assets,len(columns)).eval()
                    replay=fresh(values,mask)
                    np.testing.assert_allclose(original,replay,atol=1e-5,rtol=1e-5)
                    checks.append(dict(fold=spec['name'],view=view,channels=len(columns),features=original.shape[1],
                                       history_loss=float(loss.detach()),gradient_norm=float(norm),
                                       parameters=sum(p.numel() for p in fresh.parameters()),
                                       batch_isolation_delta=float((original[0]-actual[0]).abs().max()),
                                       fresh_reload_delta=float((original-replay).abs().max())))
                    del fresh,loss;torch.cuda.empty_cache()
                print(spec['name'],view,'qualified',flush=True)
    verify_assets(ASSETS)
    receipt=dict(status='passed',checks=checks,assets_path=ASSETS,assets_sha256=file_digest(ASSETS),
                 optimizer_updates=0,forward_backward_checks=6,device=torch.cuda.get_device_name())
    save_json(report/'qualification.json',receipt)
    return receipt
