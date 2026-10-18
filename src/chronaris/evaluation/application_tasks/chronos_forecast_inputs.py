"""Same expanded roles, exact historical anchor, and all original observation ages."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FairResidualData
from chronaris.modeling.fusion_encoders.chronos_forecast import forecast_point
from chronaris.modeling.training.rng import canonical_training_state_sha256


INPUT_REPORT=Path('docs/review/stage/eligible-fit-20261019')
VIEWS=('aircraft','joint')


def forecast_folds():return json.loads((INPUT_REPORT/'folds.json').read_text())


def forecast_data(spec,view):
    if view not in VIEWS:raise ValueError(view)
    receipt=INPUT_REPORT/(spec['name']+'_qualification.json');q=json.loads(receipt.read_text())
    data=FairResidualData(q['snapshot_path'],receipt,spec,'cpu')
    with np.load(q['snapshot_path'],allow_pickle=False) as raw:
        anchor=raw['anchor'];latest=raw['vehicle'][:,:,11];valid=raw['vehicle_mask'][:,:,11]
        assert (anchor>=0).all() and np.isfinite(anchor).all()
        zero=anchor==0
        # A valid interval mean containing its positive latest point cannot be zero.
        assert not (zero&valid&(latest<=0)).any()
        anchor_mask=torch.tensor(~zero)
        assert anchor_mask[:,-1].all()
    streams=[data.streams['vehicle']] if view=='aircraft' else [data.streams[s] for s in ('physiology','vehicle')]
    values=torch.cat([s[0] for s in streams],-1);mask=torch.cat([s[1] for s in streams],-1)
    ages=torch.cat([s[2] for s in streams],-1)
    center=data.normalizer['vehicle']['center'][11];scale=data.normalizer['vehicle']['scale'][11]
    history=((data.anchor-center)/scale).masked_fill(~anchor_mask,0)
    x=torch.cat((values,ages,history[:,:,None]),-1);m=torch.cat((mask,mask,anchor_mask[:,:,None]),-1)
    assert x.shape[-1]==(25 if view=='aircraft' else 57) and torch.isfinite(x).all()
    proof=dict(snapshot_sha256=file_digest(q['snapshot_path']),view=view,channels=x.shape[-1],
               missing_historical_means=int(zero.sum()),ambiguous_zero_means=0,center=center,scale=scale,
               input_sha256=canonical_training_state_sha256(x,m,[r['sample_id'] for r in data.rows],data.normalizer))
    return data,x,m,proof


@torch.inference_mode()
def predict_forecast(model,data,x,mask,ids,deadline=float('inf')):
    permitted=np.r_[data.roles['fit'],data.roles['select'],data.roles['score'] if data.score_open else []]
    if not np.isin(ids,permitted).all():raise PermissionError('score role remains sealed')
    center=data.normalizer['vehicle']['center'][11];scale=data.normalizer['vehicle']['scale'][11]
    model.eval();parts=[]
    for off in range(0,len(ids),4):
        if time.time()>=deadline:raise TimeoutError('forecast inference deadline')
        batch=ids[off:off+4];parts.append(forecast_point(model,x[batch],mask[batch],center,scale).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)
