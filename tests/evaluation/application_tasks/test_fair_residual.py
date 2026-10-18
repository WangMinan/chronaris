import json
import numpy as np
import pytest
import torch

from chronaris.dataset.cogpilot_performance import file_digest
from chronaris.evaluation.application_tasks.fair_residual_inputs import FairResidualData, FOLDS
from chronaris.evaluation.application_tasks.performance_pilot_data import future_query_variants
from chronaris.modeling.fusion_encoders.causal_query import CausalQueryStream
from chronaris.modeling.fusion_encoders.fair_residual import FairResidualModel
from chronaris.evaluation.application_tasks.fair_residual_verify import verify_fair_residual


def stream(features):
    x = torch.randn(2, 96, features)
    mask = torch.ones_like(x, dtype=torch.bool)
    t = torch.arange(1, 97, dtype=torch.float32)[None,:].expand(2,-1)*30/96
    return CausalQueryStream(x, mask, mask.any(-1), torch.full_like(x,.01), t, torch.zeros_like(x,dtype=torch.long))


def test_matching_capacity_zero_correction_and_history():
    torch.set_num_threads(1);torch.manual_seed(17)
    candidate = FairResidualModel('candidate').eval()
    control = FairResidualModel('control').eval();control.load_state_dict(candidate.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) == sum(p.numel() for p in control.parameters())
    p,v = stream(16),stream(12)
    for model in (candidate,control):
        s,delta=model(p,v)
        assert delta.abs().max()==0
        torch.testing.assert_close(s[...,:4],v.values[...,8:12])
        for a,b in zip(future_query_variants(p),future_query_variants(v)):
            actual,_=model(a,b)
            torch.testing.assert_close(actual[:, :48],s[:, :48],atol=1e-5,rtol=1e-5)
    modified,_=future_query_variants(p,cutoff=-1)
    torch.testing.assert_close(control(p,v)[0],control(modified,v)[0],atol=0,rtol=0)


def test_fold_normalization_sealing_and_crossfit_roles(tmp_path):
    spec=FOLDS[0]
    subjects=spec['fit']+spec['select']+spec['score']
    rows=[dict(sample_id=s,subject=s,target=float(i+1),persistence=float(i),accepted=True) for i,s in enumerate(subjects)]
    arrays={}
    for name,n in [('physiology',16),('vehicle',12)]:
        arrays[name]=np.arange(7*96*n,dtype=np.float32).reshape(7,96,n)/1000
        arrays[name+'_mask']=np.ones((7,96,n),bool);arrays[name+'_age']=np.ones((7,96,n),np.float32)*.01
    arrays['anchor']=np.zeros((7,96));arrays['metadata']=np.array(json.dumps({'windows':rows}))
    path=tmp_path/'inputs.npz';np.savez_compressed(path,**arrays)
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(dict(status='passed',gates={'test':True},snapshot_sha256=file_digest(path))))
    data=FairResidualData(path,receipt,spec,'cpu')
    with pytest.raises(PermissionError):data.indices('score')
    with pytest.raises(PermissionError):data.queries(data.roles['score'])
    inner=dict(name='inner',fit=spec['fit'][1:],select=spec['select'],score=spec['fit'][:1])
    oof=FairResidualData(path,receipt,inner,'cpu')
    assert subjects[0] not in oof.normalizer['fit_subjects']
    oof.score_open=True
    assert set(oof.subjects[oof.indices('score')])=={subjects[0]}
    with pytest.raises(PermissionError):oof.queries(data.roles['score'])


def test_verification_fails_on_changed_frozen_source(tmp_path):
    original=tmp_path/'original.txt';original.write_text('frozen')
    (tmp_path/'protected_sources.json').write_text(json.dumps({str(original):file_digest(original)}))
    original.write_text('changed')
    with pytest.raises(AssertionError):verify_fair_residual(tmp_path)
