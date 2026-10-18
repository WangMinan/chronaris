import torch
from chronaris.modeling.fusion_encoders.chronos_forecast import forecast_point


def test_native_forecast_groups_are_isolated_and_gradients_reach_pretrained_head():
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    torch.set_num_threads(1);torch.manual_seed(17)
    config=Chronos2CoreConfig(d_model=16,d_kv=8,d_ff=32,num_layers=1,num_heads=2,dropout_rate=0,
        chronos_config=dict(context_length=96,input_patch_size=16,input_patch_stride=16,output_patch_size=16,
                            quantiles=[.1,.5,.9],use_reg_token=True,use_arcsinh=True,max_output_patches=1,time_encoding_scale=96))
    model=Chronos2Model(config).eval();x=torch.randn(4,96,3);mask=torch.ones_like(x,dtype=torch.bool)
    expected=forecast_point(model,x,mask,2.,3.)
    changed=x.clone();changed[1:]=changed[1:].flip(1)*17+3
    actual=forecast_point(model,changed,mask,2.,3.)
    torch.testing.assert_close(expected[0],actual[0],atol=1e-5,rtol=1e-5)
    expected.square().mean().backward()
    assert model.output_patch_embedding.output_layer.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_native_training_resume_preserves_optimizer_rng_and_window_order(tmp_path,monkeypatch):
    import numpy as np
    from chronos.chronos2.config import Chronos2CoreConfig
    from chronos.chronos2.model import Chronos2Model
    from chronaris.evaluation.application_tasks import chronos_forecast_training as train
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def build(_):
        config=Chronos2CoreConfig(d_model=16,d_kv=8,d_ff=32,num_layers=1,num_heads=2,dropout_rate=.1,
            chronos_config=dict(context_length=96,input_patch_size=16,input_patch_stride=16,output_patch_size=16,
                                quantiles=[.1,.5,.9],use_reg_token=True,use_arcsinh=True,max_output_patches=1,time_encoding_scale=96))
        return Chronos2Model(config)
    monkeypatch.setattr(train,'load_forecast_model',build)
    class Data:
        roles=dict(fit=np.arange(8),select=np.arange(8,10),score=np.arange(10,12))
        score_open=False;target_scale=1.;fold=dict(name='tiny',fit=['a'],select=['b'],score=['c'])
        rows=[dict(sample_id=str(i)) for i in range(12)]
        normalizer=dict(vehicle=dict(center=[0.]*12,scale=[1.]*12))
        weights=torch.ones(12);y=np.zeros(12)
        def indices(self,role):
            if role=='score':raise PermissionError('sealed')
            return self.roles[role]
    data=Data();g=torch.Generator().manual_seed(99);x=torch.randn(12,96,3,generator=g);mask=torch.ones_like(x,dtype=torch.bool)
    a,b=tmp_path/'whole',tmp_path/'resumed';a.mkdir();b.mkdir()
    whole=train.fit_forecast({},data,x,mask,{},a/'weights',a,'tiny',dict(seed=17),float('inf'))
    original=train.forecast_point;calls=0
    def interrupt_before_update(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError('test checkpoint boundary')
        return original(*args,**kwargs)
    monkeypatch.setattr(train,'forecast_point',interrupt_before_update)
    partial=train.fit_forecast({},data,x,mask,{},b/'weights',b,'tiny',dict(seed=17),float('inf'))
    assert not partial['complete'] and partial['updates']==2
    monkeypatch.setattr(train,'forecast_point',original)
    resumed=train.fit_forecast({},data,x,mask,{},b/'weights',b,'tiny',dict(seed=17),float('inf'))
    assert whole['updates']==resumed['updates'] and whole['best_epoch']==resumed['best_epoch']
    left=torch.load(whole['last_path'],weights_only=True);right=torch.load(resumed['last_path'],weights_only=True)
    assert left['steps']==right['steps'] and left['history']==right['history'] and left['orders']==right['orders']
    for key in left['model']:torch.testing.assert_close(left['model'][key],right['model'][key],atol=0,rtol=0,equal_nan=True)
